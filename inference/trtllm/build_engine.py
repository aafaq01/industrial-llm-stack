"""Build a TensorRT-LLM 0.14.0 FP16 engine for Qwen2.5-1.5B-Instruct on a Tesla T4 (SM75). Runs in the Kaggle
notebook's Python 3.10 venv (notebooks/trtllm-requirements.txt).

1. Untie the embeddings: copy the HF checkpoint with lm_head.weight = model.embed_tokens.weight and
   tie_word_embeddings false. TensorRT-LLM 0.13/0.14's Qwen converter fails on tied Qwen2 checkpoints
   ("'NoneType' object has no attribute 'to'", NVIDIA/TensorRT-LLM issue 2388; Qwen2.5-1.5B support came in 0.15).
   The weights are the same tensors, so the model is unchanged.
2. Convert to a TensorRT-LLM checkpoint in FP16 with QWenForCausalLM.from_hugging_face (what
   examples/qwen/convert_checkpoint.py calls). The HF checkpoint is bf16; T4 has no bf16, as in the vLLM runs.
3. trtllm-build, as examples/qwen/README.md (FP16, --gemm_plugin float16) plus --context_fmha disable: the fused
   context attention kernels are not built for SM75 (fmhaRunner.cpp asserts "Unsupported architecture";
   NVIDIA/TensorRT-LLM issue 1515), so context attention runs unfused. The commands are written to --cmd-out.

Two engines from one checkpoint (PROFILES), because unfused context attention needs a plugin workspace that grows
with max_batch_size x max_input_len^2. TensorRT-LLM 0.14's GPTAttentionPluginCommon::getWorkspaceSizeForContext
(gptAttentionCommon.cpp), FP16, 12 query heads, 2 KV heads, head size 128: the qk buffer (2 bytes) plus its float
copy (4 bytes) take 6 x B x 12 x L^2 bytes, plus small terms. For B=32, L=3072, max_num_tokens 8192 it gives
21,978,227,456 bytes, exactly what the v3h build reported against 15,636,037,632 available
(scripts in the commit message; workspace_bytes() below). Hence:
- bench: B=32 (concurrency 32), L=1024 (longest loadgen prompt 893 tokens with the chat template), seq 1024+128:
  2.55 GB workspace.
- gsm8k: B=8, L=2048 (longest GSM8K 5-shot chat prompt 1654 tokens over the 250 items), seq 2048+256: 2.55 GB.
  Batch 8 only queues GSM8K requests (num_concurrent 16); greedy outputs do not depend on batching.
B=32 with L=2048 would need 9.85 GB, leaving too little of a 15 GB T4 for weights (about 3.1 GB) and KV cache.

python build_engine.py HF_DIR WORK_DIR --cmd-out results/server_args/trtllm-build.txt
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

BUILD_ARGS = [
    "--gemm_plugin", "float16",
    "--gpt_attention_plugin", "float16",
    "--context_fmha", "disable",
    "--remove_input_padding", "enable",
    "--kv_cache_type", "paged",
    "--max_num_tokens", "8192",
    "--workers", "1",
]  # fmt: skip
# name: (max_batch_size, max_input_len, max_seq_len)
PROFILES = {"bench": (32, 1024, 1024 + 128), "gsm8k": (8, 2048, 2048 + 256)}
HEADS, KV_HEADS, HEAD_SIZE = 12, 2, 128


def workspace_bytes(batch, input_len, max_num_tokens=8192):
    """Unfused context attention workspace of TensorRT-LLM 0.14 (getWorkspaceSizeForContext), FP16, 256-byte aligned
    buffers. Reproduces the 21,978,227,456 bytes of the failed v3h build for (32, 3072, 8192)."""
    s, b, length, t = 2, batch, input_len, max_num_tokens
    parts = [
        32 * 1024 * 1024,
        s * t * length,
        4 * (b + 1),
        4 * (b + 1),
        4 * (b + 1),
        4 * b * HEAD_SIZE // 2,
    ]
    parts += [
        s * t * HEADS * HEAD_SIZE,
        s * b * length * KV_HEADS * HEAD_SIZE,
        s * b * length * KV_HEADS * HEAD_SIZE,
    ]
    parts += [
        s * b * HEADS * length * length,
        s * t * HEADS * HEAD_SIZE,
        4 * b * HEADS * length * length,
        4 * t,
        4 * t,
    ]
    return sum((p + 255) // 256 * 256 for p in parts)


def untie(src, dst):
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("model") and f.name != "config.json":
            shutil.copy2(f, dst / f.name)
    tensors = {}
    for f in sorted(src.glob("*.safetensors")):
        tensors.update(load_file(f))
    if "lm_head.weight" not in tensors:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    save_file(tensors, dst / "model.safetensors", metadata={"format": "pt"})
    cfg = json.loads((src / "config.json").read_text())
    cfg["tie_word_embeddings"] = False
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))


def convert(untied, ckpt):
    """FP16 TensorRT-LLM checkpoint from the untied HF dir. 0.14's save_checkpoint writes into output_dir without
    creating it (failed on Kaggle 2026-10-10, v3f: "No such file or directory ... /tmp/trtllm/ckpt-fp16/.tmp..."),
    so the dir is created here. The model is deleted inside this function so PretrainedModel.__del__ (which calls
    torch.cuda.ipc_collect) runs while torch is intact, not at interpreter exit."""
    import gc

    from tensorrt_llm.models import QWenForCausalLM

    ckpt.mkdir(parents=True, exist_ok=True)
    model = QWenForCausalLM.from_hugging_face(str(untied), dtype="float16")
    model.save_checkpoint(str(ckpt), save_config=True)
    del model
    gc.collect()
    return sorted(f.name for f in ckpt.iterdir())


def build(ckpt, engine, cmd_out, profile):
    batch, input_len, seq_len = PROFILES[profile]
    engine.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(Path(sys.executable).parent / "trtllm-build"),
        "--checkpoint_dir",
        str(ckpt),
        "--output_dir",
        str(engine),
        *BUILD_ARGS,
        "--max_batch_size",
        str(batch),
        "--max_input_len",
        str(input_len),
        "--max_seq_len",
        str(seq_len),
    ]
    Path(cmd_out).parent.mkdir(parents=True, exist_ok=True)
    with open(cmd_out, "a") as f:
        f.write(
            f"# {profile}: attention workspace {workspace_bytes(batch, input_len)} bytes\n{shlex.join(cmd)}\n"
        )
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("hf_dir")
    p.add_argument("work_dir")
    p.add_argument("--cmd-out", required=True)
    args = p.parse_args()
    hf, work = Path(args.hf_dir), Path(args.work_dir)
    untied, ckpt = work / "hf-untied", work / "ckpt-fp16"

    t = time.time()
    untie(hf, untied)
    print(f"untied checkpoint in {time.time() - t:.0f}s", flush=True)
    t = time.time()
    files = convert(untied, ckpt)
    print(f"converted checkpoint in {time.time() - t:.0f}s: {files}", flush=True)
    Path(args.cmd_out).unlink(missing_ok=True)
    for profile in PROFILES:
        t = time.time()
        build(ckpt, work / f"engine-fp16-{profile}", args.cmd_out, profile)
        print(f"built {profile} engine in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
