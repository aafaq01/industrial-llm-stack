"""OpenAI-compatible server for a prebuilt TensorRT-LLM 0.14.0 engine, with Prometheus metrics at /metrics.

TensorRT-LLM 0.14.0 is the last release whose support matrix lists Turing (SM75). It has no trtllm-serve (added in
0.15.0); its OpenAI server is examples/apps/openai_server.py in the source tree. The notebook downloads that file at
tag v0.14.0 (sha256 checked); this launcher imports a copy with one patch (4.) and adds:
1. ignore_eos: 0.14's request models accept ignore_eos but to_sampling_params() drops it, and SamplingParams has no
   ignore_eos until 0.15. When a request sets it, end_id and pad_id become -1, which is what 0.14's own trtllm-bench
   does (tensorrt_llm/bench/benchmark/throughput.py) to always decode max_tokens.
2. /metrics (prometheus_client): HTTP requests by path and status, and prompt tokens submitted. 0.14's LLM API
   exposes no iteration or KV cache stats, so there are no engine-side metrics.
3. The upstream entrypoint builds an engine on the fly with max_batch_size 10; this loads the prebuilt engine.
4. Streaming: upstream chat_stream_generator marks a choice finished on the first update whose text diff is empty
   (`else: finish_reason_sent[i] = True`), so an empty diff mid-stream (e.g. a token that decodes to nothing yet)
   silences the rest of the stream while usage still reports every token, and client TPOT/ITL undercount.
   STREAM_PATCH makes that branch `elif output.finish_reason is not None:`, so only a finished sequence ends the
   stream. The load generator still flags truncated streams (benchmark/loadgen/metrics.py stream_truncated).
5. Detokenization skips special tokens (0.14's TransformersTokenizer.decode passes no skip_special_tokens, so a
   trailing <|im_end|> could reach the text). vLLM's default is also skip_special_tokens=True; affects GSM8K text.
KV cache block reuse (prefix caching) is off, as --no-enable-prefix-caching on vLLM. Temperature 0 needs no patch:
the 0.14 runtime maps it to greedy (top_k 1) in runtime/samplingConfig.h.

python serve.py ENGINE_DIR --tokenizer HF_DIR --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --openai-server-dir DIR_WITH_openai_server.py --port 8000 --kv-cache-free-gpu-memory-fraction 0.9
"""

import argparse
import asyncio
import importlib.util
from pathlib import Path

INDENT = " " * 20
STREAM_PATCH = (
    f"{INDENT}else:\n{INDENT}    finish_reason_sent[i] = True\n",
    f"{INDENT}elif output.finish_reason is not None:\n{INDENT}    finish_reason_sent[i] = True\n",
)


def patched_server_source(src):
    """Upstream openai_server.py source with STREAM_PATCH applied; raises if the snippet is not there exactly once."""
    old, new = STREAM_PATCH
    if src.count(old) != 1:
        raise ValueError(
            f"stream patch target found {src.count(old)} times, expected 1"
        )
    return src.replace(old, new)


def load_server_module(server_dir):
    path = Path(server_dir) / "openai_server_patched.py"
    path.write_text(
        patched_server_source((Path(server_dir) / "openai_server.py").read_text())
    )
    spec = importlib.util.spec_from_file_location("openai_server_patched", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("engine_dir")
    p.add_argument(
        "--tokenizer", required=True, help="HF model dir (tokenizer and chat template)"
    )
    p.add_argument("--served-model-name", required=True)
    p.add_argument(
        "--openai-server-dir",
        required=True,
        help="directory holding v0.14.0 openai_server.py",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--kv-cache-free-gpu-memory-fraction", type=float, default=0.9)
    args = p.parse_args()

    openai_server = load_server_module(args.openai_server_dir)  # v0.14.0 + STREAM_PATCH
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
    from tensorrt_llm.hlapi import LLM, KvCacheConfig
    from tensorrt_llm.hlapi.openai_protocol import (
        ChatCompletionRequest,
        CompletionRequest,
    )
    from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer
    from transformers import AutoTokenizer

    for cls in (ChatCompletionRequest, CompletionRequest):

        def to_sampling_params(self, _orig=cls.to_sampling_params):
            sp = _orig(self)
            if self.ignore_eos:
                sp.end_id, sp.pad_id = -1, -1
            return sp

        cls.to_sampling_params = to_sampling_params

    decode = TransformersTokenizer.decode

    def decode_skip_special(self, token_ids, *a, **kw):
        return decode(self, token_ids, *a, **{"skip_special_tokens": True, **kw})

    TransformersTokenizer.decode = decode_skip_special

    llm = LLM(
        args.engine_dir,
        tokenizer=args.tokenizer,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
            enable_block_reuse=False,
        ),
    )
    requests = Counter(
        "trtllm_http_requests", "HTTP requests by path and status", ["path", "status"]
    )
    prompt_tokens = Counter(
        "trtllm_prompt_tokens", "Prompt tokens submitted to the engine"
    )
    generate_async = llm.generate_async

    def counted_generate_async(*a, **kw):
        promise = generate_async(*a, **kw)
        prompt_tokens.inc(len(promise.prompt_token_ids))
        return promise

    llm.generate_async = counted_generate_async
    server = openai_server.OpenaiServer(
        llm=llm,
        model=args.served_model_name,
        hf_tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
    )

    @server.app.middleware("http")
    async def count_requests(request, call_next):
        resp = await call_next(request)
        requests.labels(request.url.path, str(resp.status_code)).inc()
        return resp

    server.app.add_api_route(
        "/metrics",
        lambda: Response(generate_latest(), media_type=CONTENT_TYPE_LATEST),
        methods=["GET"],
    )
    asyncio.run(server(args.host, args.port))


if __name__ == "__main__":
    main()
