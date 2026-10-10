"""Generate the two v3 Kaggle notebooks (docs/PLAN.md items 10 to 12), each in its own push folder:

  notebooks/kaggle-dcgm-vision/  DCGM inside Kaggle + YOLOv8n vision (ONNX Runtime GPU, TensorRT FP16, Triton)
  notebooks/kaggle-trtllm/       TensorRT-LLM 0.14.0 FP16 Qwen2.5-1.5B-Instruct, same loadgen cells as vLLM

Reuses the shared cells and build() from build_notebook.py. Re-run after changing any embedded file, after a commit
(so the embedded GIT_COMMIT is clean).

python notebooks/build_v3.py
"""

import re

from build_notebook import (
    ENV_CELL,
    HELPERS_CELL,
    PREFLIGHT_CELL,
    ROOT,
    build,
    code,
    md,
)

LOADGEN = [
    "benchmark/loadgen/run.py",
    "benchmark/loadgen/metrics.py",
    "benchmark/loadgen/prompts.jsonl",
    "benchmark/accuracy/run_gsm8k.sh",
    "benchmark/accuracy/tasks/gsm8k_pinned.yaml",
    "benchmark/requirements.txt",
]
EMBED_A = (
    LOADGEN
    + [
        "notebooks/vllm-requirements.txt",  # install_vllm in the shared step cell references it
        "notebooks/vision-requirements.txt",
        "benchmark/vision/run_vision.py",
        "benchmark/vision/fetch_coco_subset.py",
        "benchmark/vision/coco_val2017_first500.txt",
        "benchmark/vision/eval_map.py",
        "inference/vision/export.py",
        "inference/vision/triton_serve.py",
    ]
)
EMBED_B = LOADGEN + [
    "notebooks/vllm-requirements.txt",
    "notebooks/trtllm-requirements.txt",
    "notebooks/trtllm-build-constraints.txt",
    "inference/trtllm/build_engine.py",
    "inference/trtllm/serve.py",
]

# Shared by both notebooks: time budget, per-step timing, FAILED-/NOT_RUN- files, pinned downloads, server start.
STEP_CELL = """# Step runner. Every unit of work is one step(): it is skipped with NOT_RUN-<name>.txt when the remaining session
# budget cannot cover its estimate, records FAILED-<name>.txt with the traceback on any exception (the notebook
# moves on), and writes env/timing-<name>.txt. Returns fn's value (True for None) on success, None otherwise.
import hashlib
import shlex
import traceback
import urllib.request

T0 = time.time()


def time_left():
    return BUDGET_S - (time.time() - T0)


def not_run(name, reason):
    pathlib.Path(RESULTS, f"NOT_RUN-{name}.txt").write_text(f"not run: {reason}\\n")
    print(name, "not run:", reason)


def step(name, est_s, fn, *args, **kw):
    if time_left() < est_s:
        not_run(name, f"time budget: {time_left():.0f} s left of {BUDGET_S} s, step estimate {est_s} s")
        return None
    t = time.time()
    try:
        res = fn(*args, **kw)
        return True if res is None else res
    except Exception:  # noqa: BLE001 record and move on
        pathlib.Path(RESULTS, f"FAILED-{name}.txt").write_text(traceback.format_exc())
        print(f"{name} FAILED, see results/FAILED-{name}.txt")
        return None
    finally:
        pathlib.Path(RESULTS, "env", f"timing-{name}.txt").write_text(f"{time.time() - t:.0f} s wall clock\\n")
        print(f"{name}: {time.time() - t:.0f}s")


def fetch(url, sha, dest):
    \"\"\"Download url to dest and check its sha256 (raises on mismatch).\"\"\"
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise ValueError(f"{url}: sha256 {got} != pinned {sha}")
    dest.write_bytes(data)
    return dest


def install_vllm():
    \"\"\"vLLM 0.31.0 venv from the run v2 lock (notebooks/vllm-requirements.txt).\"\"\"
    log = sh(f"uv venv -q --python {PYTHON_VERSION} {VENVS}/vllm && uv pip install -q --python {VENVS}/vllm/bin/python "
             f"--torch-backend cu129 -r {BENCH}/notebooks/vllm-requirements.txt", "logs/install-vllm.log", check=True,
             timeout=2400)
    print(log[-1500:])
    sh(f"uv pip freeze --python {VENVS}/vllm/bin/python", "env/pip-freeze-vllm.txt")


def start_proc(cell, args, env, ready_url, timeout=900):
    \"\"\"Start a server in its own process group, log to logs/server-<cell>.log, wait until ready_url answers 200.\"\"\"
    cmd = shlex.join(args)
    pathlib.Path(RESULTS, "server_args", f"{cell}.txt").write_text(cmd + "\\n")
    log = open(pathlib.Path(RESULTS, "logs", f"server-{cell}.log"), "w")  # noqa: SIM115 open for the server's lifetime
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{cell}: server exited with code {proc.returncode}, see logs/server-{cell}.log")
        try:
            urllib.request.urlopen(ready_url, timeout=5)
            print(f"{cell}: ready in {time.time() - t:.0f}s")
            return proc, cmd
        except OSError:
            time.sleep(5)
    stop_server(proc, cell)
    raise TimeoutError(f"{cell}: not ready after {timeout}s")
"""

SUMMARY_CELL = """# Quick look. The real report is benchmark/report.py, run offline after downloading the results.
# Wrapped like a step: nothing in the summary may raise, so the end-of-session files are always written.
import glob


def summary():
    for f in sorted(glob.glob(f"{RESULTS}/raw/**/*.json", recursive=True)):
        try:
            d = json.loads(pathlib.Path(f).read_text())
            print(os.path.relpath(f, RESULTS), json.dumps(d.get("summary") or d.get("metrics"))[:300])
        except (OSError, ValueError, AttributeError) as e:
            print(f, "unreadable:", e)
    print(sh(f"ls {RESULTS}/FAILED-* {RESULTS}/NOT_RUN-* 2>/dev/null") or "no failed or skipped steps")


step("summary", 0, summary)
sh("nvidia-smi -q", "env/nvidia-smi-q-end.txt")  # cumulative clock event reason counters over the session
pathlib.Path(RESULTS, "env/finished_utc.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\\n", time.gmtime()))
print(f"session used {time.time() - T0:.0f} s of the {BUDGET_S} s budget")
"""

# ---------------------------------------------------------------- notebook A: DCGM + vision
INTRO_A = """# v3 on Kaggle T4: YOLOv8n vision (DCGM settled by the 2026-10-10 run)

Items 10 and 11 of `docs/PLAN.md` (`industrial-llm-platform`). One-click **Run all** on **GPU T4 x2** with internet on.
Pushed with `kaggle kernels push -p notebooks/kaggle-dcgm-vision -t 3600` (1 h hard cap); the notebook's own budget
(`BUDGET_S`) skips steps that would not finish, leaving time for the summary. Estimate: about 0.4 h (re-export, then mAP).

**Rerun scope (`RUN_SPEED = False`, `RUN_TRITON = False`: mAP only).** Triton succeeded on 2026-10-10 in
`benchmark/results/raw/kaggle-20261010-v3e/`; mAP failed there because Kaggle exports `MPLBACKEND` set to the
notebook's matplotlib_inline backend, which the vision venv lacks (now `MPLBACKEND=Agg`). The ONNX Runtime CUDA and TensorRT speed cells succeeded on 2026-10-10
(`benchmark/results/raw/kaggle-20261010-v3c/`); this session re-exports the models (Triton and mAP need them), then
runs only Triton and mAP, which failed there: mAP because the data YAML had no `train:` key (Ultralytics 8.4.174
requires it), Triton because the Python backend stub loaded Kaggle's Ubuntu libpython3.12 instead of the uv-managed
Python's (`No module named '_contextvars'`). Both fixes were reproduced and verified locally on CPU.

**A. DCGM: not run again.** The 2026-10-10 session (notebook built at 19ccae8, output in
`benchmark/results/raw/kaggle-20261010-v3a/results/dcgm/`) settled it: the basic fields work in embedded and
standalone mode, and every profiling field fails with `DCGMError_RequiresRoot: This operation is not supported when
the host engine is running as non root`. Kaggle gives no root over the host engine, so DCGM profiling (DRAM active,
SM active) cannot run there. This notebook writes `NOT_RUN-dcgm.txt` and `NOT_RUN-vllm-t4-1.5b-fp16-dcgm.txt`
pointing to that evidence and spends the session on vision. The text below on how the probe worked is kept for the
record.

How the 2026-10-10 probe worked: NVIDIA DCGM 4.7.0-2 is extracted from its pinned .deb files (no apt, no Docker; the
dcgm-exporter container cannot run here). Tested: the Python bindings in embedded mode, a standalone `nv-hostengine`,
`dcgmi discovery/profile/dmon`, each with the basic field group (clocks, clock event reasons, temperature, power,
utilisation, framebuffer) and the profiling group (GR engine, SM active, SM occupancy, tensor pipe, DRAM active, FP16
pipe). All output is kept verbatim in `results/dcgm/`. Each profiling field is then probed alone, because one
unsupported field (FP16 pipe may be one on Turing) can fail a whole watch. Only if SM active or DRAM active returns
values: vLLM 1.5B FP16 at concurrency 1 and 8 is re-run sampling the basic fields plus the profiling fields that
worked (cell `vllm-t4-1.5b-fp16-dcgm`) to see whether batch-1 decode is memory-bound (DRAM active). The probe accepts
idle zeros; `results/dcgm/rerun-check.json` then records whether SM/DRAM active were non-zero under load. Otherwise
`NOT_RUN-vllm-t4-1.5b-fp16-dcgm.txt` holds the reason.

NVIDIA proprietary but free software is used here: the DCGM proprietary package (profiling module) and TensorRT.

**B. Vision (estimate 45 min).** YOLOv8n (Ultralytics 8.4.174, **AGPL-3.0**; weights sha256-pinned) exported to ONNX
FP32 (opset 17) and TensorRT FP16 engines (one per batch size). Images/s and per-batch latency p50/p95/p99 at batch
1, 8, 32 for ONNX Runtime CUDA (FP32) and TensorRT FP16 in process, then TensorRT FP16 behind Triton via PyTriton
(Triton's Python backend; the NGC image needs Docker). mAP50-95 on the first 500 val2017 images (sha256 manifest)
for PyTorch FP32 (reference), ONNX FP32 and TensorRT FP16.

T4 is SM75: FP16 only (no bf16, no FP8). Preflight stops at once without GPU or internet (`FAILED-preflight.txt`).
"""

CONFIG_A = """# Pinned configuration. Everything else is in the embedded lock files.
GIT_COMMIT = "@GIT_COMMIT@"  # repo commit the embedded code came from
UV_VERSION = "0.11.32"
PYTHON_VERSION = "3.12.13"
VLLM_VERSION = "0.31.0"
MODELS = {"1.5b-fp16": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")}
REPEATS = 3
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.90
GSM8K_LIMIT = 250
PORT = 8000
WORK = "/kaggle/working"
RESULTS = f"{WORK}/results"
BENCH = f"{WORK}/bench"
VENVS = "/tmp/venvs"
BUDGET_S = 3000  # push -t 3600; leaves 10 min for the summary
DCGM_EVIDENCE = "benchmark/results/raw/kaggle-20261010-v3a/results/dcgm/ (notebook built at 19ccae8)"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
YOLO_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
VISION_BATCHES = (1, 8, 32)
# ORT CUDA and TensorRT speed cells succeeded in kaggle-20261010-v3c; the rerun only needs Triton and mAP.
RUN_SPEED = False
SPEED_EVIDENCE = "benchmark/results/raw/kaggle-20261010-v3c/ (notebook built at 80905ea)"
# Triton succeeded in kaggle-20261010-v3e; this rerun is mAP only.
RUN_TRITON = False
TRITON_EVIDENCE = "benchmark/results/raw/kaggle-20261010-v3e/ (notebook built at fb5ea24)"
DATA = "/tmp/data/coco500"
MODELS_DIR = f"{RESULTS}/models"  # ONNX and engines, kept in the output for traceability (git-ignored)
# MPLBACKEND=Agg: Kaggle exports the notebook's matplotlib_inline backend, which the vision venv does not have.
VISION_ENV = "YOLO_OFFLINE=1 YOLO_AUTOINSTALL=False CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg"  # no Ultralytics downloads or pip installs
PINS = {
    "vllm_version": VLLM_VERSION, "ultralytics": "8.4.174", "tensorrt": "10.16.1.11",
    "onnxruntime_gpu": "1.26.0", "nvidia_pytriton": "0.7.0", "onnx_opset": 17,
}
PREFLIGHT_URLS = (
    "https://pypi.org/simple/", "https://pypi.nvidia.com/tensorrt-cu12/", "https://github.com",
    "https://huggingface.co", "http://images.cocodataset.org/val2017/000000000139.jpg",
)
"""

BASE_INSTALL = """# uv and the client venv (loadgen, COCO fetch, lm-eval). Other venvs are installed by the steps that need them.
def install_base():
    print(sh(f"pip install -q uv=={UV_VERSION} && uv --version", "logs/install-uv.log", check=True, timeout=600))
    sh(f"uv venv -q --python {PYTHON_VERSION} {VENVS}/bench && "
       f"uv pip install -q --python {VENVS}/bench/bin/python -r {BENCH}/benchmark/requirements.txt", "logs/install-bench.log",
       check=True, timeout=1200)
    sh(f"uv pip freeze --python {VENVS}/bench/bin/python", "env/pip-freeze-bench.txt")


BASE_OK = step("install-base", 600, install_base)
"""

DCGM_SKIP_CELL = """# DCGM is not run again: profiling needs root over the host engine, which Kaggle does not give.
reason = ("DCGM profiling fails on Kaggle for every field with 'DCGMError_RequiresRoot: This operation is not supported "
          f"when the host engine is running as non root'; basic fields worked. Evidence: {DCGM_EVIDENCE}")
not_run("dcgm", reason)
not_run("vllm-t4-1.5b-fp16-dcgm", reason)
"""

VISION_CELL = """# Vision: venv, data, exports, then the benchmarks and mAP. One step each, so one failure does not stop the rest.
VPY = f"{VISION_ENV} {VENVS}/vision/bin/python"
WEIGHTS = "/tmp/data/yolov8n.pt"
ONNX = f"{MODELS_DIR}/yolov8n-fp32-opset17.onnx"
ENGINE = {b: f"{MODELS_DIR}/yolov8n-fp16-b{b}.engine" for b in VISION_BATCHES}
HW = "'1x Tesla T4 (Kaggle)'"


def vision_install():
    print(sh(f"uv venv -q --python {PYTHON_VERSION} {VENVS}/vision && uv pip install -q --python {VENVS}/vision/bin/python "
             "--torch-backend cu129 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match "
             f"-r {BENCH}/notebooks/vision-requirements.txt", "logs/install-vision.log", check=True, timeout=1800)[-1500:])
    sh(f"uv pip freeze --python {VENVS}/vision/bin/python", "env/pip-freeze-vision.txt")
    print(sh(f"{VPY} -c 'import torch, tensorrt, onnxruntime, ultralytics; print(torch.__version__, torch.version.cuda, "
             "torch.cuda.get_device_capability(0), tensorrt.__version__, onnxruntime.__version__, "
             "onnxruntime.get_available_providers(), ultralytics.__version__)'", "env/vision-versions.txt", check=True))


def vision_data():
    print(sh(f"{VENVS}/bench/bin/python {BENCH}/benchmark/vision/fetch_coco_subset.py --out {DATA}", "logs/fetch-coco.log", check=True, timeout=1200))
    fetch(YOLO_URL, YOLO_SHA256, WEIGHTS)


def vision_export():
    out = sh(f"{VPY} {BENCH}/inference/vision/export.py {WEIGHTS} {MODELS_DIR} --batches {' '.join(map(str, VISION_BATCHES))}",
             "logs/vision-export.log", check=True, timeout=3000)
    print(out[-3000:])


def vision_bench(label, backend, model, precision, batches, extra="", log=None):
    b = " ".join(map(str, batches))
    out = sh(f"{VPY} {BENCH}/benchmark/vision/run_vision.py --backend {backend} --model {model} --images {DATA}/images/val2017 "
             f"--batch {b} --repeats {REPEATS} --label {label} --precision {precision} --hardware {HW} "
             f"--git-commit {GIT_COMMIT} --out-dir {RESULTS}/raw/{label} {extra}", log or f"logs/{label}.log", check=True,
             timeout=1200)
    print(out[-2000:])


def vision_trt():
    for b in VISION_BATCHES:
        vision_bench("vision-t4-yolov8n-trt-fp16", "trt", ENGINE[b], "fp16", [b], log=f"logs/vision-t4-yolov8n-trt-fp16-b{b}.log")


def vision_triton():
    args = [f"{VENVS}/vision/bin/python", f"{BENCH}/inference/vision/triton_serve.py", *ENGINE.values(),
            "--log-file", f"{RESULTS}/logs/tritonserver.log"]
    proc, cmd = start_proc("vision-triton", args, dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                           "http://127.0.0.1:8000/v2/health/ready", timeout=600)
    try:
        for b in VISION_BATCHES:
            vision_bench("vision-t4-yolov8n-trt-fp16-triton", "triton", f"yolov8n_b{b}", "fp16", [b],
                         extra=f"--model-file {ENGINE[b]} --triton-url 127.0.0.1:8000 "
                               f"--metrics-url http://127.0.0.1:8002/metrics --server-args {shlex.quote(cmd)}",
                         log=f"logs/vision-t4-yolov8n-trt-fp16-triton-b{b}.log")
    finally:
        stop_server(proc, "vision-triton")


def vision_map(label, model, precision):
    out = sh(f"{VPY} {BENCH}/benchmark/vision/eval_map.py {model} {DATA} --label {label} --precision {precision} "
             f"--device 0 --git-commit {GIT_COMMIT} --hardware {HW} --out {RESULTS}/raw/vision-accuracy",
             f"logs/map-{label}.log", check=True, timeout=800)
    print(out[-1500:])


if BASE_OK and step("vision-install", 900, vision_install) and step("vision-data", 600, vision_data) \\
        and step("vision-export", 1500, vision_export):
    if RUN_SPEED:
        step("vision-t4-yolov8n-onnx-fp32-ortcuda", 600, vision_bench, "vision-t4-yolov8n-onnx-fp32-ortcuda", "ort-cuda",
             ONNX, "fp32", VISION_BATCHES)
        step("vision-t4-yolov8n-trt-fp16", 600, vision_trt)
    else:
        for name in ("vision-t4-yolov8n-onnx-fp32-ortcuda", "vision-t4-yolov8n-trt-fp16"):
            not_run(name, f"RUN_SPEED is False: already measured, see {SPEED_EVIDENCE}")
    if RUN_TRITON:
        step("vision-triton", 900, vision_triton)  # FAILED-vision-triton.txt + logs/server-vision-triton.log if Triton cannot run
    else:
        not_run("vision-triton", f"RUN_TRITON is False: already measured, see {TRITON_EVIDENCE}")
    for label, model, precision in (("pt-fp32", WEIGHTS, "fp32"), ("onnx-fp32", ONNX, "fp32"), ("trt-fp16", ENGINE[1], "fp16")):
        step(f"map-{label}", 400, vision_map, label, model, precision)
"""

# ---------------------------------------------------------------- notebook B: TensorRT-LLM
INTRO_B = """# v3 on Kaggle T4: TensorRT-LLM FP16 vs vLLM, Qwen2.5-1.5B-Instruct

Item 12 of `docs/PLAN.md` (`industrial-llm-platform`). One-click **Run all** on **GPU T4 x2** with internet on.
Pushed with `kaggle kernels push -p notebooks/kaggle-trtllm -t 10800` (3 h hard cap); the notebook's own budget
(`BUDGET_S`) skips steps that would not finish. Estimate: about 1.6 h (vLLM control about 0.6 h, then TensorRT-LLM). Uses one T4.

**Rerun scope (`RUN_VLLM_CONTROL = True`).** The vLLM control succeeded on 2026-10-10
(`benchmark/results/raw/kaggle-20261010-v3d/`, 9 runs and GSM8K) and runs again here, so TensorRT-LLM has a
same-session comparison. There the TensorRT-LLM venv installed but
`import tensorrt_llm` failed: datasets 2.14.4 subclasses `pyarrow.PyExtensionType`, which pyarrow 21 removed, and
the lock had pyarrow 25.0.1. The lock now pins pyarrow 20.0.0 and pynvml 11.5.3 (pynvml 13 has no `__version__`,
which `tensorrt_llm/profiler.py` reads at import: the next failure, found locally). `import tensorrt_llm` succeeded
in a local Ubuntu 24.04 container with CUDA driver and NVML stub libraries. The install step now also logs `uv pip check`. With the control on, `benchmark/report.py`
compares the TensorRT-LLM rows with this session's vLLM rows only.
The third run (v3f) got past install and import, then the checkpoint save failed because TensorRT-LLM 0.14's
`save_checkpoint` does not create its output dir; `inference/trtllm/build_engine.py` now creates it, and the
conversion of the real model ran to completion on CPU in a local container (the engine build needs the GPU).
The fourth run (v3h) failed in trtllm-build: unfused context attention needed 21.98 GB of workspace for batch 32 and
input 3072. Two engines now: `bench` (batch 32, input 1024) for loadgen, `gsm8k` (batch 8, input 2048) for GSM8K,
about 2.55 GB of workspace each (calculation in `inference/trtllm/build_engine.py`).

**Why TensorRT-LLM 0.14.0.** It is the last release whose support matrix lists Turing (SM75: FP32, FP16, INT8 (no
SmoothQuant on SM75), INT4). 0.15.0 drops Turing from the matrix and 0.16.0 stops compiling SM70 kernels (which
Turing runs). The 0.14.0 wheel is Python 3.10 only, so this notebook builds a 3.10 venv for it. FP16 only on T4.

Steps: Open MPI 4.1.6 from Ubuntu (pinned), venvs, model download (pinned revision), engine build
(`inference/trtllm/build_engine.py`: untie embeddings, FP16 checkpoint, `trtllm-build --context_fmha disable`, since
fused context attention has no SM75 kernels), OpenAI-compatible server (`inference/trtllm/serve.py` around 0.14.0's
`examples/apps/openai_server.py`, sha256-pinned, one documented stream patch, plus `/metrics`), then loadgen at
concurrency 1, 8, 32, 3 repeats, 128-token fixed decode, and GSM8K (250 items) if time allows.

**Same-session vLLM control.** vLLM 0.31.0, same model, FP16, same loadgen cells and GSM8K, with
`--generation-config vllm` (no repetition_penalty), so the TensorRT-LLM vs vLLM comparison is same hardware, same
session, same sampling. `benchmark/report.py` compares TensorRT-LLM only with vLLM rows of the same session, and
generates the like-for-like check (prompts sha256, max_tokens, actual token counts, truncated streams) from the raw
files. Run v2 took 978 s for this cell.

**Not run: Triton's TensorRT-LLM backend.** It ships only in the NGC `tritonserver:<ver>-trtllm-python-py3` image,
which needs Docker; Kaggle has none. Recorded as `NOT_RUN-triton-tensorrtllm-backend.txt`.

If TensorRT-LLM cannot be installed, built or served, the failing step writes `FAILED-<step>.txt` with the
traceback and every later step writes `NOT_RUN-<step>.txt` with the reason. Preflight stops at once without GPU or
internet (`FAILED-preflight.txt`).
"""

CONFIG_B = """# Pinned configuration. Everything else is in the embedded lock files.
GIT_COMMIT = "@GIT_COMMIT@"  # repo commit the embedded code came from
UV_VERSION = "0.11.32"
PYTHON_VERSION = "3.12.13"  # client venv (loadgen, lm-eval)
TRTLLM_PYTHON = "3.10.20"  # TensorRT-LLM 0.14.0 ships cp310 wheels only
TRTLLM_VERSION = "0.14.0"
OPENMPI_APT = "4.1.6-7ubuntu2"  # Ubuntu 24.04 (noble) universe
OPENAI_SERVER_URL = ("https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/"
                     "b0880169d0fb8cd0363049d91aa548e58a41be07/examples/apps/openai_server.py")  # tag v0.14.0
OPENAI_SERVER_SHA256 = "8ae2d1d892792bada31424bd8295dc87a5e1a902c2f723427e424a5a1b3f9f1f"
MODELS = {"1.5b-fp16": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")}
REPEATS = 3
GSM8K_LIMIT = 250
PORT = 8000
KV_FRACTION = 0.90  # of the GPU memory free after the engine loads (vLLM's 0.90 is of the whole card)
WORK = "/kaggle/working"
RESULTS = f"{WORK}/results"
BENCH = f"{WORK}/bench"
VENVS = "/tmp/venvs"
TRT_WORK = "/tmp/trtllm"  # HF snapshot, checkpoint and engine (not saved as output)
BUDGET_S = 10200  # push -t 10800; leaves 10 min for the summary
CELL = "trtllm-t4-1.5b-fp16"
VLLM_VERSION = "0.31.0"  # same-session control, as run v2 plus --generation-config vllm
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.90
VLLM_CELL = "vllm-t4-1.5b-fp16"
# Re-run the vLLM control so TensorRT-LLM has same-session vLLM rows (v3d's control is from another session).
RUN_VLLM_CONTROL = True
CONTROL_EVIDENCE = "benchmark/results/raw/kaggle-20261010-v3d/ (notebook built at 80905ea)"
PINS = {"tensorrt_llm": TRTLLM_VERSION, "tensorrt": "10.4.0", "torch": "2.4.0+cu124", "python": TRTLLM_PYTHON,
        "openmpi": OPENMPI_APT, "vllm_version": VLLM_VERSION}
PREFLIGHT_URLS = ("https://pypi.org/simple/", "https://pypi.nvidia.com/tensorrt-llm/", "https://github.com",
                  "https://huggingface.co", OPENAI_SERVER_URL)
"""

TRTLLM_CELL = """# TensorRT-LLM: install, download, build, serve, benchmark. Each step needs the previous one.
TRT_ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
TPY = f"{VENVS}/trtllm/bin/python"


def install():
    # uv and the client venv come from the base install cell, which runs before the vLLM control
    print(sh(f"apt-get update -qq && apt-get install -y -qq --no-install-recommends openmpi-bin={OPENMPI_APT} "
             f"libopenmpi-dev={OPENMPI_APT} && mpirun --version", "logs/install-openmpi.log", check=True,
             timeout=900)[-1000:])
    # about 9 GB of wheels (tensorrt_llm 1.9 GB, TensorRT 10.4 libs, torch cu124, CUDA libs); 12 GB installed
    print(sh(f"uv python install {TRTLLM_PYTHON} && uv venv -q --python {TRTLLM_PYTHON} {VENVS}/trtllm && "
             f"UV_HTTP_TIMEOUT=300 uv pip install -q --python {TPY} --torch-backend cu124 --extra-index-url https://pypi.nvidia.com "
             f"--index-strategy unsafe-best-match --build-constraints {BENCH}/notebooks/trtllm-build-constraints.txt "
             f"-r {BENCH}/notebooks/trtllm-requirements.txt", "logs/install-trtllm.log",
             check=True, timeout=3600)[-1500:])
    sh(f"uv pip freeze --python {TPY}", "env/pip-freeze-trtllm.txt")
    # dependency consistency, logged even when it passes (a pyarrow/datasets mismatch broke the import in v3d)
    print(sh(f"uv pip check --python {TPY}; echo exit=$?", "env/trtllm-pip-check.txt"))
    fetch(OPENAI_SERVER_URL, OPENAI_SERVER_SHA256, "/tmp/trtllm-example/openai_server.py")
    print(sh(f"cd /tmp && {TPY} -c 'import torch, tensorrt, tensorrt_llm; print(tensorrt_llm.__version__, tensorrt.__version__, "
             "torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))'", "env/trtllm-versions.txt", check=True))


def download():
    repo, rev = MODELS["1.5b-fp16"]
    print(sh(f"{TPY} -c \\"from huggingface_hub import snapshot_download; "
             f"print(snapshot_download('{repo}', revision='{rev}', local_dir='{TRT_WORK}/hf'))\\"", "logs/hf-download.log", check=True,
             timeout=1200)[-500:])


def build_engine():
    env = " ".join(f"{k}={TRT_ENV[k]}" for k in ("CUDA_VISIBLE_DEVICES", "OMPI_ALLOW_RUN_AS_ROOT", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"))
    out = sh(f"{env} {TPY} {BENCH}/inference/trtllm/build_engine.py {TRT_WORK}/hf {TRT_WORK} "
             f"--cmd-out {RESULTS}/server_args/trtllm-build.txt", "logs/trtllm-build.log", check=True, timeout=3600)
    print(out[-3000:])
    for p in ("bench", "gsm8k"):
        sh(f"cp {TRT_WORK}/engine-fp16-{p}/config.json {RESULTS}/env/trtllm-engine-config-{p}.json && "
           f"du -sh {TRT_WORK}/engine-fp16-{p}", f"env/trtllm-engine-size-{p}.txt", check=True)


def serve(name, profile):
    \"\"\"Start serve.py on one engine profile (build_engine.PROFILES): bench (batch 32, input 1024) for loadgen,
    gsm8k (batch 8, input 2048) for GSM8K. One engine cannot do both on a T4 (attention workspace, see build_engine).\"\"\"
    args = [TPY, f"{BENCH}/inference/trtllm/serve.py", f"{TRT_WORK}/engine-fp16-{profile}", "--tokenizer", f"{TRT_WORK}/hf",
            "--served-model-name", MODELS["1.5b-fp16"][0], "--openai-server-dir", "/tmp/trtllm-example",
            "--port", str(PORT), "--kv-cache-free-gpu-memory-fraction", str(KV_FRACTION)]
    return start_proc(name, args, TRT_ENV, f"http://127.0.0.1:{PORT}/health", timeout=900)


def serve_and_bench():
    proc, cmd = serve(CELL, "bench")
    try:
        step(f"{CELL}-loadgen", 1800, loadgen, CELL, "1.5b-fp16", "fp16", [1, 8, 32], cmd, 1,
             engine="trtllm", engine_version=TRTLLM_VERSION)
    finally:
        stop_server(proc, CELL)


def serve_and_gsm8k():
    proc, _ = serve("trtllm-t4-1.5b-fp16-gsm8k", "gsm8k")
    try:
        gsm8k("gsm8k-trtllm-1.5b-fp16", "1.5b-fp16")
    finally:
        stop_server(proc, "trtllm-t4-1.5b-fp16-gsm8k")


not_run("triton-tensorrtllm-backend", "Triton's TensorRT-LLM backend ships only in the NGC "
        "tritonserver:<ver>-trtllm-python-py3 container image, which needs Docker; Kaggle has no Docker. "
        "TensorRT-LLM is served with its own OpenAI server instead (inference/trtllm/serve.py)")
ok = True
for name, est, fn in (("install", 1800, install), ("download", 600, download), ("build-engine", 1800, build_engine),
                      (CELL, 2700, serve_and_bench), ("gsm8k-trtllm-1.5b-fp16", 1200, serve_and_gsm8k)):
    if not ok:
        not_run(name, "an earlier TensorRT-LLM step failed or was skipped (see FAILED-*/NOT_RUN-*)")
        continue
    ok = bool(step(name, est, fn))
"""

VLLM_CONTROL_CELL = """# Same-session vLLM control: run v2's cell with --generation-config vllm (start_server), same loadgen and GSM8K.
# Runs before TensorRT-LLM so a slow TensorRT-LLM step cannot squeeze it out of the budget. Needs the base install.
if not RUN_VLLM_CONTROL:
    not_run(VLLM_CELL, f"RUN_VLLM_CONTROL is False: already run, see {CONTROL_EVIDENCE}")
elif not pathlib.Path(VENVS, "bench/bin/python").exists():
    not_run(VLLM_CELL, "client venv missing (the base install step failed)")
elif step("install-vllm", 1500, install_vllm):
    step(VLLM_CELL, 1800, run_cell, VLLM_CELL, "1.5b-fp16", "fp16", [1, 8, 32], accuracy_label="gsm8k-vllm-1.5b-fp16")
"""


def notebook_a(commit, embed_cell):
    return [
        md(INTRO_A),
        code(CONFIG_A.replace("@GIT_COMMIT@", commit)),
        code(embed_cell),
        code(ENV_CELL),
        code(PREFLIGHT_CELL),
        code(HELPERS_CELL),
        code(STEP_CELL),
        md("## Install the client venv (estimate 2 min)"),
        code(BASE_INSTALL),
        md("## A. DCGM: not run again (settled by the 2026-10-10 session)"),
        code(DCGM_SKIP_CELL),
        md(
            "## B. Vision: YOLOv8n ONNX Runtime CUDA, TensorRT FP16, Triton, mAP (estimate 45 min)"
        ),
        code(VISION_CELL),
        md("## Summary"),
        code(SUMMARY_CELL),
    ]


def notebook_b(commit, embed_cell):
    return [
        md(INTRO_B),
        code(CONFIG_B.replace("@GIT_COMMIT@", commit)),
        code(embed_cell),
        code(ENV_CELL),
        code(PREFLIGHT_CELL),
        code(HELPERS_CELL),
        code(STEP_CELL),
        md("## Install the client venv (estimate 2 min)"),
        code(BASE_INSTALL),
        md(
            "## Same-session vLLM 0.31.0 control, 1.5B FP16 c1/c8/c32 + GSM8K (estimate 35 min)"
        ),
        code(VLLM_CONTROL_CELL),
        md(
            "## TensorRT-LLM 0.14.0: install, build, serve, loadgen, GSM8K (estimate 80 min)"
        ),
        code(TRTLLM_CELL),
        md("## Summary"),
        code(SUMMARY_CELL),
    ]


def checked(make_cells, embed):
    """make_cells wrapper that fails the build if a code cell references {BENCH}/<path> that is not embedded."""

    def make(commit, embed_cell):
        cells = make_cells(commit, embed_cell)
        used = {
            m
            for c in cells
            if c["cell_type"] == "code"
            for m in re.findall(r"\{BENCH\}/([\w./-]+\.\w+)", "".join(c["source"]))
        }
        missing = sorted(used - set(embed))
        assert not missing, f"cells reference files that are not embedded: {missing}"
        return cells

    return make


def main():
    build(
        ROOT / "notebooks" / "kaggle-dcgm-vision" / "dcgm_vision_t4.ipynb",
        EMBED_A,
        checked(notebook_a, EMBED_A),
    )
    build(
        ROOT / "notebooks" / "kaggle-trtllm" / "trtllm_t4.ipynb",
        EMBED_B,
        checked(notebook_b, EMBED_B),
    )


if __name__ == "__main__":
    main()
