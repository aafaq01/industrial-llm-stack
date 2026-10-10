# Inference serving paths

Two engines in core scope. Both expose the OpenAI API, so the same load generator (`benchmark/loadgen/`) and the
same accuracy harness (`benchmark/accuracy/`) run against either.

| Path | Where | Engine | Model | Precision |
|------|-------|--------|-------|-----------|
| CPU | local, `docker-compose.yml` service `llamacpp` | llama.cpp `llama-server`, image `ghcr.io/ggml-org/llama.cpp:server-b11459` | Qwen2.5-1.5B-Instruct GGUF | Q4_K_M |
| GPU | Kaggle notebook `notebooks/vllm_t4_benchmark.ipynb` | vLLM 0.31.0 (`+cu129` wheel, torch 2.13.0+cu129) | Qwen2.5-1.5B-Instruct, Qwen2.5-1.5B-Instruct-AWQ, Qwen2.5-7B-Instruct-AWQ | FP16, AWQ INT4 |

TensorRT-LLM, Triton, DCGM and the YOLOv8n vision path are v3 items 10 to 12 (`docs/PLAN.md`): two Kaggle notebooks,
run 2026-10-10 (runs v3a to v3i), results in `benchmark/RESULTS.md`. Triton's native TensorRT and TensorRT-LLM backends
were not run (they need Docker). See "v3 Kaggle notebooks" at the end.

## llama.cpp on CPU

Flags (from `docker-compose.yml`): `--ctx-size 4096 --parallel 2 --threads 4 --metrics`. The context is split
across the slots, so each request gets 2048 tokens. At most 2 requests decode at once; extra concurrent requests
queue, which shows up as TTFT in the benchmark. Prompt caching is on by default and is not limited to the 2 slots:
`llama-server --help` in this image (build 11459, commit f498f864f) lists `--cache-prompt` (default enabled),
`--cache-ram N` (host-memory prompt cache, default 8192 MiB, llama.cpp PR 16391) and `--cache-idle-slots` (save idle
slots to that cache, default enabled). So any earlier prompt can skip prefill until the server restarts; the Phase 2
CPU run measured 96% of prompt tokens served from cache on repeats over 64 distinct prompts
(`docs/evidence/phase2-cpu.txt`). The model file is pinned by commit and sha256 in `docs/VERSIONS.md`.

## vLLM on a Kaggle T4

Install: an isolated Python 3.12.13 venv (uv 0.11.32) from the lock `notebooks/vllm-requirements.txt`: the GitHub
release wheel `vllm-0.31.0+cu129` with PyTorch from the cu129 index. The +cu129 wheel's METADATA is not identical
to the PyPI one: it requires `humming-kernels[cu12]` where PyPI has `[cu13]` (and `nvidia-cutlass-dsl` without the
`[cu13]` extra, which other packages request anyway). The lock is resolved from PyPI metadata with
`notebooks/vllm-overrides.txt` fixing the humming-kernels extra. It has 197 entries: the vllm wheel URL plus 196
name==version pins, and those 196 were checked to be identical to an independent resolve of the wheel's own METADATA requirements (`docs/evidence/phase2-prep.txt`). Three packages with
13.x versions remain because the wheel's own dependency tree asks for them: `nvidia-cutlass-dsl-libs-cu13` and
`nvidia-cuda-nvdisasm` (CUTLASS DSL; some dependency requests `nvidia-cutlass-dsl[cu13]`) and `nvidia-ml-py` (NVML
bindings, via flashinfer).

Server command (one per cell, written verbatim to `results/server_args/<cell>.txt` in the run output):

```
vllm serve <repo> --revision <sha> --tokenizer-revision <sha> --served-model-name <repo>
  --host 127.0.0.1 --port 8000 --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.90
  --attention-backend TRITON_ATTN --no-enable-prefix-caching --seed 0 --generation-config vllm
  --tensor-parallel-size <1|2>
```

`--generation-config vllm` was added after run v2 (2026-10-09). Without it vLLM applies the model's
`generation_config.json` as default sampling parameters; for Qwen2.5 that put repetition_penalty 1.1 (1.05 for 7B)
on every request, including the GSM8K runs (`docs/evidence/phase2-gpu.txt`). Run v2 results were produced without it.

Prometheus metrics come from the same port at `/metrics`; the load generator snapshots them before and after each run.

| Model | HF revision |
|-------|-------------|
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | `3ecffa0ceb27851800f45519bab9c457a04405e1` |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | `b25037543e9394b818fdfca67ab2a00ecc7dd641` |

### Why vLLM 0.31.0 and the CUDA 12.9 wheel

- 0.31.0 (released 2026-10-05) is the newest release. Its `CMakeLists.txt` still builds kernels for
  compute capability 7.5, its Dockerfile `torch_cuda_arch_list` starts at 7.5, and the GPU install docs list
  "compute capability 7.5 or higher (e.g., T4)".
- The plain PyPI wheel is built against torch 2.13.0 from PyPI, which is a CUDA 13 build and needs an R580+
  driver. The Kaggle driver version is not known before the session, so the notebook uses the `+cu129` wheel
  from the GitHub release with torch 2.13.0+cu129, which runs on any R525+ driver via CUDA minor version
  compatibility. The notebook records `nvidia-smi -q` first, so the actual driver is in the evidence.
  Run v2 got driver 580.178.04 (CUDA 13.0), so the plain CUDA 13 wheel would also have worked on that session;
  the +cu129 choice still covers older drivers.
- Fallback if 0.31.0 fails on T4: set `VLLM_VERSION` and the wheel URL in `notebooks/vllm-requirements.in` to
  0.30.0 (also has a `+cu129` asset), recheck `vllm-overrides.txt` against that wheel's METADATA, recompile the
  lock, rebuild the notebook. Not needed: 0.31.0 worked in run v2.

### T4 limits and substitutions

| Want | T4 (sm75) reality | What we do |
|------|-------------------|------------|
| bf16 | no bf16 tensor cores | `--dtype float16` everywhere |
| FP8 | no FP8 hardware | not used; quantised cells are AWQ INT4 |
| FlashAttention 2 | needs sm80+ | not available |
| FlashInfer | vLLM 0.31.0 marks it broken on SM75 and skips it (flashinfer issue 3620) | `--attention-backend TRITON_ATTN`, pinned so it cannot change silently |
| AWQ kernels | vLLM `auto_awq` needs capability 75, picks the kernel (Marlin has SM75 builds) | no `--quantization` flag; the server log records the chosen kernel |
| Memory | 15 GB per T4 | `--max-model-len 4096`, `--gpu-memory-utilization 0.90`; 7B only as AWQ |
| TP=2 | Kaggle 2x T4, PCIe, no NVLink | one extra cell; skipped with "not run: 1 GPU" if the session has one GPU |

Prefix caching is off on vLLM so repeats measure prefill every time. llama.cpp's prompt cache stays at its
default; keep that in mind when comparing TTFT across engines.

The Kaggle base image's own Python (3.13 at the time of writing) is only used to pip-install the pinned uv;
its packages are not used.

### What the numbers mean

- Every request decodes a fixed 128 tokens with `ignore_eos` (temperature 0, seed 1234). Throughput and latency
  describe a fixed-length decode, not where the model would naturally stop, and are comparable across engines
  and precisions for that reason.
- "GPU mem max" in `benchmark/RESULTS.md` mostly reflects vLLM pre-allocating `--gpu-memory-utilization 0.90`
  of the card for weights and KV cache, not the memory the workload needed.

### What run v2 (2026-10-09) settled

Details and log lines: `docs/evidence/phase2-gpu.txt`.
- Driver: 580.178.04, CUDA 13.0, 2x Tesla T4.
- Phone verification and internet worked (preflight passed); run v1 had failed on both.
- The session took about 39 min from start to finish, far under the `-t 14400` cap, so whether `/kaggle/working`
  survives a cap kill is still untested.
- vLLM 0.31.0 works on T4: the server logs show the TRITON_ATTN backend, the Marlin kernel for both AWQ models
  (`MarlinLinearKernel for AutoAWQMarlinLinearMethod`), and TP=2 on both GPUs. TP=2 runs without P2P: custom
  allreduce is disabled ("platform lacks GPU P2P capability"), so all-reduce goes through NCCL.
- Every cell ran close to the 70 W power cap (64 to 67 W mean). The FP16 c1 cell ran at about half the SM clock of
  the others for an unrecorded reason. From the next run on, the GPU sampler records memory clock and clock event
  reasons and the notebook saves `nvidia-smi -q` again at the end (cumulative power vs thermal counters).

## Running the Kaggle notebook

The notebook embeds the benchmark code (`notebooks/build_notebook.py` writes it, with sha256 per file), so no
Kaggle dataset or git remote is needed. Each push uses GPU quota and needs approval first.
Before the first push, replace `<kaggle-user>` with your Kaggle username in the `id` of each
`kernel-metadata.json` under `notebooks/`, and in the commands below.

```bash
git commit ...                                                  # so the embedded GIT_COMMIT is clean
benchmark/.venv/Scripts/python notebooks/build_notebook.py      # regenerate after any change under benchmark/
kaggle kernels push -p notebooks -t 14400                        # private, GPU T4, internet on; 4 h hard cap
kaggle kernels status <kaggle-user>/vllm-t4-benchmark
kaggle kernels output <kaggle-user>/vllm-t4-benchmark -p benchmark/results/raw/kaggle-<YYYYMMDD>
rm -r benchmark/results/raw/kaggle-<YYYYMMDD>/bench                # embedded code copy, already in git
benchmark/.venv/Scripts/python benchmark/report.py
```

Measured session time (run v2): about 39 min from start to finish; per cell 978 s (1.5B FP16 incl. GSM8K), 610 s
(1.5B AWQ incl. GSM8K), 312 s (7B AWQ), 182 s (TP=2), from `results/env/timing-<cell>.txt`. The pre-run estimate was
1.6 h. The `-t 14400` cap bounds the worst case at 4 h.

## v3 Kaggle notebooks (prepared 2026-10-09, run 2026-10-10)

Two notebooks, each its own push folder, built by `notebooks/build_v3.py` with the same pattern as the vLLM one:
embedded code with sha256, preflight (GPU and every download host), environment capture, one `step()` per unit of
work with `env/timing-<step>.txt`, `FAILED-<step>.txt` (traceback) or `NOT_RUN-<step>.txt` (reason), an in-notebook
time budget (`BUDGET_S`, 10 min under the push cap) and the hard cap from `kaggle kernels push -t`. Heavy shell commands
(installs, exports, engine build, loadgen, GSM8K) run under `timeout` at about twice their estimate, so a hang ends in a
FAILED file instead of using up the session. The output layout
matches run v2, so `benchmark/report.py` reads it after download.

| Notebook | Items | Push cap | Estimate | GPUs used |
|---|---|---|---|---|
| `notebooks/kaggle-dcgm-vision/` (`<kaggle-user>/v3-dcgm-vision-t4`) | 11 vision rerun: Triton and mAP only (`RUN_SPEED = False`) | `-t 3600` (1 h) | about 0.5 h | 1 |
| `notebooks/kaggle-trtllm/` (`<kaggle-user>/v3-trtllm-t4`) | 12 TensorRT-LLM rerun with the same-session vLLM control (`RUN_VLLM_CONTROL = True`) | `-t 10800` (3 h) | about 1.6 h | 1 |

Worst case 3 GPU-h (both caps), expected about 1.5 GPU-h.

**Second runs (2026-10-10, from 80905ea; `kaggle-20261010-v3c`, `kaggle-20261010-v3d`).** Valid: the ONNX Runtime
CUDA and TensorRT speed cells (v3c) and the same-session vLLM control with GSM8K (v3d). Failed, and fixed since:
- mAP: `eval_map.py` wrote a data YAML without `train:`, which Ultralytics 8.4.174's `check_det_dataset` requires.
  `train` now points at the same image list. Verified on CPU with Ultralytics 8.4.174 and the 5-image subset for the
  .pt and ONNX models (both run; the engine needs a GPU, its YAML path is the same function, unit-tested).
- Triton: PyTriton's Python backend stub links `libpython3.12.so.1.0` by name. With the uv-managed Python it found
  Kaggle's Ubuntu libpython3.12 and then the uv Python's stdlib: `No module named '_contextvars'`. Reproduced in
  ubuntu:24.04 with `libpython3.12t64` installed and a CPU model (same error), and fixed by putting the running
  Python's `sys.base_prefix/lib` first on `LD_LIBRARY_PATH` before PyTriton starts tritonserver
  (`use_own_libpython()` in `inference/vision/triton_serve.py`): the model then loads and serves.
- TensorRT-LLM: the venv installed (mpi4py fix worked) but `import tensorrt_llm` failed in datasets 2.14.4
  (`pyarrow.PyExtensionType`, removed in pyarrow 21). The lock now pins pyarrow 20.0.0. The next import failure was
  found locally: `tensorrt_llm/profiler.py` reads `pynvml.__version__`, which pynvml 13 (a shim over nvidia-ml-py)
  lacks, so pynvml is pinned to 11.5.3. Verified in ubuntu:24.04 with the notebook's exact install command and the
  CUDA driver and NVML stub libraries (cuda-driver-dev-12-6, cuda-nvml-dev-12-6 from NVIDIA's repo, sha256-checked;
  a test-only shim makes pynvml's init a no-op): `import tensorrt_llm`, the submodules `build_engine.py` and
  `serve.py` use, and `serve.py`'s patched upstream server all import. The install step also logs `uv pip check`.
- TensorRT-LLM third run (v3f, from fb5ea24, vLLM control on and valid again): install, import and model download
  worked; the checkpoint conversion then failed to save (`No such file or directory ... /tmp/trtllm/ckpt-fp16/.tmp...`)
  because 0.14's `save_checkpoint` does not create its output dir. `build_engine.py` now creates the checkpoint and
  engine dirs (the untied copy already did), and deletes the converted model inside `convert()` so its `__del__`
  (`torch.cuda.ipc_collect`) does not run at interpreter exit, where it raised `DeferredCudaCallError`. Verified on
  CPU in the same ubuntu:24.04 container (venv from the notebook's install command, CUDA/NVML stubs,
  `CUDA_VISIBLE_DEVICES=""`): the real Qwen2.5-1.5B-Instruct at the pinned revision was downloaded, untied (9 s) and
  converted and saved by `build_engine.convert()` (20 s, 7.4 GiB peak RSS): `config.json` (Qwen2ForCausalLM, float16,
  28 layers, share_embedding_table false) and `rank0.safetensors` (3.59 GB, 230 tensors, with lm_head). Every
  `trtllm-build` argument exists in 0.14.0's `--help` (`--kv_cache_type paged` parses case-insensitively). The engine
  build itself needs the GPU and was not run locally.
- TensorRT-LLM fourth run (v3h, from 414f88e): conversion worked; `trtllm-build` failed because the attention plugin
  needed 21,978,227,456 bytes of scratch with 15,636,037,632 available. With context FMHA off (no SM75 kernels) the
  unfused context path allocates 6 x max_batch_size x 12 heads x max_input_len^2 bytes for the qk buffers
  (`getWorkspaceSizeForContext` in 0.14's gptAttentionCommon.cpp; `workspace_bytes()` in `build_engine.py`
  reproduces 21,978,227,456 exactly for batch 32, input 3072). Measured prompt lengths (Qwen tokenizer at the pinned
  revision, chat template): loadgen max 893 tokens, GSM8K 5-shot max 1654 over the 250 items (from the v3h vLLM
  samples). One engine with batch 32 and input 2048 would need 9.85 GB, too much next to 3.1 GB of weights and a KV
  cache on a 15 GB T4. So two engines are built from one checkpoint: `bench` (batch 32, input 1024, seq 1152) for
  loadgen and `gsm8k` (batch 8, input 2048, seq 2304; GSM8K requests queue, greedy outputs do not depend on batching),
  each about 2.55 GB of workspace. Checked in the container: both commands parse in 0.14's trtllm-build (it then stops
  at CUDA init); the conversion is unchanged since the v3f check.

Separate sessions, so a failure in one cannot use the other's time.

### DCGM (item 10)

**Result (Kaggle, 2026-10-10, notebook built at 19ccae8; output `benchmark/results/raw/kaggle-20261010-v3a/`):**
the basic fields work, 96 values each in embedded and standalone mode. Every profiling field (GR engine, SM active,
SM occupancy, tensor pipe, DRAM active, FP16 pipe), probed together and one by one, fails with
`DCGMError_RequiresRoot: This operation is not supported when the host engine is running as non root`. Kaggle gives no
root over the host engine, so DCGM profiling cannot run there, and the memory-bound question (DRAM active) stays open
on Kaggle. The vLLM re-run with DCGM was not run for that reason (`NOT_RUN-vllm-t4-1.5b-fp16-dcgm.txt`). The same
session then crashed in the DCGM teardown (`HOSTENGINE.wait(30)` raised TimeoutExpired, uncaught), so its vision
part never ran. The next vision notebook does not run DCGM again: it writes `NOT_RUN-dcgm.txt` pointing to this
evidence, has no DCGM teardown, and its summary cell runs inside `step()` so it cannot raise.

How the probe worked:

DCGM 4.7.0-2: five .deb files (libdcgm, nv-hostengine, dcgmi, the proprietary package with the profiling module, the
Python bindings) are downloaded from NVIDIA's Ubuntu 24.04 CUDA repo, checked against the sha256 in its Packages index
and extracted with `dpkg-deb -x` (no apt, no maintainer scripts). The dcgm-exporter container cannot run on Kaggle (no
Docker). The notebook tries, and keeps verbatim in `results/dcgm/`: the Python bindings in embedded mode, a standalone
`nv-hostengine`, `dcgmi discovery -l`, `dcgmi profile -l` and `dcgmi dmon`, each for the basic field group (clocks,
clock event reasons, temperature, power, utilisation, framebuffer) and the profiling group (GR engine, SM active, SM
occupancy, tensor pipe, DRAM active, FP16 pipe), plus the process capabilities and the driver's profiling setting,
which decide whether a container may read the profiling counters. Each profiling field is also probed alone,
because one unsupported field (FP16 pipe may be one on Turing) can fail a whole watch. Only if SM active or DRAM
active returns values, vLLM 1.5B FP16 at c1 and c8 is re-run with `run.py --dcgm-fields` set to the basic fields plus
the profiling fields that worked (cell `vllm-t4-1.5b-fp16-dcgm`; DCGM is attached, so it does not replace run v2), and
`report.py` adds a DCGM table (basic fields with units, profiling ratios): DRAM active near 1 at c1 would confirm that
batch-1 decode is memory-bound. The probe counts idle zeros as values; `results/dcgm/rerun-check.json` records
whether SM/DRAM active were non-zero under load, and `report.py` flags all-zero ratios as "probably not measuring".
The sampler CSV stores each field's DCGM tag and unit. Licence: the DCGM proprietary package (profiling module) is
NVIDIA proprietary software, free to use; the rest of DCGM is Apache-2.0.
Untested risk: the profiling module may refuse inside the container (no CAP_SYS_ADMIN, or profiling restricted to
admin on the host); then `NOT_RUN-vllm-t4-1.5b-fp16-dcgm.txt` carries DCGM's own error text.

### Vision (item 11)

YOLOv8n from Ultralytics 8.4.174, **AGPL-3.0** (code and weights; serving it over a network brings AGPL obligations).
Weights `yolov8n.pt` from the `ultralytics/assets` v8.4.0 release, sha256 in the notebook config and
`docs/VERSIONS.md`. `inference/vision/export.py`: ONNX FP32 (opset 17, dynamic batch, exported on the CPU) and one
TensorRT 10.16 FP16 engine per batch size 1, 8, 32 (static shapes, built on the T4, I/O stays FP32; engine builds are
not bit-reproducible, so their sha256 is recorded, not pinned). `benchmark/vision/run_vision.py` times one batch per
request at concurrency 1 (the model call including copies; no decode, letterbox or NMS) and writes the loadgen
envelope with `"workload": "vision"`. Backends: ONNX Runtime CUDA (fails instead of falling back to the CPU),
TensorRT in process, and Triton over HTTP. mAP50-95 and mAP50 come from Ultralytics' validator on the first 500
val2017 images (`benchmark/vision/coco_val2017_first500.txt`, sha256 per image; labels from Ultralytics'
`coco2017labels.zip`, sha256-pinned) for PyTorch FP32 (reference), ONNX FP32 and TensorRT FP16. Validation uses
`rect=False` for every model (square 640x640 letterbox, batch 1): Ultralytics defaults to rectangular batches for
PyTorch and dynamic ONNX but square for static engines, which would mix preprocessing into the FP16 vs FP32 delta.
COCO licences: annotations CC BY 4.0, images under their Flickr terms of use; nothing from COCO is committed.
TensorRT is NVIDIA proprietary software, free to use.

Triton: the NGC `tritonserver` image needs Docker, and the PyPI `tritonserver` wheel is only the in-process Python API
(it needs `libtritonserver.so` from that image). PyTriton 0.7.0 ships a `tritonserver` binary with the Python backend
only, so `inference/vision/triton_serve.py` serves a Python callable that runs the TensorRT engine; Triton's native
TensorRT and ONNX Runtime backends are not used. Clients get the KServe v2 HTTP API and Prometheus metrics on port
8002. If it does not start, `FAILED-vision-triton.txt` and `logs/server-vision-triton.log` record why. Each Triton raw
file records the serving mode and the TensorRT version, and RESULTS.md labels those rows "Python backend via PyTriton
running the TensorRT engine; not Triton's native TensorRT backend".

CUDA 12 throughout (`notebooks/vision-requirements.txt`): onnxruntime-gpu 1.26.0 is the last CUDA 12 build (1.27 and
later need CUDA 13, so an R580 driver), TensorRT 10.16 the last 10.x (TensorRT 11 is strongly typed only and
Ultralytics then needs ModelOpt for FP16). PyTriton pins numpy below 2, which holds opencv-python at 4.11.

Local CPU baseline: `benchmark/vision/cpu_baseline.sh <yolov8n-fp32-opset17.onnx from the Kaggle output> "<hardware>"`
(ONNX Runtime 1.26.0 CPU, batch 1, 8 and 32, 3 repeats, same loadgen settings as the GPU cells). Run on 2026-10-10 on
a quiet machine with the v3c ONNX file (the one the ORT CUDA cells used): results in `benchmark/RESULTS.md`, evidence
in `docs/evidence/v3-vision-cpu.txt`.

### TensorRT-LLM (item 12)

Pinned: TensorRT-LLM 0.14.0 (TensorRT 10.4.0, torch 2.4.0+cu124, Python 3.10.20 venv, Open MPI 4.1.6-7ubuntu2 from
apt). mpi4py 3.1.6 builds from source, and its `conf/mpidistutils.py` calls `new_compiler(dry_run=...)`, which
setuptools 81.0.0 removed ("Removed support for the --dry-run parameter", #4872); with uv's default (newest) setuptools
in the build env the 2026-10-10 install failed with `TypeError: new_compiler() got an unexpected keyword argument
'dry_run'`. The install now passes `--build-constraints notebooks/trtllm-build-constraints.txt` (setuptools 80.10.2,
the last 80.x) and `UV_HTTP_TIMEOUT=300` for the 1.9 GB tensorrt_llm wheel. Both were tested in an Ubuntu 24.04
container with the same Open MPI packages (commands and results in the fix commit message). The 2026-10-10 vLLM control also
failed: `notebooks/vllm-requirements.txt` was not embedded in that notebook; `build_v3.py` now fails the build if a
cell references a `{BENCH}/` file that is not embedded. Why this version:
- The 0.14.0 support matrix (`docs/source/reference/support-matrix.md` at tag v0.14.0) lists the Turing architecture
  and "Turing (SM75) - FP32, FP16, INT8, INT4" (INT8 SmoothQuant not on SM75). 0.15.0's matrix lists neither Turing
  nor Volta, and a maintainer wrote on NVIDIA/TensorRT-LLM issue 2619 (T4, 0.15 dev build) "we don't test the latest
  TRT-LLM functionality on T4 platform as we remove its supporting since 0.14 release".
- Kernels: 0.14.0 and 0.15.0 compile `70-real` (`cpp/CMakeLists.txt`), which a T4 runs by binary compatibility; 0.16.0
  drops 70, so no later wheel has kernels a T4 can run. Today's matrix lists Ampere and newer only.
- The 0.14.0 wheel on pypi.nvidia.com is cp310 only, hence the 3.10 venv (the project's 3.12 rule is for host code).

T4 limits and workarounds, each from a source:
- Fused context attention (FMHA) has no SM75 kernels: `fmhaRunner.cpp` asserts "Unsupported architecture" and the
  maintainers say to build with `--context_fmha disable` (issue 1515). Context attention runs unfused.
- 0.14's Qwen converter fails on tied embeddings (issue 2388; Qwen2.5-1.5B support came in 0.15).
  `inference/trtllm/build_engine.py` writes an untied copy (lm_head = embed_tokens, the same tensors).
- FP16 only: no bf16 (the bf16 checkpoint is cast, as on vLLM), no FP8.
- No `trtllm-serve` before 0.15: `inference/trtllm/serve.py` loads the prebuilt engine into 0.14.0's
  `examples/apps/openai_server.py` (downloaded at the tag commit, sha256-pinned) and adds `ignore_eos` (dropped by
  0.14's request mapping; mapped to end_id = pad_id = -1, as 0.14's trtllm-bench does) and `/metrics` (HTTP requests
  and prompt tokens; 0.14's LLM API exposes no engine stats). KV cache block reuse is off.
- Stream patch: upstream `chat_stream_generator` stops streaming a choice on the first update with an empty text
  diff, while usage still counts every token, so client TPOT and ITL would undercount. `serve.py` imports a copy with
  that one branch changed to end the stream only when the sequence has a finish reason (`STREAM_PATCH`; it raises if
  the upstream text is not found exactly once). Detokenization skips special tokens, as vLLM does by default, so a
  trailing `<|im_end|>` cannot reach GSM8K answers.
- Triton's TensorRT-LLM backend (the "+ Triton" of PLAN item 12) is not run: it ships only in the NGC
  `tritonserver:<ver>-trtllm-python-py3` image, which needs Docker. The notebook records
  `NOT_RUN-triton-tensorrtllm-backend.txt`.
- Build: two engines from one checkpoint (after run v3h): `bench` `--max_batch_size 32 --max_input_len 1024
  --max_seq_len 1152` for loadgen and `gsm8k` `--max_batch_size 8 --max_input_len 2048 --max_seq_len 2304` for GSM8K,
  both with `--gemm_plugin float16 --gpt_attention_plugin float16 --context_fmha disable --remove_input_padding enable
  --kv_cache_type paged --max_num_tokens 8192`; one GPU (TP=2 would need mpirun; not attempted). Exact commands in
  `server_args/trtllm-build.txt`.

Loadgen cell `trtllm-t4-1.5b-fp16`: concurrency 1, 8, 32, 3 repeats, 128-token fixed decode; GSM8K 250 items if the
budget allows. The same session first runs a vLLM 0.31.0 control (before TensorRT-LLM, so a slow TensorRT-LLM step
cannot squeeze it out of the budget; `vllm-t4-1.5b-fp16`, same cells and GSM8K, with
`--generation-config vllm`, so no repetition_penalty), and `report.py` compares TensorRT-LLM only with vLLM rows of
the same session, never with run v2. The like-for-like statement is generated from the raw files: prompts sha256,
max_tokens, ignore_eos, concurrency levels, how many requests decoded a token count other than max_tokens, and how
many streams were truncated; any mismatch prints "Not compared" instead of a ratio. One difference stays: TensorRT-LLM's
KV cache takes 0.90 of the memory left after the engine loads, vLLM's 0.90 is of the whole card.

Risks that need a GPU to test: Open MPI from apt as root, unfused attention speed, and whether 0.14's runtime honours
end_id -1 through the LLM API as it does in trtllm-bench. The last one would not fail on its own: requests would stop
at EOS and still return 200, giving silent numbers. The checks catch it instead: `run.py` counts requests whose output
tokens differ from max_tokens (`summary.decode_checks`), and `report.py` flags any such cell and refuses the
TensorRT-LLM vs vLLM comparison. Truncated streams (fewer chunks than tokens and a long silent tail) are flagged the
same way, and their TPOT and ITL show as "invalid" while throughput, TTFT and end-to-end latency stay.

**Result (run v3i, 2026-10-10, from 693b44b; `benchmark/results/raw/kaggle-20261010-v3i/`):** both engines built and
served; every measured request decoded 128 tokens (so `end_id` -1 held) with 0 truncated streams; TensorRT-LLM vs the
same-session vLLM control: 68.9 vs 55.7 out tok/s at c1 (vLLM c1 ran at a 642.5 MHz SM clock vs 1040 MHz for TensorRT-LLM, so part of that gap may be clock), 368.3 vs 359.7 at c8, 613.8 vs 732.5 at c32; GSM8K strict-match
+0.012 (95% CI [-0.012, +0.036]). Tables, clock caveats and integrity checks: `benchmark/RESULTS.md`,
`docs/evidence/v3-trtllm.txt`.

### Push and fetch (after review; each push uses GPU quota)

```bash
git commit ...                                         # code first, so the embedded GIT_COMMIT is clean
benchmark/.venv/Scripts/python notebooks/build_v3.py   # rebuild both notebooks, then commit them
kaggle kernels push -p notebooks/kaggle-dcgm-vision -t 3600
kaggle kernels push -p notebooks/kaggle-trtllm -t 10800
kaggle kernels status <kaggle-user>/v3-dcgm-vision-t4
kaggle kernels output <kaggle-user>/v3-dcgm-vision-t4 -p benchmark/results/raw/kaggle-<YYYYMMDD>-v3a
kaggle kernels output <kaggle-user>/v3-trtllm-t4 -p benchmark/results/raw/kaggle-<YYYYMMDD>-v3b
rm -r benchmark/results/raw/kaggle-<YYYYMMDD>-v3*/bench    # embedded code copy, already in git
benchmark/.venv/Scripts/python benchmark/report.py
```
