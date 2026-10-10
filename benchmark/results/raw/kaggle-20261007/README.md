# Kaggle run v1 (2026-10-07): failed, not a result

Run v1 failed before any benchmark: the session had no GPU and no internet. No benchmark or accuracy number exists
for this run.

Evidence (all in this folder, downloaded with `kaggle kernels output`):

- `results/env/run.json`: `"gpu_count": 0`, started 2026-10-07T16:28:49Z, git commit `3953d846`.
- `results/env/nvidia-smi.txt`: `/bin/sh: 1: nvidia-smi: not found`.
- `vllm-t4-benchmark.log`: `pip install uv==0.11.32` failed with "Temporary failure in name resolution" for pypi.org.
- `results/logs/install-vllm.log`: uv timed out downloading python-build-standalone from github.com.

Likely cause: the Kaggle account was not phone verified, so the kernel ran on CPU without internet. The notebook
now has a preflight cell that stops at once in this situation and writes `FAILED-preflight.txt`.

The embedded code copy (`bench/`) was deleted because it duplicates committed files.
