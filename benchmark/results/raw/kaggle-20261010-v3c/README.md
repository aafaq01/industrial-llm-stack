# Kaggle run v3c (2026-10-10 09:30-09:49Z), notebook dcgm_vision_t4 from 80905ea: partial

- Valid: YOLOv8n speed cells on 1x T4, ONNX Runtime CUDA (FP32) and TensorRT FP16 engines, batch 1, 8, 32,
  3 repeats each (results/raw/). Subject to review before reporting.
- Failed: all three mAP steps (eval_map.py data YAML had no `train:` key, rejected by ultralytics 8.4.174), and
  Triton via PyTriton (the Python backend stub could not import `_contextvars` from the uv-managed standalone
  CPython 3.12.13). See FAILED-*.txt and results/logs/.
- DCGM not re-run on purpose: see NOT_RUN-dcgm.txt and kaggle-20261010-v3a.
- results/models/ (exported .onnx and .engine) is git-ignored.
