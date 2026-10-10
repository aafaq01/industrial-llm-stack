# Kaggle run v3e (2026-10-10, ~12:16-12:37Z), notebook dcgm_vision_t4 from fb5ea24: partial

- Valid: YOLOv8n TensorRT FP16 engine served through Triton via PyTriton (Python backend running the TensorRT
  engine, not Triton's native TensorRT backend), batch 1, 8, 32 x 3 (results/raw/vision-t4-yolov8n-trt-fp16-triton/).
  Subject to review. The `_contextvars` fix (own libpython first on LD_LIBRARY_PATH) worked.
- Failed: all three mAP steps. Kaggle exports MPLBACKEND=module://matplotlib_inline.backend_inline into child
  processes; the vision venv has no matplotlib_inline, so importing matplotlib failed. Fixed in the next run
  (MPLBACKEND=Agg).
- Not run on purpose: speed cells (see kaggle-20261010-v3c) and DCGM (see kaggle-20261010-v3a).
- results/models/ is git-ignored.
