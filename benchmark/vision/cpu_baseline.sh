#!/usr/bin/env bash
# Local CPU baseline for YOLOv8n: ONNX Runtime CPU (FP32), batch 1, 8 and 32, 3 repeats, same loadgen settings as the
# Kaggle GPU cells (first 64 images, 640 images per run, 10 warmup requests, imgsz 640). ONNX Runtime's default
# intra-op thread count unless THREADS=N is set. Run on a quiet machine (no kind cluster, no compose stack).
# Usage: benchmark/vision/cpu_baseline.sh ONNX_FILE "HARDWARE LABEL" [OUT_DIR] [extra run_vision.py args]
#   ONNX_FILE  yolov8n-fp32-opset17.onnx from the Kaggle output (results/models/); its sha256 lands in every raw
#              JSON, so it can be matched to results/models/exports.json of the GPU run.
# The venv comes from benchmark/vision/requirements.txt (onnxruntime 1.26.0, same version as the GPU run).
set -euo pipefail
ONNX=$1 HW=$2 OUT=${3:-benchmark/results/raw/cpu-vision-$(date -u +%Y%m%d)}
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=$HERE/.venv
[ -d "$VENV" ] || { uv venv -q --python 3.12 "$VENV" && uv pip install -q --python "$VENV" -r "$HERE/requirements.txt"; }
PY=$VENV/bin/python
[ -x "$PY" ] || PY=$VENV/Scripts/python.exe  # Windows
DATA=${DATA:-$HERE/.data/coco500}
THREADS=${THREADS:-0}
"$PY" "$HERE/fetch_coco_subset.py" --out "$DATA" --limit 64
"$PY" "$HERE/run_vision.py" --backend ort-cpu --model "$ONNX" --images "$DATA/images/val2017" \
  --batch 1 8 32 --repeats 3 --images-per-run 640 --warmup 10 --num-images 64 --threads "$THREADS" \
  --label vision-cpu-yolov8n-onnx-fp32 --precision fp32 --hardware "$HW" --out-dir "$OUT" "${@:4}"
