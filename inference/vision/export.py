"""Export YOLOv8n (Ultralytics, AGPL-3.0) to an ONNX FP32 model and TensorRT FP16 engines, with sha256 of each.

- ONNX: opset 17, dynamic batch and image size, onnxslim simplified, exported on the CPU (FP32). Opset 17 rather
  than Ultralytics' default 18+: ONNX Runtime CUDA lacks some opset 19+ kernels (Ultralytics' own note).
- TensorRT: one engine per static batch size (FP16 builder flag; inputs and outputs stay FP32), built on the GPU
  it will run on. Static shapes so each batch size gets tactics tuned for it. T4: FP16 only (no FP8, no bf16).
  Engine builds are not bit-reproducible (tactic timing), so each sha256 is recorded, not pinned.

python export.py yolov8n.pt OUT_DIR --batches 1 8 32
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

OPSET = 17
IMGSZ = 640


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("weights")
    p.add_argument("out_dir")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    args = p.parse_args()

    import onnx
    import tensorrt
    import torch
    import ultralytics
    from ultralytics import YOLO

    out = Path(args.out_dir)
    work = (
        out / "work"
    )  # Ultralytics writes exports next to the weights; the engine export reuses yolov8n.onnx
    work.mkdir(parents=True, exist_ok=True)
    src = work / "yolov8n.pt"
    shutil.copy2(args.weights, src)
    files = {}

    f = YOLO(src).export(
        format="onnx",
        opset=OPSET,
        dynamic=True,
        simplify=True,
        imgsz=IMGSZ,
        device="cpu",
    )
    dst = out / f"yolov8n-fp32-opset{OPSET}.onnx"
    shutil.move(f, dst)
    files[dst.name] = {"sha256": sha256(dst), "precision": "fp32", "batch": "dynamic"}

    for b in args.batches:
        f = YOLO(src).export(
            format="engine",
            quantize=16,
            batch=b,
            dynamic=False,
            opset=OPSET,
            simplify=True,
            imgsz=IMGSZ,
            device=0,
        )
        dst = out / f"yolov8n-fp16-b{b}.engine"
        shutil.move(f, dst)
        (work / "yolov8n.onnx").unlink(
            missing_ok=True
        )  # FP32 intermediate of the engine build
        files[dst.name] = {"sha256": sha256(dst), "precision": "fp16", "batch": b}

    (out / "exports.json").write_text(
        json.dumps(
            {
                "weights": Path(args.weights).name,
                "weights_sha256": sha256(args.weights),
                "opset": OPSET,
                "imgsz": IMGSZ,
                "versions": {
                    "ultralytics": ultralytics.__version__,
                    "torch": torch.__version__,
                    "onnx": onnx.__version__,
                    "tensorrt": tensorrt.__version__,
                },
                "files": files,
            },
            indent=1,
        )
        + "\n"
    )
    shutil.rmtree(work)
    print(json.dumps(files, indent=1))


if __name__ == "__main__":
    main()
