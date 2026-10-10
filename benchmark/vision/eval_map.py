"""COCO mAP50-95 and mAP50 of a YOLOv8n model file on the fixed val2017 subset (fetch_coco_subset.py), with
Ultralytics' own validator (square 640 letterbox, NMS, COCO-style metrics), batch 1, rect off, conf 0.001, IoU 0.7.

python benchmark/vision/eval_map.py MODEL DATA_DIR --label trt-fp16 --precision fp16 --out results/raw/vision-accuracy
Writes <out>/<label>.json marked "workload": "vision-accuracy".
"""

import argparse
import datetime as dt
import hashlib
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_yaml_dict(data, names):
    """Ultralytics dataset YAML for the subset. check_det_dataset (8.4.174) requires a train key even for val-only
    use, so train points at the same image list; nothing is trained."""
    return {"path": str(data), "train": "val.txt", "val": "val.txt", "names": names}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("model")
    p.add_argument("data_dir")
    p.add_argument("--label", required=True)
    p.add_argument("--precision", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--git-commit", default=None)
    p.add_argument("--hardware", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO

    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    names = yaml.safe_load(
        (Path(ultralytics.__file__).parent / "cfg/datasets/coco.yaml").read_text()
    )["names"]
    data = Path(args.data_dir).resolve()
    images = len((data / "val.txt").read_text().splitlines())
    work = Path(tempfile.mkdtemp())
    data_yaml = work / "coco-subset.yaml"
    data_yaml.write_text(yaml.safe_dump(data_yaml_dict(data, names)))
    val_args = {
        "imgsz": 640,
        "batch": 1,
        # Square 640x640 letterbox for every model. Model.val defaults rect=True and the validator only turns it
        # off for non-PyTorch static models, so without this PyTorch and dynamic ONNX would use rectangular
        # batches while the static TensorRT engine uses square ones, mixing preprocessing into the FP16 delta.
        "rect": False,
        "conf": 0.001,
        "iou": 0.7,
        "plots": False,
        "verbose": False,
    }
    r = YOLO(args.model, task="detect").val(
        data=str(data_yaml),
        device=args.device,
        project=str(work),
        name=args.label,
        **val_args,
    )
    res = {
        "schema": 1,
        "workload": "vision-accuracy",
        "label": args.label,
        "model": Path(args.model).name,
        "model_sha256": sha256(args.model),
        "precision": args.precision,
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "dataset": {
            "name": "COCO val2017, first 500 sorted image names",
            "images": images,
            "manifest_sha256": sha256(HERE / "coco_val2017_first500.txt"),
        },
        "val_args": val_args,
        "env": {
            "hardware": args.hardware,
            "git_commit": args.git_commit,
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
        },
        "metrics": {k: float(v) for k, v in r.results_dict.items()},
        "speed_ms": r.speed,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(args.label, json.dumps(res["metrics"]))


if __name__ == "__main__":
    main()
