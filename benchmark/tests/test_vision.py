"""Vision benchmark summary, the report's vision and DCGM tables, and the workload split in load_runs."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vision"), str(ROOT / "loadgen"), str(ROOT)]

from report import (  # noqa: E402
    HERE,
    dcgm_means,
    dcgm_section,
    load_runs,
    vision_section,
)
from run_vision import letterbox, summarize  # noqa: E402


def req(start, end, batch=8, warmup=False, error=None):
    return {
        "start": start,
        "end": end,
        "batch": batch,
        "warmup": warmup,
        "error": error,
    }


def test_summary_counts_images_and_ignores_warmup_and_errors():
    recs = [
        req(0, 5, warmup=True),
        req(10, 10.1),
        req(10.1, 10.3),
        req(10.3, 10.4, error="boom"),
    ]
    s = summarize(recs)
    assert (
        s["requests"] == 3 and s["ok"] == 2 and s["error_rate"] == pytest.approx(1 / 3)
    )
    assert s["images"] == 16
    assert s["wall_s"] == pytest.approx(0.4)
    assert s["images_s"] == pytest.approx(16 / 0.4)
    assert s["latency_ms"]["p50"] == pytest.approx(150)  # 100 ms and 200 ms


def test_letterbox_keeps_aspect_and_pads(tmp_path):
    from PIL import Image

    p = tmp_path / "wide.jpg"
    Image.new("RGB", (200, 100), (255, 0, 0)).save(p)
    x = letterbox(p, 64)
    assert x.shape == (3, 64, 64) and x.dtype.name == "float32"
    assert x[:, 0, 0] == pytest.approx([114 / 255] * 3)  # padded top row
    assert x[0, 32, 32] == pytest.approx(1.0, abs=0.02)  # red image centre


def vrun(label, batch, rep, ips, session="kaggle-20261012", workload="vision"):
    return {
        "schema": 1,
        "workload": workload,
        "label": label,
        "engine": "trt",
        "engine_version": "10.16.1.11",
        "model": "yolov8n-fp16-b8.engine",
        "model_sha256": "abc",
        "precision": "fp16",
        "batch": batch,
        "repeat": rep,
        "started_utc": f"2026-10-12T10:00:0{rep}+00:00",
        "config": {"requests": 80},
        "env": {"hardware": "1x Tesla T4 (Kaggle)", "git_commit": "c0ffee"},
        "server_args": None,
        "gpu_samples": None,
        "summary": {
            "images_s": ips,
            "error_rate": 0.0,
            "latency_ms": {"p50": 8.0, "p95": 9.0, "p99": 10.0},
        },
        "_path": HERE / "results" / "raw" / session / f"{label}-b{batch}-r{rep}.json",
        "_session": session,
        "_cell": f"{session}/{label}",
    }


def test_vision_cpu_rows_show_threads_and_model_sha():
    runs = []
    for rep, threads in ((1, 0), (2, 0), (3, 4)):
        r = vrun("vision-cpu", 1, rep, 20.0, session="cpu-vision-x")
        r.update(
            engine="ort-cpu", engine_version="1.26.0", model_sha256="c3a4c32fa3939177"
        )
        r["env"]["backend"] = {"intra_op_num_threads": threads}
        runs.append(r)
    md = "\n".join(vision_section(runs[:2], []))
    assert "ort-cpu 1.26.0 (intra-op threads ORT default)" in md
    assert "sha256 c3a4c32fa393" in md
    assert "intra-op threads 4" in "\n".join(vision_section(runs[2:], []))


def test_vision_section_absent_without_data():
    assert vision_section([], []) == []


def test_vision_section_table_flags_and_map_delta():
    runs = [vrun("vision-t4-trt", 8, r, ips) for r, ips in ((1, 1000.0), (2, 1010.0))]
    acc = [
        {
            "label": label,
            "model": "m",
            "precision": prec,
            "dataset": {"images": 500},
            "metrics": {"metrics/mAP50-95(B)": m, "metrics/mAP50(B)": m + 0.15},
            "_session": "kaggle-20261012",
            "_path": HERE / "results" / "raw" / "kaggle-20261012" / f"{label}.json",
        }
        for label, prec, m in (
            ("onnx-fp32", "fp32", 0.370),
            ("trt-fp16", "fp16", 0.368),
        )
    ]
    md = "\n".join(vision_section(runs, acc))
    assert (
        "| kaggle-20261012/vision-t4-trt | 1x Tesla T4 (Kaggle) | trt 10.16.1.11 | fp16 | 8 | 2 | 1005.0 ± 7.1 |"
        in md
    )
    assert "only 2 repeat(s), need 3" in md
    assert "TensorRT FP16 minus ONNX FP32 mAP50-95 = -0.0020" in md


def test_dcgm_means_and_section():
    samples = [
        {"utc": "t", "gpu": "0", "field": "1005", "value": "0.8"},
        {"utc": "t", "gpu": "0", "field": "1005", "value": "0.6"},
        {"utc": "t", "gpu": "0", "field": "1002", "value": "bad"},
    ]
    run = {"dcgm_samples": samples, "dcgm_error": None, "repeat": 1, "_rep": "r1"}
    assert dcgm_means(run) == {1005: pytest.approx(0.7)}
    md = "\n".join(dcgm_section({("kaggle-x/vllm", 1): [run]}))
    # 5 basic fields n/a, then GR, SM active, occupancy, tensor n/a, DRAM 0.700, FP16 n/a; 2 parsed values
    assert "### DCGM" in md
    assert "| kaggle-x/vllm | 1 | 1 |" + " n/a |" * 9 + " 0.700 | n/a | 2 |" in md
    zero = {
        "dcgm_samples": [{"field": "1005", "value": "0"}],
        "repeat": 1,
        "_rep": "r1",
    }
    assert "probably not measuring" in "\n".join(dcgm_section({("k/v", 1): [zero]}))
    assert dcgm_section({("kaggle-x/vllm", 1): [{"repeat": 1}]}) == []


def test_triton_rows_say_python_backend():
    r = vrun("vision-t4-trt-triton", 8, 1, 900.0)
    r["engine"], r["engine_version"] = "triton", "2.59.0"
    r["env"]["backend"] = {"tensorrt": "10.16.1.11"}
    md = "\n".join(vision_section([r], []))
    assert (
        "triton 2.59.0 (Python backend via PyTriton running the TensorRT engine; not Triton's native"
        in md
    )
    assert "TensorRT 10.16.1.11)" in md


def test_load_runs_keeps_only_llm_workload():
    docs = [
        {"label": "a"},
        {"label": "b", "workload": "llm"},
        {"label": "c", "workload": "vision"},
    ]
    assert [d["label"] for d in load_runs(docs)] == ["a", "b"]


def test_data_yaml_has_train_and_val_for_every_model_format():
    # Ultralytics 8.4.174 check_det_dataset raises "'train:' key missing" without train (Kaggle v3c, 2026-10-10).
    # The same dict is written for .pt, .onnx and .engine models; the CPU check ran .pt and .onnx for real.
    from eval_map import data_yaml_dict

    d = data_yaml_dict(Path("/data/coco500"), {0: "person"})
    assert d["train"] == d["val"] == "val.txt" and d["names"] == {0: "person"}
