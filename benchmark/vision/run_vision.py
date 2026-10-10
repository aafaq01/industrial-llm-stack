"""Closed-loop batch benchmark for an image model: ONNX Runtime (CPU or CUDA), TensorRT, or a Triton HTTP endpoint.

One request = one batch through the model, sent as soon as the previous one returns (concurrency 1). Images are
letterboxed to --imgsz once up front, so the timings cover the model call including host to device copies and
the output copy back, not JPEG decode, resize or NMS. Writes one raw JSON per (batch, repeat) in the loadgen
envelope (config, env, git commit, per-request records, summary), marked "workload": "vision".

python benchmark/vision/run_vision.py --backend ort-cpu --model yolov8n.onnx --images /tmp/coco500/images/val2017 \
  --label vision-cpu-yolov8n-onnx-fp32 --precision fp32 --batch 1 8 --repeats 3 --out-dir benchmark/results/raw
"""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "loadgen"))
from metrics import dist  # noqa: E402
from run import GpuSampler, cpu_model, git_commit, sh, utcnow  # noqa: E402


def letterbox(path, size):
    """RGB float32 CHW in [0, 1], resized to fit size x size with aspect kept, padded with 114 (Ultralytics' value)."""
    im = Image.open(path).convert("RGB")
    s = size / max(im.size)
    im = im.resize(
        (max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR
    )
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0


class OrtModel:
    def __init__(self, path, device, threads):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        if device == "cuda":
            ort.preload_dlls()  # CUDA and cuDNN from the nvidia-* wheels in the venv
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, so, providers=providers)
        active = self.sess.get_providers()
        if device == "cuda" and active[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                f"CUDA requested but ONNX Runtime runs on {active}"
            )  # no silent CPU fallback
        self.inp = self.sess.get_inputs()[0].name
        self.info = {
            "onnxruntime": ort.__version__,
            "providers": active,
            "provider_options": self.sess.get_provider_options(),
            "intra_op_num_threads": so.intra_op_num_threads,
        }

    def __call__(self, x):
        return self.sess.run(None, {self.inp: x})


class TrtModel:
    """TensorRT engine runner with torch CUDA tensors as buffers. Accepts Ultralytics engine files (4-byte length +
    JSON metadata before the plan) and plain plans."""

    def __init__(self, path):
        import tensorrt as trt
        import torch

        data = Path(path).read_bytes()
        n = int.from_bytes(data[:4], "little", signed=True)
        try:
            self.meta = json.loads(data[4 : 4 + n])
            data = data[4 + n :]
        except (ValueError, UnicodeDecodeError):
            self.meta = None
        self.engine = trt.Runtime(
            trt.Logger(trt.Logger.WARNING)
        ).deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {path}")
        self.ctx = self.engine.create_execution_context()
        names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.inp = next(
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self.outs = [
            n
            for n in names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.dtype = {n: dtypes[self.engine.get_tensor_dtype(n)] for n in names}
        self.torch = torch
        self.info = {
            "tensorrt": trt.__version__,
            "io": {
                n: [
                    str(self.engine.get_tensor_dtype(n)),
                    list(self.engine.get_tensor_shape(n)),
                ]
                for n in names
            },
            "metadata": self.meta,
        }

    def __call__(self, x):
        torch = self.torch
        t = torch.from_numpy(x).to("cuda", self.dtype[self.inp])
        self.ctx.set_input_shape(self.inp, tuple(t.shape))
        self.ctx.set_tensor_address(self.inp, t.data_ptr())
        outs = []
        for n in self.outs:
            o = torch.empty(
                tuple(self.ctx.get_tensor_shape(n)), dtype=self.dtype[n], device="cuda"
            )
            self.ctx.set_tensor_address(n, o.data_ptr())
            outs.append(o)
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("execute_async_v3 returned False")
        return [o.cpu().numpy() for o in outs]  # .cpu() synchronizes


TRITON_SERVING = "Python backend via PyTriton running the TensorRT engine; not Triton's native TensorRT backend"


class TritonModel:
    def __init__(self, url, name):
        import tritonclient.http as th

        self.th, self.name = th, name
        self.client = th.InferenceServerClient(url)
        md = self.client.get_model_metadata(name)
        self.inp = md["inputs"][0]["name"]
        self.outs = [o["name"] for o in md["outputs"]]
        import tensorrt

        self.info = {
            "triton_server": self.client.get_server_metadata(),
            "model_metadata": md,
            # inference/vision/triton_serve.py runs in this same venv, so this is the server's TensorRT
            "tensorrt": tensorrt.__version__,
            "serving": TRITON_SERVING,
        }

    def __call__(self, x):
        th = self.th
        i = th.InferInput(self.inp, list(x.shape), "FP32")
        i.set_data_from_numpy(x, binary_data=True)
        r = self.client.infer(
            self.name,
            [i],
            outputs=[th.InferRequestedOutput(o, binary_data=True) for o in self.outs],
        )
        return [r.as_numpy(o) for o in self.outs]


def load_model(args):
    if args.backend in ("ort-cpu", "ort-cuda"):
        return OrtModel(args.model, args.backend.removeprefix("ort-"), args.threads)
    if args.backend == "trt":
        return TrtModel(args.model)
    return TritonModel(args.triton_url, args.model)


def summarize(records):
    """Summary over measured (non-warmup) batch requests. Latency is per batch, in ms."""
    measured = [r for r in records if not r["warmup"]]
    ok = [r for r in measured if not r["error"]]
    wall = (
        (max(r["end"] for r in measured) - min(r["start"] for r in measured))
        if measured
        else 0
    )
    images = sum(r["batch"] for r in ok)
    return {
        "requests": len(measured),
        "ok": len(ok),
        "errors": len(measured) - len(ok),
        "error_rate": (len(measured) - len(ok)) / len(measured) if measured else None,
        "wall_s": wall,
        "images": images,
        "images_s": images / wall if wall > 0 else None,
        "latency_ms": dist([(r["end"] - r["start"]) * 1000 for r in ok]),
    }


def run_once(model, batches, n_warm, n, batch, t0):
    """n_warm + n requests; request i uses batch slice i (cycling through the preloaded images)."""
    records = []
    for i in range(n_warm + n):
        x = batches[i % len(batches)]
        rec = {"i": i, "batch": batch, "warmup": i < n_warm, "error": None}
        rec["start"] = round(time.perf_counter() - t0, 6)
        try:
            model(x)
        except Exception as e:  # noqa: BLE001 recorded, never raised: a failed request is data
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["end"] = round(time.perf_counter() - t0, 6)
        records.append(rec)
    return records


def scrape(url):
    if not url:
        return None
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return [
                ln
                for ln in r.read().decode().splitlines()
                if ln and not ln.startswith("#")
            ]
    except OSError as e:
        return f"scrape failed: {e}"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--backend", required=True, choices=["ort-cpu", "ort-cuda", "trt", "triton"]
    )
    p.add_argument(
        "--model", required=True, help="ONNX or engine file; for triton the model name"
    )
    p.add_argument(
        "--model-file",
        default=None,
        help="triton: the engine file behind the model, for its sha256",
    )
    p.add_argument("--triton-url", default="127.0.0.1:8000")
    p.add_argument(
        "--metrics-url",
        default=None,
        help="Prometheus /metrics to snapshot before and after",
    )
    p.add_argument(
        "--images",
        required=True,
        help="directory of .jpg files (sorted, first --num-images used)",
    )
    p.add_argument("--num-images", type=int, default=64)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, nargs="+", default=[1])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--images-per-run",
        type=int,
        default=640,
        help="measured requests = max(20, ceil(this / batch))",
    )
    p.add_argument("--warmup", type=int, default=10, help="warmup requests per run")
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op threads (0 = its default)",
    )
    p.add_argument("--label", required=True)
    p.add_argument("--precision", required=True)
    p.add_argument("--hardware", default=None)
    p.add_argument(
        "--server-args", default=None, help="server command line, stored verbatim"
    )
    p.add_argument("--git-commit", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="mark as smoke run (excluded from RESULTS.md)",
    )
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    files = sorted(Path(args.images).glob("*.jpg"))[: args.num_images]
    if not files:
        raise SystemExit(f"no .jpg files in {args.images}")
    imgs = np.stack([letterbox(f, args.imgsz) for f in files])
    model_file = args.model_file or (args.model if args.backend != "triton" else None)
    model = load_model(args)
    env = {
        "hardware": args.hardware,
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": args.git_commit or git_commit(),
        "nvidia_smi": sh(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        )
        if shutil.which("nvidia-smi")
        else None,
        "backend": model.info,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for batch in args.batch:
        # Contiguous batches cycling through the preloaded images; built once so the loop only times the model.
        batches = [
            np.ascontiguousarray(imgs[np.arange(i, i + batch) % len(imgs)])
            for i in range(0, len(imgs), batch)
        ]
        n = max(20, math.ceil(args.images_per_run / batch))
        for rep in range(1, args.repeats + 1):
            t0 = time.perf_counter()
            started = utcnow()
            warm = run_once(model, batches, args.warmup, 0, batch, t0)
            before = scrape(args.metrics_url)
            gpu = GpuSampler()
            measured = run_once(model, batches, 0, n, batch, t0)
            gpu_samples = gpu.stop()
            res = {
                "schema": 1,
                "workload": "vision",
                "smoke": args.smoke,
                "label": args.label,
                "engine": args.backend,
                "engine_version": model.info["triton_server"].get("version")
                if args.backend == "triton"
                else model.info.get("onnxruntime") or model.info.get("tensorrt"),
                "model": Path(model_file).name if model_file else args.model,
                "model_sha256": hashlib.sha256(
                    Path(model_file).read_bytes()
                ).hexdigest()
                if model_file
                else None,
                "precision": args.precision,
                "batch": batch,
                "concurrency": 1,
                "repeat": rep,
                "started_utc": started,
                "finished_utc": utcnow(),
                "config": {
                    "images_dir": Path(args.images).name,
                    "images": [f.name for f in files],
                    "imgsz": args.imgsz,
                    "requests": n,
                    "warmup_requests": args.warmup,
                    "timed": "model call incl. host-device copies; excludes decode, letterbox, NMS",
                    "notes": args.notes,
                },
                "env": env,
                "server_args": args.server_args,
                "gpu_samples": gpu_samples,
                "server_metrics_before": before,
                "server_metrics_after": scrape(args.metrics_url),
                "summary": summarize(warm + measured),
                "requests": warm + measured,
            }
            stamp = started.replace(":", "").replace("-", "").replace("+0000", "Z")
            path = out_dir / f"{args.label}-b{batch}-r{rep}-{stamp}.json"
            path.write_text(
                json.dumps(res, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            s = res["summary"]
            print(
                f"{path.name}: ok {s['ok']}/{s['requests']} images/s {s['images_s'] or 0:.1f} "
                f"p50 {s['latency_ms']['p50'] or 0:.2f} ms p99 {s['latency_ms']['p99'] or 0:.2f} ms",
                flush=True,
            )


if __name__ == "__main__":
    main()
