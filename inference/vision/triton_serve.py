"""Serve YOLOv8n TensorRT FP16 engines through Triton Inference Server on Kaggle, without Docker.

The NGC tritonserver image cannot run on Kaggle (no Docker), and the PyPI `tritonserver` wheel is only the
in-process Python API: it needs libtritonserver.so from that image. PyTriton (nvidia-pytriton, Apache-2.0) ships a
tritonserver binary with the Python backend only, so Triton here serves a Python callable that runs the TensorRT
engine (TrtModel from benchmark/vision/run_vision.py); Triton's own TensorRT and ONNX Runtime backends are not
used. Clients get Triton's HTTP/gRPC KServe v2 API and Prometheus metrics on port 8002.

One model per static-batch engine: yolov8n_b<N>, max_batch_size N, so a request of N images runs the engine built
for N (dynamic batching cannot form larger batches than N).

python triton_serve.py yolov8n-fp16-b1.engine yolov8n-fp16-b8.engine ... --log-file results/logs/triton.log
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark" / "vision"))


def use_own_libpython():
    """Make PyTriton's Python backend stub load this interpreter's libpython3.12.so.

    The stub links libpython3.12.so.1.0 by name. With a uv-managed (python-build-standalone) Python, the loader
    either finds no libpython (error at stub start) or, on Kaggle, the image's Ubuntu libpython3.12, whose stdlib
    extension layout differs: "ModuleNotFoundError: No module named '_contextvars'" (Kaggle 2026-10-10, v3c). PyTriton
    copies os.environ into tritonserver's environment and only appends its own libs to LD_LIBRARY_PATH, so putting
    sys.base_prefix/lib first makes the stub use the same Python as this process. Reproduced and verified in
    ubuntu:24.04 with a CPU model (2026-10-10)."""
    lib = str(Path(sys.base_prefix) / "lib")
    old = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = lib + (f":{old}" if old else "")
    return lib


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "engines", nargs="+", help="Ultralytics engine files named ...-b<N>.engine"
    )
    p.add_argument("--log-file", default=None)
    args = p.parse_args()

    import numpy as np
    from pytriton.decorators import batch
    from pytriton.model_config import ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig
    from run_vision import TrtModel

    def make_infer(model):
        @batch
        def infer(**inputs):
            (x,) = inputs.values()
            return {
                o: y.astype(np.float32, copy=False)
                for o, y in zip(model.outs, model(x), strict=True)
            }

        return infer

    print("libpython for the Python backend stub:", use_own_libpython(), flush=True)
    config = TritonConfig(
        http_port=8000, grpc_port=8001, metrics_port=8002, log_file=args.log_file
    )
    with Triton(config=config) as triton:
        for path in args.engines:
            n = int(re.search(r"-b(\d+)\.engine$", path).group(1))
            model = TrtModel(path)
            shapes = {k: v[1] for k, v in model.info["io"].items()}
            triton.bind(
                model_name=f"yolov8n_b{n}",
                infer_func=make_infer(model),
                inputs=[
                    Tensor(
                        name=model.inp,
                        dtype=np.float32,
                        shape=tuple(shapes[model.inp][1:]),
                    )
                ],
                outputs=[
                    Tensor(name=o, dtype=np.float32, shape=tuple(shapes[o][1:]))
                    for o in model.outs
                ],
                config=ModelConfig(max_batch_size=n),
            )
        triton.serve()


if __name__ == "__main__":
    main()
