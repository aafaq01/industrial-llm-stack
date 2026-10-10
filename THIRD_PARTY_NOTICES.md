# Third-party notices

The code in this repo is Apache-2.0 (see [LICENSE](LICENSE)). The parts below are not part of that grant. They are
downloaded or pulled at run time, unless the table says otherwise, and keep their own licences. Check each one
before you redistribute an image or ship a product.

## Models and data

| Component | Licence | How it is used here |
|-----------|---------|---------------------|
| Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct (FP16, AWQ, GGUF) | Apache-2.0 | downloaded by URL and pinned revision. Other Qwen2.5 sizes (3B, 72B) use the Qwen licence, so check before swapping |
| BAAI/bge-small-en-v1.5 (embedder) | MIT | **baked into the rag image** (`rag/Dockerfile`); keep this notice with the image |
| cross-encoder/ms-marco-MiniLM-L6-v2 (reranker) | Apache-2.0 | **baked into the rag image**; keep this notice with the image |
| YOLOv8n weights | AGPL-3.0 (Ultralytics) | downloaded for the optional vision benchmark; exported ONNX and TensorRT files are not committed |
| GSM8K | MIT | question samples appear in committed lm-eval outputs under `benchmark/results/raw/` |
| COCO val2017 | annotations CC BY 4.0; images under Flickr terms | only file names are committed (`benchmark/vision/coco_val2017_first500.txt`); images are downloaded |
| NASA and NIST documents (RAG corpus) | US government works, public domain in the US | URLs and sha256 only, see [rag/data/SOURCES.md](rag/data/SOURCES.md); short excerpts appear in `rag/eval/` |

## Software with copyleft or non-OSS terms

| Component | Licence | How it is used here |
|-----------|---------|---------------------|
| Ultralytics 8.4.174 | AGPL-3.0 | imported by `inference/vision/export.py` and `benchmark/vision/eval_map.py` only. If you embed or serve that path over a network, AGPL terms apply (or buy an Ultralytics licence). The rest of the platform does not import it |
| Elasticsearch 9.5.5 | AGPL-3.0, SSPL or Elastic License 2.0 (your choice) | unmodified image. OpenSearch (Apache-2.0) is the usual swap and needs a Haystack store change |
| Grafana, Loki, Tempo, Alloy | AGPL-3.0 | unmodified images |
| MinIO (Langfuse profile only) | AGPL-3.0 | unmodified image, optional |
| TensorRT, tensorrt-cu12, the tritonserver bundled in PyTriton | NVIDIA proprietary (free to use) | pip-installed inside Kaggle notebooks only; engines and wheels are not committed or redistributed |
| DCGM 4.7.0 | Apache-2.0, except the `datacenter-gpu-manager-4-proprietary` package (NVIDIA proprietary) | installed inside a Kaggle notebook only |

## Permissive software (pulled, not modified)

llama.cpp (MIT), vLLM (Apache-2.0), TensorRT-LLM (Apache-2.0), Haystack (Apache-2.0), FastAPI (MIT),
PostgreSQL (PostgreSQL licence), Prometheus, OpenTelemetry Collector, ArgoCD, KEDA, Traefik, Jenkins (Apache-2.0 or MIT),
Langfuse (MIT core). Python dependencies are pinned in the `requirements*.txt` files and `rag/uv.lock`; each package
carries its own licence.
