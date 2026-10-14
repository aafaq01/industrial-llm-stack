# Pinned versions

Pinned 2026-10-07. Every tag below was checked to exist in its registry on that date.

## Container images

| Service | Image |
|---------|-------|
| model-init | `curlimages/curl:8.22.0` |
| llamacpp | `ghcr.io/ggml-org/llama.cpp:server-b11459` (CPU build) |
| postgres | `postgres:18.6-alpine3.24` |
| elasticsearch | `elasticsearch:9.5.5` |
| prometheus | `prom/prometheus:v3.15.0` |
| grafana | `grafana/grafana:13.2.3` |
| tempo | `grafana/tempo:3.1.0` |
| otel-collector | `otel/opentelemetry-collector-contrib:0.162.0` |
| rag-api, rag-migrate, rag-ingest (built from `rag/Dockerfile`) | `python:3.12.15-slim-trixie`, `ghcr.io/astral-sh/uv:0.11.32` (build stage) |
| judge-init (profile eval) | `curlimages/curl:8.22.0` |
| llamacpp-judge (profile eval) | `ghcr.io/ggml-org/llama.cpp:server-b11459` |

Base bump 2026-10-09: rag/Dockerfile moved from `python:3.12.13-slim-trixie` to `python:3.12.15-slim-trixie` to clear
3 CRITICAL perl-base CVEs found by Trivy in Phase 5a (docs/evidence/phase5a-platform.txt). The compose stack was rebuilt
on it and the orchestrator re-checked smoke and chat on 2026-10-09. Phase 3 and 4 evidence was recorded on 3.12.13.

The observability tags match [observability/VERSIONS.md](../observability/VERSIONS.md), which is the source of truth for them.
`promtool` (rule checks and unit tests) runs from the same `prom/prometheus:v3.15.0` image.
`observability/gpu-replay/convert.py` is stdlib only, run with uv 0.11.32 on Python 3.12; its backfill uses the same promtool image.

Note: llama.cpp does not publish a `server-bNNNN` image for every build. `b11459` was the newest CPU
server image on GHCR at pin time.

## Model

| Model | File | Source |
|-------|------|--------|
| Qwen2.5-1.5B-Instruct GGUF Q4_K_M | `qwen2.5-1.5b-instruct-q4_k_m.gguf` (1,117,320,736 bytes) | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` at commit `91cad51170dc346986eccefdc2dd33a9da36ead9`, sha256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e` |
| Qwen2.5-7B-Instruct GGUF Q4_K_M (eval judge only) | `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` (3,993,201,344 bytes) + `-00002-of-00002.gguf` (689,872,288 bytes) | `Qwen/Qwen2.5-7B-Instruct-GGUF` at commit `bb5d59e06d9551d752d08b292a50eb208b07ab1f`, sha256 `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` and `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a` |
| BAAI/bge-small-en-v1.5 (MIT, embeddings, baked into the rag image) | `model.safetensors` + config | commit `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0, reranker, baked into the rag image) | `model.safetensors` + config | commit `233902d25c440f23af6f7d6e94d2946bac0bee0a` |

Model licences, as reported by the Hugging Face model API (`cardData.license`, fetched 2026-10-10 for docs/LEARN.md;
current repo values, not checked at the pinned revisions): Apache-2.0 for
[Qwen2.5-1.5B-Instruct](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct),
[Qwen2.5-1.5B-Instruct-AWQ](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct-AWQ),
[Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF),
[Qwen2.5-7B-Instruct](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct),
[Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct-AWQ),
[Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct-GGUF) and
[ms-marco-MiniLM-L6-v2](https://huggingface.co/api/models/cross-encoder/ms-marco-MiniLM-L6-v2); MIT for
[bge-small-en-v1.5](https://huggingface.co/api/models/BAAI/bge-small-en-v1.5). YOLOv8n: AGPL-3.0 (see the v3 section).

## RAG Python packages

Locked in `rag/uv.lock` (Python 3.12). Direct pins in `rag/pyproject.toml`:
haystack-ai 2.31.0 (latest 2.x; 3.x exists but the plan specifies 2.x), elasticsearch-haystack 6.4.1
(pulls elasticsearch-py 8.19.3, talks to the 9.5.5 server through REST compatibility), sentence-transformers 5.6.1,
torch 2.14.1 (CPU wheel index), pypdf 6.19.0, fastapi 0.142.2, uvicorn 0.54.0, sqlalchemy 2.1.3, alembic 1.20.0,
psycopg 3.3.6, opentelemetry-sdk and -exporter-otlp-proto-grpc 1.45.1, prometheus-client 0.26.0.
Added with sign-in: pyjwt 2.15.1 with the crypto extra (cryptography 50.0.2, cffi 2.1.1, pycparser 3.11).
Dev group: pytest 9.1.1, httpx 0.28.1. Eval group: deepeval 4.2.8.

Optional sign-in overlay (`docker-compose.auth.yml`): Keycloak 26.8.1 (`quay.io/keycloak/keycloak`) and oauth2-proxy v7.15.5, alpine variant, from quay.io.
Optional OCR build (`docker-compose.ocr.yml`, `WITH_OCR=1`): Debian trixie packages from the base image's release, not pinned
(see the Dockerfile comment). Installed when recorded 2026-10-14: tesseract-ocr 5.5.0-1+b1, tesseract-ocr-eng 1:4.1.0-2,
poppler-utils 25.03.0-5+deb13u4.
hadolint `hadolint/hadolint:v2.15.1` was first run by hand on `rag/Dockerfile`; it now runs as a pre-commit local hook (`.pre-commit-config.yaml`) and in `ci/run.sh lint`.

## pre-commit

| Tool | Rev |
|------|-----|
| pre-commit (runner, via uvx on Python 3.12) | `4.6.2` |
| pre-commit/pre-commit-hooks | `v6.0.0` |
| astral-sh/ruff-pre-commit | `v0.16.10` |
| adrienverge/yamllint | `v1.38.0` |
| gitleaks (local hook, image `zricethezav/gitleaks`) | `v8.30.1` |

hadolint now runs as a pre-commit local hook (`.pre-commit-config.yaml`) and in `ci/run.sh lint`.

## Phase 2 benchmark and GPU serving (pinned 2026-10-07)

| Component | Version | Lock / source |
|-----------|---------|---------------|
| Client tools (loadgen, report, tests, accuracy), Python 3.12 venv | httpx 0.28.1, matplotlib 3.11.2, pytest 9.1.1, lm-eval[api] 0.4.13 | `benchmark/requirements.txt` (uv pip compile, universal) |
| GSM8K dataset | `openai/gsm8k` main, revision `740312add88f781978c0658806c59bc2815b9866` | `benchmark/accuracy/tasks/gsm8k_pinned.yaml` |
| vLLM (Kaggle T4) | 0.31.0 `+cu129` wheel from the GitHub release, torch 2.13.0+cu129, humming-kernels[cu12] 0.1.16 | `notebooks/vllm-requirements.txt` (linux x86_64, Python 3.12), resolved with `notebooks/vllm-overrides.txt` |
| uv on Kaggle | 0.11.32, venv Python 3.12.13 | `notebooks/build_notebook.py` |
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Hugging Face commit |
| `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | `3ecffa0ceb27851800f45519bab9c457a04405e1` | Hugging Face commit |
| `Qwen/Qwen2.5-7B-Instruct-AWQ` | `b25037543e9394b818fdfca67ab2a00ecc7dd641` | Hugging Face commit |

Why 0.31.0 and cu129: [inference/README.md](../inference/README.md).

## Phase 5a platform (pinned 2026-10-09)

Versions live in `scripts/tools.sh`; this table mirrors them. How each is pinned: the kind and kubeconform
binaries are downloaded into `.tools/bin` (git-ignored) and checked against the sha256 below; the kind node image
is pinned by digest; the JSON schemas by git commit. Everything else (ArgoCD `install.yaml`, the KEDA and Traefik
Helm charts, registry, alpine, Trivy) is pinned by version or tag only, with no checksum or digest.

| Component | Version | Notes |
|-----------|---------|-------|
| kind | `v0.33.0` | `kind-windows-amd64` sha256 `4b22adaa135368c5a465d56bbd8e520cbea87272a06ca00b6078e7b81515c9fc`, `kind-linux-amd64` sha256 `aee6151561422756b764a4ae28e7f44cda5af5a9eead3cc9985112b1de8d8e0d` |
| kind node image | `kindest/node:v1.36.4@sha256:099e049362a1526b2db71494e1947aae99bd16290d7c895f2b7ea312e3cbfaed` | `infra/kind/cluster.yaml`; built for kind v0.33.0 |
| Local registry | `registry:3.1.2` | container `kind-registry`, `localhost:5001` |
| Git server | `alpine:3.23.0` + `git-daemon=2.52.0-r0` | `infra/git-server/Dockerfile`, image `industrial-llm-platform/git-server:0.1.0` |
| ArgoCD | `v3.5.4` (`manifests/install.yaml`; images `quay.io/argoproj/argocd:v3.5.4`, `ghcr.io/dexidp/dex:v2.45.1`, `public.ecr.aws/docker/library/redis:8.2.3-alpine`) | |
| KEDA | Helm chart `kedacore/keda` `2.21.0` (images `2.21.0`) | |
| Traefik (kind ingress only) | Helm chart `traefik/traefik` `41.6.1` (image `traefik:v3.7.13`) | |
| kubeconform | `v0.8.0` | `windows-amd64.zip` sha256 `e3f56102bcf4f50b034a567e2482a1c5330799983ddd655952310211aef73d93`, `linux-amd64.tar.gz` sha256 `9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883` |
| Kubernetes JSON schemas | `yannh/kubernetes-json-schema` commit `39a87cd29966bc0b3434e8cfbc2061782c8cb69a` (v1.36.4) | `scripts/helm-check.sh` |
| CRD schemas (ScaledObject, Application) | `datreeio/CRDs-catalog` commit `63669a570e231d4f1f8396d229a1de512bcf0a34` | |
| OpenShift Route schema | `melmorabity/openshift-json-schemas` commit `62ad608b18edcd8ef4b6dd3741fc36115813fd18` (v4.20) | |
| Trivy | `aquasec/trivy:0.75.0` | vulnerability DB is downloaded at scan time, so results depend on the scan date |
| Helm (host) | 4.2.4 | already installed; ArgoCD uses its own bundled Helm |

In-cluster workloads reuse the compose images above (llama.cpp, curl, postgres, elasticsearch, prometheus, rag).

## Phase 5b CI (pinned 2026-10-09)

Images live in `ci/run.sh` (tools), `ci/jenkins/Dockerfile` (controller) and `.github/workflows/ci.yml` (actions).
CI also reuses images pinned above: postgres, prometheus (promtool), Trivy, gitleaks, hadolint; uv 0.11.32 and
pre-commit 4.6.2 as in the pre-commit section.

| Component | Version | Notes |
|-----------|---------|-------|
| Jenkins controller | `jenkins/jenkins:2.580.1-lts-jdk21@sha256:a660310e39ade10631f774bacd5219de767dfd08947ea5c32a739b5e5bb382c1` | newest LTS on 2026-10-09; built into `industrial-llm-platform/jenkins:2.580.1` |
| Docker CLI + buildx in the controller | `docker:29.6.2-cli@sha256:be132a9f282288de4afaf63379dff75711fda0147c6b72a9df44e51841402144` (buildx v0.35.0) | matches the Docker Desktop engine 29.6.2 on the dev machine |
| Jenkins plugins | 63 plugins in `ci/jenkins/plugins.yaml` | configuration-as-code 2133.ve570217cdb_cd, job-dsl 3732.v9a_c49a_61a_313, workflow-aggregator 608.v67378e9d3db_1, git 5.10.1 and every required dependency, each pinned |
| CI tool image (pre-commit, pytest) | `ghcr.io/astral-sh/uv:0.11.32-python3.12-trixie@sha256:5c14184583b8da2956605104e364053c863e4de51f8a83b230a89d50e34e9ab0` | full (non-slim) Debian image because pre-commit needs git; checked 2026-10-09: git 2.47.3, Python 3.12.13 |
| actionlint | `rhysd/actionlint:1.7.12@sha256:b1934ee5f1c509618f2508e6eb47ee0d3520686341fec936f3b79331f9315667` | its bundled shellcheck also checked `ci/run.sh` and `scripts/gitops-publish.sh` by hand |
| `actions/checkout` | `3d3c42e5aac5ba805825da76410c181273ba90b1` (v7.0.1) | only action used; GitHub runner image `ubuntu-24.04` |
| kubectl in `scripts/gitops-publish.sh` (CI deploy and kind-up.sh step 8) | the binary inside the kind node image above (v1.36.4) | run through `docker exec`, so no separate pin |
| Helm for the `charts` stage | `alpine/helm:4.2.4@sha256:76c375eed56144c68d6197c55bc5a4552fb42002190b796729901cbab3ae6e51` | same Helm as the host; the image has bash and curl, which `scripts/helm-check.sh` needs. Checked 2026-10-09: helm-check.sh passes in it as UID 1000 |

The CI tool image's Python (3.12.13) can differ from the 3.12 patch uv picks on a developer host
(`uvx --python 3.12`); CI results are for 3.12.13.

## v3 item 9: Loki and Langfuse (pinned 2026-10-09)

Details and notes: `observability/VERSIONS.md`. Latest GitHub release of each project on 2026-10-09.

| Component | Version | Notes |
|-----------|---------|-------|
| Loki | `grafana/loki:3.7.8` | default profile |
| Alloy (log shipper) | `grafana/alloy:v1.20.1` | default profile |
| Langfuse web, worker | `langfuse/langfuse:4.56.0`, `langfuse/langfuse-worker:4.56.0` | profile `langfuse` |
| ClickHouse | `clickhouse/clickhouse-server:25.12.11.4` | profile `langfuse` |
| Redis | `redis:7.4.11-alpine` | profile `langfuse` |
| MinIO | `cgr.dev/chainguard/minio@sha256:6476f81b9739912fa9f50ceae2bb049a30336527cccd24beccfbeb4145f8f696` | profile `langfuse`; `RELEASE.2026-09-22T19-25-18Z` |
| Postgres for Langfuse | `postgres:18.6-alpine3.24` | reuses the pinned image above |

## v3 Kaggle notebooks: DCGM, vision, TensorRT-LLM (pinned 2026-10-09, run 2026-10-10)

Notebooks `notebooks/kaggle-dcgm-vision/` and `notebooks/kaggle-trtllm/`, built by `notebooks/build_v3.py`. Reasons
and sources for each choice: [inference/README.md](../inference/README.md), section "v3 Kaggle notebooks".

| Component | Version | Lock / source |
|-----------|---------|---------------|
| DCGM | 4.7.0-2: `libdcgm`, `nv-hostengine`, `dcgmi`, `datacenter-gpu-manager-4-proprietary`, `datacenter-gpu-manager-4-python3` (.deb, extracted, not apt-installed); licence: Apache-2.0 except the proprietary package (NVIDIA proprietary, free) | NVIDIA CUDA repo `ubuntu2404/x86_64`, sha256 per file from its Packages index, in the notebook config |
| Vision venv (Python 3.12.13) | ultralytics 8.4.174 (AGPL-3.0), torch 2.13.0+cu129, torchvision 0.28.0+cu129, onnx 1.23.2, onnxslim 0.1.98, onnxruntime-gpu 1.26.0, tensorrt-cu12 10.16.1.11, nvidia-pytriton 0.7.0 (bundled tritonserver, Python backend), numpy 1.26.4; TensorRT is NVIDIA proprietary software (free), ultralytics AGPL-3.0 | `notebooks/vision-requirements.txt` |
| YOLOv8n weights | `yolov8n.pt` from `ultralytics/assets` release v8.4.0, sha256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | notebook config |
| ONNX export | opset 17, dynamic batch, onnxslim | `inference/vision/export.py` |
| COCO subset | first 500 sorted val2017 names; images from `images.cocodataset.org`, sha256 per image; labels `coco2017labels.zip` (Ultralytics assets v0.0.0) sha256 `51a5175c894a7a1010f90eb4cba613473445f02633b684ed46c0292a997d0234`; licences: COCO annotations CC BY 4.0, images under their Flickr terms of use (nothing from COCO is committed) | `benchmark/vision/coco_val2017_first500.txt`, `benchmark/vision/fetch_coco_subset.py` |
| Vision CPU baseline venv (Python 3.12) | onnxruntime 1.26.0, numpy 2.5.3, pillow 12.3.0, httpx 0.28.1 | `benchmark/vision/requirements.txt` |
| TensorRT-LLM venv (Python 3.10.20) | tensorrt-llm 0.14.0 (Apache-2.0; its TensorRT 10.4.0 is NVIDIA proprietary, free), tensorrt 10.4.0, torch 2.4.0+cu124, transformers 4.45.1, cuda-python 12.5.0, mpi4py 3.1.6 (built from source), fastapi 0.112.0, prometheus-client 0.26.0 | `notebooks/trtllm-requirements.txt` |
| pynvml (TensorRT-LLM venv) | 11.5.3 (13.x has no `__version__`, read by tensorrt_llm/profiler.py at import) | `notebooks/trtllm-requirements.in` |
| pyarrow (TensorRT-LLM venv) | 20.0.0 (last with `PyExtensionType`, which datasets 2.14.4 needs) | `notebooks/trtllm-requirements.in` |
| setuptools in the mpi4py build env (TensorRT-LLM venv) | 80.10.2 (last before 81.0.0 removed dry_run, which mpi4py 3.1.6 passes) | `notebooks/trtllm-build-constraints.txt`, `uv pip install --build-constraints` |
| Open MPI (TensorRT-LLM) | `openmpi-bin`, `libopenmpi-dev` 4.1.6-7ubuntu2 (Ubuntu 24.04 universe, apt) | notebook config |
| TensorRT-LLM OpenAI server | `examples/apps/openai_server.py` at tag v0.14.0 (commit `b0880169d0fb8cd0363049d91aa548e58a41be07`), sha256 `8ae2d1d892792bada31424bd8295dc87a5e1a902c2f723427e424a5a1b3f9f1f` | notebook config; wrapped by `inference/trtllm/serve.py` |
| vLLM (same-session control in the TensorRT-LLM notebook), client venv, uv, Qwen2.5-1.5B-Instruct revision | as in "Phase 2 benchmark and GPU serving" above | the control adds `--generation-config vllm` |

## v3 OKD via OpenShift Local (pinned 2026-10-10)

`scripts/okd-up.sh`; evidence `docs/evidence/v3-openshift-okd.txt`. The charts, the KEDA chart and the app images are
the ones pinned above.

| Component | Version | Notes |
|-----------|---------|-------|
| OpenShift Local (crc) | `2.64.0+458092` | `crc-windows-installer.zip` sha256 `2f3ffed8e72a9b8b8363d664eed54db6bee3154a24dd28ca4802ff1a34c2ec01` (release sha256sum.txt); MSI inside sha256 `18d59aac59650d68c35fd97ee1f02f183bb6168a365e2b2b7b84ac4873825a7c`; installed by the user (admin) |
| OKD bundle | `4.22.0-okd-scos.10` (Kubernetes v1.35.6, CentOS Stream CoreOS 10.0.20260806-0) | `quay.io/crcont/okd-bundle@sha256:58dde1e5fa21ea10faf64a18ff44687b1288f61a332ed59c4df60a881bbed0e7`; windows/amd64 layer sha256 `170b7fc908bfbd59deaa6f5b2798359eb17e471e55b198f21a1d033f1ba8bcd2` (6,194,525,758 bytes) holds `crc_okd_hyperv_4.22.0-okd-scos.10_amd64.crcbundle` and its `.sig`. The PGP signature was verified once by hand (Good signature, key B84655BFAC1C8E067FD9CA87DC7EAC400A1BFDFB, crc@crc.dev, from crc v2.64.0 `pkg/crc/constants/keys.go`; `docs/evidence/v3-openshift-okd/session2-bundle-signature.txt`); `okd-up.sh` checks only the layer sha256 |
| oc | `4.22.0-okd-scos.10` | shipped in the bundle (`~/.crc/bin/oc`) |
| ArgoCD operator | `argocd-operator.v0.19.0` (community-operators, channel alpha, manual InstallPlan approval). Only the CSV is pinned: the `community-operators` catalog image comes with the bundle and is not pinned by digest here | runs ArgoCD v3.4.4, `quay.io/argoproj/argocd@sha256:2fb3efa9eaa42382acb39c9a1485eaac7a1ad7a09e8aa7da7143ca256fa6da09` |
| KEDA | Helm chart `kedacore/keda` `2.21.0` | same as kind; the OKD catalog's KEDA is 2.7.1 (observed, no raw log kept) |
| skopeo (rag image push) | `quay.io/skopeo/stable:v1.22.3@sha256:e5d5d2815b94d74767d40be7111647c3e9cdc84406a638ffe1c7ea1f286f52a3` | |
