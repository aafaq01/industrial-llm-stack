# Plan (slim scope, v2)

Status lives in [STATUS.md](STATUS.md). A phase is done only when `reviewer` returns PASS.
v1 had 9 phases and ~19 GPU hours. v2 cuts to 6 core phases and ~8 GPU hours. Cut items are listed under
"Stretch" and are only started if the core is done.

## Environment facts (checked 2026-10-07)
- Host: Windows 11, 16 logical cores, 63 GB RAM, AMD integrated GPU (no CUDA locally).
- Installed: git 2.53, Docker 29.6 (Docker Desktop, WSL2 backend, currently stopped), kubectl 1.36.1, helm 4.2.4, Python 3.14.3.
- Missing: kind, k3d, make.
- Python 3.14 is too new for the ML stack. Every Python service and tool runs in a container on a pinned
  `python:3.12-slim` base, or in a `uv` venv pinned to 3.12.
- The "one command" is `docker compose up -d --wait`. The Makefile wraps it but is optional on Windows.

## GPU options (free only)
| Option | Hardware | Quota | Use here |
|--------|----------|-------|----------|
| Kaggle notebooks | 2x T4 or 1x P100 | ~30 GPU-h/week, 12 h sessions | Primary: vLLM benchmarks, TP=2 cell |
| Lightning AI free Studio | T4/L4 (interruptible) | 15 credits/month | Backup if Kaggle quota runs out |
| Google Colab free | 1x T4 | Unpublished, varies | Backup for short runs |
| Local AMD iGPU | Radeon integrated, shared RAM | Unlimited | Optional llama.cpp Vulkan build, labelled "iGPU", never compared as if it were CUDA |
| SageMaker Studio Lab | T4 | Closed to new users since 2026-07-30 | Not usable |

## Global decisions
- Models: Qwen2.5-1.5B-Instruct everywhere. Qwen2.5-7B-Instruct AWQ only as one extra GPU cell.
- Engines: llama.cpp (CPU, local) and vLLM (T4, Kaggle). TensorRT-LLM + Triton moved to Stretch: long engine
  builds, and INT4 AWQ/GPTQ are not supported on T4 (sm75) in TRT-LLM, so the quantisation story is weak there.
- Corpus (confirm in Phase 4): public domain US government engineering documents (NASA NTRS or NIST SPs).
  Source and licence in `rag/data/SOURCES.md`.
- Everything pinned. Versions in `docs/VERSIONS.md`.

## Core phases

| # | Phase | Owner(s) | Depends on | GPU |
|---|-------|----------|-----------|-----|
| 1 | Skeleton + CPU inference | generalist, observability-engineer | none | no |
| 2 | GPU inference + benchmarks | inference-engineer, benchmark-analyst | 1 | Kaggle, ~8 h |
| 3 | RAG | rag-engineer | 1 | no |
| 4 | Observability + SLOs | observability-engineer | 1, 3 | DCGM from Phase 2 run |
| 5 | Platform + CI/CD | platform-engineer | 1, 3 | no |
| 6 | Docs | generalist | all | no |

Parallel: after Phase 1, run 2 and 3 together. Then 4 and 5 together.

### Phase 1: Skeleton + CPU inference
Deliverables: `git init`, layout, `.gitignore`, `.env.example`, pre-commit (ruff, yamllint, hadolint, gitleaks),
`docker-compose.yml` with llama.cpp `llama-server` (Qwen2.5-1.5B GGUF Q4_K_M, downloaded by an init container),
PostgreSQL, Elasticsearch, Prometheus, Grafana, Tempo, OTel Collector. `scripts/smoke.sh`.
Acceptance:
1. `docker compose up -d --wait` from a clean clone brings every service to healthy.
2. Smoke test passes: chat completion returns text, Postgres query works, Elasticsearch health green or yellow,
   Grafana healthy, Prometheus scrapes llama.cpp `/metrics`.
3. `pre-commit run --all-files` passes. All image tags pinned. No secrets.

### Phase 2: GPU inference + benchmarks
Deliverables: `benchmark/loadgen/` (async OpenAI-API client, fixed prompts + seed), `benchmark/accuracy/`
(lm-evaluation-harness, fixed 250-item GSM8K subset), `notebooks/vllm_t4_benchmark.ipynb`, raw JSON in
`benchmark/results/raw/`, `benchmark/report.py` producing charts + `benchmark/RESULTS.md`.
Matrix (12 cells + 2 extras):
- llama.cpp CPU, GGUF Q4_K_M, concurrency {1, 4, 8}
- vLLM 1x T4, 1.5B {FP16, AWQ INT4}, concurrency {1, 8, 32}
- Extras: vLLM 7B AWQ on 1x T4 at concurrency 8; vLLM 1.5B FP16 TP=2 on 2x T4 at concurrency 32
Metrics: TTFT, inter-token latency, tokens/s, p50/p95/p99, error rate, GPU memory/util (DCGM or nvidia-smi).
Acceptance:
1. Notebook runs top to bottom on Kaggle with "Run all" and writes raw JSON + `pip freeze` + `nvidia-smi`.
2. 3 repeats per cell after warmup; spread reported; suspicious cells flagged.
3. FP16 vs AWQ accuracy delta on the same subset.
4. `python benchmark/report.py` regenerates everything offline. Every number traces to a raw file.

### Phase 3: RAG
Deliverables: Haystack 2.x pipeline (ingest, BM25 + dense hybrid in Elasticsearch, reranker, generation via
llama.cpp OpenAI API), Postgres chat history with Alembic, FastAPI with SSE streaming, eval set of 30 labelled questions.
Acceptance:
1. RAG API in compose; one ingest command indexes the corpus.
2. Chat streams an answer with citations; history persists.
3. Recall@5 and faithfulness scored with a local judge (RAGAS or DeepEval), results in `rag/eval/results/`.
4. Unit tests pass. OTel spans per pipeline step visible in Tempo.

### Phase 4: Observability + SLOs
Deliverables: provisioned Grafana dashboards (inference, RAG, GPU from captured run), recording + alert rules,
`observability/SLOs.md`.
Acceptance:
1. Dashboards load from files: latency, throughput, queue depth, tokens, error rate, RAG scores.
2. Two SLOs (RAG availability, p95 TTFT) with multi-window burn-rate alerts and an error-budget panel;
   `promtool test rules` passes.
3. GPU dashboard clearly labelled as replaying data captured in Phase 2.

### Phase 5: Platform + CI/CD
Deliverables: non-root multi-stage Dockerfiles, Helm charts with `values-kind.yaml` and `values-openshift.yaml`,
ArgoCD app-of-apps, KEDA ScaledObject on llama.cpp pending requests, `scripts/kind-up.sh`,
`.github/workflows/ci.yml`, `ci/Jenkinsfile` with local Jenkins compose.
Acceptance:
1. `scripts/kind-up.sh` creates the cluster, installs ArgoCD + KEDA, all apps Healthy.
2. Load test triggers KEDA scale up then down; evidence committed.
3. Images run as a random UID; `helm lint` + `kubeconform` pass; Trivy has no unexplained critical CVEs.
4. Jenkins pipeline green locally (lint, test, build, scan, push to local registry, chart bump, ArgoCD sync);
   GitHub Actions passes `actionlint`. OpenShift: documented as "manifests provided, not tested" unless you approve a Sandbox account.

### Phase 6: Docs
README with results table first, Mermaid diagram, quickstart, reproduce steps, decisions, limitations.
Every number traces to a raw file; anything not run says so.

## Stretch (only after core passes)
- TensorRT-LLM + Triton FP16 on T4 (~4 GPU-h)
- Vision: YOLOv8n (AGPL-3.0) via ONNX Runtime CPU, then Triton on T4 (~3 GPU-h)
- bitbucket-pipelines.yml mirror of the CI
- Loki logs, Langfuse
- OpenShift Developer Sandbox test
- llama.cpp Vulkan on the local AMD iGPU

## Approvals I will ask for
- Each GPU session (tier, notebook, expected hours).
- Creating any account (Kaggle, Lightning AI, Red Hat Sandbox, registries).
- Any push to a remote.

## v3 additions (approved 2026-10-09, after the core passed)

| # | Item | Owner | GPU | Notes |
|---|------|-------|-----|-------|
| 7 | GitHub private repo + GitHub Actions run | orchestrator, platform-engineer | no | push needs the user's go on the repo name; Actions runs lint/test/build/scan only |
| 8 | OpenShift via OpenShift Local (CRC) OKD preset | platform-engineer | no | no Red Hat account; Hyper-V; run while kind is stopped (RAM) |
| 9 | Loki logs + self-hosted Langfuse | observability-engineer | no | compose profile, off by default |
| 10 | DCGM inside Kaggle | inference-engineer | Kaggle ~0.5 h | experiment: report what works; no fabricated metrics |
| 11 | Vision: YOLOv8n (AGPL-3.0) CPU ONNX baseline + T4 (ONNX/TensorRT FP16, Triton) | inference-engineer, benchmark-analyst | Kaggle ~1-3 h | CPU run needs a quiet machine |
| 12 | TensorRT-LLM FP16 + Triton on T4 | inference-engineer, benchmark-analyst | Kaggle ~2-4 h | no INT4 AWQ/GPTQ on sm75 |

Acceptance follows the same rules as the core: raw output committed, every number traced, reviewer PASS, "not run" with reason
where something does not work on free hardware. GPU sessions are bundled where possible; the user approved these items, and
each push to Kaggle is still announced with its expected hours.
