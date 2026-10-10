# Learning guide: industrial-llm-platform

A single guide to everything in this repo: what it does, every tool and model, every file that matters, how each
number was measured, what the numbers mean, what is done and what is not, and how to run
each part.

Snapshot: written at repo HEAD `30673df` and updated at `53ccd0a` (2026-10-10) for TensorRT-LLM run v3i. Every number is copied from a committed file, and the file is named
next to it. Where the repo holds no number, this guide says "not recorded". Ratios that are not in the repo are
marked "derived: x / y" with both sources.

## 0. How to read this

**In plain English.** Read it like a textbook with two voices. Each topic starts with "In plain English", which uses
an everyday picture (a kitchen, a library, a post room). Then "Technically" gives the precise version with file
paths. If a word is new, look it up in the [Glossary](#2-glossary). If you only have ten minutes, read
[1. Big picture](#1-big-picture), [7. Results explained](#7-results-explained) and
[8. Status, honestly](#8-status-honestly).

**Technically.**
- Paths in backticks (`rag/app/rag.py`) are relative to the repo root. Links like
  [../benchmark/RESULTS.md](../benchmark/RESULTS.md) are relative to this file in `docs/`.
- "Evidence file" means a committed log under `docs/evidence/` or a raw result under `benchmark/results/raw/` or
  `rag/eval/results/`.
- "Not run" means no measurement exists. "Partial" means some steps worked and the rest failed or were skipped, with
  the failure recorded. "Estimate" means an opinion, not a measurement.
- Section numbers in cross-references point inside this file, for example "see 6.3".

### Table of contents

0. [How to read this](#0-how-to-read-this)
1. [Big picture](#1-big-picture)
   - [1.1 What the platform does](#11-what-the-platform-does)
   - [1.2 Architecture diagram](#12-architecture-diagram)
   - [1.3 Flow: one chat request](#13-flow-one-chat-request)
   - [1.4 Flow: one benchmark run](#14-flow-one-benchmark-run)
   - [1.5 Flow: one code change reaching the cluster](#15-flow-one-code-change-reaching-the-cluster)
   - [1.6 How the project was built](#16-how-the-project-was-built)
   - [1.7 The evidence rule and the review catches](#17-the-evidence-rule-and-the-review-catches)
2. [Glossary](#2-glossary)
3. [Tools and open-source projects](#3-tools-and-open-source-projects)
   - [3.1 Master table](#31-master-table)
   - [3.2 Serving](#32-serving)
   - [3.3 Benchmarking](#33-benchmarking)
   - [3.4 RAG](#34-rag)
   - [3.5 Observability](#35-observability)
   - [3.6 Platform](#36-platform)
   - [3.7 CI and hygiene](#37-ci-and-hygiene)
4. [Models](#4-models)
5. [File-by-file code walkthrough](#5-file-by-file-code-walkthrough)
   - [5.1 Root files](#51-root-files)
   - [5.2 rag/](#52-rag)
   - [5.3 inference/](#53-inference)
   - [5.4 notebooks/](#54-notebooks)
   - [5.5 benchmark/](#55-benchmark)
   - [5.6 observability/](#56-observability)
   - [5.7 deploy/](#57-deploy)
   - [5.8 infra/](#58-infra)
   - [5.9 scripts/](#59-scripts)
   - [5.10 ci/](#510-ci)
   - [5.11 .github/](#511-github)
   - [5.12 docs/](#512-docs)
6. [Methods](#6-methods)
   - [6.1 LLM benchmarks](#61-llm-benchmarks)
   - [6.2 The llama.cpp prompt cache issue](#62-the-llamacpp-prompt-cache-issue)
   - [6.3 Quantisation accuracy testing](#63-quantisation-accuracy-testing)
   - [6.4 Making vLLM vs TensorRT-LLM fair](#64-making-vllm-vs-tensorrt-llm-fair)
   - [6.5 Vision speed, accuracy and provenance](#65-vision-speed-accuracy-and-provenance)
   - [6.6 RAG retrieval](#66-rag-retrieval)
   - [6.7 RAG evaluation](#67-rag-evaluation)
   - [6.8 Tracing, metrics and logs](#68-tracing-metrics-and-logs)
   - [6.9 SLOs and burn-rate alerts](#69-slos-and-burn-rate-alerts)
   - [6.10 Testing the alert rules](#610-testing-the-alert-rules)
   - [6.11 GPU replay dashboard](#611-gpu-replay-dashboard)
   - [6.12 Deployment and GitOps](#612-deployment-and-gitops)
   - [6.13 KEDA autoscaling](#613-keda-autoscaling)
   - [6.14 CI pipeline and secrets](#614-ci-pipeline-and-secrets)
7. [Results explained](#7-results-explained)
   - [7.1 How to read benchmark columns](#71-how-to-read-benchmark-columns)
   - [7.2 CPU llama.cpp](#72-cpu-llamacpp)
   - [7.3 GPU vLLM run v2](#73-gpu-vllm-run-v2)
   - [7.4 The GPU power-cap story](#74-the-gpu-power-cap-story)
   - [7.5 Same-session vLLM controls](#75-same-session-vllm-controls)
   - [7.6 GSM8K accuracy](#76-gsm8k-accuracy)
   - [7.7 Vision speed](#77-vision-speed)
   - [7.8 Vision accuracy](#78-vision-accuracy)
   - [7.9 The memory-limit story (TensorRT-LLM)](#79-the-memory-limit-story-tensorrt-llm)
   - [7.10 TensorRT-LLM vs vLLM (run v3i)](#710-tensorrt-llm-vs-vllm-run-v3i)
   - [7.11 Kaggle run history](#711-kaggle-run-history)
   - [7.12 RAG results](#712-rag-results)
   - [7.13 Observability results](#713-observability-results)
   - [7.14 Platform and CI results](#714-platform-and-ci-results)
8. [Status, honestly](#8-status-honestly)
   - [8.1 Every part, its state, its evidence](#81-every-part-its-state-its-evidence)
   - [8.2 Docs that lag the results](#82-docs-that-lag-the-results)
   - [8.3 How to run each part locally](#83-how-to-run-each-part-locally)
   - [8.4 GPU needs](#84-gpu-needs)
9. [Things not fully understood](#9-things-not-fully-understood)

---

## 1. Big picture

### 1.1 What the platform does

**In plain English.** Think of a small restaurant kitchen built to learn how big restaurant chains work.

- The **cook** is a language model: a program that reads text and writes text back.
- The **recipe library** is a shelf of public engineering documents (NASA and NIST). When a customer asks a question,
  a helper finds the most relevant pages and hands them to the cook, so the answer is based on real pages and says
  which page it came from.
- The **order book** remembers each customer's earlier questions, so "what about that value?" makes sense.
- **Cameras and counters** on the walls record how long every step took and whether anything failed.
- A **stopwatch test kitchen** measures how fast different cooks (engines) and ingredient grades (precisions) work,
  using borrowed free ovens (free GPUs on Kaggle).
- A **franchise blueprint** (Kubernetes, Helm, ArgoCD) shows how to copy the kitchen into a bigger managed building,
  and a **quality inspector** (CI: Jenkins and GitHub Actions) checks every recipe change before it gets there.

Everything is free and open source, and every claim must point at the saved measurement behind it.

**Technically.** An open-source platform for serving, benchmarking and operating small LLMs (`README.md`,
`CLAUDE.md`):

| Layer | What runs | Where it lives |
|-------|-----------|----------------|
| Serving | llama.cpp `llama-server` on CPU (Qwen2.5-1.5B-Instruct, GGUF Q4_K_M); vLLM 0.31.0 on Kaggle T4 GPUs; TensorRT-LLM 0.14.0 on one Kaggle T4 (run v3i) | `docker-compose.yml`, `inference/README.md` |
| RAG app | FastAPI + Haystack 2.x: BM25 + dense retrieval in Elasticsearch, RRF fusion, cross-encoder rerank, streamed answer with citations, PostgreSQL chat history | `rag/app/main.py`, `rag/app/rag.py` |
| Observability | OpenTelemetry traces to Tempo, Prometheus metrics with SLO burn-rate alerts, Grafana dashboards, Loki logs via Alloy, optional Langfuse | `observability/`, `observability/SLOs.md` |
| Benchmarks | async load generator, lm-evaluation-harness GSM8K, YOLOv8n vision timing and mAP, `report.py` that rebuilds every table from raw files | `benchmark/` |
| Platform | kind cluster, Helm charts, ArgoCD app-of-apps, KEDA autoscaling, Traefik ingress; OpenShift values tested on OKD | `deploy/`, `scripts/kind-up.sh`, `scripts/okd-up.sh` |
| CI/CD | Jenkins in Docker (full pipeline including deploy), GitHub Actions (lint, test, build, scan) | `ci/`, `.github/workflows/ci.yml` |

Terms used here ([RAG](#2-glossary), [BM25](#2-glossary), [RRF](#2-glossary), [SSE](#2-glossary) and so on) are in
the glossary.

### 1.2 Architecture diagram

**In plain English.** There are four places where things run: your laptop (Docker Compose), a mini Kubernetes
cluster on your laptop (kind), a bigger OpenShift-style cluster in a virtual machine on your laptop (OKD), and
borrowed GPUs on Kaggle. CI checks code. Benchmark results from Kaggle come back as files, and a script turns them
into tables.

**Technically.** Ports from `docs/LAYOUT.md`; all compose ports bind to 127.0.0.1.

```
+=========================== LAPTOP: docker compose (CPU) ============================+
|  user/curl --127.0.0.1:8000--> [rag-api FastAPI, SSE]                               |
|                                   |   |   |    \                                    |
|                     chat history  |   |   |     \ OTLP spans (gRPC 4317)           |
|                                   v   |   |      v                                 |
|                         [PostgreSQL]  |   |   [OTel Collector] ---> [Tempo :3200]   |
|                           :5432       |   |          \                             |
|                    BM25 + dense search|   |           \--(profile langfuse)-->     |
|                                       v   |            [Langfuse web :3001, worker,|
|                           [Elasticsearch] |             ClickHouse, Redis, MinIO,  |
|                             :9200         |             Postgres]                  |
|                                           | OpenAI-style /v1/chat/completions      |
|                                           v                                        |
|                      [llama.cpp server :8080, Qwen2.5-1.5B Q4_K_M]                  |
|  [model-init] one-shot: downloads + sha256-checks the GGUF                          |
|  [rag-migrate] one-shot: Alembic schema   [rag-ingest] one-shot: index if empty     |
|  [llamacpp-judge :8081] profile eval: Qwen2.5-7B Q4_K_M judge for rag/eval          |
|  [Prometheus :9090] scrapes rag-api, llama.cpp, collector, tempo; SLI rules, alerts |
|  [Grafana :3000] reads Prometheus, Tempo, Loki                                      |
|  [Alloy :12345] reads every container's logs via Docker socket --> [Loki :3100]     |
+=====================================================================================+

+==================== LAPTOP: kind cluster "llm" (scripts/kind-up.sh) ================+
|  user --127.0.0.1:8090--> [Traefik ingress] --> rag-api.localhost -> [rag-api]       |
|                                             \--> (catch-all) -----> [llama.cpp]      |
|  [ArgoCD] root Application (app-of-apps), sync waves:                               |
|      wave 0 platform-deps: Postgres, Elasticsearch, minimal Prometheus              |
|      wave 1 llamacpp: Deployment + model PVC + KEDA ScaledObject (1 to 3 replicas)   |
|      wave 2 rag-api: Deployment + migrate Job (PreSync) + ingest Job                |
|  [KEDA] reads in-flight requests from that Prometheus, scales llama.cpp             |
|  outside the cluster, on the kind Docker network:                                   |
|      [kind-registry localhost:5001]  rag image, deployed by digest                  |
|      [kind-git: git daemon]  serves this repo's committed HEAD to ArgoCD            |
+=====================================================================================+
        ^ image by digest                 ^ CI bump commit
+=======+== Jenkins in Docker :8088 ======+==+   +==== GitHub Actions (private repo) ====+
| ci/Jenkinsfile -> ci/run.sh:              |   | .github/workflows/ci.yml -> ci/run.sh:|
| lint, charts, test, build, scan, push,    |   | lint, charts, test, build, scan.      |
| bump chart, argocd sync                   |   | No push/bump/sync (cannot reach the   |
+===========================================+   | laptop cluster)                       |
                                                +=======================================+
+===== LAPTOP: OpenShift Local (crc) VM, OKD 4.22 (scripts/okd-up.sh) =====+
| ArgoCD operator, same 3 charts with values-openshift.yaml, Routes (TLS),  |
| restricted-v2 SCC, KEDA chart, rag image in the internal registry.        |
| Reads a git daemon on host 127.0.0.1:9418, branch okd. Run with kind off. |
+===========================================================================+
+================ KAGGLE (free, 2x Tesla T4, internet on) ===================+
| notebooks/vllm_t4_benchmark.ipynb  -> vLLM 0.31.0 cells + GSM8K           |
| notebooks/kaggle-trtllm/           -> TensorRT-LLM 0.14.0 + vLLM control  |
| notebooks/kaggle-dcgm-vision/      -> DCGM probe, YOLOv8n ONNX/TensorRT,   |
|                                       PyTriton, mAP                        |
| code is embedded in the notebook with a sha256 per file                   |
+=============================+=============================================+
                              | kaggle kernels output (raw JSON, logs,
                              v nvidia-smi samples, pip freeze)
            benchmark/results/raw/kaggle-*/  (+ cpu-*/ from laptop runs)
                              v  python benchmark/report.py (offline)
            benchmark/RESULTS.md + benchmark/charts/*.png
                              v  observability/gpu-replay/backfill.sh
            Prometheus "GPU replay" dashboard (not live)
```

Sources: `README.md` (Mermaid diagram), `docs/LAYOUT.md`, `deploy/README.md`, `ci/jenkins/README.md`,
`.github/workflows/ci.yml`, `docs/evidence/v3-openshift-okd.txt`, `inference/README.md`.

### 1.3 Flow: one chat request

**In plain English.** You ask a question at the counter. The waiter writes it in your order book, runs to the library,
grabs pages two ways (by keywords and by meaning), merges the piles, has an expert pick the best five, hands them to the
cook with your question, and brings the answer back a few words at a time. The answer goes in your order book. The
cameras record every step.

**Technically.** Code: `rag/app/main.py` (`run_chat`) and `rag/app/rag.py`. Evidence of a real request and its trace:
`docs/evidence/phase3-rag.txt` sections 4 to 6.

1. Client sends `POST /chat` with `{"message": ..., "session_id": optional, "user": optional}` to `127.0.0.1:8000`
   (compose) or `rag-api.localhost:8090` (kind). FastAPI validates it (`message` 1 to 4000 characters).
2. A worker thread opens the OpenTelemetry span `rag.chat` under the FastAPI request span `POST /chat`.
3. Postgres: `get_or_create_session`, read the last `HISTORY_MESSAGES` messages (default 4), store the user message.
4. First SSE event `session` carries `session_id` and `trace_id`; the same id is logged as
   `chat session_id=... trace_id=<32 hex>`, which Alloy ships to Loki and Grafana links to Tempo.
5. bge-small-en-v1.5 embeds the question (span `rag.embed_query`).
6. Elasticsearch: BM25 top 10 (`rag.bm25_retrieve`) and dense top 10 (`rag.dense_retrieve`); `RETRIEVE_K = 10`.
7. RRF join (`rag.join`).
8. Cross-encoder ms-marco-MiniLM-L6-v2 reranks to the top 5 (`TOP_K = 5`, span `rag.rerank`).
9. SSE `sources` event lists the 5 chunks with title, page and URL.
10. Prompt build (`rag.prompt_build`): system prompt "use only the numbered sources", sources, history, question plus
    a citation reminder.
11. Generate (`rag.generate`): llama.cpp `/v1/chat/completions`, streaming, `MAX_TOKENS` default 384. Each token is
    an SSE `token` event; the first one records TTFT in `rag_stage_duration_seconds{stage="ttft"}`.
12. `cited()` keeps the sources whose numbers appear in the answer.
13. Postgres write of the answer and citations (`rag.db_write`).
14. SSE `done` with answer, citations and token usage. A failure after the stream started sends SSE `error` and
    increments `rag_chat_errors_total`.
15. Spans go over OTLP gRPC to the collector, then Tempo (and Langfuse if enabled). Prometheus scrapes `/metrics`.

A real first request after a restart (`docs/evidence/phase3-rag.txt` section 5): `POST /chat` 109047.1 ms, of which
`rag.generate` 86117.3 ms, `rag.rerank` 14924.1 ms, `rag.embed_query` 6778.3 ms (cold caches). In kind there is no
collector, so `trace_id` is all zeros (`deploy/README.md`).

### 1.4 Flow: one benchmark run

**In plain English.** A stopwatch robot fires a fixed set of questions at the cook, several at once, and writes down
exactly when each word came back. It saves everything without rounding. Later a separate script reads all the saved
files and builds the tables. Nobody types numbers by hand.

**Technically.** Details in [6.1](#61-llm-benchmarks).

1. Server up: llama.cpp in compose (CPU) or `vllm serve` inside a Kaggle notebook (GPU), pinned model revisions
   (`docs/VERSIONS.md`).
2. `benchmark/loadgen/run.py`: fixed prompts (`prompts.jsonl`, sha256 recorded), seed 1234, `max_tokens` 128 with
   `ignore_eos`, warmup excluded, concurrency 1, 4, 8 on CPU and 1, 8, 32 on GPU, 3 repeats per cell.
3. Per request: TTFT, ITL, TPOT, E2E, errors. GPU sampled every 500 ms with nvidia-smi on Kaggle.
4. One raw JSON per repeat with config, git commit, server args and environment, under
   `benchmark/results/raw/<session>/`. Kaggle output is kept byte-exact.
5. GSM8K accuracy via `benchmark/accuracy/run_gsm8k.sh` against the same server (GPU sessions).
6. `python benchmark/report.py` rebuilds `benchmark/RESULTS.md` and `benchmark/charts/*.png` ("Do not edit by hand").
7. `README.md` copies headline numbers from `RESULTS.md`.

### 1.5 Flow: one code change reaching the cluster

**In plain English.** You change a recipe and file it (commit). The inspector (Jenkins) checks spelling, tastes it
(tests), bakes a sealed box (container image), scans the box for known problems, stores it in the local warehouse with
a unique fingerprint (digest), writes the fingerprint into the blueprint, and the site manager (ArgoCD) swaps the old
box for the new one.

**Technically.** Sources: `ci/Jenkinsfile`, `ci/run.sh`, `ci/jenkins/README.md`, `docs/evidence/phase5b-ci.txt`.
Details in [6.12](#612-deployment-and-gitops) and [6.14](#614-ci-pipeline-and-secrets).

1. Commit on `main` (Jenkins builds only committed `main`).
2. Jenkins runs `ci/run.sh` stages: `lint`, `charts`, `test`, `build` (from `git archive`, `--provenance=false`),
   `scan` (Trivy, fails on CRITICAL), `push` to `localhost:5001`, `bump chart` (CI-only commit with the digest),
   `argocd sync` (publish to `kind-git`, wait for Synced/Healthy).
3. GitHub Actions runs lint, charts, test, build and scan on push to `main` or a pull request, but not push, bump or
   sync.
4. ArgoCD sees the new revision, runs the migrate Job (PreSync), rolls rag-api, and recreates the ingest Job, which
   deletes and rebuilds the `corpus` index (about 9 minutes with no index in build #1, `docs/LIMITATIONS.md`).

Measured: Jenkins builds #1 2471 s, #2 2452 s, #3 220 s with all layers cached (`docs/evidence/phase5b-ci.txt`).

### 1.6 How the project was built

**In plain English.** The project was cut into chapters, each with a "done means" checklist. A site foreman (the
orchestrator) handed each job to a specialist, who could only touch their own part. When a specialist said "finished",
an inspector who did not build it walked through with a checklist and either signed it or wrote a numbered defect list.
Only a signature moved a job to "done".

**Technically.** Plan: `docs/PLAN.md` (slim scope v2: 6 core phases and about 8 GPU hours, down from 9 phases and about
19 GPU hours). Status: `docs/STATUS.md`. Agent briefs: `.claude/agents/*.md`.

| # | Phase or item | Owner agent(s) | Key commits | Outcome |
|---|---|---|---|---|
| 1 | Skeleton + CPU inference | generalist, observability-engineer | `dddb386` | reviewer PASS 2026-10-07 |
| 2 | GPU inference + benchmarks | inference-engineer, benchmark-analyst | `3953d84`, `fc81af2` (failed run v1), `8c01b32`, `c0ad34e` (run v2), `dc6c21e` (CPU rerun) | reviewer PASS 2026-10-09 |
| 3 | RAG | rag-engineer | `707cf34` | reviewer PASS 2026-10-09 after 2 rounds |
| 4 | Observability + SLOs | observability-engineer | `edbc797`, `48b1714` | reviewer PASS 2026-10-09 |
| 5a | Platform: kind, Helm, ArgoCD, KEDA | platform-engineer | `bea3e53`, `785a946` | reviewer PASS |
| 5b | CI: Jenkins + GitHub Actions | platform-engineer | `b8dacd5`, `50a9ece` to `e1c067a`, `332ba71` | reviewer PASS 2026-10-09 |
| 6 | Docs | generalist | `7098f15` | reviewer PASS 2026-10-09 after 2 rounds |
| 7 | Private GitHub repo + Actions | orchestrator, platform-engineer | `13781fd` | first run green |
| 8 | OpenShift via OpenShift Local (OKD) | platform-engineer | `66413fc`, `446b72f`, `de208a9` | reviewer PASS |
| 9 | Loki + Alloy, Langfuse | observability-engineer | `c1d4aa2`, `41aa01f`, `2820268` | runtime PASS, doc fixes applied |
| 10 | DCGM on Kaggle | inference-engineer | `8affb13`, run v3a | experiment answered: basic fields yes, profiling fields need root |
| 11 | Vision YOLOv8n | inference-engineer, benchmark-analyst | runs v3c, v3e, v3g; CPU `a195fd9` | speed, PyTriton and mAP rows committed |
| 12 | TensorRT-LLM FP16 + Triton on T4 | inference-engineer, benchmark-analyst | runs v3b, v3d, v3f, v3h, v3i; `693b44b`, `30673df`, `944586c`, `782b343` | TensorRT-LLM done in v3i (reviewer PASS per `docs/STATUS.md`), see [7.10](#710-tensorrt-llm-vs-vllm-run-v3i); Triton's TensorRT-LLM backend not run |

Agents and what they own:

| Agent | Owns | Brief in short |
|-------|------|----------------|
| generalist | scaffolding, scripts, docs | build only what is assigned, touch only named files, one runnable check |
| inference-engineer | `inference/`, vision, GPU notebooks | vLLM, TensorRT-LLM, Triton, llama.cpp, quantisation; free hardware; T4 limits |
| benchmark-analyst | `benchmark/` | async loadgen, lm-eval, raw JSON per run; never report a number without a raw file |
| rag-engineer | `rag/` | Haystack, ES hybrid retrieval, Postgres with migrations, streaming, licensed corpus, eval |
| observability-engineer | `observability/` | Prometheus, Grafana, Loki, Tempo, OTel, DCGM, Langfuse, SLOs |
| platform-engineer | `deploy/`, `ci/`, `infra/` | non-root images, Helm for kind and OpenShift, ArgoCD, KEDA, Jenkins, Trivy |
| reviewer | read-only | run tests, check acceptance, trace every number to a raw file; PASS or FAIL with numbered defects |

Approvals required from the owner (`docs/PLAN.md`, "Approvals"): each GPU session, creating any account, any push to a
remote. Timeline from `git log`: first commit 2026-10-07, core phases done 2026-10-09, v3 work 2026-10-09 to
2026-10-10.

### 1.7 The evidence rule and the review catches

**In plain English.** Like a lab notebook: if you say "the cake took 30 minutes", you keep the recipe card, the oven
timer printout and the steps, so anyone can redo it. The inspector earned the job several times by finding that a
result looked fine but measured the wrong thing.

**Technically.** Rule (`CLAUDE.md`): "Every number in docs comes from a committed run: config + raw output + script.
If something was not run, say 'not run' and why." In practice: each raw JSON embeds its config, git commit, server
args, model revision and prompts sha256; raw output is byte-exact; failed runs are kept with a README saying "not a
result"; Kaggle notebooks write `FAILED-<step>.txt` or `NOT_RUN-<step>.txt`; numbers seen on screen but not saved are
labelled "observed, no raw log kept".

| Catch | What was wrong | Fix | Source |
|-------|----------------|-----|--------|
| CPU prompt cache | repeats of 64 prompts were 96% cached, so TTFT measured cache hits | 3 cold + 3 warm repeats; `report.py` splits at 50% cache share | `docs/DECISIONS.md`; `docs/evidence/phase2-cpu.txt` |
| Raw files altered | pre-commit rewrote whitespace in Kaggle files | re-downloaded byte-exact; global `exclude`; `.gitattributes -text` | `docs/DECISIONS.md` |
| SLO tests too weak | first promtool tests did not pin thresholds or windows | per-window-pair tests; 9 mutants must FAIL; later exact-value tests | `docs/evidence/phase4-observability.txt` sections 2, 2b, 7 |
| Rebuild did not roll out | tag `0.1.0` is mutable | deploy by digest; `--provenance=false` | `docs/evidence/phase5a-platform.txt` section 8 |
| CI and kind built different digests | file modes 0755 vs 0644, then CRLF | `git -c core.autocrlf=false archive` with fixed mtime | `docs/evidence/phase5b-ci.txt` |
| Lost failure logs | ingest Job `restartPolicy: OnFailure` | `restartPolicy: Never`, `backoffLimit: 3` | `docs/LIMITATIONS.md` |
| Judge self-contradiction | 7B judge gave 0.0 with a reason saying "fully relevant" | read every reason; report headline plus "excluding contradicted" | `docs/evidence/phase3-rag.txt` section 9 |
| Corpus licence | NIST TN 2148 had no proof of federal authorship | dropped, questions replaced, eval re-run | `rag/data/SOURCES.md` |
| Loki claim too narrow | one-shot containers missing from Loki | doc corrected; Alloy socket risk documented | `docs/evidence/v3-loki-langfuse.txt` |
| OKD numbers without logs | some timings had no raw log | raw logs committed, others labelled "observed, no raw log kept" | `docs/evidence/v3-openshift-okd.txt` |
| v3 notebooks before GPU time | silent EOS stop would fake TRT-LLM numbers; cross-session comparison; mAP preprocessing differed by backend | `decode_checks`, same-session vLLM control, stream patch, `rect=False` | commit `3335293` |

---

## 2. Glossary

**In plain English.** A dictionary for this guide. Each term has a one-line everyday meaning and a one-line precise
meaning.

**Technically.** One merged, deduplicated, alphabetised list from all five drafts. Where a value is given (for example
"4096 here"), it comes from the repo file named in the section that uses it.

| Term | Plain | Technical |
|------|-------|-----------|
| AGPL-3.0 | A "share your changes" licence that also covers software you only offer over a network. | GNU Affero GPL v3. Ultralytics YOLOv8 code and weights use it; `inference/README.md` notes serving it over a network brings AGPL obligations. |
| Air gap | A network with no wire to the internet. | Every image, model, package, schema and database must be mirrored inside. See [9.5](#95-purpose-5-air-gapped-and-on-prem-ai). |
| Alembic | A logbook of changes to database table layouts. | Python schema migration tool; one revision `0001` in `rag/migrations/versions/`. Runs as the `rag-migrate` one-shot job. |
| Alert rule | A rule that rings a bell when a condition is true. | Prometheus rule whose PromQL creates an alert (pending, then firing) when it returns series. Five in `observability/prometheus/rules/alerts.yml`. |
| Alertmanager | The switchboard that phones people when an alarm rings. | Prometheus component that routes alerts to receivers. Not run here, so alerts are only visible. |
| All-reduce, NCCL | GPUs adding up their partial answers together. | Collective sum across GPUs; NCCL is NVIDIA's library for it. Used by vLLM TP=2 on Kaggle without P2P. |
| Alloy | A courier that copies every container's diary into the log store. | Grafana Alloy v1.20.1, reads Docker logs through the socket and pushes to Loki (`observability/alloy/config.alloy`). |
| Answer relevancy | "Did the answer stay on the question?" | DeepEval metric: share of answer statements the judge marks relevant to the input. 0.798 (n = 12) here. |
| Apache-2.0 | A permissive licence: use, change, sell, keep the notice. | Licence of the Qwen2.5 models used (Hugging Face API, see section 4) and of ms-marco-MiniLM-L6-v2. |
| App-of-apps | One master blueprint that lists other blueprints. | ArgoCD Application (`deploy/argocd/root.yaml`) whose chart renders three child Applications. |
| ArgoCD | A foreman who keeps the building matching the blueprint in git. | GitOps controller for Kubernetes. v3.5.4 on kind; operator v0.19.0 (ArgoCD v3.4.4) on OKD. |
| Arbitrary UID | The landlord picks your ID number, not you. | OpenShift runs pods as a high UID assigned from the namespace's range, with group 0; images must make writable paths group-writable. |
| ASGI | The plug between a Python web server and a web framework. | Async server gateway interface; uvicorn serves FastAPI through it. |
| AWQ | A 4-bit compression method that protects the most important weights. | Activation-aware Weight Quantisation, INT4 weights with FP16 activations. vLLM runs it on T4 with the Marlin kernel. |
| Backfill | Pasting old readings into the ledger at their original dates. | Writing historical samples into Prometheus storage with `promtool tsdb create-blocks-from openmetrics`. |
| Batching | Cooking several orders in one pan. | Running several requests through one model pass to use the GPU better. |
| Bi-encoder | Gives question and passage separate postcodes, then compares them. | Embeds query and document independently (bge-small here), so document vectors can be precomputed. |
| BM25 | Classic keyword search scoring. | Ranks chunks by query term overlap weighted by term rarity and normalised by chunk length. Top 10 here. |
| BuildKit, provenance | Docker's builder, and the "made on this date" sticker it adds. | BuildKit attaches a provenance attestation with a timestamp, so every build got a new digest; the repo uses `--provenance=false`. |
| Burn rate, burn-rate alerts | How fast you are spending your monthly allowance of failures. | Error ratio divided by (1 minus target). Multi-window alerts fire when a long and a short window both exceed a burn rate (14.4, 6, 3, 1 here). |
| Cardinality | How many separate counters exist. | Number of distinct label combinations; raw URLs as labels would explode it, so rag-api uses route templates. |
| CC BY 4.0 | Free to use with credit to the author. | Creative Commons Attribution 4.0. COCO annotations use it; COCO images keep Flickr terms; nothing from COCO is committed. |
| Cell (benchmark) | One test setting. | One (session, engine, model, precision, concurrency or batch) combination, measured 3 times. |
| Chat template | The fixed wrapper around a conversation before the model reads it. | Role markers and special tokens (Qwen's `<\|im_start\|>`, `<\|im_end\|>`) added before tokenising. |
| Checkpoint | A saved copy of a model's numbers. | Weights plus config files on disk, for example the converted TensorRT-LLM `rank0.safetensors` (3.59 GB). |
| Chunk, chunking | Cutting books into short overlapping strips. | 150-word windows with 30-word overlap (`rag/app/rag.py`, `splitter()`); 3129 chunks for the 9 documents. |
| CI, CD | Automatic checking (CI) and automatic shipping (CD) of every change. | Here Jenkins does both up to an ArgoCD sync; GitHub Actions does CI only. |
| Citation | A footnote number like [2] pointing at a source. | `cited()` in `rag/app/rag.py` extracts `[n]` and keeps 1 <= n <= number of sources. |
| Clock event reasons | The GPU's own note saying why it slowed down. | NVML bitmask `clocks_event_reasons.active`; 0x4 = SW power cap, 0x20/0x40 thermal. Sampled from run v3 onward. |
| Closed loop | Each pretend user asks again only after the last answer finished. | Load pattern where concurrency equals in-flight requests (`benchmark/loadgen/run.py`, `closed_loop`). |
| Compute capability, SM75, Turing | A GPU's generation number. | T4 is 7.5 (Turing): no BF16, no FP8, no FlashAttention 2, no TensorRT-LLM context FMHA kernels. |
| Concurrency | How many users are waiting at once. | Number of requests in flight (c1, c4, c8, c32 in the tables). |
| Confidence interval (95%) | The range the true value is likely in. | Here a normal-approximation interval: mean ± 1.96 x stdev / sqrt(n) (`report.py`, `paired_delta`). |
| Container | A sealed box of software that runs the same everywhere. | Process isolated by namespaces, started from an image. |
| Context propagation | Passing the job folder number along so every step files into the same folder. | Passing OTel context to a new thread (`chat()` passes `ctx` to `run_chat`) so spans join one trace. |
| Context window | How much text the model can hold in mind at once. | Max tokens per request (prompt plus output). 4096 in llama.cpp, split into 2 slots of 2048. |
| Continuous batching | Adding new orders to the pan between stirs instead of waiting for the pan to empty. | Scheduler adds and removes requests from the running batch every decode step (vLLM). |
| Counter, gauge, histogram | Tally that only rises; dial that moves; tally sheet by size buckets. | Prometheus metric types. Histograms expose `_bucket{le=...}`, `_sum`, `_count`. |
| Cross-encoder | Reads question and passage together and scores the match. | Full attention over the concatenated pair; ms-marco-MiniLM-L6-v2 reranks up to 20 candidates to 5. |
| CUDA | NVIDIA's software layer for programming its GPUs. | Runtime and toolkit; versions (12.x, 13.x) must match the driver (R525+, R580+). |
| CV (coefficient of variation) | How jumpy repeated measurements are. | stdev / mean across repeats; `report.py` flags CV > 0.10. |
| CVE | A public ID for a known security hole. | Common Vulnerabilities and Exposures; Trivy reports them by severity (CRITICAL, HIGH). |
| Datasource (Grafana) | A plug from the dashboard wall to one ledger. | Grafana connection to Prometheus, Tempo or Loki (`observability/grafana/provisioning/datasources/`). |
| DCGM | NVIDIA's professional GPU sensor kit. | Data Center GPU Manager 4.7.0-2. On Kaggle, basic fields worked; profiling fields failed `DCGMError_RequiresRoot`. |
| Decode | Writing the answer one word at a time. | Token-by-token generation after prefill; memory-bound at batch 1. |
| DeepEval | A marking kit for AI answers. | LLM evaluation library 4.2.8; `FaithfulnessMetric`, `AnswerRelevancyMetric` with a local judge. |
| Deployment, Job, StatefulSet | "Keep N copies running"; "run once to completion"; "keep N copies with their own storage". | Kubernetes workload objects used by the charts in `deploy/helm/`. |
| Derived field | A pattern in a log line that becomes a clickable link. | Grafana Loki setting: regex `trace_id=([0-9a-f]{32})` links to the Tempo datasource. |
| Digest | The fingerprint of an image; cannot be moved. | `sha256:` content hash. The repo deploys rag-api by digest. |
| Docker, Docker Compose | The crane that moves software boxes; a seating plan for many boxes. | Docker Engine 29.6.2; Compose v5.3.1 runs `docker-compose.yml` with profiles `ingest`, `eval`, `langfuse`. |
| E2E latency | Total time from order to last bite. | End minus start of a request, in seconds. |
| Elasticsearch | A card catalogue that searches by words and by meaning. | Search engine 9.5.5; index `corpus` holds text, metadata and vectors; BM25 and kNN queries. |
| Embeddings | A list of numbers that captures what a text means. | Fixed-length vector from an encoder; bge-small-en-v1.5 outputs 384 numbers (Hugging Face config, section 4). |
| Engine (TensorRT) | A model pre-compiled for one exact GPU. | Serialized TensorRT plan; not bit-reproducible, so sha256 is recorded, not pinned. |
| EOS token, ignore_eos | The model's "I am done" signal, and the option to keep going anyway. | `ignore_eos: true` forces exactly `max_tokens` (128) output tokens in benchmarks. |
| Error budget | The allowed number of failures per month. | 1 minus the SLO target: 1% for availability, 5% for TTFT here. |
| Faithfulness | "Is every claim in the answer backed by the pages shown?" | DeepEval metric: share of answer claims not contradicted by the retrieved context. 0.592 (n = 12). |
| FastAPI | The shop counter that accepts and checks requests. | Python web framework 0.142.2 with pydantic validation; served by uvicorn. |
| Few-shot (5-shot) | Showing worked examples before the question. | GSM8K runs 5 solved examples as multi-turn chat before each item. |
| FMHA (context FMHA) | A fast attention method. | Fused multi-head attention kernels for prefill. TensorRT-LLM 0.14 has none for SM75, so `--context_fmha disable`. |
| FP32, FP16, BF16, FP8, INT8, INT4 | How many bits each number gets. | 32-bit float (full), 16-bit float (T4's fast format), bfloat16 (FP32 range, no T4 support), 8-bit float (no T4 support), 8-bit and 4-bit integers (quantised weights). |
| GGUF, Q4_K_M | llama.cpp's single-file model format, and its "about 4 bits, medium" grade. | GGUF stores weights plus metadata; Q4_K_M is a 4-bit k-quant mixed block scheme averaging about 4.8 bits per weight (some tensors at Q6_K). |
| git daemon | A tiny read-only library desk that hands out copies of the repo. | Unauthenticated `git://` server; `kind-git` serves the repo to ArgoCD. |
| GitHub Actions | GitHub's own checklist runner. | `.github/workflows/ci.yml` on `ubuntu-24.04`: lint, charts, test, build, scan. |
| gitleaks | A sniffer dog for passwords in commits. | Secret scanner v8.30.1: staged changes in pre-commit, full history in CI. |
| GitOps | The cluster changes only when git changes. | An agent (ArgoCD) applies what git declares and reverts drift. |
| GPTQ | Another 4-bit compression method. | Post-training INT4 weight quantisation; named as unsupported by TensorRT-LLM on SM75 (`docs/DECISIONS.md`). |
| GPU | A chip with thousands of small cores, good at the maths models need. | Here NVIDIA Tesla T4 on Kaggle. |
| Grafana | The wall of screens that draws charts from the ledgers. | Grafana 13.2.3 with provisioned datasources and dashboards as JSON in git. |
| Greedy decoding, temperature | Always pick the most likely next word. | Temperature 0; deterministic in principle (llama.cpp was not fully deterministic here, q11). |
| GSM8K | A school maths word-problem exam. | Benchmark dataset; 250 items, strict-match (number after `####`) and flexible-extract (last number). |
| Haystack | Lego bricks for search and question answering. | Python framework 2.31.0; components joined into indexing, retrieval and generation pipelines. |
| Helm | A form letter for Kubernetes files. | Chart = templates + `values.yaml`; values files per target (`values-kind.yaml`, `values-openshift.yaml`). Helm 4.2.4. |
| HPA | The built-in Kubernetes "add or remove copies" controller. | HorizontalPodAutoscaler; KEDA creates `keda-hpa-llamacpp` and feeds it the metric. |
| Hybrid retrieval | Search by words and by meaning, then merge. | BM25 + dense kNN, fused with RRF, then reranked. Gain over single methods not measured here. |
| Image (container) | The sealed box in the warehouse. | Read-only filesystem layers plus metadata; built from a Dockerfile. |
| Inference | Using a trained model to get answers. | Running forward passes to produce outputs, as opposed to training. |
| Ingress, Traefik | The receptionist who sends visitors to the right office. | Ingress rules route HTTP by host; Traefik 3.7.13 (chart 41.6.1) implements them on kind at `127.0.0.1:8090`. |
| Init container | A helper that runs before the main program starts. | Runs to completion first; llama.cpp's `model` init container downloads and checks the GGUF. |
| ITL | The gap between two words arriving. | Inter-token latency: every gap between consecutive streamed chunks, pooled over requests. |
| Jenkins, Jenkinsfile, JCasC | A factory foreman, his checklist, and his written job description. | Jenkins 2.580.1 LTS; `ci/Jenkinsfile` runs `ci/run.sh`; `ci/jenkins/casc.yaml` configures it as code. |
| Judge model | An exam marker that is itself an AI. | LLM-as-a-judge; here Qwen2.5-7B-Instruct GGUF Q4_K_M on llama.cpp, JSON-schema output. |
| K3s | A lightweight Kubernetes that fits small or offline sites. | CNCF Kubernetes distribution, common on edge and air-gapped sites. The repo does not target it (kind is used). |
| Kaggle | A website that lends free GPU time in notebooks. | Free 2x T4, about 30 GPU-h/week, 12 h sessions; phone verification needed for GPU and internet. |
| KEDA | A host who calls in more cooks when the queue grows. | Event-driven autoscaler 2.21.0; `ScaledObject` scales llama.cpp 1 to 3 on in-flight requests. |
| kind | A doll's house Kubernetes inside one Docker container. | Kubernetes in Docker v0.33.0, node image v1.36.4 pinned by digest, one node. |
| kubeconform | A spell checker for Kubernetes YAML. | Validates rendered manifests against pinned JSON schemas, `-strict`, no `-ignore-missing-schemas`. |
| Kubernetes | A building manager who keeps reality matching the written plan. | Container orchestrator; controllers reconcile desired and actual state. |
| KV cache | The model's notes on what it already read, so it does not reread. | Stored attention keys and values of earlier tokens; vLLM had 9.08 GiB for it on 1.5B FP16. |
| Langfuse | A viewer built for AI app traces. | Self-hosted Langfuse 4.56.0 (profile `langfuse`); gets generic spans, so its LLM views stay empty here. |
| Latency | How long one request takes. | TTFT, ITL, TPOT and E2E are kinds of latency. |
| llama.cpp | A lean program that runs language models on an ordinary processor. | C/C++ engine, image `server-b11459`; OpenAI API and `/metrics`; `--parallel 2` gives 2 slots. |
| LLM | A very well read autocomplete. | Large language model: a neural network that predicts the next token. |
| lm-evaluation-harness | A standard exam paper and marking scheme for models. | lm-eval 0.4.13; `local-chat-completions` against any OpenAI-compatible server. |
| Lock file | An exact shopping list of package versions. | `rag/uv.lock`, `notebooks/*-requirements.txt`, `benchmark/requirements.txt`. |
| Loki, LogQL | A diary box indexed only by the label on the cover; its search language. | Grafana Loki 3.7.8, 7-day retention; queries like `{service="rag-api"} \|= "trace_id="`. |
| mAP (mAP50, mAP50-95) | How well the detector's boxes match the true boxes. | Mean average precision at IoU 0.5, and averaged over IoU 0.5 to 0.95. 0.401 mAP50-95 here on 500 images. |
| Marlin kernel | A fast GPU routine for 4-bit maths. | vLLM kernel for AWQ matrix multiplies ("Using MarlinLinearKernel"). |
| McNemar test | A test that looks only at questions where two models disagree. | Exact test on discordant pairs; p = 0.105 strict, 0.068 flexible (`docs/evidence/phase2-gpu.txt`). |
| Memory-bound vs compute-bound | Waiting for ingredients vs waiting for the cook's hands. | Limited by reading weights from memory (batch-1 decode) vs by arithmetic (large batches). |
| Metrics, traces and logs | Counters, journeys and diaries: the three kinds of telemetry. | Metrics in Prometheus, traces in Tempo, logs in Loki here. |
| MIT licence | A very short permissive licence. | Licence of bge-small-en-v1.5 (`docs/VERSIONS.md`, Hugging Face API). |
| Model | A file of learned numbers plus the code that uses them. | Weights + architecture config + tokenizer; see section 4. |
| Multi-stage build | Cook in a messy prep kitchen, serve only the plate. | Dockerfile with several `FROM` stages; only the last ships (`rag/Dockerfile`). |
| Mutation testing | Breaking the rules on purpose to check the tests notice. | Mutants with one deliberate bug; all 9, then all 17, made `promtool test rules` fail. |
| Namespace | A named room in the cluster. | Partition for objects and policies; `llm` here, with PSA restricted on kind. |
| NMS | Removing duplicate boxes around the same object. | Non-maximum suppression; excluded from vision timing here. |
| nvidia-smi | The basic GPU status command. | Sampled every 500 ms during measured windows by `GpuSampler` in `run.py`. |
| Observability | Being able to tell what a running system is doing from what it emits. | Metrics, traces and logs plus SLOs and alerts (`observability/`). |
| ONNX, ONNX Runtime | A universal model file format, and a universal player for it. | Opset 17 export; ORT 1.26.0 runs it on CPU or CUDA, with no silent CPU fallback in this repo. |
| OpenAI-compatible API | A web interface shaped like OpenAI's, so standard clients work. | `/v1/chat/completions`; llama.cpp, vLLM and the TensorRT-LLM wrapper all expose it. |
| OpenShift, OKD | Kubernetes with stricter house rules (Red Hat's product); OKD is the free community edition. | Adds SCC, Routes, internal registry, OLM. Tested: OKD 4.22.0-okd-scos.10 single node via OpenShift Local (crc) 2.64.0. |
| OpenTelemetry (OTel, OTLP) | A common language for "what happened and how long it took". | Vendor-neutral telemetry standard; SDK 1.45.1 in rag-api, Collector 0.162.0, OTLP on 4317 (gRPC) and 4318 (HTTP). |
| Operator | An agent who installs and maintains an app for you. | Controller driven by a custom resource; `argocd-operator` on OKD via OLM. |
| p50, p95, p99 | The typical case, the unlucky one in twenty, the unlucky one in a hundred. | Percentiles; `metrics.py` uses linear interpolation like numpy's default. |
| Paged attention | Storing the model's notes in fixed-size pages handed out on demand. | vLLM's KV cache in fixed-size blocks allocated on demand; waste is limited to the last block of each request, and blocks can be shared between requests. |
| Paired comparison | Comparing two models question by question. | AWQ minus FP16 per GSM8K item, then mean and CI; removes "some questions are just hard". |
| Parameters, weights | The learned numbers inside a model. | Count sets size: Qwen2.5-1.5B-Instruct 1,543,714,304; YOLOv8n 3,151,904 (sources in section 4). |
| Pod | One or more containers that live and die together. | Smallest Kubernetes unit. |
| PostgreSQL | A filing cabinet for conversations. | Relational database 18.6; tables `users`, `sessions`, `messages` with JSONB citations. |
| Power cap | The GPU's maximum allowed power draw. | 70 W on a T4; the driver lowers clocks to stay under it (SW power cap, bit 0x4). See [7.4](#74-the-gpu-power-cap-story). |
| Pre-commit | A checklist that runs every time you commit. | `.pre-commit-config.yaml`: whitespace, YAML, JSON, ruff, yamllint, gitleaks, hadolint; runner pre-commit 4.6.2. |
| Prefill | Reading the whole question before writing the first word. | One pass over the prompt tokens; the main TTFT cost on CPU. |
| Probe | A regular "are you ok?" check. | Startup, readiness and liveness HTTP checks in the Helm charts. |
| Prometheus, PromQL | A clerk who reads every counter every 15 s; the language for asking the ledger questions. | Pull-based time series database v3.15.0 with rules; PromQL functions like `rate()` and `histogram_quantile`. |
| Prompt caching | Remembering questions already read so they need not be reread. | llama.cpp `--cache-prompt` and `--cache-ram` (on by default) reuse KV for repeated prefixes; vLLM prefix caching is off here. Unrelated to hosted-API prompt caching. |
| Public domain | Nobody owns the copyright. | US federal works (17 U.S.C. 105); basis for the 9 NIST and NASA documents. |
| PVC (RWO, RWX) | A request for a storage cupboard; one-room or shared-by-many-rooms. | PersistentVolumeClaim; model PVC is ReadWriteOnce on kind, ReadWriteMany on OpenShift. |
| PyTriton | A way to get a Triton front desk from Python without Docker. | `nvidia-pytriton` 0.7.0, ships `tritonserver` with only the Python backend. |
| Quantisation | Storing each number with fewer bits, like saving a photo as a smaller JPEG. | Fewer bits per weight (AWQ INT4, GGUF Q4_K_M) cuts memory and memory traffic, at some quality risk. |
| RAG | Look things up first, then answer from what you found. | Retrieval-augmented generation: retrieve chunks, put them in the prompt, generate with citations. |
| Recall@k (recall@5, doc recall@5) | "Was the right page among the k shown?" | Here: gold document and page `p` or `p - 1` in the top 5 (0.867); doc recall: gold document in top 5 (1.000). |
| Recording rule | Doing a long sum once and writing down the answer. | Prometheus rule storing a query result as a new series; 18 in `rules/recording.yml`. |
| Registry | A warehouse for software boxes. | Image store: `kind-registry` (`localhost:5001`), OpenShift internal registry, Docker Hub, ghcr.io, quay.io. |
| repetition_penalty, generation_config.json | A setting that discourages repeating words; the model's default settings file. | vLLM applies `generation_config.json` unless `--generation-config vllm`; for Qwen2.5-1.5B that is repetition_penalty 1.1. |
| Reranking | A careful second reader reorders the shortlist. | Cross-encoder rescoring of fused candidates down to the top 5. |
| Route (OpenShift) | OpenShift's front-door rule with free HTTPS. | `route.openshift.io/v1` with edge TLS and a 600 s router timeout for SSE. |
| RRF | Merge two ranked lists by giving points for each place. | Reciprocal rank fusion: score = sum of 1 / (k + rank); Haystack's default k, not set in this repo. |
| SBOM | A list of every ingredient in the box. | Software bill of materials; not produced here. |
| SCC (restricted-v2) | The strict landlord rule: no root, no special keys, ID handed out at the door. | OpenShift Security Context Constraint; verified pods ran as `uid=1000670000 gid=0`. |
| securityContext | The per-container house rules. | `runAsNonRoot`, `readOnlyRootFilesystem`, `capabilities: drop [ALL]`, seccomp `RuntimeDefault` in the charts. |
| Session (chat) | One conversation thread. | UUID row in `sessions`; no access control on it in this repo. |
| sha256 | A tamper-proof fingerprint of a file. | Cryptographic hash used to pin downloads, embedded notebook code and images. |
| SLI, SLO, SLA | What you measure, what you promise yourself, what you promise a customer with penalties. | SLI = good / valid ratio; SLO = target over 30 days (99% availability, 95% TTFT within 60 s); no SLA exists here. |
| SM, SM clock | A GPU compute block, and how fast it ticks. | Streaming multiprocessor; T4 max SM clock 1590 MHz (`nvidia-smi-q-end.txt`). |
| Smoke test | Switch it on and check nothing smokes. | `scripts/smoke.sh`; also `--smoke` runs that `report.py` never reports. |
| Span, trace ID | One timed step, and the ticket number shared by all steps of a request. | OTel span with name, times, parent; trace ID is 128 bits (32 hex). |
| SSE | A ticker tape: the server pushes pieces of the answer as they are ready. | Server-sent events over one HTTP response (`text/event-stream`): `session`, `sources`, `token`, `done`, `error`. |
| Sync wave | The order of building: foundations, then walls, then roof. | ArgoCD annotation `argocd.argoproj.io/sync-wave` 0, 1, 2 for the three apps. |
| T4 | A free, modest data centre GPU. | NVIDIA Tesla T4: 15360 MiB, 70 W cap, Turing SM75, 320 GB/s memory bandwidth (`docs/evidence/phase2-gpu.txt`). |
| Tag (image) | A movable name sticker on a box. | Human name like `0.1.0`; can be moved, unlike a digest. |
| Tempo, TraceQL | A filing cabinet for request journeys; its search language. | Grafana Tempo 3.1.0 single binary; queries like `{ resource.service.name = "rag-api" && name = "rag.chat" }`. |
| Tensor | A grid of numbers, possibly with many dimensions. | Basic data unit of neural networks; YOLOv8n input is [1, 3, 640, 640]. |
| Tensor parallelism (TP) | Two cooks each doing half of every step and passing notes. | Splitting each layer's matrices across GPUs with two all-reduces per transformer layer; TP=2 on 2x T4. |
| TensorRT | NVIDIA's compiler that tailors a network to one GPU. | Builds engines; `tensorrt-cu12` 10.16.1.11 for vision. NVIDIA proprietary but free. |
| TensorRT-LLM | TensorRT for language models. | 0.14.0 (last with Turing); `trtllm-build` compiles a checkpoint into an engine. Measured once against same-session vLLM on a T4 (run v3i, [7.10](#710-tensorrt-llm-vs-vllm-run-v3i)). |
| Throughput, tokens per second | How many words the whole shop produces per second. | Aggregate output tokens / wall time ("out tok/s"); per-request tok/s is 128 / E2E. |
| Tokenizer | The cutter that turns text into numbered pieces. | Maps text to token ids; all Qwen variants share one tokenizer. |
| Tokens | The pieces of text a model reads and writes. | Sub-word units; 64 benchmark prompts total 15485 prompt tokens. |
| TPOT | Average time per word once the answer has started. | (last token time minus first) / (output tokens minus 1), per request. |
| Training vs fine-tuning | Learning from scratch vs extra lessons for a specific job. | Training sets all weights from data; fine-tuning adjusts a trained model. This repo does neither; it only runs inference. |
| Triton Inference Server | A standard front desk for models over the network. | NVIDIA server with KServe v2 HTTP/gRPC; 2.51.0 via PyTriton here (Python backend only). |
| Trivy | An inspector that checks the box against a list of known defects. | Scanner 0.75.0; CI fails on any CRITICAL vulnerability. |
| TTFT | How long until the first word appears. | Time to first token: first streamed content minus request start. |
| uv | A very fast, exact Python installer. | uv 0.11.32: lock files, venvs, `uv sync --frozen`. |
| Vector search, dense retrieval | Finding passages with "postcodes" close to the question's. | kNN over embeddings in Elasticsearch; similarity function is the integration default. |
| Vision models | Models that look at pictures. | Here YOLOv8n object detection, 640x640 input. |
| vLLM | A high-throughput GPU server for language models. | 0.31.0 `+cu129`; continuous batching, paged attention, `TRITON_ATTN` backend on T4. |
| VRAM | The memory on the graphics card. | GPU memory where weights and KV cache must fit; 15360 MiB per T4. |
| Warmup | The first customers of the day are slower, so they are not counted. | `max(2, concurrency)` unmeasured requests before each measured window. |
| Workspace (scratch memory) | A scratch pad a calculation needs while running. | Temporary GPU memory; TensorRT-LLM's attention workspace sank the v3h build. |
| YOLO, YOLOv8n | A one-glance object detector; the nano size. | Ultralytics YOLOv8n, 3,151,904 parameters, COCO classes ("person", "bicycle", "car", ...). |

---

## 3. Tools and open-source projects

**In plain English.** A catalogue of every tool in the kitchen: what it is, why it was picked over other brands, which
model number, and where it is used. The table is the quick reference; the sections after it explain each tool twice.

**Technically.** Versions come from `docs/VERSIONS.md`, `observability/VERSIONS.md`, `rag/pyproject.toml`,
`notebooks/*.in`, `benchmark/requirements.in`, `scripts/tools.sh` and the evidence files. "Not stated" means the repo
gives no reason or alternative. Every tool is open source except TensorRT ("NVIDIA proprietary software, free to use")
and DCGM's profiling package (NVIDIA proprietary, free to use; the rest of DCGM is Apache-2.0) (`inference/README.md`).

### 3.1 Master table

In plain English: one row per tool, so you can find any tool at a glance.

| Tool | What it is | Why chosen | Alternatives (repo's words) | Version | Files |
|------|-----------|------------|-----------------------------|---------|-------|
| llama.cpp | CPU LLM server (GGUF) | free CPU path required by the rules | Vulkan on iGPU (stretch, not done) | image `server-b11459` (commit f498f864f) | `docker-compose.yml`, `deploy/helm/llamacpp/` |
| vLLM | GPU LLM server | AWQ on T4 via Marlin; newest release at pinning (2026-10-05) and still builds SM75 kernels | 0.30.0 fallback; CUDA 13 PyPI wheel (driver risk); FlashInfer, FA2 backends (not on SM75) | 0.31.0 `+cu129`, torch 2.13.0+cu129 | `notebooks/build_notebook.py`, `notebooks/vllm-requirements.txt` |
| TensorRT-LLM | NVIDIA LLM compiler and runtime | v3 stretch item; 0.14.0 is the last with Turing | rejected for core (long builds, no INT4 AWQ/GPTQ on SM75) | 0.14.0, TensorRT 10.4.0, torch 2.4.0+cu124 | `inference/trtllm/`, `notebooks/kaggle-trtllm/` |
| TensorRT | NVIDIA network compiler (vision) | FP16 engines per batch | TensorRT 11 (strongly typed, Ultralytics then needs ModelOpt) | `tensorrt-cu12` 10.16.1.11 | `inference/vision/export.py`, `benchmark/vision/run_vision.py` |
| ONNX Runtime | ONNX model player, CPU or CUDA | same file on CPU and GPU | 1.27+ needs CUDA 13 and R580 | 1.26.0 (CPU and GPU builds) | `benchmark/vision/run_vision.py` |
| Triton via PyTriton | network front desk for models | only Triton that runs without Docker on Kaggle | NGC container (needs Docker); native TensorRT backend (not in PyTriton) | `nvidia-pytriton` 0.7.0, Triton 2.51.0 | `inference/vision/triton_serve.py` |
| Ultralytics | YOLO library: export and mAP validator | official exporter and validator | not stated | 8.4.174 | `inference/vision/export.py`, `benchmark/vision/eval_map.py` |
| lm-evaluation-harness | standard LLM exam runner | named in the stack | not stated | `lm-eval[api]` 0.4.13 | `benchmark/accuracy/` |
| httpx | async HTTP client | streaming load generator | not stated | 0.28.1 | `benchmark/loadgen/run.py` |
| matplotlib | charts | report charts | not stated | 3.11.2 | `benchmark/report.py` |
| uv | Python installer and lock tool | exact locks, fast venvs | not stated | 0.11.32 | everywhere Python is installed |
| Kaggle CLI and notebooks | free T4 sessions | free GPU quota | Colab (allowed, not used); Kaggle dataset or git clone (rejected for embedding) | CLI 2.2.4 | `notebooks/` |
| nvidia-smi | basic GPU sampler | always present | DCGM | driver 580.178.04 on Kaggle | `benchmark/loadgen/run.py` |
| DCGM | detailed GPU telemetry | test "is batch-1 decode memory-bound?" | `dcgm-exporter` (needs Docker); nvidia-smi | 4.7.0-2 | `benchmark/loadgen/dcgm_sampler.py` |
| Haystack | RAG pipeline framework | named in the plan (2.x) | Haystack 3.x exists, plan says 2.x | `haystack-ai` 2.31.0 | `rag/app/rag.py` |
| Elasticsearch | BM25 + vector store | one engine for hybrid search | official Helm chart (deprecated), ECK (too heavy) | 9.5.5; `elasticsearch-haystack` 6.4.1 (client 8.19.3) | `docker-compose.yml`, `deploy/helm/platform-deps/` |
| PostgreSQL | chat history database | named in the plan | not stated | `postgres:18.6-alpine3.24` | `rag/app/db.py` |
| SQLAlchemy, Alembic | ORM and migrations | "Postgres chat history with Alembic" | not stated | 2.1.3, 1.20.0 | `rag/app/db.py`, `rag/migrations/` |
| FastAPI, uvicorn | web API and server | "FastAPI with SSE streaming" | WebSockets not discussed | 0.142.2, 0.54.0 | `rag/app/main.py` |
| sentence-transformers, torch | runs embedder and reranker | wrapped by Haystack | not stated | 5.6.1; torch 2.14.1 CPU wheel | `rag/pyproject.toml` |
| pypdf | PDF to text | used by `PyPDFToDocument` | not stated | 6.19.0 | `rag/app/rag.py` |
| DeepEval | judge-based RAG metrics | plan allowed "RAGAS or DeepEval" | RAGAS (reason not stated); hosted judge (paid, excluded) | 4.2.8 | `rag/eval/run_eval.py` |
| OpenTelemetry SDK | span creation in rag-api | plan: spans per pipeline step in Tempo | not stated | 1.45.1 | `rag/app/main.py` |
| prometheus-client | metrics in rag-api | plain library plus own middleware | OTLP metrics (collector has traces only) | 0.26.0 | `rag/app/main.py` |
| Prometheus, promtool | metric store, rules, tests, backfill | stack | not stated | `prom/prometheus:v3.15.0` | `observability/prometheus/` |
| Grafana | dashboards | stack | not stated | 13.2.3 | `observability/grafana/` |
| Tempo | trace store | stack | Jaeger not discussed | 3.1.0 | `observability/tempo/tempo.yaml` |
| OTel Collector | span router | one stream to Tempo and Langfuse | not stated | contrib 0.162.0 | `observability/otel-collector/` |
| Loki | log store | stack (stretch, v3 item 9) | not stated | 3.7.8 | `observability/loki/loki.yaml` |
| Alloy | log shipper | "Promtail is deprecated in favour of Alloy" | Promtail | v1.20.1 | `observability/alloy/config.alloy` |
| Langfuse (+ ClickHouse, Redis, MinIO, Postgres) | LLM trace UI | stack (stretch) | not stated | 4.56.0; ClickHouse 25.12.11.4; Redis 7.4.11; MinIO by digest | `docker-compose.yml` profile `langfuse` |
| Docker, BuildKit, Compose | containers and builds | local, free | not stated | Engine 29.6.2, Compose v5.3.1 | `docker-compose.yml`, Dockerfiles |
| kind | local Kubernetes | CLAUDE.md allows kind or k3d | k3d (why kind not stated) | v0.33.0, node v1.36.4 by digest | `infra/kind/cluster.yaml` |
| registry | local image store | push and deploy by digest | not stated | `registry:3.1.2` | `scripts/kind-up.sh` |
| git daemon | local git server for ArgoCD | no remote; push needs approval | a hosted remote | alpine 3.23.0, `git-daemon` 2.52.0-r0 | `infra/git-server/Dockerfile` |
| Helm | chart templating | kind and OpenShift values from one chart | Bitnami charts (images no longer free) | 4.2.4 | `deploy/helm/` |
| kubeconform | manifest schema check | strict validation in CI | not stated | v0.8.0 | `scripts/helm-check.sh` |
| ArgoCD (+ argocd-operator) | GitOps controller | plan | upstream `install.yaml` on OKD (redis UID 999 rejected) | v3.5.4 kind; operator v0.19.0 (ArgoCD v3.4.4) OKD | `deploy/argocd/` |
| KEDA | metric-driven autoscaler | "ScaledObject on llama.cpp pending requests" | Red Hat Custom Metrics Autoscaler (not tested); OKD catalog KEDA 2.7.1 | chart 2.21.0 | `deploy/helm/llamacpp/templates/scaledobject.yaml` |
| Traefik | ingress on kind | not stated | not stated | chart 41.6.1, `traefik:v3.7.13` | `scripts/kind-up.sh` |
| OKD, OpenShift Local (crc) | OpenShift in a VM | free, no Red Hat account with preset `okd` | Developer Sandbox (stretch, not done) | OKD 4.22.0-okd-scos.10, crc 2.64.0 | `scripts/okd-up.sh` |
| skopeo | image copier | push to OKD internal registry | not stated | v1.22.3 by digest | `scripts/okd-up.sh` |
| Jenkins | CI/CD server | plan: "Jenkins (Jenkinsfile)" | separate build agent (not done) | 2.580.1 LTS jdk21 by digest, 63 plugins | `ci/jenkins/`, `ci/Jenkinsfile` |
| GitHub Actions | hosted CI | v3 item 7 | not applicable | `actions/checkout` v7.0.1 by SHA | `.github/workflows/ci.yml` |
| Trivy | vulnerability scanner | CRITICAL gate | not stated | 0.75.0 (tag) | `ci/run.sh` |
| gitleaks | secret scanner | pre-commit and full-history scan | upstream hook (Go bootstrap fails on Windows) | v8.30.1 | `.pre-commit-config.yaml`, `.gitleaks.toml` |
| hadolint, actionlint, shellcheck | Dockerfile, workflow and shell linters | CI lint | not stated | v2.15.1; 1.7.12 by digest; bundled | `ci/run.sh` |
| yamllint, ruff, pre-commit | YAML and Python lint, hook runner | lint before commit | not stated | 1.38.0; 0.16.10; pre-commit 4.6.2 | `.pre-commit-config.yaml`, `.yamllint`, `ruff.toml` |

### 3.2 Serving

**In plain English.** Four kinds of kitchen. llama.cpp is a small efficient kitchen that needs no special oven (CPU).
vLLM is a big restaurant that serves many tables in one trip (GPU). TensorRT and TensorRT-LLM are tailors that make a
suit for one exact GPU: perfect fit, slow to make. Triton is a standard front desk that takes orders over the network
and passes them to whichever kitchen is behind it.

**Technically.**
- **llama.cpp** serves the GGUF model with `--ctx-size 4096 --parallel 2 --threads 4 --metrics`, so each request gets
  2048 tokens of context and at most 2 decode at once (`inference/README.md`). Prompt caching is on by default
  (`--cache-prompt`, `--cache-ram` default 8192 MiB, `--cache-idle-slots`), which matters for benchmarks
  ([6.2](#62-the-llamacpp-prompt-cache-issue)).
- **vLLM 0.31.0** runs with `--dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.90 --attention-backend
  TRITON_ATTN --no-enable-prefix-caching --seed 0` and, from v3, `--generation-config vllm`
  (`notebooks/build_notebook.py`, `start_server`). The `+cu129` wheel runs on R525+ drivers; the PyPI wheel needs
  CUDA 13 and R580+, unknown before a session. Run v2 got driver 580.178.04, so either would have worked that day
  (`docs/DECISIONS.md`). Resolving the lock from the GitHub wheel URL "hung for about 3 h", so it is resolved from
  PyPI metadata plus `notebooks/vllm-overrides.txt` (`docs/evidence/phase2-prep.txt`).
- **TensorRT-LLM 0.14.0** (cp310 only, so a Python 3.10.20 venv; Open MPI 4.1.6-7ubuntu2). 0.15.0 drops Turing and
  0.16.0 stops compiling the SM70 kernels a T4 can run (`inference/README.md`). `trtllm-serve` does not exist before
  0.15, so `inference/trtllm/serve.py` wraps the 0.14 example OpenAI server ([5.3](#53-inference)). Run v3i built two
  engines (3.4G each) and served them this way; results in [7.10](#710-tensorrt-llm-vs-vllm-run-v3i).
- **TensorRT 10.16** builds one FP16 engine per batch size (1, 8, 32) for YOLOv8n, inputs and outputs FP32. The last
  10.x; TensorRT 11 is "strongly typed only and Ultralytics then needs ModelOpt for FP16" (`inference/README.md`).
- **ONNX Runtime 1.26.0** is the last CUDA 12 build. The same version runs on the laptop CPU, on purpose. With CUDA
  requested the code raises if ORT silently falls back to CPU.
- **Triton via PyTriton 0.7.0** binds one model per engine (`yolov8n_b1`, `_b8`, `_b32`), exposes KServe v2 HTTP on
  8000 and metrics on 8002. Only the Python backend is available, so these rows measure Python plus HTTP, not Triton's
  native TensorRT backend. The TensorRT-LLM Triton backend ships only in the NGC `-trtllm-python-py3` container,
  which needs Docker; Kaggle has none, recorded as `NOT_RUN-triton-tensorrtllm-backend.txt`.
- **Ultralytics 8.4.174** exports ONNX and engines and runs the mAP validator. `YOLO_OFFLINE=1
  YOLO_AUTOINSTALL=False` stop it downloading or pip-installing at run time (`notebooks/build_v3.py`).

### 3.3 Benchmarking

**In plain English.** The stopwatch, the exam paper, the drawing pens and the borrowed ovens. A robot (the load
generator) plays impatient users; an exam (lm-eval GSM8K) checks the model did not get dumber; sensors (nvidia-smi,
DCGM) watch the GPU; a chart library draws the results; Kaggle lends the GPUs; uv builds clean Python rooms for each
tool.

**Technically.**
- **httpx 0.28.1** `AsyncClient` streams responses with connection limits of concurrency + 4 (`run.py`, `run_cell`).
- **lm-eval 0.4.13** runs `--model local-chat-completions` on task `gsm8k_pinned`, a copy of lm-eval's GSM8K with the
  dataset revision pinned (`benchmark/accuracy/tasks/gsm8k_pinned.yaml`).
- **nvidia-smi** is sampled every 500 ms with timestamp, memory, utilisation, SM clock, temperature, power, memory
  clock and `clocks_event_reasons.active`; the last two were added after run v2.
- **DCGM 4.7.0-2** is five `.deb` files checked by sha256 and unpacked with `dpkg-deb -x` (no apt). Profiling fields
  need root, which Kaggle does not give ([7.11](#711-kaggle-run-history)).
- **matplotlib 3.11.2** with the `Agg` backend draws the two charts.
- **uv 0.11.32** compiles locks (`uv pip compile`, commands at the top of each `.in` file), builds venvs with pinned
  Pythons, installs with `--torch-backend cu129` or `cu124`, and uses `--build-constraints` for the mpi4py build. On
  Kaggle the base Python is only used to install the pinned uv.
- **Kaggle** notebooks are private push folders with `kernel-metadata.json` (`enable_gpu`, `enable_internet`,
  `machine_shape: "NvidiaTeslaT4"`). Kernel ids `<kaggle-user>/vllm-t4-benchmark`, `<kaggle-user>/v3-dcgm-vision-t4`,
  `<kaggle-user>/v3-trtllm-t4`.

### 3.4 RAG

**In plain English.** The library and the front desk. Haystack is the Lego kit for the search line. Elasticsearch is
the card catalogue. PostgreSQL is the filing cabinet for conversations, with SQLAlchemy and Alembic managing its
drawers. FastAPI and uvicorn are the counter and the building. sentence-transformers runs the two small clerks
(embedder and reranker). pypdf scans PDFs. DeepEval is the marking scheme for the exam marker.

**Technically.**
- **Haystack 2.31.0** builds three pipelines in `rag/app/rag.py`: indexing, retrieval, generation
  ([5.2](#52-rag)). Haystack 3.x exists; the plan specifies 2.x.
- **Elasticsearch 9.5.5** runs single node, security on, TLS off, heap `-Xms1g -Xmx1g`, user `elastic` with
  `ELASTIC_PASSWORD`. The Python client is elasticsearch-py 8.19.3 through REST compatibility (`docs/VERSIONS.md`).
- **PostgreSQL 18.6** with `psycopg[binary]` 3.3.6; SQLAlchemy 2.1.3 ORM; Alembic 1.20.0 migrations run as a separate
  one-shot step so the API never changes the schema.
- **FastAPI 0.142.2** validates `ChatRequest`; **uvicorn 0.54.0** runs `app.main:app` on port 8000.
- **sentence-transformers 5.6.1** with torch 2.14.1 from the CPU-only index keeps CUDA out of the image (final image
  626,209,896 bytes, `docs/evidence/phase3-rag.txt` section 1). `OMP_NUM_THREADS` is 4 for the API ("leave CPU for
  llama.cpp") and 8 for ingest.
- **pypdf 6.19.0** puts a form feed between pages, which the splitter counts for `page_number`.
- **DeepEval 4.2.8** is only in the `eval` dependency group, so it is not in the production image. Telemetry off with
  `DEEPEVAL_TELEMETRY_OPT_OUT=1`.
- **OpenTelemetry SDK 1.45.1** and **prometheus-client 0.26.0** instrument rag-api ([6.8](#68-tracing-metrics-and-logs)).

### 3.5 Observability

**In plain English.** A tally clerk (Prometheus) reads counters every 15 seconds. A wall of screens (Grafana) draws
charts. A filing cabinet (Tempo) keeps request journeys; a post room (OTel Collector) delivers them. A diary box
(Loki) keeps logs; a courier (Alloy) copies container diaries into it. An optional AI-specific viewer (Langfuse) needs
its own back office (ClickHouse, Redis, MinIO, Postgres).

**Technically.**
- **Prometheus v3.15.0**: `scrape_interval` and `evaluation_interval` 15 s; retention 31 days so a 30-day SLO window
  fits; `--web.enable-lifecycle` for reloads. promtool from the same image checks config, unit-tests rules and
  backfills GPU samples. No Alertmanager.
- **Grafana 13.2.3**: provisioned datasources (Prometheus default, Tempo, Loki with a `TraceID` derived field) and
  three dashboards (`rag.json`, `llamacpp.json`, `gpu.json`), `allowUiUpdates: false`. Docker Desktop on Windows did
  not deliver file change events, so new dashboards needed a restart (`docs/evidence/phase4-observability.txt`).
- **Tempo 3.1.0** single binary, OTLP 4317/4318 inside the network, HTTP 3200; distroless, so no compose
  healthcheck; retention not set in `tempo.yaml` (default applies, value not stated in the repo).
- **OTel Collector contrib 0.162.0**: OTLP receiver, `batch` processor, `otlp/tempo` exporter, own metrics on 8888. A
  second config file adds a Langfuse pipeline only when `LANGFUSE_OTEL_ENDPOINT` is set ([6.8](#68-tracing-metrics-and-logs)).
- **Loki 3.7.8** single binary, TSDB schema v13, 24 h index periods, retention 168 h enforced by the compactor.
- **Alloy v1.20.1** discovers containers with label `com.docker.compose.project=industrial-llm-platform`. The Docker
  socket is mounted `:ro`, but read-only does not restrict API calls, so Alloy has root-equivalent access to the
  engine (`docs/LIMITATIONS.md`).
- **Langfuse 4.56.0** web and worker need `HOSTNAME=0.0.0.0` (Docker sets `$HOSTNAME` to the container id); v4 runs in
  "events_only" mode, so read spans with `/api/public/v2/observations` (`observability/VERSIONS.md`). About 2.5 GiB
  RAM for its 6 containers.

### 3.6 Platform

**In plain English.** Shipping containers (Docker), a seating plan (Compose), a building manager (Kubernetes) in a
doll's house (kind) or a strict landlord's building (OpenShift/OKD in a VM), a warehouse next door (registry), a
library desk for blueprints (git daemon), form letters (Helm), a spell checker (kubeconform), a foreman who follows
the blueprint (ArgoCD), a host who calls in more cooks (KEDA), a receptionist (Traefik) and a forklift (skopeo).

**Technically.**
- **Docker** builds `rag/Dockerfile` in three stages with `--provenance=false` (the default attestation has a
  timestamp, so every build got a new digest: `07c50a6f...` vs `5fa06ab9...`, `docs/evidence/phase5a-platform.txt`
  section 8). No SBOM is attached.
- **kind** uses one node because the model PVC is ReadWriteOnce and shared by all llama.cpp replicas;
  `extraPortMappings` maps `127.0.0.1:8090` to Traefik NodePort 30080.
- **registry 3.1.2** runs as `kind-registry` on `127.0.0.1:5001`; containerd in the node maps `localhost:5001` to
  `http://kind-registry:5000`.
- **git daemon** (`kind-git`) runs as UID 65534 with the bare repo mounted read-only; ArgoCD reads
  `git://kind-git/industrial-llm-platform`, branch `main`.
- **Helm 4.2.4**: three charts (`platform-deps`, `llamacpp`, `rag-api`) plus `deploy/argocd/apps`; each has
  `values.yaml` (safe for OpenShift), `values-kind.yaml`, `values-openshift.yaml`.
- **kubeconform v0.8.0** with schemas pinned by commit (Kubernetes, CRDs, OpenShift 4.20 Route).
- **ArgoCD**: automated prune and self-heal; child apps retry 10 times with backoff 10 s, factor 2, max 3 min; a Lua
  health check makes the parent wait for child health so sync waves work.
- **KEDA 2.21.0** creates HPA `keda-hpa-llamacpp`; `image.pullPolicy=IfNotPresent` because `Always` would re-contact
  ghcr.io at every pod start.
- **Traefik 41.6.1** as NodePort, with `--providers.kubernetesingress.ingressendpoint.ip=127.0.0.1` ("without it
  ArgoCD keeps them Progressing").
- **OKD 4.22 via crc 2.64.0**: preset `okd`, 6 CPUs, 16384 MiB, 60 GB disk, `host-network-access=true`. Install needs
  admin once (MSI, Hyper-V groups, reboot).
- **skopeo v1.22.3** copies the image to the OKD registry route; the token is passed as an environment variable, not
  on a command line.

### 3.7 CI and hygiene

**In plain English.** Two foremen run the same checklist (Jenkins on the laptop, GitHub in the cloud). Inspectors
check for known defects (Trivy) and leaked passwords (gitleaks). Grammar checkers check each language. A doorman
(pre-commit) runs the quick checks before anything enters the history.

**Technically.**
- **Jenkins 2.580.1 LTS** (image pinned by digest) plus Docker CLI and buildx; JCasC defines one admin, no anonymous
  read, 1 executor, URL `http://127.0.0.1:8088/`, and one Job DSL pipeline that clones `file:///repo` `*/main`.
  63 plugins pinned in `ci/jenkins/plugins.yaml` (YAML because "gitleaks reads `<name>-api:<version>` lines as API
  keys").
- **GitHub Actions**: `permissions: contents: read`; `concurrency` cancels older runs; `actions/checkout` pinned by SHA
  with `persist-credentials: false`; jobs `lint` and `test` in parallel, then `image`.
- **Trivy 0.75.0**: `--scanners vuln --severity CRITICAL --exit-code 1`; `.trivyignore` path never exercised.
- **gitleaks v8.30.1**: `.gitleaks.toml` keeps default rules plus one value-only allowlist for public Hugging Face
  revision SHAs in vLLM logs (`regexTarget = "match"`).
- **hadolint v2.15.1** (pre-commit local hook and CI), **actionlint 1.7.12** (runs shellcheck on `run:` steps),
  **yamllint 1.38.0** (relaxed, line length 160, Helm templates excluded), **ruff 0.16.10** (rules E, F, W, I, B, UP;
  explicit "so a ruff upgrade cannot silently change what is enforced").
- **pre-commit 4.6.2** via `uvx`: a global `exclude` keeps every hook off byte-exact evidence folders; in CI
  `SKIP=gitleaks,hadolint` because pre-commit "cannot mount the workspace from inside a container", and they run
  directly instead.

---

## 4. Models

**In plain English.** Seven specialist workers. A small writer (Qwen2.5-1.5B) in three outfits: a compact one for the
laptop (GGUF Q4_K_M), a full-detail one for the GPU (FP16) and a compressed one for the GPU (AWQ). A bigger writer
(Qwen2.5-7B) in two outfits: compressed for the GPU (AWQ) and compact for the laptop, where it works as the exam marker
(judge). Two small clerks for search: one gives texts "postcodes" (bge-small), one reads question and passage side by
side (MiniLM reranker). And a picture spotter (YOLOv8n).

**Technically.** Revisions, file sizes and sha256 values come from `docs/VERSIONS.md`, `docker-compose.yml`,
`rag/Dockerfile`, `inference/README.md` and `notebooks/build_*.py`. Parameter counts and licences not recorded in the
repo were fetched from the Hugging Face model API on 2026-10-10 (field `safetensors.total` and `cardData.license`);
each such value names its URL. For AWQ repos the API's `safetensors.total` counts packed INT32 storage elements, not
real parameters, so the base model's count is the meaningful one.

### 4.0 Summary table

In plain English: one row per model, with its size, licence, source and memory in one place.

| Model | Role | Params | Licence | Pinned source | Engine | Memory evidence |
|-------|------|--------|---------|---------------|--------|-----------------|
| Qwen2.5-1.5B-Instruct GGUF Q4_K_M | CPU serving, CPU benchmark, RAG generator | 1,543,714,304 (base, HF API) | Apache-2.0 | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` @ `91cad511...`, 1,117,320,736 bytes, sha256 `6a1a2eb6...` | llama.cpp | container 2.885 GiB in one `docker stats` snapshot |
| Qwen2.5-1.5B-Instruct FP16 | GPU benchmark, GSM8K, TensorRT-LLM comparison (v3i) | same | Apache-2.0 | `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa798...` | vLLM, TensorRT-LLM | "Model loading took 2.98 GiB" |
| Qwen2.5-1.5B-Instruct-AWQ | GPU benchmark, GSM8K | same base | Apache-2.0 | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` @ `3ecffa0c...` | vLLM (Marlin) | "Model loading took 1.1 GiB" |
| Qwen2.5-7B-Instruct-AWQ | one GPU benchmark cell (c8) | 7,615,616,512 (base, HF API) | Apache-2.0 | `Qwen/Qwen2.5-7B-Instruct-AWQ` @ `b2503754...` | vLLM (Marlin) | "Model loading took 5.29 GiB" |
| Qwen2.5-7B-Instruct GGUF Q4_K_M | RAG eval judge | same base | Apache-2.0 | `Qwen/Qwen2.5-7B-Instruct-GGUF` @ `bb5d59e0...`, 2 shards 3,993,201,344 + 689,872,288 bytes | llama.cpp | "stopping the judge frees about 5 GB RAM" |
| bge-small-en-v1.5 | embeddings | 33,360,512 (HF API) | MIT | `BAAI/bge-small-en-v1.5` @ `5c38ec7c...` | sentence-transformers | inside rag-api 503.2 MiB snapshot |
| ms-marco-MiniLM-L6-v2 | reranker | 22,714,113 (HF API) | Apache-2.0 | `cross-encoder/ms-marco-MiniLM-L6-v2` @ `233902d2...` | sentence-transformers | same |
| YOLOv8n | vision detection | 3,151,904 | AGPL-3.0 | `ultralytics/assets` v8.4.0 `yolov8n.pt`, sha256 `f59b3d83...` | ONNX Runtime, TensorRT, PyTriton | ONNX initializers 12,607,848 bytes |

Hugging Face API URLs used (fetched 2026-10-10):
[Qwen2.5-1.5B-Instruct](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct),
[Qwen2.5-7B-Instruct](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct),
[Qwen2.5-1.5B-Instruct-AWQ](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct-AWQ),
[Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct-AWQ),
[Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF),
[Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct-GGUF),
[bge-small-en-v1.5](https://huggingface.co/api/models/BAAI/bge-small-en-v1.5),
[ms-marco-MiniLM-L6-v2](https://huggingface.co/api/models/cross-encoder/ms-marco-MiniLM-L6-v2). All six Qwen repos
report `apache-2.0`; bge reports `mit`; MiniLM reports `apache-2.0`. These are the current API values for the repos,
not values checked at the pinned revisions. bge and MiniLM both have `hidden_size` 384 in their `config.json` on the
main branch at fetch time (bge 12 layers, MiniLM 6 layers).

### 4.1 Qwen2.5-1.5B-Instruct, GGUF Q4_K_M (CPU)

**In plain English.** The small chat model squeezed into one compact file so a laptop processor can run it, like a
high-quality photo saved as a smaller JPEG.

**Technically.**
- **Quantisation:** pre-quantised by Qwen (the repo downloads Qwen's own GGUF, it does not quantise). Q4_K_M is nominally
  4-bit; it averages about 4.8 bits per weight in a mixed block scheme (some tensors at Q6_K).
- **Download:** compose service `model-init` (`curlimages/curl:8.22.0`) skips if the file exists; otherwise
  `curl -fL --retry 3` to a `.part` file, `sha256sum -c`, then rename. Volume `models`. In Kubernetes the init container
  resumes downloads (`-C -`) and re-checks sha256 on every start.
- **Loading and serving:** `llamacpp` service, image `server-b11459`, starts only after `model-init` succeeded, mounts
  the volume read-only, runs `--host 0.0.0.0 --port 8080 --metrics -m /models/qwen2.5-1.5b-instruct-q4_k_m.gguf
  --ctx-size 4096 --parallel 2 --threads 4`. Port bound to 127.0.0.1.
- **Memory:** 2 slots of 2048 tokens (server log `n_slots = 2, n_ctx_slot = 2048`, `docs/evidence/phase2-cpu.txt`).
  Container at 2.885 GiB in one snapshot and 1.705 GiB in another (`phase2-cpu.txt`, `phase2-cpu-rerun.txt`); not
  broken into weights, KV cache and prompt cache. The prompt cache may use up to 8192 MiB host RAM by default. The Helm
  chart requests 1536Mi, limits 2Gi, comment "~1.5 GB RAM per replica".

### 4.2 Qwen2.5-1.5B-Instruct, FP16 (GPU)

**In plain English.** The same small model in full detail, 16 bits per number; needs a GPU.

**Technically.**
- **Stored precision:** the checkpoint is bf16; vLLM logs "Casting torch.bfloat16 to torch.float16" because the T4
  has no bf16 (`docs/evidence/phase2-gpu.txt`).
- **Download:** vLLM fetches it at server start, pinned by `--revision` and `--tokenizer-revision`. The TensorRT-LLM
  notebook uses `huggingface_hub.snapshot_download(repo, revision=rev)`.
- **Memory:** "Model loading took 2.98 GiB"; "Available KV cache memory: 9.08 GiB"; "GPU KV cache size: 340,064 tokens"
  (`benchmark/results/raw/kaggle-20261009/results/logs/server-vllm-t4-1.5b-fp16.log`), enough for 83 requests of 4096
  tokens at once against the 32 used (`docs/evidence/phase2-gpu.txt`). With TP=2 each card held 1.5 GiB of weights
  and the KV cache totalled 774,272 tokens (`server-vllm-2xt4-1.5b-fp16-tp2.log`).
- **TensorRT-LLM:** converted checkpoint `rank0.safetensors` 3.59 GB, 230 tensors, 28 layers, float16
  (`inference/README.md`, v3f section). The converter fails on tied embeddings, so `build_engine.py` unties them first
  ([5.3](#53-inference)). In run v3i the bench server log shows "loaded engine 3410 MiB" and a 7.58 GiB paged KV cache
  (283,776 tokens) (`docs/evidence/v3-trtllm.txt`).

### 4.3 Qwen2.5-1.5B-Instruct-AWQ (GPU, INT4)

**In plain English.** The small model compressed to about 4 bits by Qwen with a method that protects the important
weights. Smaller file, less memory traffic, usually faster.

**Technically.** Pre-quantised by Qwen; no `--quantization` flag, vLLM detects AWQ and logs "Using MarlinLinearKernel
for AutoAWQMarlinLinearMethod". FP16 activations. "Model loading took 1.1 GiB"; KV cache 10.98 GiB, 411,376 tokens
(`server-vllm-t4-1.5b-awq.log`). Accuracy cost: [7.6](#76-gsm8k-accuracy).

### 4.4 Qwen2.5-7B-Instruct-AWQ (GPU, INT4)

**In plain English.** The bigger sibling, compressed so it fits on one T4. Used for one benchmark cell.

**Technically.** "7B only as AWQ" because a T4 has 15 GB (`inference/README.md`). "Model loading took 5.29 GiB"; KV
cache 6.07 GiB, 113,744 tokens, enough for 27.8 requests of 4096 tokens (`server-vllm-t4-7b-awq.log`,
`docs/evidence/phase2-gpu.txt`). Its `generation_config.json` brought repetition_penalty 1.05.

### 4.5 Qwen2.5-7B-Instruct GGUF Q4_K_M (judge)

**In plain English.** The exam marker. Bigger than the writer because marking needs more reasoning.

**Technically.** Two GGUF shards, downloaded by `judge-init` (compose profile `eval`) with `curl -fL --retry 10
--retry-all-errors -C -` and a sha256 per shard. Served by `llamacpp-judge` with `--ctx-size 8192 --parallel 1
--threads 8` on host port 8081; `-m` points at shard `00001-of-00002` and llama.cpp reads the second itself. Called by
`LocalJudge` in `rag/eval/run_eval.py` at temperature 0 with `response_format` `json_schema`. First run downloads 4.7
GB; stopping it frees about 5 GB RAM (`rag/eval/README.md`). Licence: the GGUF repo reports Apache-2.0 in the Hugging
Face API; `docs/VERSIONS.md` does not list it.

### 4.6 bge-small-en-v1.5 (embeddings)

**In plain English.** Gives every paragraph a postcode based on meaning, so paragraphs about the same thing sit close
together even with different words.

**Technically.** Used as `SentenceTransformersDocumentEmbedder(batch_size=64)` at ingest and
`SentenceTransformersTextEmbedder(prefix=QUERY_PREFIX)` at query time; the prefix is "Represent this sentence for
searching relevant passages: " (bge v1.5's recommended retrieval instruction, documents embedded without it). Baked
into the image at build time by `snapshot_download` (only `*.json`, `*.txt`, `model.safetensors`, `1_Pooling/*`) to
`/opt/models/bge-small-en-v1.5`; `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` at runtime. Loaded once by
`warm_up()` at API start. Memory not measured alone; the whole `rag-api` container was 503.2 MiB in one snapshot
(`docs/evidence/phase2-cpu.txt`); the chart requests 1Gi and limits 3Gi.

### 4.7 ms-marco-MiniLM-L6-v2 (reranker)

**In plain English.** The careful clerk who reads the question and each shortlisted passage side by side and reorders
them. Slower per passage, so it only reads the shortlist.

**Technically.** `SentenceTransformersSimilarityRanker(top_k=5)`. Trained on MS MARCO (name only; training details
are not in the repo). Baked into the image the same way as bge. Cost on a cold request: `rag.rerank` 14924.1 ms
(`docs/evidence/phase3-rag.txt` section 5).

### 4.8 YOLOv8n (vision)

**In plain English.** A tiny fast object detector: give it a photo and it draws boxes around people, cars and dogs
and names them. "n" means nano.

**Technically.**
- **Library and licence:** Ultralytics 8.4.174, AGPL-3.0 for code and weights (`inference/README.md`).
- **Weights:** `yolov8n.pt` from the `ultralytics/assets` v8.4.0 release, sha256 `f59b3d83...` (`notebooks/build_v3.py`,
  `docs/VERSIONS.md`); `fetch()` raises on a mismatch.
- **Size:** "72 layers, 3,151,904 parameters, 0 gradients, 8.7 GFLOPs" (fused), printed by the validator in
  `benchmark/results/raw/kaggle-20261010-v3c/results/FAILED-map-pt-fp32.txt`.
- **I/O:** input `images` [1, 3, 640, 640] float; output `output0` [1, 84, 8400] float (b1 engine metadata in the v3c
  raw JSON).
- **Conversion:** ONNX FP32 opset 17 (ORT CUDA lacks some opset 19+ kernels), dynamic batch, simplified with onnxslim,
  exported on CPU. TensorRT FP16 engines per static batch 1, 8, 32, built on the T4 they run on; inputs and outputs
  stay FP32. `exports.json` records every sha256.
- **Memory:** ONNX initializers 12,607,848 bytes (`benchmark/results/raw/onnx-compare-20261010/onnx-compare.json`).
  GPU memory during vision runs is in the raw `gpu_samples` but not summarised in RESULTS.md.

---

## 5. File-by-file code walkthrough

**In plain English.** A walk through the building, room by room, in the order the directories appear: what each file
is for and the settings that matter.

**Technically.** Ordered: root files, `rag/`, `inference/`, `notebooks/`, `benchmark/`, `observability/`, `deploy/`,
`infra/`, `scripts/`, `ci/`, `.github/`, `docs/`. Methods behind the code are in section 6; this section is about what
the code does and its key values.

### 5.1 Root files

**In plain English.** The front office: the seating plan for the laptop stack, the blank password form, and the
house rules for tidy commits.

**Technically.**

**`docker-compose.yml`.** The laptop stack. Default services and profiles:

| Service | Image | Host port (127.0.0.1) | Health check | Profile |
|---|---|---|---|---|
| model-init | `curlimages/curl:8.22.0` | none | one-shot | default |
| llamacpp | `ghcr.io/ggml-org/llama.cpp:server-b11459` | 8080 | `curl /health` | default |
| postgres | `postgres:18.6-alpine3.24` | 5432 | `pg_isready` | default |
| elasticsearch | `elasticsearch:9.5.5` | 9200 | cluster health yellow | default |
| prometheus | `prom/prometheus:v3.15.0` | 9090 | `/-/ready` | default |
| grafana | `grafana/grafana:13.2.3` | 3000 | `/api/health` | default |
| tempo | `grafana/tempo:3.1.0` | 3200 | none (distroless) | default |
| otel-collector | `otel/opentelemetry-collector-contrib:0.162.0` | 4317, 4318, 8888 | none (distroless) | default |
| loki | `grafana/loki:3.7.8` | 3100 | none (distroless) | default |
| alloy | `grafana/alloy:v1.20.1` | 12345 | bash `/dev/tcp` probe | default |
| rag-migrate | `industrial-llm-platform/rag:0.1.0` (built from `rag/`) | none | one-shot `alembic upgrade head` | default |
| rag-api | same | 8000 | Python `urlopen /healthz`, start 60 s | default |
| rag-ingest | same | none | one-shot | default |
| judge-init, llamacpp-judge | curl, llama.cpp | 8081 | `/health` | `eval` |
| langfuse-web, worker, postgres, clickhouse, minio, redis | pinned, MinIO by digest | 3001 (web) | each has one | `langfuse` |

Key points: `depends_on` conditions order startup (`service_completed_successfully` for model-init and rag-migrate,
`service_healthy` for postgres, elasticsearch and llamacpp); `${POSTGRES_PASSWORD:?...}` refuses to start without a
value; YAML anchor `&rag-env` shares `DATABASE_URL`, `ELASTIC_PASSWORD`, `ES_URL`, `LLM_BASE_URL`,
`OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`, `OTEL_SERVICE_NAME=rag-api`, `OMP_NUM_THREADS=4`;
`model-init` runs as `user: "0"` because a fresh named volume is root-owned; Prometheus runs with
`--storage.tsdb.retention.time=31d`. The collector's second `--config` uses
`config${LANGFUSE_OTEL_ENDPOINT:+-langfuse}.yaml` as a switch ([6.8](#68-tracing-metrics-and-logs)).

**`.env.example`.** Three dev placeholders (`POSTGRES_PASSWORD`, `ELASTIC_PASSWORD`, `GRAFANA_ADMIN_PASSWORD`) and
empty Langfuse keys with the `openssl rand -hex` commands to fill them. `.env` is git-ignored.

**Hygiene files.**

| File | Key content |
|---|---|
| `.gitattributes` | `* text=auto eol=lf`; byte-exact evidence folders marked `-text` |
| `.gitignore` | `.env`, `*.gguf`, `.venv/`, `.tools/`, caches, exported ONNX and engines |
| `.pre-commit-config.yaml` | hooks in [3.7](#37-ci-and-hygiene); global `exclude` for evidence folders |
| `.gitleaks.toml` | default rules plus one value-only allowlist |
| `.yamllint` | extends `relaxed`, line length 160 |
| `ruff.toml` | Python 3.12 target, explicit rule set |
| `Makefile` | `up`, `down`, `smoke`, `lint`; untested (no make on the dev machine) |
| `CLAUDE.md` | project rules: free and open source only, no fabricated results, pin versions, no dashes in prose |

### 5.2 rag/

**In plain English.** The library and its front desk: the search and answer logic, the web counter, the conversation
filing cabinet, the delivery-day script that builds the catalogue, the reading list with licence receipts, the exam,
and the tests.

**Technically.**

| Path | Role |
|------|------|
| `rag/app/rag.py` | Haystack pipelines, prompt, citations |
| `rag/app/main.py` | FastAPI: `/chat`, `/sessions/{id}/messages`, `/healthz`, `/metrics`; tracing and metrics |
| `rag/app/db.py` | SQLAlchemy models and queries |
| `rag/app/ingest.py` | downloads PDFs, verifies sha256, rebuilds the index |
| `rag/migrations/` | Alembic revision `0001` |
| `rag/data/sources.json`, `SOURCES.md` | the 9 documents, URLs, sha256, licence evidence |
| `rag/eval/` | 30 questions, eval and judge scripts, results |
| `rag/tests/` | chunking and API tests |
| `rag/Dockerfile`, `pyproject.toml`, `uv.lock` | image and pinned packages |

**`rag/Dockerfile`.** Three stages: `ghcr.io/astral-sh/uv:0.11.32` (only for the uv binary); build stage
`python:3.12.15-slim-trixie` with `UV_COMPILE_BYTECODE=1`, `UV_LINK_MODE=copy`, `UV_PYTHON_DOWNLOADS=never`, then
`uv sync --frozen --no-default-groups --no-cache` and `snapshot_download` of both small models at pinned revisions;
final stage copies `.venv`, `/opt/models`, `alembic.ini`, `migrations/`, `data/sources.json`, `app/`. User 1001,
group 0; `/data/corpus` owned `1001:0`, mode `2775` (group-writable, setgid) so any UID in group 0 can write it, the
OpenShift pattern. `HOME=/tmp` so caches work with a read-only root filesystem. Checked with `docker run --user
123456:0` and `--read-only --tmpfs /tmp` (`docs/evidence/phase5a-platform.txt` section 1). The base moved from 3.12.13
to 3.12.15 to clear 3 CRITICAL perl-base CVEs (`docs/VERSIONS.md`). One image serves API, migration and ingest.

**`rag/app/rag.py`.** Module constants:

| Constant | Value | Meaning |
|----------|-------|---------|
| `ES_URL`, `ES_INDEX` | `http://elasticsearch:9200`, `corpus` | store address and index |
| `EMBED_MODEL`, `RERANK_MODEL` | bge-small-en-v1.5, ms-marco-MiniLM-L6-v2 (image sets local paths) | models |
| `LLM_BASE_URL`, `LLM_MODEL` | `http://llamacpp:8080/v1`, `qwen2.5-1.5b-instruct-q4_k_m` | generator |
| `MAX_TOKENS` | 384 | answer cap |
| `RETRIEVE_K`, `TOP_K` | 10, 5 | per retriever, after rerank |
| `QUERY_PREFIX` | "Represent this sentence for searching relevant passages: " | bge query instruction |
| `FOOTER` | regex for the NIST "available free of charge" footer | noise removed |

Functions: `document_store()` (Elasticsearch with basic auth); `cleaner()` (`DocumentCleaner(remove_regex=FOOTER)`);
`splitter()` (`split_by="word", split_length=150, split_overlap=30`, with a `ponytail:` comment naming
"sentence-aware splitting if answers get cut mid-thought" as the upgrade path and a note that each llama.cpp slot has
2048 tokens); `chunker()` (clean plus split, unit-testable without models); `indexing_pipeline()` (PDF to document,
clean, split, embed with batch 64, write with `DuplicatePolicy.OVERWRITE`); `retrieval_pipeline()` (embed query, BM25
and dense retrieve, RRF join, rerank top 5); `generation_pipeline()` (`OpenAIChatGenerator`, `max_tokens 384`,
`temperature 0`, `stream_options.include_usage`, timeout 300 s, dummy key "not-needed"); `source_label()`;
`build_messages()`; `cited()` (regex `\[(\d+)\]`, keeps 1 <= n <= number of docs, deduplicated and sorted, so a
made-up `[9]` is dropped). Class `Rag` builds the store and pipelines once and calls `warm_up()`.

The prompt: system message `SYSTEM_PROMPT` with numbered sources `[i] <title>, page <n>:` then the chunk; then
history; then the question plus `CITE_REMINDER` ("Answer in one to three sentences. End each sentence with the source
number(s) it came from, like [2]."). Code comment: "The 1.5B model drops citations unless the reminder sits next to
the question."

**`rag/app/main.py`.** `ChatRequest`: `message` 1 to 4000 characters, optional `session_id` UUID, `user` 1 to 200
characters, default "anonymous". `HISTORY_TURNS = int(os.environ.get("HISTORY_MESSAGES", "4"))` counts messages, not
turns. `chat()` starts `run_chat` in a daemon thread and returns `StreamingResponse(iter(out.get, None))`, so blocking
Haystack and database calls run off the event loop. SSE events in order: `session`, `sources`, `token`, `done`, or
`error` if anything fails after the stream started ("stream already started, so report in-band").

Metrics:

| Name | Type | Labels | Note |
|------|------|--------|------|
| `rag_http_requests_total` | Counter | method, path (route template or `unmatched`), status class | `/chat` series pre-created at 0 |
| `rag_http_request_duration_seconds` | Histogram | method, path | time to response headers, so little use for SSE |
| `rag_stage_duration_seconds` | Histogram | stage = retrieval, generation, ttft, total | buckets 0.05 to 120 s |
| `rag_llm_tokens_total` | Counter | kind = prompt, completion | from llama.cpp usage |
| `rag_chat_errors_total` | Counter | none | in-stream failures |

The middleware sets status 500 before calling the handler and records in `finally`, so unhandled exceptions count as
5xx. TTFT is measured from the start of `run_chat`, so it includes database work and retrieval. Tracing: FastAPI's
built-in telemetry (`auto_configure False`, metrics and logs off, `/healthz` and `/metrics` excluded), `StepTracer`
renames Haystack spans to `rag.<component>`, manual spans `rag.prompt_build` and `rag.db_write`. The `trace_id` line is
logged through `uvicorn.error` because that logger prints INFO here.

**`rag/app/db.py` and `rag/migrations/versions/0001_chat_history.py`.**

| Table | Columns | Constraints |
|-------|---------|-------------|
| `users` | `id` UUID PK, `name` varchar(200) unique, `created_at` | |
| `sessions` | `id` UUID PK, `user_id` FK, `created_at` | indexed |
| `messages` | `id` integer PK, `session_id` FK ON DELETE CASCADE, `role` varchar(16), `content` text, `citations` JSONB, `created_at` | `role IN ('user', 'assistant')` |

`get_or_create_session()` creates a session with the client's `session_id` if it does not exist (no access control).
`history()` returns the newest `limit` messages, oldest first. The user message is stored before generation, so a
failed generation leaves a question without an answer in the history.

**`rag/app/ingest.py`.** `fetch()` reuses `<id>.pdf` if its sha256 matches; otherwise up to 4 attempts with urllib,
timeout 120 s, sleeping 5, 10, 15, 20 s ("NTRS sometimes answers 503"), writes `.part`, renames only after a hash
match. `main()` prints `corpus ok: N files, sha256 verified`, deletes index `corpus` ("full rebuild keeps the index
consistent"), runs the indexing pipeline with metadata key `doc_id` (not `source_id`, because the splitter overwrites
`source_id`). Evidence: `indexed 9 documents, 3129 chunks in 1115s` (`docs/evidence/phase3-rag.txt` section 3).

**`rag/data/sources.json`, `SOURCES.md`.** 9 public domain US government documents, 1,115 pages, about 27 MB; PDFs
not committed. Pages: NASA SP-2016-6105 Rev2 356, NIST SP 800-82r3 316, TN 1900 105, SP 811 90, SP 1500-201 79,
AMS 100-37 60, AMS 300-1 47, AMS 200-9 37, TN 1297 25. Rejected: SP 330 (BIPM copyright), AMS 100-1 (contractor),
AMS 100-8 and NISTIR 8183r1 (non-federal co-authors), AMS 300-8 (withdrawn); TN 2148 dropped 2026-10-08.

**`rag/eval/`.** `questions.jsonl` (30 questions with `reference`, `source_id`, `page`); `run_eval.py` (asks via the
real API, scores recall, optionally judges, writes `raw.jsonl`, `config.json`, `summary.md`); `judge_run.py` (judges
chosen ids of an existing run by re-asking only to fetch chunk text, checking the retrieval is equal, and judging the
stored answer; `--summarize` with `--contradicted id:metric` flags). Method in [6.7](#67-rag-evaluation).

**`rag/tests/`.** `test_chunking.py` (chunks at most 150 words, 30-word overlap, page numbers kept, footer and blank
lines removed); `test_api.py` (skipped unless `TEST_PG_URL`; throwaway database, `alembic upgrade head`, fake RAG;
checks health, SSE order, citations, history, roles, 404, 5xx counting, metric names). Evidence: `7 passed in
30.58s` (`docs/evidence/phase3-rag.txt` section 7).

### 5.3 inference/

**In plain English.** The notes and adapters for each serving kitchen: why each version was chosen, the TensorRT-LLM
prep and serving wrapper, and the vision export and Triton scripts.

**Technically.**

**`inference/README.md`.** Serving paths, T4 limits table, version reasons (vLLM 0.31.0 `+cu129`, TensorRT-LLM 0.14.0,
TensorRT 10.16, ORT 1.26.0, DCGM), Kaggle commands, and the v3 notebooks. Parts of its text lag the results
([8.2](#82-docs-that-lag-the-results)).

**`inference/trtllm/build_engine.py`.** Three stages:
1. `untie(src, dst)`: TensorRT-LLM 0.14's Qwen converter fails on tied embeddings (issue 2388), so it writes a copy with
   `lm_head.weight` cloned from `model.embed_tokens.weight` and `tie_word_embeddings: false`.
2. `convert(untied, ckpt)`: `QWenForCausalLM.from_hugging_face(..., dtype="float16")` then `save_checkpoint`; it
   creates the output directory first (0.14 does not, which failed run v3f) and deletes the model inside the function
   so its `__del__` runs while torch is alive.
3. `build(ckpt, engine, cmd_out, profile)`: `trtllm-build` with `--gemm_plugin float16 --gpt_attention_plugin float16
   --context_fmha disable --remove_input_padding enable --kv_cache_type paged --max_num_tokens 8192 --workers 1` plus
   the profile sizes; appends the command and its computed workspace to `server_args/trtllm-build.txt`.

`PROFILES = {"bench": (32, 1024, 1024 + 128), "gsm8k": (8, 2048, 2048 + 256)}` as (max_batch_size, max_input_len,
max_seq_len). `workspace_bytes()` reproduces 0.14's context workspace formula; for (32, 3072, 8192) it returns
21,978,227,456 bytes, "exactly what the v3h build reported", pinned by a unit test. Why the sizes: the longest loadgen
prompt is 893 tokens and the longest GSM8K 5-shot prompt is 1654 tokens. Story in [7.9](#79-the-memory-limit-story-tensorrt-llm).

**`inference/trtllm/serve.py`.** Loads 0.14.0's `examples/apps/openai_server.py` (sha256-checked) with:
1. `ignore_eos`: 0.14 drops it, so the wrapper sets `end_id = pad_id = -1` when true, as 0.14's `trtllm-bench` does.
2. `/metrics`: counters `trtllm_http_requests` and `trtllm_prompt_tokens` (0.14 exposes no engine stats).
3. Loads the prebuilt engine instead of building one on the fly.
4. Stream patch: upstream marks a stream finished on the first empty text diff, silencing the rest while usage still
   counts every token, which would undercount TPOT and ITL. The patch changes `else:` to `elif output.finish_reason is
   not None:`; `patched_server_source()` raises unless the target text appears exactly once.
5. Skips special tokens when detokenising, as vLLM does.
KV block reuse is off (`enable_block_reuse=False`), matching vLLM's prefix caching off.

**`inference/vision/export.py`.** ONNX FP32 opset 17 with dynamic batch and onnxslim; TensorRT FP16 engines for batch
1, 8 and 32 on the target T4; `exports.json` with every sha256.

**`inference/vision/triton_serve.py`.** PyTriton binding per engine with `max_batch_size` equal to its batch.
`use_own_libpython()` puts `sys.base_prefix/lib` first on `LD_LIBRARY_PATH`; without it the Python backend stub loaded
Kaggle's Ubuntu `libpython3.12` with uv Python's standard library and failed with "No module named '_contextvars'"
(run v3c). The fix was reproduced in `ubuntu:24.04` before the next run.

### 5.4 notebooks/

**In plain English.** Scripts that write the Kaggle notebooks for you, with the benchmark code pasted inside, plus
safety checks so a broken session fails fast instead of eating GPU time.

**Technically.**

**`notebooks/build_notebook.py`** (runs v1 and v2). Embeds each file in `EMBED` (`run.py`, `metrics.py`,
`prompts.jsonl`, `run_gsm8k.sh`, `gsm8k_pinned.yaml`, `benchmark/requirements.txt`, `notebooks/vllm-requirements.txt`)
as `{path: (sha256, text)}`; the embed cell asserts each sha256 on Kaggle. `GIT_COMMIT` gets `-dirty` if any embedded
file differs from HEAD. Formats with ruff 0.16.10. Cells: intro; `CONFIG` (uv 0.11.32, Python 3.12.13, vLLM 0.31.0,
model revisions, `REPEATS = 3`, `MAX_MODEL_LEN = 4096`, `GPU_MEM_UTIL = 0.90`, `GSM8K_LIMIT = 250`); embed;
`ENV_CELL` (saves `nvidia-smi -q`, `lscpu`, `free -m`, os-release, pip freeze; defines `sh()` with `timeout -k 60`);
`PREFLIGHT_CELL` (writes `FAILED-preflight.txt` and stops if there is no GPU or no internet: "Run v1 (2026-10-07) had
no GPU and no internet"); `INSTALL_CELL` (venvs `vllm` with `--torch-backend cu129`, and `bench`); `HELPERS_CELL`:
`start_server` (writes the exact command to `server_args/<cell>.txt`, waits for `/health` up to 1200 s),
`stop_server` (kills the process group, then waits until every GPU uses under 1 GiB), `loadgen`, `gsm8k` (limit 250,
concurrency 16), `run_cell` (any exception writes `FAILED-<cell>.txt` and moves on). Cells: 1.5B FP16 c1/c8/c32 +
GSM8K, 1.5B AWQ c1/c8/c32 + GSM8K, 7B AWQ c8, TP=2 c32. The summary cell saves `nvidia-smi -q` again as
`env/nvidia-smi-q-end.txt`.

**`notebooks/build_v3.py`** (runs v3a to v3h). Reuses those cells and adds `STEP_CELL`: `time_left()` =
`BUDGET_S - elapsed` (`BUDGET_S` 3000 for vision with push `-t 3600`; 10200 for TensorRT-LLM with `-t 10800`); each
`step(name, est_s, fn)` writes `NOT_RUN-<name>.txt` if it would not fit, `FAILED-<name>.txt` on exception, and
`env/timing-<name>.txt` always. `checked()` fails the build if a cell references an un-embedded file (added after
v3b). Notebook A (`kaggle-dcgm-vision/`): DCGM skip cell citing v3a, `VISION_CELL` steps (install 900 s, data 600 s,
export 1500 s, speed, Triton, then mAP for pt, onnx, trt at 400 s each); `RUN_SPEED = False` and `RUN_TRITON = False`
in the current build. Notebook B (`kaggle-trtllm/`): `CONFIG_B` (TensorRT-LLM 0.14.0, Python 3.10.20, `KV_FRACTION =
0.90`, `RUN_VLLM_CONTROL = True`), the vLLM control first "so a slow TensorRT-LLM step cannot squeeze it out of the
budget", then `install` 1800 s, `download` 600 s, `build-engine` 1800 s, `trtllm-t4-1.5b-fp16` 2700 s (bench engine,
loadgen c1/c8/c32), `gsm8k-trtllm-1.5b-fp16` 1200 s (gsm8k engine). A failed step makes every later step write
`NOT_RUN`.

**Lock and constraint files.** `vllm-requirements.in/.txt` (197 entries) plus `vllm-overrides.txt`;
`trtllm-requirements.in/.txt` (pins such as pyarrow 20.0.0 and pynvml 11.5.3 from runs v3d);
`trtllm-build-constraints.txt` (setuptools 80.10.2 for the mpi4py build after v3b);
`vision-requirements.in/.txt`. Each `kernel-metadata.json` sets `enable_gpu`, `enable_internet`, `machine_shape:
"NvidiaTeslaT4"`, `is_private: true`.

### 5.5 benchmark/

**In plain English.** The stopwatch robot, its rule book, the question sheet, the accountant who builds the tables,
the maths exam, the vision timers and markers, and the tests that keep them honest.

**Technically.**

**`benchmark/loadgen/run.py`.** Closed-loop streaming client for any OpenAI-compatible endpoint.

| Function | What it does |
|---|---|
| `one_request` | streaming request with `stream_options.include_usage`, `temperature 0`, `seed`, `max_tokens`, `ignore_eos`; records start, each content chunk time, end, usage; errors are stored, not raised ("a failed request is data") |
| `closed_loop` | `concurrency` workers share one iterator; each sends its next request when the last ends; prompts reused round-robin |
| `GpuSampler`, `DcgmSampler` | nvidia-smi every 500 ms; DCGM sampler subprocess |
| `docker_stats`, `scrape` | one `docker stats` snapshot; server `/metrics` before and after |
| `run_cell` | warmup, metrics snapshot, samplers, measured loop, samplers stop, snapshot |
| `main` | prompts sha256, environment, loops, one JSON per run `<label>-c<conc>-r<rep>-<UTC>.json` |

Defaults: measured requests `max(64, 4 x concurrency)` (64 at c1 to c8, 128 at c32); warmup `max(2, concurrency)`
taken from the end of the list "so they do not prime the first measured prompts"; `max_tokens` 128; seed 1234;
timeout 600 s; `--smoke` marks runs that the report skips.

**`benchmark/loadgen/metrics.py`.** TTFT = first token time minus start; ITL = every gap between chunks, pooled; TPOT =
(last minus first token time) / (output tokens minus 1); E2E = end minus start; out tok/s = successful output tokens /
wall time from first measured start to last measured end. `percentile` interpolates like numpy's default.
`stream_truncated()` flags a stream with fewer chunks than tokens and a tail gap over 0.25 s and over 10x the median
gap. `decode_checks()` counts requests not decoding `max_tokens` and truncated streams. `cv()` is stdev / mean.

**`benchmark/loadgen/dcgm_sampler.py`.** `DcgmReader` against `nv-hostengine` or embedded; `InitWrapped()` so
connection errors raise; writes `utc,gpu,field,tag,unit,value` rows every 0.5 s; "Errors propagate unchanged".

**`benchmark/loadgen/prompts.jsonl`, `make_prompts.py`.** 48 short industrial-maintenance questions and 16 long fake
maintenance logs (36 days of events on assets like "pump P-101"), seeded `random.Random(1234)`, shuffled; 64 lines;
sha256 starts `72fb01ab9477`. Longest prompt 893 tokens, shortest 34; 15485 prompt tokens in total per 64 requests on
both servers (`docs/evidence/phase2-prep.txt`, `phase2-gpu.txt`).

**`benchmark/report.py`.** Offline report builder:

| Piece | Role |
|---|---|
| `load_docs` | every raw JSON with a `schema` key, skipping smoke; cell name `<session>/<label>` so sessions never pool |
| `cache_share`, `regime` | llama.cpp only: cached / (cached + processed) prompt tokens; >= `WARM_SHARE = 0.5` is warm |
| `SUPERSEDED` | the 2026-10-08 CPU session stays listed but out of charts and headline |
| `flags` | fewer than 3 repeats; CV > 0.10 on out tok/s, TTFT p50 or E2E p95; config disagreement; SM clock p50 < 0.6 x run max; temperature >= 85 C; errors; decode mismatches; truncated streams; first repeat > 10% slow |
| `clock_reasons` | decodes the NVML bitmask (0x1 idle, 0x2 app clocks, 0x4 SW power cap, 0x8 HW slowdown, 0x10 sync boost, 0x20 SW thermal, 0x40 HW thermal, 0x80 HW power brake, 0x100 display clock) |
| `low_clock_notes` | cell note when SM clock p50 < 0.6 x median of other GPU cells |
| `session_notes` | server-arg differences across sessions, TensorRT-LLM status per session |
| `trtllm_comparison` | only same session, model and precision; prints "Not compared" on any like-for-like mismatch |
| `paired_delta` | AWQ minus FP16 per GSM8K item, 95% normal-approximation CI |
| `dcgm_section`, `vision_section` | DCGM table if present; vision speed, flags, provenance, mAP |
| `charts` | out tok/s, TTFT p50, E2E p95 vs concurrency on a log2 axis |

**`benchmark/accuracy/run_gsm8k.sh`, `tasks/gsm8k_pinned.yaml`.** `lm_eval run --model local-chat-completions ...
--tasks gsm8k_pinned --apply_chat_template --fewshot_as_multiturn --limit $LIMIT --seed 0,1234,1234,1234
--log_samples`; dataset `openai/gsm8k` main at `740312add8...`, 5-shot, greedy, stops at `Question:`, `</s>` or
`<|im_end|>`.

**`benchmark/vision/`.** `run_vision.py` (`letterbox()` to 640x640 with grey 114 once before timing; `OrtModel`,
`TrtModel` reading Ultralytics engine metadata and using `execute_async_v3`, `TritonModel` over HTTP; closed loop at
concurrency 1; 640 requests at b1, 80 at b8, 20 at b32); `eval_map.py` (validator with `imgsz 640`, `batch 1`,
`rect False`, `conf 0.001`, `iou 0.7`; `data_yaml_dict()` adds `train:` because 8.4.174 requires it);
`fetch_coco_subset.py` (stdlib, labels zip sha256 pinned, each image checked against `coco_val2017_first500.txt`);
`cpu_baseline.sh` (venv, 64 images, ORT CPU at batch 1, 8, 32, 3 repeats, `--threads ${THREADS:-0}`);
`compare_onnx.py` (canonical graph and initializer hashes; other fields compared one by one).

**`benchmark/tests/`.** `test_metrics.py`, `test_decode_checks.py` (including
`test_attention_workspace_formula_matches_the_failed_v3h_build`), `test_vision.py`. Last recorded run in the evidence: "26 passed"
(`docs/evidence/v3-vision-cpu.txt`); HEAD now has 29 test functions, and the reviewer of this guide ran them: 29 passed (observed during review, no raw log kept); the first GitHub Actions run had "benchmark 9 passed"
(`docs/evidence/v3-github-actions.txt`), before later tests were added.

**`benchmark/results/raw/`.** One folder per session (`cpu-20261009`, `cpu-vision-20261010`, `kaggle-20261007` to
`kaggle-20261010-v3h`, `onnx-compare-20261010`), the superseded 2026-10-08 CPU files at the top level, and `smoke/`.
Each Kaggle folder has a README stating what is valid and what failed.

### 5.6 observability/

**In plain English.** The control room: the ledger's settings and rules, the exams for the rules, the filing cabinets
for journeys and diaries, the courier's route, the wall screens, and the tool that replays old GPU readings.

**Technically.**

```
observability/
  VERSIONS.md                    pinned images and gotchas
  SLOs.md                        SLO definitions, rationale, tests (the runbook)
  prometheus/prometheus.yml      5 scrape jobs, rule files
  prometheus/rules/recording.yml 18 SLI and error-budget rules
  prometheus/rules/alerts.yml    5 alert rules
  prometheus/tests/slo_test.yml          34 test groups
  prometheus/tests/slo_windows_test.yml  3 test groups
  tempo/tempo.yaml  otel-collector/config.yaml, config-langfuse.yaml
  loki/loki.yaml    alloy/config.alloy
  grafana/provisioning/  grafana/dashboards/rag.json, llamacpp.json, gpu.json
  gpu-replay/convert.py, backfill.sh
```

Counts from `docs/evidence/phase4-observability.txt` sections 2 and 7.

- **`prometheus.yml`:** jobs prometheus, llamacpp, otel-collector, tempo, rag-api; each `job_name` becomes the `job`
  label; `up` per target feeds `RagApiDown`.
- **`rules/recording.yml`:** groups `slo-rag-availability` and `slo-rag-ttft`, each 8 ratio rules
  (`slo:sli_error:ratio_rate<W>` for 5m, 30m, 1h, 2h, 6h, 1d, 3d, 30d) plus one budget rule in the same group, so the
  budget reads this evaluation's ratio. PromQL in [6.9](#69-slos-and-burn-rate-alerts).
- **`rules/alerts.yml`:** `RagAvailabilityErrorBudgetBurnFast` (page), `...Slow` (ticket), `RagTTFTErrorBudgetBurnFast`,
  `...Slow`, `RagApiDown` (`up == 0` for 2m). Thresholds written as `burn rate * (1 - target)`; no `for:` on SLO alerts;
  every alert has `runbook: observability/SLOs.md`.
- **`tests/`:** scenario and per-window-pair tests, and an exact-value staircase test ([6.10](#610-testing-the-alert-rules)).
- **`tempo/tempo.yaml`:** HTTP 3200, OTLP receivers, local backend, WAL under `/var/tempo/wal`, usage reporting off.
- **`otel-collector/config.yaml`:** OTLP in, `batch`, `otlp/tempo` out (`tls.insecure: true`), own metrics on 8888;
  `debug` exporter defined but unused. `config-langfuse.yaml` adds `basicauth/langfuse` and `otlphttp/langfuse` in a
  second pipeline.
- **`loki/loki.yaml`:** auth off, single instance, in-memory ring, TSDB v13 from 2026-01-01, 24h periods, retention
  168h with the compactor; effective config showed `retention_period: 1w` (`docs/evidence/v3-loki-langfuse.txt`).
- **`alloy/config.alloy`:** `discovery.docker`, `discovery.relabel` (compose service to `service`, container name to
  `container`), `loki.source.docker`, `loki.write`.
- **`grafana/provisioning/`:** datasources Prometheus (default), Tempo, Loki with derived field `TraceID` (regex
  `trace_id=([0-9a-f]{32})`, URL `$${__value.raw}` so provisioning does not expand it); dashboards provider with
  `disableDeletion: true`, `allowUiUpdates: false`.
- **`dashboards/rag.json`** ("RAG API and SLOs"): rows SLOs (budget remaining, SLI 30d, burn rate with lines at 6 and
  14.4, firing alerts), Traffic and errors, Latency (TTFT and total p50/p95, stage means), Traces and quality (recent
  `rag.chat` traces, a static "offline eval run, not live" table), Logs (lines per service, rag-api logs without probe
  noise). Evidence counted 15 panels before the Logs row was added; the committed file now has 18 entries in `panels`,
  5 of them rows (derived: counted from `observability/grafana/dashboards/rag.json` for this guide).
- **`dashboards/llamacpp.json`:** throughput gauges, queue depth (`requests_processing`, `requests_deferred`; build
  b11459 exposes no KV cache usage metric), token rates, busy slots per decode, `up`.
- **`dashboards/gpu.json`** ("GPU replay: Kaggle T4 run 2026-10-09 (nvidia-smi, not live)"): fixed range 05:55Z to
  06:40Z, variables `cell` and `gpu`, panels for utilisation, memory, power with `max_over_time` and a 70 W line, SM
  clock, temperature, and a per-run median table.
- **`gpu-replay/convert.py`:** stdlib; reads `vllm-*/*.json` in the Kaggle raw folder, writes OpenMetrics with six
  gauges in base units (`gpu_replay_*`), labels `source`, `cell`, `concurrency`, `repeat`, `run`, `gpu`; exits if any
  sample falls outside its run window plus or minus 60 s (timezone safety); values written with `repr` (the first
  version's `:g` rounded to 6 digits); asserts `n_samples == n_rows * len(METRICS)`.
- **`gpu-replay/backfill.sh`:** refuses without the volume or a reachable Prometheus; idempotent check; `promtool tsdb
  create-blocks-from openmetrics` into the volume; restarts Prometheus. Output `out/` is git-ignored.
- **`SLOs.md`, `VERSIONS.md`:** the runbook and the pins with gotchas (Tempo 3.x config, distroless healthchecks,
  `llamacpp:` prefix, Alloy over Promtail, Langfuse `HOSTNAME`, events_only).

### 5.7 deploy/

**In plain English.** The blueprints for the cluster: a master list that orders three sub-blueprints, and the three
sub-blueprints themselves (support services, the model server, the chat API), each with kind and OpenShift variants.

**Technically.**

**`deploy/argocd/root.yaml` and `apps/`.** Application `root` in namespace `argocd`, source
`git://kind-git/industrial-llm-platform` `main`, path `deploy/argocd/apps`, Helm `valuesObject` `{repoURL,
targetRevision, env: kind}`, automated prune and self-heal. `apps/values.yaml`: `env` picks `values-<env>.yaml`,
namespace `llm`, `ragImage: ""` (set to the digest by scripts), apps with waves 0, 1, 2.
`apps/templates/applications.yaml` adds Helm parameter `image=<digest>` only for rag-api and only when set.

**`deploy/helm/llamacpp/`.**
- PVC `llamacpp-models` 2Gi, ReadWriteOnce on kind, ReadWriteMany on OpenShift ("replicas may land on different nodes").
- Deployment without `replicas` when autoscaling (the HPA owns it); `progressDeadlineSeconds: 1800` ("first start
  downloads 1.1 GB; the 600 s default marks it Degraded"); `automountServiceAccountToken: false`.
- Init container `model`: download to `$f.$HOSTNAME.part` with `curl -fL --retry 5 --retry-all-errors -C -`; "the
  checksum decides" (partial kept, complete-but-wrong deleted); re-check on every start.
- Main container: same flags as compose; model read-only; `/tmp` emptyDir.
- Probes: readiness `/health` every 5 s; liveness from 60 s, every 20 s, 6 failures.
- Resources: requests 1 CPU and 1536Mi, limits 4 CPU and 2Gi.
- securityContext: `runAsNonRoot`, seccomp `RuntimeDefault`, no privilege escalation, read-only root, `drop [ALL]`;
  kind adds `runAsUser: 1000650000`, `runAsGroup: 0` (the image has no `USER`).
- `templates/scaledobject.yaml`: min 1, max 3, polling 5 s, `AverageValue` target = `server.parallel` = 2, query
  `sum(llamacpp:requests_processing{job="llamacpp"} + llamacpp:requests_deferred{job="llamacpp"})`; scale-down
  stabilisation 60 s then 1 pod per 30 s; `cooldownPeriod: 60`.
- `templates/expose.yaml`: kind Ingress with empty host (catch-all, because "Windows does not resolve `*.localhost`
  for every client"); OpenShift Route, edge TLS, redirect, `haproxy.router.openshift.io/timeout: 600s`.

**`deploy/helm/platform-deps/`.** Postgres StatefulSet (PVC 2Gi, emptyDir for socket and `/tmp`); Elasticsearch
StatefulSet (PVC 4Gi, heap 1g, security on, plain HTTP; the one exception to read-only root because "ES writes its
keystore under config/ at start"); a minimal Prometheus Deployment with a namespaced Role to discover pods
`app=llamacpp`, scrape every 5 s, retention 2d in emptyDir ("metrics are only needed live for KEDA"), and a
`checksum/config` annotation. Passwords from Secret `llm-secrets` via `secretKeyRef`.

**`deploy/helm/rag-api/`.** `_helpers.tpl` builds `DATABASE_URL` with `$(POSTGRES_PASSWORD)`; Deployment with
`startupProbe` `/healthz` every 5 s up to 60 failures, readiness every 10 s, liveness every 20 s; requests 500m and
1Gi, limits 4 CPU and 3Gi. `rag-migrate` Job as Helm hook `pre-install,pre-upgrade` (ArgoCD PreSync), `backoffLimit
6`. `rag-ingest` Job with `restartPolicy: Never`, `backoffLimit: 3`, and `argocd.argoproj.io/sync-options:
Force=true,Replace=true` because a Job template is immutable; requests 1 CPU and 1Gi, limits 8 CPU and 4Gi; each retry
gets a new emptyDir and downloads the corpus again. `Chart.yaml` is what the CI bump commit edits.

**`deploy/README.md`.** Charts, kind vs OpenShift, timings, and how ArgoCD reads the repo without a remote.

### 5.8 infra/

**In plain English.** Two small building parts: the doll's house frame and the library desk.

**Technically.** `infra/kind/cluster.yaml`: one node, image `kindest/node:v1.36.4@sha256:099e0493...`,
`extraPortMappings` from `127.0.0.1:8090` to NodePort 30080. `infra/git-server/Dockerfile`: `alpine:3.23.0` plus
`git-daemon=2.52.0-r0`, user 65534, serves `/srv/git` on 9418 with `--export-all`.

### 5.9 scripts/

**In plain English.** The buttons: start or stop the clusters, publish to GitOps, run the scaling demo, check the
charts, and the shared toolbox.

**Technically.**
- **`tools.sh`:** pins `KIND_VERSION=v0.33.0`, `KUBECONFORM_VERSION=v0.8.0`, `REGISTRY_IMAGE=registry:3.1.2`,
  `ARGOCD_VERSION=v3.5.4`, `KEDA_CHART_VERSION=2.21.0`, `TRAEFIK_CHART_VERSION=41.6.1`; `fetch` (download, sha256,
  rename); `d()` runs docker with `MSYS_NO_PATHCONV=1`; Windows and Linux amd64 checksums only.
- **`kind-up.sh`:** `wait_rollout` (retries `rollout status --timeout 15m` for up to 30 minutes per resource); `retry`
  (5 attempts, 15 s apart); download cache in `.tools/downloads`; KEDA `IfNotPresent`; CoreDNS `force_tcp`; image
  preloading (render manifests, extract images, pull for linux/amd64, `docker save | ctr images import`, skip present);
  git source `head` or `worktree`; reproducible build from `git archive`; no `helm --wait`; namespace with PSA
  `restricted`; secrets from `.env` or random 16 bytes hex, created once. Method in [6.12](#612-deployment-and-gitops).
- **`gitops-publish.sh`:** force-pushes `<rev>` to the bare repo `main`; renders `root.yaml` with `ragImage` via
  `kubectl patch --local` and applies it; hard-refreshes every Application; polls every 15 s until 4 apps are
  Synced/Healthy at `<rev>`; deadline `KIND_UP_TIMEOUT` (default 3600 s; kind-up passes 10800 s). kubectl runs inside
  the kind node via `docker exec`.
- **`keda-demo.sh [concurrency] [requests]`:** defaults 6 and 120 (evidence used 6 and 60); runs `run.py` through the
  ingress; prints time, in-flight metric, HPA desired/current and pods every 10 s; output marked "not a benchmark".
- **`helm-check.sh`:** renders every chart for both value sets into `kubeconform -strict`; checks the digest parameter
  appears in the apps chart.
- **`kind-down.sh`:** deletes cluster, `kind-registry`, `kind-git`, `.tools/git`.
- **`smoke.sh`:** compose checks (chat, Postgres, Elasticsearch, Grafana, `up` for llamacpp and otel-collector, Tempo,
  Loki, Alloy, collector metrics); does not check the rag-api scrape job, rule health or dashboards.
- **`okd-up.sh`:** resumable sha256-checked bundle download from quay.io; version guards (crc 2.64.0, `oc`); guard for
  `host-network-access`; `until_ok` waits; ArgoCD operator via OLM Subscription with manual InstallPlan approval;
  ArgoCD CR with `resourceHealthChecks`; skopeo push; final wait `OKD_UP_TIMEOUT` default 7200 s; saves the router CA.

### 5.10 ci/

**In plain English.** One script holds every CI step; Jenkins and GitHub both call it, so they cannot drift apart.

**Technically.**

**`ci/run.sh`.** Every tool runs in a pinned container with the workspace mounted at the same path (`CI_MOUNT`).

| Stage | What runs | Gate |
|---|---|---|
| lint | pre-commit (all files), gitleaks full history, hadolint per Dockerfile, actionlint | any finding |
| charts | `scripts/helm-check.sh` in `alpine/helm:4.2.4` | lint or schema failure |
| test | benchmark pytest; rag pytest against a throwaway `postgres:18.6-alpine3.24` with trust auth on a per-build network; `promtool check config` and `test rules` | any failure |
| build | `git archive HEAD:rag` to `docker build --provenance=false` | build failure |
| scan | Trivy CRITICAL, exit code 1 | any CRITICAL |
| push | `docker push` to `localhost:5001`; prints the digest | needs `kind-registry` |
| bump | edits `deploy/helm/rag-api/Chart.yaml` to `version: 0.1.0+ci.<build>`, `appVersion: '<digest>'`, commits in the workspace only | |
| deploy | `scripts/gitops-publish.sh <CI commit> <digest>` | Synced/Healthy |

**`ci/Jenkinsfile`.** Calls those stages; `disableConcurrentBuilds()`, 120-minute timeout, keep 20 builds, `post {
always }` removes the CI tag.

**`ci/jenkins/`.** `Dockerfile` (Jenkins 2.580.1 LTS jdk21 by digest plus Docker CLI and buildx from `docker:29.6.2-cli`
by digest); `casc.yaml` (JCasC); `plugins.yaml` (63 pins); `docker-compose.yml` (`127.0.0.1:8088`, Docker socket,
`group_add: ["0"]`, repo read-only at `/repo`, admin password as a compose secret file); `.env.example`; `README.md`.

### 5.11 .github/

**In plain English.** GitHub's copy of the checklist, minus the steps that need the laptop.

**Technically.** `.github/workflows/ci.yml`: on push to `main` and pull requests; `permissions: contents: read`;
`concurrency` cancels older runs; jobs `lint` (lint, charts) and `test` in parallel, `image` (build, scan) after both;
`actions/checkout` pinned by SHA (v7.0.1) with `persist-credentials: false`; runner `ubuntu-24.04`. The header
explains why push, bump and sync are skipped.

### 5.12 docs/

**In plain English.** The project's notebook: the plan, the status board, why things were chosen, what is weak,
where things are, every version pin, and the evidence logs.

**Technically.**

| File | Content |
|------|---------|
| `docs/PLAN.md` | scope v2, phases, acceptance criteria, approvals, environment facts, stretch list |
| `docs/STATUS.md` | phase and v3 item states with evidence links (partly lagging, see [8.2](#82-docs-that-lag-the-results)) |
| `docs/DECISIONS.md` | what was chosen, why, what was rejected |
| `docs/LIMITATIONS.md` | not run, measurement limits, platform limits, security notes, future work |
| `docs/LAYOUT.md` | layout, compose services and ports |
| `docs/VERSIONS.md` | every pin with checksums and reasons |
| `docs/evidence/*.txt` | per-phase logs: commands, outputs, review rounds |
| `docs/evidence/phase5a-review/`, `phase5b-ci/`, `v3-github-actions/`, `v3-openshift-okd/`, `phase5a-keda/` | raw logs behind those files |
| `docs/LEARN.md` | this guide |

---

## 6. Methods

**In plain English.** How each measurement and each mechanism works: the rules of the experiment, not the results.

**Technically.** Results are in section 7. Code is in section 5. Each method cites its source file.

### 6.1 LLM benchmarks

**In plain English.** Start the server. Send a few practice questions to warm it up. Then send a fixed batch of
questions with a fixed number of pretend users, timing every word, while sensors watch the GPU. Save everything. Do it
three times. Only then compute averages.

**Technically.**
1. The server starts with pinned arguments; the exact command is saved to `server_args/<cell>.txt` and copied into
   every raw file (`notebooks/build_notebook.py`, `start_server`).
2. `run.py` records the environment and the prompts sha256.
3. Per (concurrency, repeat): warmup (`max(2, concurrency)` requests from the end of the list), a `/metrics` snapshot,
   samplers start, the measured closed loop (`max(64, 4 x concurrency)` requests), samplers stop, a second snapshot.
4. Every request: temperature 0, seed 1234, `max_tokens` 128, `ignore_eos`, so every request decodes exactly 128
   tokens; `decode_checks` counts any that did not.
5. 3 repeats per cell.
6. `report.py` groups by session, label, concurrency (and cache regime for llama.cpp) and prints mean ± stdev, flags
   and per-repeat rows.

Why a fixed 128-token decode: throughput and latency become comparable across engines and precisions; the cost is
that the numbers do not reflect natural answer lengths (`docs/DECISIONS.md`). Why closed loop: concurrency is exactly
the number of in-flight requests, so "c8" means 8 users waiting at all times. An open-loop (fixed arrival rate) test
was not run; the repo does not say whether it was considered.

Warmup matters: vLLM printed "Triton kernel JIT compilation during inference" once per server, inside warmup; the TP=2
r1 warmup TTFT mean was 1.811 s vs 0.709 and 0.726 s for r2 and r3 (`docs/evidence/phase2-gpu.txt`). `report.py`
still flags a first repeat more than 10% slower than the rest.

vLLM's `--gpu-memory-utilization 0.90` reserves 90% of each card at start ("Desired GPU memory utilization is (0.9,
13.11 GiB)"), so "GPU mem max" in every vLLM row (12927 to 13731 MiB) shows the booking, not the need. Plain picture: a
restaurant that books 90% of its tables at opening whether or not guests come.

### 6.2 The llama.cpp prompt cache issue

**In plain English.** llama.cpp remembers questions it has already read. Ask the same 64 questions again and it skips
the reading and starts answering almost at once. That made later repeats look much faster, not because the server
warmed up but because it had memorised the questions.

**Technically.**
- llama.cpp b11459 has `--cache-prompt` (on), `--cache-ram` (host prompt cache, default 8192 MiB) and
  `--cache-idle-slots` (on), so any earlier prompt can skip prefill until restart, beyond the 2 slots
  (`docs/evidence/phase2-cpu.txt`, N6).
- First session (2026-10-08): repeat 1 had 10% to 26% of prompt tokens cached, repeats 2 and 3 had 96%. At c1, r1
  spent 175 s of its 530 s in prefill, r2 and r3 spent 11 to 12 s; decode speed was the same (ITL p50 41.4, 40.6,
  41.6 ms) (`docs/evidence/phase2-cpu.txt`). Averaging mixed two regimes and produced CV and "warmup" flags.
- Why repeat 1 is not 0% cached: every prompt shares the chat template prefix, and the warmup prompts are measured
  again later in the run, so c8 (8 warmups) has the highest cold share.
- Fix in reporting: `report.py` computes each repeat's cache share and splits at 50% into warm and cold rows.
- Fix in measurement (session `cpu-20261009`): kind and other compose services stopped; per concurrency, 3 cold
  repeats each after a container restart and as its own `run.py --repeats 1` call, then a discarded warming pass, then
  3 warm repeats. A PowerShell host sampler recorded CPU %, clock, temperature and passive limit every ~10 s. Cold runs
  were 10% (c1), 16% (c4), 26% (c8) cached; warm runs 96% (`docs/evidence/phase2-cpu-rerun.txt`).
- The 2026-10-08 session is superseded but kept in RESULTS.md: busier machine, too few comparable repeats.
- vLLM avoids this with `--no-enable-prefix-caching` (`vllm:prefix_cache_queries_total` delta 0 in every run).

### 6.3 Quantisation accuracy testing

**In plain English.** Compressing a model might make it dumber. Both versions sat the same 250 maths word problems
under the same rules, and each question was compared one to one: did the full model get it right, did the compressed
one?

**Technically.**
- GSM8K first 250 test items, dataset revision pinned, 5-shot as multi-turn chat with the chat template, greedy.
- Two scores: strict-match (needs the `#### N` format) and flexible-extract (last number). Strict also measures
  format: 113 (FP16) and 123 (AWQ) of 250 responses had no `#### N` answer (`benchmark/RESULTS.md`).
- Paired comparison: per item, AWQ score minus FP16 score; mean and 95% CI by normal approximation (`paired_delta`).
  Pairing removes "some questions are just hard", so it is tighter than comparing two independent averages.
- McNemar test (only discordant items): strict 45 vs 30, p = 0.105; flexible 42 vs 26, p = 0.068
  (`docs/evidence/phase2-gpu.txt`). The script that computed it is not committed.
- Caveat for run v2: both servers applied `generation_config.json` (repetition_penalty 1.1), so "the delta compares
  like with like but absolute scores are not stock GSM8K settings" (`benchmark/RESULTS.md`).
- Vision quantisation (FP16 TensorRT vs FP32) is tested with mAP ([6.5](#65-vision-speed-accuracy-and-provenance)).

### 6.4 Making vLLM vs TensorRT-LLM fair

**In plain English.** Timing two runners on different tracks on different days is unfair. So vLLM runs again in the
same session, on the same GPU, right before TensorRT-LLM, with identical settings, and only that pair is compared.

**Technically.**
- Same-session control: the TensorRT-LLM notebook runs vLLM 1.5B FP16 c1/c8/c32 x 3 and GSM8K 250 first.
- `--generation-config vllm` stops vLLM applying `generation_config.json` (repetition_penalty 1.1), which TensorRT-LLM
  would not apply; run v2 lacked it, and RESULTS.md prints this as a "Session differences" note from stored server
  args.
- `report.py` compares TensorRT-LLM only with vLLM rows of the same session, model and precision, after checking
  prompts sha256, `max_tokens`, `ignore_eos` and decode counts; otherwise it prints "Not compared".
- Known remaining difference: TensorRT-LLM's KV cache takes 0.90 of memory left after the engine loads; vLLM's 0.90 is
  of the whole card (`inference/README.md`).
- Known risk: if 0.14 does not honour `end_id = -1`, requests stop at EOS and still return 200; `decode_checks` would
  catch it and block the comparison.
- Stream patch in `serve.py` prevents undercounted TPOT and ITL ([5.3](#53-inference)).
- Used in run v3i: the generated like-for-like line passed (prompts sha256 72fb01ab9477 on both; `max_tokens` 128;
  `ignore_eos` on; concurrency 1, 8, 32; 0 requests off 128 tokens and 0 truncated streams on both), so `end_id = -1`
  held and the ratio was printed (`benchmark/RESULTS.md`). TensorRT-LLM GSM8K used the `gsm8k` engine, vLLM the same
  server as its loadgen (`docs/evidence/v3-trtllm.txt`).

### 6.5 Vision speed, accuracy and provenance

**In plain English.** Same idea as the language benchmark, but each request is one batch of 1, 8 or 32 photos, and
only the model's own work is timed, not photo preparation. For accuracy, the detector runs on 500 standard photos with
human-drawn boxes, in three formats with identical preparation, so any difference comes from the format. And before
trusting comparisons, the three exported model files were checked to be the same model.

**Technically.**
- Speed: images letterboxed once; one batch per request at concurrency 1; timed from host arrays to host outputs
  including copies; JPEG decode, letterbox and NMS excluded; 10 warmup batches, then `max(20, ceil(640 / batch))`
  measured; 3 repeats; the same first 64 images in the same order on every backend. The CPU baseline used the exact
  ONNX file the ORT CUDA cells used (sha256 `c3a4c32fa393...`), with ORT default threads (about 8) and 4 threads
  (`docs/evidence/v3-vision-cpu.txt`).
- Accuracy: Ultralytics validator on the first 500 sorted COCO val2017 images, imgsz 640, batch 1, `rect=False`, conf
  0.001, IoU 0.7, for `pt-fp32`, `onnx-fp32` and `trt-fp16` (b1 engine). `rect=False` because Ultralytics defaults to
  rectangular batches for PyTorch and dynamic ONNX but square ones for static engines, which "would mix preprocessing
  into the FP16 delta". Not comparable to published YOLOv8n figures (5,000 images).
- Provenance: `compare_onnx.py` on the v3c, v3e and v3g exports found identical graphs (321 nodes) and weights
  (12,607,848 initializer bytes), differing only in `metadata_props` (`date`) and `value_info`. Engines are not
  bit-reproducible, so RESULTS.md names the timed engine (0b5ffbf7), the Triton one (15be2884) and the mAP one
  (c9b34364) as sibling builds.

### 6.6 RAG retrieval

**In plain English.** You cannot hand the writer a 356-page book, so each book is cut into short overlapping strips.
Two search styles find candidate strips: exact words (great for "TN 1297", weak on paraphrases) and meaning (great on
paraphrases, weak on rare codes). Their lists are merged by giving points for each place in each list. A careful reader
then picks the best five. The writer must answer only from those five and footnote each sentence.

**Technically.**
- Chunking: 150-word windows, 30-word overlap, page number = page the chunk starts on; small so 5 chunks plus the
  prompt fit in the 2048 tokens per slot. Cold eval prompts were 1211 to 1731 tokens (`observability/SLOs.md`).
- BM25 top 10 and dense kNN top 10 (`RETRIEVE_K`); dense similarity uses the integration default (not set).
- RRF: score = sum of 1 / (k + rank) across lists; uses ranks only, so BM25 and cosine scores need no calibration;
  `k` is Haystack's default. Up to 20 candidates.
- Rerank: the bi-encoder embeds query and documents separately (precomputable); the cross-encoder runs full attention
  over each (question, passage) pair at query time and keeps the top 5. Cold-trace cost: `rag.rerank` 14924.1 ms vs
  `rag.bm25_retrieve` 833.4 ms and `rag.dense_retrieve` 64.8 ms (`docs/evidence/phase3-rag.txt` section 5).
- Grounding is prompt-only: nothing checks that a cited source supports its sentence; faithfulness is the after-the-
  fact check.
- What hybrid bought is not measured: "BM25-only and dense-only were not evaluated" (`docs/DECISIONS.md`).

### 6.7 RAG evaluation

**In plain English.** A homemade exam: for each book, find a few distinctive facts, read the page, write a question
and a model answer from that page only, and note the page as the answer key. Retrieval is marked by a fixed rule ("was
the right page shown?"). Answer quality is marked by a bigger AI model acting as exam marker, whose marks a person then
read for obvious mistakes.

**Technically.**
- Question set: 30 questions over 9 documents (3 each, plus one extra for three documents), written 2026-10-07 from
  pypdf text; one author, no second reviewer; questions 10 to 12 replaced on 2026-10-08 after the TN 2148 drop;
  factual lookups only (`rag/eval/README.md`).
- recall@5: gold `source_id` and page `p` or `p - 1` in the top 5 (`p - 1` because a chunk carries the page it starts
  on). doc recall@5: gold document in the top 5. Deterministic.
- Faithfulness: the judge extracts claims and checks each against the 5 chunk texts; score = share not contradicted.
  Answer relevancy: share of answer statements relevant to the question. Both `async_mode=False`, local judge at
  temperature 0 with JSON-schema output; errors recorded as `<metric>_error`.
- Sample: 12 ids (1, 4, 7, 10, 11, 12, 13, 16, 19, 22, 25, 28) covering all 9 documents. Per-question judge time
  157.2 to 297.8 s for faithfulness and 19.5 to 176.9 s for relevancy (`judge-20261008T120952Z/raw.jsonl`).
- Contradiction review: a person read all 24 reasons; 2 contradicted their own score (q11 relevancy, q22
  faithfulness). A keyword rule tried first flagged 9 scores, mostly wrongly, and was dropped. Headline means keep both;
  "excluding contradicted" is a labelled secondary view.
- The stored `reference` answers are not used by any metric; there is no correctness score.
- Superseded runs: `results/20261008T095250Z/` used 10 documents including TN 2148; inside the current run, the n = 3
  judge summary is superseded by the 12-id sample.

### 6.8 Tracing, metrics and logs

**In plain English.** A tally sheet by the door counts plates out and plates back (metrics). A waiter follows one
order through the kitchen and notes each station's time (a trace). Each cook writes a diary (logs). The order number
is printed on your receipt and in the diary, so either leads to the full journey.

**Technically.**
- Traces: FastAPI's root span `POST /chat`; `chat()` passes `otel_context.get_current()` to the worker thread because
  a new thread does not inherit the current span; `rag.chat` is its child; Haystack spans are renamed
  `rag.<component>`; manual `rag.prompt_build` and `rag.db_write`. `BatchSpanProcessor` sends OTLP gRPC to the
  collector, which batches and forwards to Tempo. Chat 2 in `docs/evidence/v3-loki-langfuse.txt` had 16 spans,
  identical in Tempo and Langfuse. No trace context goes on to llama.cpp or Elasticsearch, so the trace stops at
  rag-api's side.
- Metrics: Prometheus pulls `/metrics` every 15 s; counters only rise (reset on restart, handled by `rate()`);
  histograms expose `_bucket`, `_sum`, `_count`. `path` uses the route template to avoid a series per session id.
- Logs: rag-api prints `chat session_id=... trace_id=<32 hex>`; Alloy ships stdout to Loki with labels `service` and
  `container`; Grafana's derived field links the id to Tempo. Verified 2026-10-09: Tempo returned HTTP 200 for the id
  from the SSE stream (`docs/evidence/v3-loki-langfuse.txt`, follow-up).
- Langfuse switch: compose profile `langfuse` starts its six services; setting `LANGFUSE_OTEL_ENDPOINT` makes the
  collector's second `--config` resolve to `config-langfuse.yaml` through `${VAR:+-langfuse}`; unset, it resolves to
  `config.yaml` again, and merging a file with itself changes nothing. rag-api is untouched.

### 6.9 SLOs and burn-rate alerts

**In plain English.** An SLO is a promise with a small allowance for failure. "99 of 100 chats work" leaves 1 in 100
as the error budget. Burn rate is how fast you spend it. There are two alarms: a loud one (page) when the budget burns
very fast, and a quiet one (ticket) when it burns slowly but steadily. Each alarm checks a long window ("is this
serious?") and a short window ("is it still happening?") and rings only if both agree.

**Technically.** From `observability/SLOs.md`:

| SLO | Target | Window | Budget |
|-----|--------|--------|--------|
| RAG availability | 99% of valid `/chat` requests succeed | 30d | 1% |
| RAG TTFT | 95% of `/chat` requests get the first token within 60 s | 30d | 5% |

These are "CPU development targets for one llama.cpp replica", not attainment. Availability SLI: (5xx + in-stream
errors) / non-4xx requests. In-stream errors count because SSE sends HTTP 200 before retrieval and generation run, so a
later failure cannot change the status code. Example: 5 chats sent during a re-index each got HTTP 200 and `event:
error` with `index_not_found_exception` (`docs/evidence/phase4-observability.txt` section 3). TTFT SLI: 1 minus chats
with TTFT within the `le="60.0"` bucket over all chats that got a first token.

Availability rule, per window W (`rules/recording.yml`):

```promql
(
  (sum(rate(rag_http_requests_total{job="rag-api",path="/chat",status=~"5.."}[W])) or vector(0))
  + (sum(rate(rag_chat_errors_total{job="rag-api"}[W])) or vector(0))
)
/
sum(rate(rag_http_requests_total{job="rag-api",path="/chat",status!~"4.."}[W]))
```

`le=~"60|60.0"` in the TTFT rule accepts both spellings (prometheus_client writes `60.0`). Budget remaining = 1 minus
30d ratio / (1 minus target).

Why 60 s: the threshold must be a bucket bound, so 30 s or 60 s. 30 s held for the cached eval (29 of 30) but not for
the mostly cold run (23 of 30, 77%) or the cold subset (2 of 3); 60 s held for all 33 cold requests measured
(`observability/SLOs.md`). A 40 or 45 s bucket would allow tighter; open item.

Why 99%: one replica of everything and no failover, so tighter needs a second replica first (`docs/DECISIONS.md`).

Alerts (Google SRE workbook "Alerting on SLOs", approach 6):

| Alert | Severity | Long / short window | Burn rate | Availability threshold | TTFT threshold | Budget spent when it fires |
|-------|----------|---------------------|-----------|------------------------|----------------|----------------------------|
| `*ErrorBudgetBurnFast` | page | 1h / 5m | 14.4 | 14.4% | 72% | 2% in 1h |
| | page | 6h / 30m | 6 | 6% | 30% | 5% in 6h |
| `*ErrorBudgetBurnSlow` | ticket | 1d / 2h | 3 | 3% | 15% | 10% in 1d |
| | ticket | 3d / 6h | 1 | 1% | 5% | 10% in 3d |

Threshold = burn rate x budget; budget spent = burn rate x window / 720 h (`SLOs.md`). `RagApiDown` exists because the
SLI is blind when rag-api is down (no requests means an empty ratio). Low traffic caveat: at dev volume one failed chat
can page; the suggested fix (a minimum request count) is not implemented. No Alertmanager, so "page" and "ticket" are
labels only.

### 6.10 Testing the alert rules

**In plain English.** Passing tests prove little if they would also pass for broken rules. So the reviewer broke the
rules on purpose, one small bug at a time, and checked that the tests caught each one. The first test set let bugs
through; it was rewritten until every planted bug failed a test.

**Technically.**
- Round 1: `slo_test.yml` had 10 scenario groups and passed, but "did not pin thresholds or windows"
  (`docs/evidence/phase4-observability.txt` section 2).
- Fix 1: per-window-pair tests: inputs at 1.1x a threshold must fire only that alert; long or short window at 0.95x
  must fire nothing. 2 SLOs x 4 pairs x 3 = 24, plus 10 scenarios = 34 groups. The reviewer's 9 mutants (thresholds to
  100x, 60x, 30x; 5m window to 1h; TTFT budget to 1%; 4xx in the denominator) all failed (section 2b).
- Fix 2: `slo_windows_test.yml`: a 3-day staircase of error rates (counters start at 1,000,000) so every window has a
  distinct exact answer, asserted to 4 decimals; it notes Prometheus 3 range selectors are left-open, so a window W on
  a 2m grid holds (W - 2m) / 2m increments. 3 groups.
- Round 2: 17 mutants all fail; four of them (6h computed over 1h, 30m over 5m, 1d dropping in-stream errors, 1d over
  6h) only the exact-value file catches. A tolerance probe (14.4x to 15x) passes as expected: thresholds are pinned to
  within about 5 to 10% (sections 7, 7b).

### 6.11 GPU replay dashboard

**In plain English.** The GPU test ran on a borrowed machine this stack could not watch live. The test had saved GPU
readings every half second. Those readings were pasted into the ledger at their original times, so Grafana can chart
them, clearly labelled "not live".

**Technically.** `convert.py` writes OpenMetrics with timestamps; `promtool tsdb create-blocks-from openmetrics` writes
blocks into the Prometheus volume; a restart loads them. No SLO or alert uses this data. The blocks age out with the
31-day retention around 2026-11-09 (`docs/evidence/phase4-observability.txt` section 8). Power samples above 70 W are
expected: "power.draw on a T4 is an instantaneous reading, while the 70 W board limit is enforced on averaged power"
(`gpu.json`), which is why the panel uses `max_over_time`.

### 6.12 Deployment and GitOps

**In plain English.** Set up a private warehouse, build the doll's house, stock it with every box in advance at the
loading dock (where waiting is cheap), bake the app's box from the exact committed recipe, put the blueprint on the
library shelf, install the receptionist, the queue watcher and the foreman, then hand the foreman the master list and
wait. The label "0.1.0" can be stuck on a new box, so the scripts tell the foreman the box's fingerprint instead.

**Technically.** `scripts/kind-up.sh`:
0. Cache ArgoCD `install.yaml` and the KEDA and Traefik charts in `.tools/downloads`.
1. Start `kind-registry`.
2. Create cluster `llm`; write the containerd mirror file; connect the registry; patch CoreDNS to `force_tcp`.
3. Render every manifest, extract images, pull on the host (linux/amd64), import into the node unless present.
4. Pick the revision (`GIT_SOURCE=head` default, or `worktree` for a synthetic commit of uncommitted work); build the
   rag image from `git -c core.autocrlf=false archive --format=tar --mtime=1970-01-01T00:00:00Z <rev>:rag` with
   `--provenance=false`; push; read the digest.
5. Start `kind-git`; install KEDA and Traefik (no `--wait`); install ArgoCD and patch `argocd-cm` (Application health
   check, 60 s reconciliation).
6. Create namespace `llm` with PSA restricted and `llm-secrets` once.
7. `gitops-publish.sh <rev> <digest>` with a 3-hour deadline.

Why each build flag: `git archive` gives committed file modes (the Windows tree gave 0755, Linux 0644, so caches
missed); `core.autocrlf=false` stops CRLF conversion; an explicit date because git reads `--mtime=@0` as "now";
`--provenance=false` drops the timestamped attestation. A changed digest changes the pod template, so rag-api rolls and
the ingest Job is recreated, deleting and rebuilding the index. Same digest means nothing rolls: build #3 pushed the
same `rag@sha256:6f10a3e7...` and "argocd sync 3 s (no rollout)" (`docs/evidence/phase5b-ci.txt`). Equality relies on
the shared BuildKit cache; the RUN steps are not bit-reproducible on their own.

OpenShift differences (`scripts/okd-up.sh`, `docs/evidence/v3-openshift-okd.txt`):

| Topic | kind | OpenShift/OKD |
|---|---|---|
| Exposure | Traefik Ingress on `127.0.0.1:8090` | Route with edge TLS |
| UID | fixed 1000650000, group 0 | assigned by restricted-v2 (1000670000) |
| Model PVC | RWO | RWX |
| rag image | `localhost:5001/...@sha256` | internal registry `@sha256` via skopeo |
| ArgoCD | upstream v3.5.4 | operator v0.19.0 (ArgoCD v3.4.4) |
| Git | `git://kind-git/...` `main` | `git://host.crc.testing/...` `okd`, host `127.0.0.1:9418` |

The charts needed no change for OpenShift; the setup script did the extra paperwork.

### 6.13 KEDA autoscaling

**In plain English.** Each cook can handle two orders at once. KEDA counts orders being cooked plus orders waiting,
divides by two, rounds up, and asks for that many cooks, between one and three. It adds cooks fast and sends them home
slowly: one every 30 seconds after a minute of calm.

**Technically.** llama.cpp exports `llamacpp:requests_processing` and `llamacpp:requests_deferred`; the in-cluster
Prometheus scrapes every 5 s; KEDA polls the sum every 5 s; `AverageValue` target 2 gives desired replicas =
ceil(in-flight / 2), clamped to 1..3, so 6 concurrent requests ask for 3. Scale-down: 60 s stabilisation window, then
1 pod per 30 s. New replicas start fast because the model is already on the shared PVC: "sha256 check plus load about
20 s" (`docs/evidence/phase5a-platform.txt` section 6). `cooldownPeriod: 60` governs scaling to zero in KEDA, which
`minReplicaCount: 1` never does; the repo does not discuss it.

### 6.14 CI pipeline and secrets

**In plain English.** Every run: check spelling and secrets, check the blueprints, run the tests, bake the box,
inspect it, ship it to the warehouse, write the new fingerprint into the blueprint, tell the foreman. GitHub does the
first five; only the laptop's Jenkins can reach the laptop's cluster. Real passwords live only in files git ignores,
or are made up at random inside the cluster.

**Technically.**

| | Jenkins (local) | GitHub Actions |
|---|---|---|
| Trigger | manual via jenkins-cli | push to `main`, pull requests |
| Source | `file:///repo`, committed `main` | checkout of the pushed ref |
| Stages | lint, charts, test, build, scan, push, bump, argocd sync, clean | lint+charts, test, then build+scan |
| Deploys | yes, to kind | no, by design |
| Credentials | admin password as compose secret file | `contents: read`, no persisted credentials |

CI-only bump commit: ArgoCD reads git, so a new image must become a git change. The bump commit lives only in the
Jenkins workspace and is published to `kind-git` as `main`; "it is never pushed to the developer's branch"; the next
`kind-up.sh` publishes HEAD again (`ci/jenkins/README.md`).

Secrets: compose `.env` with `${VAR:?}`; kind `llm-secrets` from `.env` or random, kept across re-runs "because
Postgres and Elasticsearch store the password on first start"; OKD always random; charts use `existingSecret` and
`secretKeyRef`; CI tests use trust auth on a throwaway Postgres; the OKD registry token is an environment variable.
Known gap: builds run on the Jenkins controller and a probe build could read `/run/secrets/jenkins_admin_password`;
fixing it needs a separate agent, not done (`docs/evidence/phase5b-ci.txt`). gitleaks scanned 21 commits in build #1
and 31 in the first Actions run, no leaks.

---

## 7. Results explained

**In plain English.** Every result table in the repo, with what it means in everyday words. Read the plain paragraph
first; the technical one explains why the numbers look the way they do.

**Technically.** Benchmark tables are in [../benchmark/RESULTS.md](../benchmark/RESULTS.md) unless another file is
named. Values are mean ± stdev across 3 repeats unless stated.

### 7.1 How to read benchmark columns

**In plain English.** Think of a coffee shop. "Out tok/s" is how many cups the whole shop makes per second. "TTFT" is
how long until your first sip. "ITL" and "TPOT" are the time between sips. "E2E" is order to empty cup. "p50" is the
typical customer, "p95" the unlucky one in twenty.

**Technically.**

| Column | Meaning |
|---|---|
| cell | `<session>/<label>`, plus `[warm cache]` or `[cold cache]` for llama.cpp |
| conc | closed-loop concurrency |
| out tok/s | aggregate output tokens per second over the measured window |
| TTFT p50 / p95 s | median and 95th percentile time to first token |
| ITL p50 ms | median gap between streamed chunks |
| TPOT p50 ms | median over requests of each request's average time per output token |
| E2E p50 / p95 / p99 s | request latency percentiles |
| tok/s per req p50 | median of each request's 128 / E2E |
| GPU mem max MiB, util mean %, SM clock p50 MHz, temp max C | nvidia-smi every 500 ms during the measured window |

Each repeat's percentile is over its own measured requests; the table then shows mean ± stdev across repeats.

### 7.2 CPU llama.cpp

Session `cpu-20261009`, Qwen2.5-1.5B Q4_K_M, `--threads 4 --parallel 2`, AMD Ryzen 7 7730U laptop.

| conc | warm out tok/s | cold out tok/s | warm TTFT p50 s | cold TTFT p50 s | warm TTFT p95 s | cold TTFT p95 s | warm ITL p50 ms | cold ITL p50 ms |
|---|---|---|---|---|---|---|---|---|
| 1 | 24.2 ± 0.4 | 17.2 ± 0.1 | 0.210 | 0.241 | 0.307 | 9.333 | 39.2 | 38.2 |
| 4 | 31.0 ± 0.1 | 19.9 ± 0.1 | 8.584 | 18.0 | 9.181 | 29.3 | 60.9 | 61.2 |
| 8 | 29.8 ± 0.2 | 20.1 ± 0.1 | 25.8 | 38.2 | 27.1 | 66.3 | 63.3 | 63.9 |

**In plain English.** With a warm cache the server skips re-reading known questions, so the shop makes more words per
second. The typical first-word wait at c1 barely changes because most questions are short, but the slowest waits (p95)
collapse from about 9 seconds to 0.3 seconds, because the long maintenance-log questions no longer need reading from
scratch. Once words flow, their speed is the same warm or cold. The server has two cooks (slots): with one customer one
cook works; with four, two work and two wait; with eight, six wait. More customers mostly make the line longer.

**Technically.** The cache removes prefill; decode (ITL) is untouched, so warm and cold ITL agree at every
concurrency. Warm throughput rises from 24.2 (c1) to 31.0 (c4) and is flat at c8 (29.8) while TTFT p50 grows from
0.210 to 8.584 to 25.8 s: with `--parallel 2` at most 2 requests decode, the rest queue, and the queue shows up as
TTFT. ITL rises from 39.2 to about 61 to 63 ms because two slots share 4 threads, which is why throughput does not
double. E2E p50 at c8 (34.1 s) is mostly queueing (`docs/evidence/phase2-cpu-rerun.txt`). No cpu-20261009 cell has a
flag; CV on out tok/s at most 0.017. The superseded 2026-10-08 session shows the same shape with lower numbers and
flags (fewer than 3 comparable repeats). CPU and GPU rows are not like for like; no ratio is reported
(`docs/LIMITATIONS.md`).

### 7.3 GPU vLLM run v2

Session `kaggle-20261009`, vLLM 0.31.0 on Kaggle T4 (driver 580.178.04).

| cell | conc | out tok/s | TTFT p50 s | ITL p50 ms | TPOT p50 ms | E2E p95 s | SM clock p50 MHz |
|---|---|---|---|---|---|---|---|
| 1.5B FP16 | 1 | 51.6 ± 1.9 | 0.034 | 19.2 | 19.3 | 2.646 | 575 |
| 1.5B FP16 | 8 | 356.0 ± 0.9 | 0.302 | 18.3 | 19.1 | 3.731 | 1070 |
| 1.5B FP16 | 32 | 716.3 ± 1.1 | 1.171 | 29.5 | 35.6 | 6.077 | 1080 |
| 1.5B AWQ | 1 | 115.2 ± 0.6 | 0.030 | 8.017 | 7.928 | 1.361 | 1045 |
| 1.5B AWQ | 8 | 567.1 ± 2.3 | 0.286 | 9.844 | 10.8 | 2.686 | 1250 |
| 1.5B AWQ | 32 | 907.6 ± 2.6 | 1.197 | 19.6 | 25.2 | 4.868 | 1230 |
| 7B AWQ | 8 | 181.9 ± 0.3 | 0.970 | 29.9 | 31.4 | 8.349 | 1010 |
| 1.5B FP16 TP=2 | 32 | 1121.0 ± 7.1 | 0.761 | 18.9 | 22.5 | 3.832 | 1345 |

**In plain English.**
- Batching is the big win: from 1 to 32 users, FP16 throughput rises from 51.6 to 716.3 tok/s while the typical word
  gap only grows from 19.2 to 29.5 ms. With one user the GPU was mostly idle.
- AWQ (4-bit) is about twice as fast with one user: each word needs the whole model read from GPU memory, and the
  4-bit model is about 1.1 GB against about 3.1 GB for FP16 (`docs/evidence/phase2-gpu.txt`).
- The AWQ advantage shrinks with more users (115.2 vs 51.6 at c1, 907.6 vs 716.3 at c32): with many users the weights
  are read once per step for everyone, and arithmetic becomes the limit (likely, not measured).
- 7B AWQ at c8 makes about half the words per second of 1.5B FP16 at c8: a much bigger model.
- Two GPUs (TP=2) beat one at c32 but not by double, likely because the cards must exchange results over a
  slow link after every layer (not measured).

**Technically.**
- Memory-bound check: FP16 c1 about 50 tok/s x about 3 GB is about 150 GB/s; AWQ c1 115 tok/s x about 1.1 GB is about
  130 GB/s; both under the T4's 320 GB/s; weight size alone predicts up to about 2.8x, observed 2.2x; "Nothing exceeds a
  physical bound" (`docs/evidence/phase2-gpu.txt`).
- TP=2 is 1.57x "on two cards without P2P"; custom all-reduce was disabled ("platform lacks GPU P2P capability"), so
  all-reduce goes through NCCL (same file).
- At c32 TPOT p50 is above ITL p50 (FP16 35.6 vs 29.5 ms). ITL is the typical gap; TPOT averages each request's gaps
  including occasional long ones. A likely reason, not analysed in the repo, is new requests' prefill delaying the next
  decode step.
- Every run averaged 64 to 67 W against the 70 W cap at 75 to 82 C, so all GPU numbers are power limited to some
  degree.
- Flags: AWQ c32 CV 0.11 on TTFT p50 ("small, not a measurement error"); 7B TTFT p95 2.887 ± 0.645 s is noisy but
  unflagged because flags check p50.
- The FP16 c1 cell ran at a low clock: next section.

### 7.4 The GPU power-cap story

**In plain English.** In the first GPU run, the one-user FP16 test ran with the GPU's engine at roughly half its usual
speed, and nobody knew why. The team added a sensor that records the GPU's own reason for slowing down. In every later
session, that sensor said "software power cap" in essentially every sample, not only for the one-user tests but for
the 8-user and 32-user tests too. The power cap is like a speed governor on a delivery van: the van slows itself to stay
under a fuel limit. So the cap explains why the GPU never ran at full speed in those sessions. It does not explain why
the one-user test sits lower than the busier ones; that is still unknown. Sessions whose GPU ran faster also produced
more words per second.

**Technically.**
- Run v2: FP16 c1 SM clock p50 575 MHz (per repeat 615, 555, 555 vs max 1305, 1200, 1245), flagged in all 3 repeats.
  Clock event reasons were then captured only at session start, so RESULTS.md says "cause not recorded".
- Two percentages have existed for the same cell. Before this docs update `README.md` and `docs/LIMITATIONS.md` said
  "about 53%" (575 vs 1080 MHz), computed when only run v2 cells existed (`docs/evidence/phase2-gpu.txt`, review round 1). The current
  `benchmark/RESULTS.md` says "about 56% of the SM clock of the other GPU cells (p50 575 MHz vs median 1030 MHz)",
  because the median moved as v3 cells were added. README.md and LIMITATIONS.md now match it.
- After run v2 the sampler added `clocks_event_reasons.active` (bitmask; 0x4 = SW power cap).
- v3d, v3f and v3h each re-ran FP16 c1 as the vLLM control. Every repeat is flagged with "SW power cap active in 100%
  of samples" (one v3d repeat 99%); thermal bits are not reported active. A raw sample shows `0x0000000000000004` in
  `kaggle-20261010-v3h/.../vllm-t4-1.5b-fp16-c1-r1-20261010T142748Z.json`.
- The c8 and c32 runs of the same sessions show the same bit. Counting bit 0x4 in the raw `gpu_samples` of every
  v3d, v3f and v3h vLLM run (`benchmark/results/raw/kaggle-20261010-v3{d,f,h}/results/raw/vllm-t4-1.5b-fp16/*.json`,
  27 files, counted for this guide): all c8 and c32 runs have it in 100% of samples, and the c1 runs in 99% to 100%.
  RESULTS.md prints reason bits only for flagged cells, which is why only c1 shows it there.
- What this proves: in v3 sessions the clocks were held below max by the SW power cap, not by heat (thermal bits are
  not reported active). What it does not prove: why c1 sits lower than c8 and c32, since all three are capped.
- RESULTS.md cell notes: v3d c1 "about 44% ... (p50 450 MHz vs median 1030 MHz; SW power cap active in 100% of
  samples)"; v3f c1 "about 55% ... (p50 565 MHz ...)". v3h c1 (715 MHz) gets per-run flags (p50 below 60% of the run's
  max) but no cell note, because 715 MHz is above 60% of the 1030 MHz median; the same holds for v3i vLLM c1 at 642.5 MHz.
- c1 throughput follows the c1 clock across sessions: v3d 42.5 tok/s at 450 MHz, v3f 50.2 at 565, v2 51.6 at 575, v3i
  55.7 at 642.5, v3h 58.9 at 715. A pattern in five sessions, not a controlled experiment.
- Run v3i (both engines): "SW power cap bit set in 91 to 100% of samples in every cell (TRT-LLM c32 91 to 94%, all
  others 100%)" (`docs/evidence/v3-trtllm.txt`). This is the clock confound in the engine comparison
  ([7.10](#710-tensorrt-llm-vs-vllm-run-v3i)).
- Not explained: why batch 1 settles at 450 to 715 MHz while c8 and c32 sit near 1000 MHz, when all three show the SW
  power cap in about 100% of samples.
- Consequence: any FP16 vs AWQ comparison at c1 carries a clock confound "whose size and direction are unknown",
  though "a large effect is unlikely" because batch-1 decode is memory-bound (`docs/evidence/phase2-gpu.txt`).
- Vision runs saw it too: ORT CUDA cells had SW power cap active in 96% to 100% of samples; TensorRT cells 80% (b1),
  50% (b8), 28% (b32); PyTriton cells "no clock event reason active" ([7.7](#77-vision-speed)).

### 7.5 Same-session vLLM controls

| session | c1 out tok/s | c8 | c32 | c1 SM p50 | c8 SM p50 | c32 SM p50 |
|---|---|---|---|---|---|---|
| kaggle-20261009 (v2, no `--generation-config vllm`) | 51.6 | 356.0 | 716.3 | 575 | 1070 | 1080 |
| kaggle-20261010-v3d | 42.5 | 344.1 | 684.5 | 450 | 970 | 1005 |
| kaggle-20261010-v3f | 50.2 | 346.5 | 689.6 | 565 | 990 | 1020 |
| kaggle-20261010-v3h | 58.9 | 363.9 | 743.3 | 715 | 1125 | 1145 |
| kaggle-20261010-v3i | 55.7 | 359.7 | 732.5 | 642.5 | 1090.0 | 1105.0 |

**In plain English.** The same test on five different borrowed GPUs gives similar but not identical numbers; faster
clocks give more words per second. That is exactly why a fair TensorRT-LLM comparison must use vLLM numbers from the
same session.

**Technically.** Same model, precision, prompts and settings; differences are session hardware state and, for v2,
the generation config. RESULTS.md prints the server-arg difference ("lacks" / "adds" `--generation-config vllm`). The
v3h c32 cell is flagged CV 0.11 on TTFT p50.

### 7.6 GSM8K accuracy

| session | label | strict-match | flexible-extract |
|---|---|---|---|
| kaggle-20261009 | 1.5B AWQ | 0.300 ± 0.029 | 0.528 ± 0.032 |
| kaggle-20261009 | 1.5B FP16 | 0.360 ± 0.030 | 0.592 ± 0.031 |
| v3d, v3f, v3h, v3i | 1.5B FP16 vLLM, `--generation-config vllm` | 0.344 ± 0.030 | 0.620 ± 0.031 |
| v3i | 1.5B FP16 TensorRT-LLM | 0.356 ± 0.030 | 0.620 ± 0.031 |

AWQ minus FP16, paired over 250 items: strict -0.060, 95% CI [-0.128, +0.008]; flexible -0.064, 95% CI [-0.128,
+0.000]. TensorRT-LLM minus vLLM (v3i, same 250 items): strict +0.012, 95% CI [-0.012, +0.036]; flexible +0.000, 95%
CI [-0.019, +0.019] (`benchmark/RESULTS.md`).

**In plain English.** The compressed model scored about 6 points lower. With only 250 questions, that gap could still
be luck: the plausible true gap runs from about 13 points worse to roughly no difference. Honest summary: "AWQ looks a
bit worse, and this test is too small to prove it". The ± after each score is the standard error: each score is
uncertain by about 3 points.

**Technically.** A 95% CI that includes 0 means the data do not reject "no difference" at the 5% level; the exact
McNemar test agrees (p = 0.105 strict, 0.068 flexible). "A larger subset was not run" (`docs/evidence/phase2-gpu.txt`).
Strict is far lower than flexible because over 100 of 250 answers lacked the `#### N` format. v2 vs v3 FP16 differ
(0.360 vs 0.344 strict, 0.592 vs 0.620 flexible); the stated setting difference is repetition_penalty 1.1 in v2; no
paired test between them exists. The four v3 vLLM sessions give identical scores to three decimals, consistent with greedy decoding
with the same settings (batched GPU greedy decoding is not guaranteed deterministic); the repo does not comment on it.

### 7.7 Vision speed

| backend | precision | b1 images/s | b8 | b32 | b1 latency p50 ms | GPU util at b32 % |
|---|---|---|---|---|---|---|
| ORT CPU, laptop, default threads | fp32 | 23.1 | 19.8 | 19.3 | 43.2 | n/a |
| ORT CPU, 4 threads | fp32 | 21.3 | 18.6 | 18.2 | 44.8 | n/a |
| ORT CUDA, T4 (v3c) | fp32 | 162.3 | 187.0 | 180.5 | 6.118 | 82.7 |
| TensorRT FP16, T4 (v3c) | fp16 | 295.1 | 390.4 | 227.7 | 3.341 | 34.1 |
| Triton via PyTriton, T4 (v3e) | fp16 engine | 32.8 | 28.2 | 23.3 | 30.9 | 4.937 |

**In plain English.**
- At batch 1 the laptop CPU does 23.1 images/s, ORT CUDA on the T4 162.3 and TensorRT 295.1. Different machines, so a
  side-by-side, not a like-for-like comparison.
- On the same GPU, TensorRT FP16 beats ONNX Runtime FP32 at b1 and b8 (295.1 vs 162.3, 390.4 vs 187.0): it compiles
  for the exact chip and uses half-size numbers.
- TensorRT gets slower per image at batch 32 than at batch 8, with the GPU only a third busy. Likely culprit: copying
  32 full-size photos as 32-bit numbers into the GPU.
- Triton is much slower here because it had to run as a Python program behind a web server receiving 32-bit photos
  over HTTP; the GPU sat almost idle.
- CPU does not improve with batch.

**Technically.**
- Generated note: "images/s falls from 390 at b8 to 228 at b32, with GPU util 34%. Likely, not proven: the FP32 input
  copy (157 MB per b32 batch, from pageable host memory) dominates the batch time."
- Triton note: "these rows measure the PyTriton Python backend over HTTP with FP32 payloads (GPU util 4 to 13%, SM clock
  p50 about 585 MHz), not Triton's native TensorRT backend; direct TensorRT FP16 rows are 9 to 14x faster at the same
  batch sizes (different sessions and engine builds)". Triton b1 is flagged CV 0.15.
- CPU: "A plausible reason is cache and memory pressure from larger activations on CPU; not investigated". Default
  threads (about 8) beat 4 threads by only 6% to 9%; the 4-thread runs came second "with the ACPI zone at 79 to 81 C
  vs 76 to 79 C", a run-order confound (`docs/evidence/v3-vision-cpu.txt`).

### 7.8 Vision accuracy

| model | precision | mAP50-95 | mAP50 |
|---|---|---|---|
| yolov8n.pt (PyTorch) | fp32 | 0.401 | 0.556 |
| yolov8n-fp32-opset17.onnx | fp32 | 0.401 | 0.556 |
| yolov8n-fp16-b1.engine (TensorRT) | fp16 | 0.401 | 0.555 |

TensorRT FP16 minus ONNX FP32 mAP50-95 = +0.0003 (`benchmark/RESULTS.md`).

**In plain English.** All three formats find objects equally well. Half precision lost nothing measurable; +0.0003 is
noise in the fourth decimal.

**Technically.** Raw values: PyTorch 0.40068282933505744, ONNX 0.4006828264389918, TensorRT 0.40100439381936354
(`benchmark/results/raw/kaggle-20261010-v3g/results/raw/vision-accuracy/`). The mAP engine is a sibling build of the
timed engine, not the same file. 500 images, not the 5,000 of published figures.

### 7.9 The memory-limit story (TensorRT-LLM)

**In plain English.** The T4 cannot run the fast attention method TensorRT-LLM normally uses, so it falls back to a
slow one that needs a large scratch pad. The first engine was sized for 32 long prompts at once, and its scratch pad
needed about 22 GB on a card with about 15.6 GB free. The fix is two smaller engines: one for the speed test (many
short prompts) and one for the maths exam (fewer, longer prompts). Because the scratch pad grows with the square of the
prompt length, shorter limits shrink it fast.

**Technically.**
- v3h log: the GPT attention plugin "requires 21978227456 bytes of scratch space, but only 15636037632 is available"
  (`benchmark/results/raw/kaggle-20261010-v3h/README.md`, `logs/trtllm-build.log`).
- Cause: `--context_fmha disable` because SM75 has no FMHA kernels in 0.14, so the unfused context path allocates the
  FP16 QK buffer (2 bytes) plus its float copy (4 bytes) for every batch slot, each of 12 heads and every pair of input
  positions: `6 x B x 12 x L^2` bytes plus smaller terms. With B 32 and L 3072 that is 21,978,227,456 bytes.
- `workspace_bytes()` in `inference/trtllm/build_engine.py` reproduces the formula exactly, and a unit test pins the
  figure.
- Fix (`693b44b`): profiles `bench` (batch 32, input 1024, seq 1152) and `gsm8k` (batch 8, input 2048, seq 2304), each
  about 2.55 GB of workspace; batch 32 with input 2048 would need 9.85 GB. Inputs fit: longest loadgen prompt 893
  tokens, longest GSM8K 5-shot prompt 1654 tokens. Batch 8 only makes GSM8K requests queue, and "greedy outputs do not
  depend on batching".
- Outcome (run v3i, notebook from 693b44b): both engines built, with attention workspace 2,550,211,328 bytes
  (`bench`) and 2,550,205,184 bytes (`gsm8k`), engine size 3.4G each; the `build-engine` step took 131 s
  (`docs/evidence/v3-trtllm.txt`). Results in [7.10](#710-tensorrt-llm-vs-vllm-run-v3i).

### 7.10 TensorRT-LLM vs vLLM (run v3i)

| conc | TRT-LLM out tok/s | vLLM out tok/s | TRT-LLM / vLLM | TRT-LLM TTFT p50 s | vLLM TTFT p50 s | TRT-LLM E2E p95 s | vLLM E2E p95 s | TRT-LLM SM clock p50 MHz | vLLM SM clock p50 MHz |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 68.9 ± 0.2 | 55.7 ± 1.3 | 1.24 | 0.025 | 0.033 | 2.029 | 2.433 | 1040.0 | 642.5 |
| 8 | 368.3 ± 2.5 | 359.7 ± 0.3 | 1.02 | 0.357 | 0.294 | 3.334 | 3.722 | 972.5 | 1090.0 |
| 32 | 613.8 ± 7.5 | 732.5 ± 1.8 | 0.84 | 2.860 | 1.109 | 6.978 | 5.998 | 850.0 | 1105.0 |

Source: `benchmark/RESULTS.md`, TensorRT-LLM vs vLLM table (session `kaggle-20261010-v3i`, Qwen2.5-1.5B FP16, one T4,
3 repeats per cell); integrity and caveats in `docs/evidence/v3-trtllm.txt`.

**In plain English.** Two engines raced on the same borrowed GPU on the same day with the same questions. With one
user, the tailor-made engine (TensorRT-LLM) was about a quarter faster, but its GPU happened to run at a much higher
clock in that test, so part of that win may be the clock, not the engine. With 8 users they tied. With 32 users vLLM
was faster, and TensorRT-LLM made each user wait much longer for the first word. Both engines got the maths exam
equally right. So on a T4, the tailor-made engine is not a free win.

**Technically.**
- Like for like, from the raw files: prompts sha256 72fb01ab9477 on both, `max_tokens` 128, `ignore_eos` on, 0 requests
  off 128 tokens, 0 truncated streams, 18 runs (`benchmark/RESULTS.md`; `docs/evidence/v3-trtllm.txt`).
- c1: 1.24x, but vLLM c1 ran at SM clock p50 642.5 MHz against 1040.0 for TensorRT-LLM, both power-capped: "part of
  the c1 gap may be clock" (`docs/evidence/v3-trtllm.txt`). This is the same unexplained low-c1-clock pattern as
  [7.4](#74-the-gpu-power-cap-story), now on vLLM only.
- c8: 1.02x, effectively a tie.
- c32: 0.84x, with TTFT p50 2.860 vs 1.109 s. Evidence reading: "likely the unfused context attention
  (--context_fmha disable on SM75) making prefill of 32 queued prompts slower, not proven". TensorRT-LLM's SM clock
  p50 at c32 was 850.0 MHz vs 1105.0 for vLLM, which is a second possible contributor; neither is isolated.
- GSM8K: TensorRT-LLM minus vLLM +0.012 strict (95% CI [-0.012, +0.036]) and +0.000 flexible: "no difference
  measurable on 250 items". Strict-match invalid answers: 112 (vLLM) and 105 (TensorRT-LLM).
- What is measured is engine plus server: TensorRT-LLM ran behind its own patched OpenAI example server (not
  `trtllm-serve`, not Triton); vLLM behind `vllm serve`. Its KV cache took 0.90 of memory free after loading; vLLM's
  0.90 is of the whole card. Triton's TensorRT-LLM backend was not run (needs Docker).
- Session cost: 15:50:24Z to 16:28:31Z (38 min); TensorRT-LLM install 169 s, engine build 131 s, loadgen 545 s
  (`docs/evidence/v3-trtllm.txt`).
- Unreconciled: the end-of-session `nvidia-smi -q` counter shows SW Power Capping 22,301,052 us (22.3 s), "far less
  than the sampled share over ~25 min of load implies; the two sources disagree" (same file).

### 7.11 Kaggle run history

| Run | Notebook, commit | Outcome | Lesson | Source |
|---|---|---|---|---|
| v1, 2026-10-07 | vllm-t4-benchmark | failed: no GPU, no internet | account likely not phone verified; add a preflight cell that stops at once | `benchmark/results/raw/kaggle-20261007/README.md` |
| v2, 2026-10-09 | vllm-t4-benchmark at 8c01b326 | success: 24 loadgen files, 2 GSM8K runs, 39 min 17 s | vLLM 0.31.0 works on T4 (TRITON_ATTN, Marlin AWQ, TP=2 via NCCL); found repetition_penalty 1.1; FP16 c1 low clock, cause unknown; add clock event reasons | `docs/evidence/phase2-gpu.txt` |
| v3a | dcgm_vision_t4 from 87ecf0a | partial: DCGM basic fields 96 values each; every profiling field `DCGMError_RequiresRoot`; crashed in DCGM teardown, so vision never ran | no DCGM profiling on Kaggle; drop the DCGM re-run | `kaggle-20261010-v3a/README.md` |
| v3b | trtllm_t4 from 87ecf0a | failed: vLLM lock not embedded; mpi4py 3.1.6 build failed (`dry_run`) | pin setuptools 80.10.2 via build constraints; `checked()` added | `kaggle-20261010-v3b/README.md` |
| v3c | dcgm_vision_t4 from 80905ea | partial: ORT CUDA and TensorRT speed valid; mAP failed (no `train:`); Triton failed (`_contextvars`) | add `train:`; `use_own_libpython()` | `kaggle-20261010-v3c/README.md` |
| v3d | trtllm_t4 from 80905ea | partial: vLLM control + GSM8K valid; `import tensorrt_llm` failed (pyarrow `PyExtensionType`) | pin pyarrow 20.0.0; pynvml 11.5.3 | `kaggle-20261010-v3d/README.md` |
| v3e | dcgm_vision_t4 from fb5ea24 | partial: Triton valid; mAP failed (Kaggle's inline `MPLBACKEND`) | set `MPLBACKEND=Agg` | `kaggle-20261010-v3e/README.md` |
| v3f | trtllm_t4 from fb5ea24 | partial: vLLM control valid; install, import, download OK; checkpoint save failed (dir not created) | create dirs in `convert()` | `kaggle-20261010-v3f/README.md` |
| v3g | dcgm_vision_t4 from 60d35f2 | mAP valid for PyTorch, ONNX and TensorRT | vision item complete on GPU | `kaggle-20261010-v3g/README.md` |
| v3h, 14:20 to 14:45Z | trtllm_t4 from 414f88e | partial: vLLM control valid; conversion OK; `trtllm-build` out of scratch memory | two engines ([7.9](#79-the-memory-limit-story-tensorrt-llm)) | `kaggle-20261010-v3h/README.md` |
| v3i, 15:50:24Z to 16:28:31Z | trtllm_t4 from 693b44b | complete: both engines built, TensorRT-LLM and same-session vLLM control, loadgen and GSM8K | the two-engine sizing worked; see [7.10](#710-tensorrt-llm-vs-vllm-run-v3i) | `docs/evidence/v3-trtllm.txt`, `kaggle-20261010-v3i/README.md` |

**In plain English.** Most failures were about the environment, not the models: a missing phone verification, a
packaging tool that removed an old option, two libraries out of step, a missing folder, a leftover plotting setting.
Each failure was recorded, fixed, reproduced locally where possible, and the next run got one step further. The last
failure was real engineering: the memory story above. The fifth TensorRT-LLM session (v3i) completed.

**Technically.** Each raw README says what is valid and what failed. RESULTS.md session notes say "TensorRT-LLM rows:
not produced" with the FAILED file for v3d (`FAILED-install.txt`), v3f and v3h (`FAILED-build-engine.txt`), and
"present" for v3i. The Triton TensorRT-LLM backend has `NOT_RUN-triton-tensorrtllm-backend.txt` (needs Docker).

### 7.12 RAG results

| Metric | Value | n | Source |
|--------|-------|---|--------|
| recall@5 (page level) | 0.867 | 30 | `rag/eval/results/20261008T113955Z/summary.md` |
| doc recall@5 | 1.000 | 30 | same |
| faithfulness | 0.592 | 12 | `.../judge-20261008T120952Z/summary.md` |
| answer relevancy | 0.798 | 12 | same |
| faithfulness, excluding contradicted | 0.645 | 11 | same |
| answer relevancy, excluding contradicted | 0.870 | 11 | same |
| latency mean / median | 7.999 s / 4.9 s | 30 | run `summary.md` |
| TTFT median | 2.6 s (2.58 s in `observability/SLOs.md`) | 30 | same |

**In plain English.**
- For most questions (26 of 30; derived: 0.867 x 30, sources above) the right page or the page before was among the 5
  shown to the writer. For every question the right book was there.
- The marker judged about 59% of answer claims as backed by the shown pages and most answers on topic. But the marker
  is a small compressed model with no human check, so read these as a thermometer for "better or worse than last
  time", not a grade.
- Most answers came in about 5 seconds only because the writer had seen nearly all those questions before. A brand-new
  question waited 22 to 35 seconds for its first word.

**Technically.**
- Page-level misses: q12, q15, q18, q26, all still the right document (`docs/evidence/phase3-rag.txt` section 9). The
  metric is strict about pages: q15's answer matches its reference although retrieval showed pages 9, 8, 1, 6, 6
  against gold page 12 (`rag/eval/results/20261008T113955Z/raw.jsonl`). A hit only
  means the page was shown, not used: q1 is a hit with a wrong answer.
- Per id faithfulness / relevancy: q1 0.0/0.75, q4 1.0/1.0, q7 0.5/1.0, q10 0.0/1.0, q11 0.0/0.0, q12 0.6/0.2, q13
  1.0/1.0, q16 1.0/1.0, q19 1.0/1.0, q22 0.0/0.75, q25 1.0/0.875, q28 1.0/1.0. q1 is a real error; q10 and q7 are judge
  misreadings (`rag/eval/README.md`, "Judge limitations"); q11 relevancy and q22 faithfulness are self-contradictions; q12 follows a retrieval miss. q10 got
  relevancy 1.0 though it answered a different phase range: the metric checks topic, not correctness.
- n = 12: one question moves a mean by about 0.08. llama.cpp was not fully deterministic at temperature 0 (q11
  re-asked gave different text); the judge scores the stored answer.
- Cache effect: 27 of 30 prompts had `cached_tokens = prompt_tokens - 1`; the 3 new questions (10 to 12) had TTFT 21.72,
  26.7, 34.54 s; all 30 nearest-rank p95 TTFT 26.7 s (`observability/SLOs.md`). The superseded mostly cold run had
  latency mean 28.967 s (`rag/eval/results/20261008T095250Z/summary.md`) and TTFT median 22.8 s, p95 41.08 s, max
  59.09 s (`observability/SLOs.md`).
- First request after a restart: `POST /chat` 109047.1 ms (`docs/evidence/phase3-rag.txt` section 5).
- Ingest: 9 documents, 3129 chunks in 1115 s.

### 7.13 Observability results

**In plain English.** The alarm rules were tested by planting bugs, and every bug was caught. During testing a real
failure happened (chats sent while the index was rebuilt), and the right alarms went off. The replayed GPU dashboard
matched the raw files exactly. Langfuse costs about 2.5 GB of memory to switch on.

**Technically.**

| Item | Value | Source |
|------|-------|--------|
| Recording rules, alert rules | 18, 5 | `docs/evidence/phase4-observability.txt` sections 2, 7 |
| Test groups | 34 (`slo_test.yml`) + 3 (`slo_windows_test.yml`) | same |
| promtool test time | 77 s (one file), 83 s (both) | same; `README.md` |
| Mutants killed | 9 of 9, then 17 of 17 | sections 2b, 7b |
| Live rule health after reload | 23 rules `ok` | sections 6, 7c |
| Real alert | `RagAvailabilityErrorBudgetBurnFast` firing from 11:24:01Z, value 0.3825, after 5 in-stream failures during a re-index | section 4 |
| Budget remaining at that time | -37.25 (Prometheus had under an hour of data) | section 4 |
| Tempo search | 17 `rag.chat` traces in 30 min; one matched the SSE `session` id | section 5 |
| Chat 1 with Langfuse | 5 spans (one batch lost to `DEADLINE_EXCEEDED` after a collector recreate); `rag.generate` 19.741 s of 20.949 s | `docs/evidence/v3-loki-langfuse.txt` section 3 |
| Loki services | 16; one-shot `model-init`, `rag-migrate` missing | same, section 2 |
| Langfuse RAM (6 containers) | 2565.6, 2541.6, 2618.1 MiB in three samples | same, section 4 |
| Loki, Alloy RAM | about 0.2 GiB, about 0.4 GiB | same |
| GPU replay | 24 files, 2265 samples, 13590 points, 0 mismatches vs raw | `docs/evidence/phase4-observability.txt` section 8 |
| Power samples over 70 W | 100 of 2265, max 81.82 W (instantaneous) | same |
| Not measured | SLO attainment over 30 days; Tempo or Prometheus RAM; `backfill.sh` time | `README.md` |

### 7.14 Platform and CI results

**In plain English.** The cluster comes up in under half an hour when images are already on the laptop, and in about a
minute when nothing changed. Autoscaling asked for 3 copies within seconds of load, had all 3 ready about half a minute
after load started, and was back to 1 within a minute of quiet. The CI pipeline takes about 40 minutes when it has to rebuild and re-index, and under 4 minutes when nothing
changed. No critical vulnerabilities were found in the shipped image.

**Technically.**

kind bring-up (`deploy/README.md`, `docs/evidence/phase5a-platform.txt`, `phase5b-ci.txt`):

| Run | Time | Why |
|---|---|---|
| Clean runs before the network fixes | 34m25s, 39m05s, 46m28s | model download, ingest 9m52s to 12m46s |
| Proof run on bea3e53 | failed at 716 s, then 42m21s | slow image pulls hit 5-minute waits |
| Clean run after fixes (a290dea) | 28m46s, first attempt | all 12 images already on the host |
| Re-runs, nothing changed | 54 s, 38 s, 36 s, 60 s, 64 s | cached; nothing rolls |
| Re-run with a new image | 10m59s | ingest rebuild 9m43s |
| Recovery from ProgressDeadlineExceeded | 442 s | ghcr.io blocked for 2 minutes |
| Compose up, cached / cold | 773 s / 3994 s | `docs/evidence/phase1-smoke.txt` |

KEDA (`README.md`; `docs/evidence/phase5a-platform.txt` section 6; `docs/evidence/phase5a-review/clean-run2-keda.log`):

| Run | Load start | Scale up | Load end | Back to 1 |
|-----|------------|----------|----------|-----------|
| 1 | 06:05:04Z | size 3 at 06:05:06Z (2 s, derived) | 06:11:28Z | 06:12:22Z (54 s after, derived) |
| 2 | 07:04:20Z | size 2 at 07:04:38Z, size 3 at 07:05:38Z (78 s, derived) | 07:11:34Z | 07:12:25Z (51 s after, derived) |

60/60 requests ok in run 1; tokens per pod 3352, 2576, 2575 (load spread). Demo 2 stepped through 2 because sampled
in-flight was 4 then 2 for about 30 s; why it was below 6 is not explained. A demo, not a benchmark.

CI (`docs/evidence/phase5b-ci.txt`, `docs/evidence/v3-github-actions.txt`):

| Run | Total | Notable stages |
|---|---|---|
| Jenkins #1 | 2471 s | test 448 s, build 595 s, argocd sync 1243 s |
| Jenkins #2 | 2452 s | charts 296 s (slow schema downloads), build 1328 s (cache miss), sync 659 s |
| Jenkins #3 | 220 s | build 8 s, sync 3 s (same digest) |
| GitHub Actions run 37970874178 | lint 91 s, test 138 s, image 175 s | Trivy DB downloaded at run time |

Trivy (`docs/evidence/phase5a-platform.txt` section 2): rag on python 3.12.13 had 3 CRITICAL (perl-base) and 91 HIGH
(47 fixable); on 3.12.15, 0 CRITICAL and 44 HIGH, none fixable, Python packages 0; llama.cpp image 0 and 0. Jenkins #1
to #3 and the GitHub run: 0 CRITICAL. Results depend on the DB date.

OKD (`docs/evidence/v3-openshift-okd.txt`): first `crc start` 9m48.853s; root Application to all Synced/Healthy 38m57s
(ingest "3129 chunks in 992s"); `okd-up.sh` re-runs 1m52s and 117 s; chat through the Route 24.9 s then 3.9 s; node
requests reached cpu 4731m (81%) and memory 14722Mi (94%) of allocatable.

---

## 8. Status, honestly

**In plain English.** A report card. "Done" means the measurement exists and was checked. "Partial" means some of it
works and the rest is written down as failed or untested. "Planned" means on the stretch list with no work yet. "Not
run" means no number exists, with the reason.

**Technically.** States as committed at `782b343`, judged from the committed raw results; the status docs were
brought in line in the same docs update as this section (see [8.2](#82-docs-that-lag-the-results)).

### 8.1 Every part, its state, its evidence

In plain English: one row per part of the project, with its grade and the file that proves it.

| Part | State | Evidence |
|------|-------|----------|
| Compose stack + smoke test | done | `docs/evidence/phase1-smoke.txt` |
| llama.cpp CPU benchmark (c1/c4/c8, 3 cold + 3 warm) | done | `docs/evidence/phase2-cpu-rerun.txt`, `benchmark/results/raw/cpu-20261009/` |
| vLLM on T4: 1.5B FP16/AWQ c1/c8/c32, 7B AWQ c8, TP=2 c32 | done (run v2) | `docs/evidence/phase2-gpu.txt`, `benchmark/results/raw/kaggle-20261009/` |
| GSM8K FP16 vs AWQ (250 items) | done | same |
| Same-session vLLM controls (v3d, v3f, v3h, v3i) | done (3 sessions; raw READMEs say "Subject to review") | `benchmark/RESULTS.md` |
| RAG API, ingest, chat, history, traces | done | `docs/evidence/phase3-rag.txt` |
| RAG eval: recall (n = 30), judge (n = 12) | done, small n, local judge | `rag/eval/results/20261008T113955Z/` |
| SLOs, burn-rate alerts, promtool tests | done | `docs/evidence/phase4-observability.txt` |
| GPU replay dashboard | done (replay, not live) | same |
| kind + ArgoCD + Helm + KEDA demo | done | `docs/evidence/phase5a-platform.txt`, `docs/evidence/phase5a-review/` |
| Jenkins pipeline, all stages | done, builds #1 to #3 green | `docs/evidence/phase5b-ci.txt` |
| GitHub Actions (lint, charts, test, build, scan) | done, first run green | `docs/evidence/v3-github-actions.txt` |
| OpenShift charts on OKD 4.22 (OpenShift Local) | partial: single node; not Red Hat OpenShift; KEDA not load-tested; `okd-up.sh` not run end to end fresh | `docs/evidence/v3-openshift-okd.txt` |
| Loki + Alloy logs | done (compose only) | `docs/evidence/v3-loki-langfuse.txt` |
| Langfuse (profile) | partial: receives spans, but as generic SPANs, so LLM views stay empty | same; `docs/LIMITATIONS.md` |
| DCGM on Kaggle | done as an experiment: basic fields yes, profiling fields `DCGMError_RequiresRoot` | `benchmark/results/raw/kaggle-20261010-v3a/README.md` |
| Vision CPU baseline (ONNX Runtime) | done | `docs/evidence/v3-vision-cpu.txt`, `benchmark/results/raw/cpu-vision-20261010/` |
| Vision T4 speed (ORT CUDA FP32, TensorRT FP16) | done (reviewer PASS per `docs/STATUS.md`) | `benchmark/results/raw/kaggle-20261010-v3c/` |
| Vision via Triton | partial: PyTriton Python backend running the TensorRT engine, not Triton's native TensorRT backend | `benchmark/results/raw/kaggle-20261010-v3e/` |
| Vision mAP (500-image COCO subset) | done (reviewer PASS per `docs/STATUS.md`) | `benchmark/results/raw/kaggle-20261010-v3g/` |
| TensorRT-LLM FP16 on T4 | done (reviewer PASS per `docs/STATUS.md`): two engines, same-session vLLM control, 18 runs, GSM8K; one session, one T4 | `docs/evidence/v3-trtllm.txt`, `benchmark/results/raw/kaggle-20261010-v3i/` |
| Triton TensorRT-LLM backend | not run: needs the NGC container, so Docker; Kaggle has none | `NOT_RUN-triton-tensorrtllm-backend.txt` in v3b, v3d, v3f, v3i |
| Alertmanager, alert delivery | not set up | `observability/SLOs.md` |
| Traces in kind | not available (no collector; `trace_id` zeros) | `deploy/README.md` |
| Makefile | untested (no make on the dev machine) | `docs/STATUS.md` |
| Bitbucket Pipelines | planned (stretch), not written | `docs/PLAN.md` |
| OpenShift Developer Sandbox test | planned (stretch) | `docs/LIMITATIONS.md` |
| llama.cpp Vulkan on the AMD iGPU | planned (stretch), not run | `docs/PLAN.md` |

`docs/STATUS.md` records reviewer PASS for v3 items 11 and 12 and the DCGM answer for item 10 (updated in this docs sync); the raw READMEs still say "Subject to review" because raw output is kept byte-exact.

### 8.2 Docs that lag the results

**In plain English.** Some notice boards had fallen behind the experiments. They were brought up to date in the same
docs update that added the TensorRT-LLM results here, so this list records what was fixed.

**Technically.** Fixed in that update (not code, docs only):
- `docs/STATUS.md`: v3 rows 10 to 12 now carry the results and evidence; the sampler-fields deviation is marked
  resolved (the fields returned samples in runs v3c to v3i).
- `README.md`: TensorRT-LLM and vision headline rows added; "Not run" now lists only the Triton native backends, Red
  Hat OpenShift and Bitbucket; the clock line uses the RESULTS.md median-based figure (56%, 575 vs median 1030 MHz)
  instead of 53%.
- `docs/LIMITATIONS.md`: vision, TensorRT-LLM and DCGM wording, the sampler-fields line removed, the clock confound
  updated.
- `inference/README.md`: header, the TensorRT-LLM "Build:" bullet (two engines), the vLLM control order (it runs
  first), and a v3i result paragraph.
- `docs/VERSIONS.md`: hadolint lines; model licences with Hugging Face API URLs.
- `observability/SLOs.md` and `docs/DECISIONS.md`: "Kaggle has no DCGM" replaced with "basic fields work, profiling
  needs root (run v3a)".
- `deploy/README.md` and `docs/DECISIONS.md`: "no git remote" now says ArgoCD reads the local git daemon and the code is
  also on a private GitHub repo used only by GitHub Actions.

Remaining known lag: none found at `782b343` plus this update.

### 8.3 How to run each part locally

**In plain English.** A menu of "how do I start this", with how long it took on the owner's laptop and whether a GPU
is needed.

**Technically.** Dev machine: Windows 11, AMD Ryzen 7 7730U (16 logical CPUs), Docker Desktop WSL2, Git Bash.
Prerequisites: Docker with Compose v2, Git Bash on Windows, `cp .env.example .env`; Python tools use `uv` and Python
3.12.

| Part | Command(s) | Measured time | Memory evidence | GPU? |
|------|------------|---------------|-----------------|------|
| Compose stack + smoke | `docker compose up -d --wait && bash scripts/smoke.sh` | 773 s cached, 3994 s cold | one snapshot: llamacpp 2.885 GiB, elasticsearch 1.944 GiB, rag-api 503.2 MiB, grafana 267.3 MiB, smaller others (`phase2-cpu.txt`); Loki about 0.2 GiB, Alloy about 0.4 GiB | no |
| RAG ingest | `docker compose run --rm -e INGEST_SKIP_IF_PRESENT=0 rag-ingest` | 1115 s | not recorded | no |
| Chat | `curl -N http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message": "..."}'` | first after restart 109 s; cold TTFT 21.7 to 34.5 s | as stack | no |
| RAG eval + judge | `docker compose --profile eval up -d --wait llamacpp-judge`, then `rag/eval/run_eval.py`, `judge_run.py` | run 11:39:55 to 12:03:16; judge 12:09:52 to 13:07:15 | judge frees about 5 GB when stopped | no |
| Langfuse | fill `.env` Langfuse values; `docker compose --profile langfuse up -d --wait` | 38.5 s (second attempt) | about 2.5 GiB | no |
| Tests and lint | pytest for `benchmark/tests` and `rag` (with `TEST_PG_URL`); promtool `test rules`; `uvx --python 3.12 pre-commit@4.6.2 run --all-files` | promtool 83 s | small | no |
| CPU LLM benchmark | driver script in `docs/evidence/phase2-cpu-rerun.txt`, then `benchmark/report.py` | 2 h 22 min 31 s | as llamacpp | no (quiet machine) |
| GPU replay | `observability/gpu-replay/backfill.sh` | not recorded | small | no |
| kind cluster | `bash scripts/kind-up.sh`; `bash scripts/kind-down.sh` | 28m46s clean with images cached; re-run about 60 s | not recorded as a total | no |
| KEDA demo | `bash scripts/keda-demo.sh 6 60` | 06:05:03Z to 06:12:39Z | as kind | no |
| Jenkins | `cp ci/jenkins/.env.example ci/jenkins/.env`; `docker compose -f ci/jenkins/docker-compose.yml --env-file ci/jenkins/.env up -d --build --wait`; needs kind-up first | controller 3m36s; builds 2471, 2452, 220 s | not recorded | no |
| OKD | install crc 2.64.0 (admin once), then `bash scripts/okd-up.sh` with kind stopped | first boot 9m48s; full sync 38m57s | crc 16384 MiB, 6 CPUs, 60 GB disk | no |
| Vision CPU baseline | `benchmark/vision/cpu_baseline.sh <onnx from Kaggle output> "<hardware>"` | 14:01:01Z to 14:12:54Z | not recorded | no, but needs the ONNX file from a Kaggle run |

### 8.4 GPU needs

**In plain English.** Only three things need a GPU, and all three run on free Kaggle T4s with the owner's approval per
session.

**Technically.**

| Part | Command | Measured or planned time | Hardware |
|------|---------|--------------------------|----------|
| vLLM benchmark + GSM8K | `python notebooks/build_notebook.py`, `kaggle kernels push -p notebooks -t 14400`, `kaggle kernels output <kaggle-user>/vllm-t4-benchmark -p <raw dir>`, `benchmark/report.py` | 39 min 17 s session (pre-run estimate 1.6 h) | 2x T4 |
| DCGM + vision | `python notebooks/build_v3.py`, `kaggle kernels push -p notebooks/kaggle-dcgm-vision -t 3600` | v3c 09:30 to 09:49Z; v3g about 12:46 to 13:01Z | 1x T4 |
| TensorRT-LLM + vLLM control | `python notebooks/build_v3.py`, `kaggle kernels push -p notebooks/kaggle-trtllm -t 10800` | v3i 15:50:24Z to 16:28:31Z (38 min), complete; v3h ran 14:20 to 14:45Z before failing at the build | 1x T4 |

Kaggle rules: the account must be phone verified or the session gets no GPU and no internet (run v1); quota about 30
GPU-h per week with 12 h sessions (`docs/PLAN.md`); each push needs the owner's approval; commit before building the
notebook so the embedded commit is clean; delete the embedded `bench/` copy after download. Colab free T4 is allowed by
the rules but not used.

---

## 9. Things not fully understood

**In plain English.** The honest list of loose ends: things the repo does not explain, did not measure, or where two
files disagree. Each is one line.

**Technically.** Consolidated and deduplicated from the open-question lists of all five drafts. Items the drafts
asked about that this guide could settle are marked "answered here" with where. Stale-doc items are listed once in
[8.2](#82-docs-that-lag-the-results) and once here as a single fix item.

**Process and docs**
1. The status docs were brought in line in an uncommitted docs update ([8.2](#82-docs-that-lag-the-results)); it still needs to be committed.
2. Run v3i's end-of-session SW Power Capping counter (22,301,052 us, 22.3 s) disagrees with the 91 to 100% capped samples over about 25 min of load; not reconciled (`docs/evidence/v3-trtllm.txt`).
3. Reviewer PASS for v3 items 11 and 12 is recorded only in `docs/STATUS.md`; no evidence file holds the vision review itself.
4. Total RAM for the full compose stack, kind and the 7B judge is not recorded; only per-container snapshots exist.
5. Times for `report.py`, the GPU replay backfill and the vision CPU venv setup are not recorded.
6. No log records how many review rounds each item took beyond STATUS notes.
7. Parameter counts are not in `docs/VERSIONS.md` (licences now are, from the Hugging Face API; counts are only in section 4 of this guide).
8. The image id recorded by the RAG eval run (`sha256:d661f3f5...`) differs from the judge run's (`sha256:483e3833...`) and the reviewed tree's (`sha256:a69db154...`); retrieval was checked equal on re-ask, but no clean run on the reviewed image exists.

**Inference and benchmarks**
9. Why FP16 c1 settles at a lower SM clock (450 to 715 MHz) than c8 and c32 (near 1000 MHz) when raw samples show the SW power cap in about 100% of samples for all three.
10. The v3h end-of-session "SW Power Capping" counter (3,677,884 us) is far smaller than v3d (524,242,817 us) and v3f (321,000,792 us) despite about 100% capped samples (`results/env/nvidia-smi-q-end.txt` in each session folder).
11. The McNemar p-values (0.105, 0.068) are in the evidence but the script that computed them is not committed.
12. TPOT p50 above ITL p50 at high concurrency is not analysed (prefill of joining requests is a likely reading, not a finding).
13. Why TensorRT-LLM is slower at c32 (613.8 vs 732.5 tok/s, TTFT p50 2.860 vs 1.109 s) is not proven: unfused context attention on SM75 is the likely cause, a lower SM clock (850 vs 1105 MHz) a possible one.
14. How much of the v2 vs v3 GSM8K gap is repetition_penalty and how much is session noise is not quantified.
15. Why the TensorRT b32 throughput drops is "likely, not proven" (FP32 input copy from pageable memory); no pinned-memory or FP16-input test.
16. Why CPU vision throughput falls with batch size is not investigated.
17. Unexplained host activity during runs: an 83% CPU burst with a `uv` process, and vmmemwsl bursts during the vision CPU run.
18. Why uv, httpx and matplotlib were chosen over alternatives is not stated.
19. No open-loop (fixed arrival rate) load test was run, and the repo does not say whether one was considered.

**RAG**
20. No metric compares answers with the stored `reference` answers; faithfulness and relevancy missed wrong answers (q1, q10).
21. Only 12 of 30 questions are judged; judging all 30 costs up to 297.8 s per question per metric on CPU.
22. Generator and judge are both Qwen2.5; possible self-preference bias is not discussed.
23. Hybrid retrieval and reranking gains are not measured (no BM25-only or dense-only ablation).
24. A fully cold eval on the current 9-document corpus is not recorded (27 of 30 prompts were cache hits).
25. The env var `HISTORY_MESSAGES` counts messages while the code name is `HISTORY_TURNS`.
26. A failed generation leaves an unanswered user message in history, which feeds the next prompt; intended or not is not stated.
27. An unknown `session_id` creates a session with that id; combined with no access control, intended behaviour is not stated.
28. The dense retrieval similarity function relies on the integration default; whether to pin it is open.

**Observability**
29. Where the two `search` spans in chat 2 come from is not stated (likely the Elasticsearch client, unverified).
30. Tempo retention is not set; its default is not stated in the repo.
31. Logs keep 7 days and metrics 31, so a 30-day SLO investigation cannot reach early logs or possibly traces.
32. The GPU replay blocks age out around 2026-11-09; after that the dashboard is empty until `backfill.sh` is re-run.
33. Page rules have no minimum request count, so one failed chat can page at dev traffic.
34. No Alertmanager: whether alert delivery is future work or the intended end state is not stated.
35. A 40 or 45 s TTFT bucket would allow a tighter threshold; not done.
36. `rag_http_request_duration_seconds` (time to headers) is unused by any panel and says little for SSE.
37. Langfuse LLM views are empty because spans carry no model, token or cost attributes.
38. `scripts/smoke.sh` does not check the rag-api scrape job, rule health or dashboards, and its header still says "Phase 1".
39. The first chat after recreating the collector lost one span batch (`DEADLINE_EXCEEDED`); no retry or ordering fix.
40. Langfuse secrets default to empty (`:-`) unlike the core services (`:?`); starting the profile without values was not tested.
41. Grafana (compose sets no `GF_ANALYTICS_*` opt-out) and Alloy (runs without `--disable-reporting`) send usage reporting by default; not disabled, which matters for data residency and air gaps.
42. The 60 s threshold rests on client-side TTFT while the SLI is server-side; the gap was not measured.

**Platform and CI**
43. Why kind rather than k3d, and why Traefik, is not stated (k3d would be closer to the owner's K3s).
44. `--provenance=false` drops build provenance and no SBOM is produced; whether that trade-off is accepted is not stated.
45. KEDA `cooldownPeriod: 60` likely has no effect with `minReplicaCount: 1`; not discussed or verified in the repo.
46. KEDA demo 2 stepped 1, 2, 3 with in-flight samples of 4 and 2 under 6 clients; not explained.
47. On OKD, wave 1 started 2 s after wave 0 while wave 0 was still Progressing; not investigated.
48. On OKD the second chat through the Route took 3.9 s against 24.9 s for the first; not explained in the evidence.
49. A slow first pull after the kind-up network fixes has not been repeated (the one post-fix clean run had images cached).
50. Digest equality on a second machine (without the shared BuildKit cache) is untested.
51. The re-index outage (about 9 minutes with no index) has a known fix (alias swap) that is not built.
52. Jenkins builds can read the admin password file; a separate agent would fix it, not done.
53. ArgoCD pods use `imagePullPolicy: Always`, which contacts registries despite preloading.
54. Supply chain pinning is partial: ArgoCD `install.yaml`, KEDA and Traefik charts, registry, alpine and Trivy by tag or version only.
55. A failing Trivy scan and the `.trivyignore` path were never exercised.
56. The Makefile was never run (no make on the dev machine).
57. The cluster has no Grafana, Tempo or collector, so `trace_id` is zeros in kind; whether a cluster observability chart is in scope is not stated.

Total: 57 items.
