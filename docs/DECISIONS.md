# Design decisions

Each entry: what was chosen, why, and what was rejected. Sources in brackets.

## Scope and engines

**Slim scope v2.** v1 had 9 phases and about 19 GPU hours; v2 has 6 core phases and about 8 GPU hours. Cut to
"Stretch": TensorRT-LLM + Triton, YOLOv8n vision, Bitbucket pipelines, Loki, Langfuse, the OpenShift Sandbox test and
llama.cpp Vulkan on the iGPU. Reason: the free Kaggle quota (about 30 GPU-h/week) and finishing the core first. The
GPU session actually used took 39 min 17 s. [docs/PLAN.md; docs/evidence/phase2-gpu.txt]

**llama.cpp on CPU plus vLLM on T4, not TensorRT-LLM.** TensorRT-LLM needs long engine builds and does not support
INT4 AWQ/GPTQ on T4 (sm75), so the quantisation story would be weak there.
vLLM runs AWQ on T4 with the Marlin kernel (seen in both AWQ server logs). [docs/PLAN.md; docs/evidence/phase2-gpu.txt]

**vLLM 0.31.0, `+cu129` wheel, `TRITON_ATTN`.** 0.31.0 still builds kernels for compute capability 7.5. The plain
PyPI wheel is a CUDA 13 build that needs an R580+ driver, unknown before the session; the `+cu129` wheel runs on
R525+. FlashInfer is marked broken on SM75 and FlashAttention 2 needs sm80, so the attention backend is pinned to
`TRITON_ATTN` rather than left to auto-selection. Run v2 got driver 580.178.04, so the CUDA 13 wheel would also have
worked that day. [inference/README.md; docs/evidence/phase2-prep.txt; docs/evidence/phase2-gpu.txt]

**Kaggle notebook with embedded code.** `notebooks/build_notebook.py` writes the benchmark code into the notebook with
a sha256 per file, so no Kaggle dataset or git remote is needed and the code that ran is checked byte for byte
against the repo. Not used: a Kaggle dataset, or cloning a git remote (there was none then; the private GitHub repo came later; any push to a remote needs
approval). A preflight cell stops the run early: run v1 failed on no GPU and no internet (phone verification).
[inference/README.md; docs/evidence/phase2-prep.txt]

## Measurement

**Cache-aware CPU reporting.** llama.cpp keeps a host-memory prompt cache, so repeats of the 64 prompts were 96%
cached and their TTFT measured cache hits, not prefill. The rerun does 3 cold repeats (server restarted before each,
10 to 26% cached) and 3 warm repeats per concurrency, and `report.py` splits rows at a 50% cache share. Rejected:
averaging all repeats, which mixes two regimes. vLLM runs with prefix caching off instead, so it is not split.
[docs/evidence/phase2-cpu-rerun.txt; benchmark/RESULTS.md]

**Fixed 128-token decode.** Every request uses `ignore_eos` and `max_tokens` 128, so throughput is comparable across
precisions and engines. The cost: numbers do not reflect natural answer lengths. [inference/README.md]

**Raw evidence kept byte-exact.** A pre-commit pass once rewrote whitespace in the downloaded Kaggle files; they were
re-downloaded byte-exact. Now a global `exclude` in `.pre-commit-config.yaml` keeps every hook off
`benchmark/results/raw/kaggle-*` and `docs/evidence/phase5b-ci/` (gitleaks still runs: `always_run`), and
`.gitattributes` marks both `-text`. What each evidence set is: Kaggle raw output and the phase5b-ci logs are
byte-exact, the CI logs apart from Jenkins' hidden console notes, which were removed. `phase5a-platform.txt` strips
non-ASCII glyphs from kind's progress output and applies the filters it names. The `phase5a-review/` logs had
trailing spaces removed and were converted to LF. [.pre-commit-config.yaml; .gitattributes; docs/evidence/phase2-gpu.txt;
docs/evidence/phase5b-ci.txt; docs/evidence/phase5a-platform.txt; docs/evidence/phase5a-review/README.txt]

## RAG

**Corpus licence rule.** Only NIST documents by federal staff or with an explicit public domain statement, plus a
NASA handbook marked `PUBLIC_USE_PERMITTED`. NIST TN 2148 was dropped on 2026-10-08 (no copyright statement, DOE
contract funding, federal authorship not confirmed); SP 330, AMS 100-1, AMS 100-8, NISTIR 8183r1 and AMS 300-8 were
rejected too. The eval questions for TN 2148 were replaced. [rag/data/SOURCES.md; docs/evidence/phase3-rag.txt]

**Hybrid retrieval with a reranker.** BM25 top 10 plus bge-small-en-v1.5 dense top 10, joined with RRF, reranked by
ms-marco-MiniLM-L6-v2 to the top 5. Result: recall@5 0.867, doc recall@5 1.000 (n = 30). The cost is latency on CPU:
the rerank span took 14924.1 ms on the first request after a restart. BM25-only and dense-only were not evaluated, so
the gain over them is not measured. [rag/eval/results/20261008T113955Z/config.json; docs/evidence/phase3-rag.txt]

**Local 7B judge.** Faithfulness and answer relevancy are scored by Qwen2.5-7B-Instruct Q4_K_M on llama.cpp with
JSON-schema output, because paid APIs are excluded. It is much weaker than the hosted judges DeepEval is tuned for,
so it is used to compare runs of this pipeline. The judged sample is 12 of 30 questions, covering all 9 documents.
[CLAUDE.md; rag/eval/README.md]

## Observability

**SLO targets.** Availability 99%: one replica of everything and no failover, so tighter needs a second replica first.
TTFT 95% within 60 s: the threshold must be a histogram bucket, 30 s missed the cold runs (77% of mostly cold
requests), and 60 s held for all 33 cold requests measured. Set from cold measurements because real users send new
questions. [observability/SLOs.md; docs/evidence/phase4-observability.txt]

**GPU dashboard as a replay.** On Kaggle DCGM's profiling fields need root (basic fields work, run v3a), and the GPU run happens outside this stack, so the nvidia-smi samples (every
500 ms) in the raw files are converted to OpenMetrics and backfilled with `promtool tsdb create-blocks-from openmetrics`.
The dashboard is labelled "not live" and no SLO uses it. The Grafana table matched the raw samples with 0 mismatches.
[docs/evidence/phase4-observability.txt; observability/SLOs.md]

## Platform and CI

**Hand-written `platform-deps` chart.** Single-instance Postgres, Elasticsearch and a minimal Prometheus for KEDA.
Rejected: Bitnami charts (images no longer free) and the official Elasticsearch chart (deprecated; ECK is too heavy
here). Dev only: no HA, no backups. [deploy/README.md]

**ArgoCD reads a local git daemon, `GIT_SOURCE=head` by default.** ArgoCD reads no git remote (the code is also on a private GitHub
repo, used only by GitHub Actions) and pushing to one needs approval, so `kind-up.sh` publishes the committed HEAD to a bare repo served by `git daemon` on the kind network.
`GIT_SOURCE=worktree` publishes a synthetic commit of the working tree; Phase 5a evidence used it before the commit.
The trade-off: ArgoCD sees only what was last published, with no review flow and no auth. [deploy/README.md;
docs/PLAN.md; docs/evidence/phase5a-platform.txt]

**Deploy by digest, build with `--provenance=false`.** The tag `0.1.0` is mutable, so a rebuilt image did not roll out;
the pushed digest is now passed to the rag-api Application. With the default provenance attestation, every build got
a new digest even when fully cached, which would roll rag-api and rebuild the index on every re-run. The ingest Job
uses `Force=true,Replace=true` because a Job template is immutable. [docs/evidence/phase5a-platform.txt, section 8]

**Build the rag image from `git archive`.** Building from the working tree gave a different digest on the Windows host
(every file mode 0755) than in the Jenkins Linux checkout (0644), so each CI deploy after a kind-up deploy rolled
rag-api and rebuilt the index. kind-up and `ci/run.sh` now build from `git -c core.autocrlf=false archive
--mtime=1970-01-01T00:00:00Z`. The first archive version still differed: Git for Windows converted `rag/` text to CRLF
(BuildKit's cache key ignores mtimes, so CRLF was the remaining difference). Both now produce the same digest
(rag@sha256:6f10a3e7...). Equality still relies on the shared BuildKit cache; the RUN steps are not bit-reproducible
alone. [docs/evidence/phase5b-ci.txt; scripts/kind-up.sh; ci/run.sh]

**Pre-pull images on the host and import them into the kind node.** Pulls inside the node on a slow network ran into
the 600 s Deployment progress deadline, and `helm --wait` and `kubectl rollout status` give up there. kind-up now
pulls every image on the host, imports it with `docker save | ctr import` (`kind load docker-image` failed on Docker
Desktop's containerd image store), drops `helm --wait` and retries rollout waits past the deadline. Tested by
recovering a ProgressDeadlineExceeded KEDA install (442 s). [docs/evidence/phase5b-ci.txt]

**Jenkins builds committed `main` and makes a CI-only bump commit.** The job clones the repo read-only, sets the
pushed digest in `deploy/helm/rag-api/Chart.yaml` in a commit inside the workspace, and publishes that commit to
kind-git. It never touches your branch; the next `kind-up.sh` publishes your HEAD again. [ci/jenkins/README.md; docs/evidence/phase5b-ci.txt]
