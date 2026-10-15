# Limitations

What this repo does not show, and where its numbers are weak. Sources in brackets.

## Not run or not tested

- **OpenShift:** `values-openshift.yaml` (Routes, no fixed UID, RWX model PVC) is tested on OKD 4.22 via OpenShift
  Local (single node, through the ArgoCD app-of-apps); not tested on Red Hat OpenShift, multi-node or with a real RWX
  storage class, and KEDA scaling under load was not run there. The first deployment was done step by step;
  `scripts/okd-up.sh` was not run end to end on a fresh machine (only re-runs on the running cluster).
  [deploy/README.md; docs/evidence/v3-openshift-okd.txt]
- **GitHub Actions:** runs on the private GitHub repo; the first run (lint, charts, test, build, Trivy) passed. Push, chart bump and ArgoCD sync are skipped there by design: a hosted runner cannot reach the local cluster. [docs/evidence/v3-github-actions.txt]
- **Bitbucket Pipelines:** not written (stretch). [docs/PLAN.md]
- **TensorRT-LLM:** run on one T4 against a same-session vLLM control (run v3i): one session, 3 repeats, fixed
  128-token decode, served by TRT-LLM 0.14's patched OpenAI example server. Triton's TensorRT-LLM backend not run: it
  needs the NGC container (Docker), which Kaggle lacks. [docs/evidence/v3-trtllm.txt; benchmark/RESULTS.md]
- **Vision (YOLOv8n):** speed (ONNX Runtime CPU and CUDA, TensorRT FP16) and mAP on 500 COCO images were run. "Triton"
  rows are PyTriton's Python backend over HTTP, not Triton's native TensorRT backend (not run: needs Docker).
  [benchmark/RESULTS.md; docs/evidence/v3-vision-cpu.txt]
- **Loki, Langfuse:** compose only, not in the Helm charts. rag-api logs `trace_id=<32 hex>` per chat, which
  Grafana links from Loki to Tempo. Langfuse (profile `langfuse`) gets the same OTel spans as Tempo, all as
  generic SPAN observations (no model, token or cost fields), so its LLM views stay empty. One-shot containers (model-init, rag-migrate) finish before Alloy's Docker
  discovery refreshes, so their logs are not in Loki. The first chat after the collector is recreated can lose one span
  batch (export DEADLINE_EXCEEDED in rag-api). [docs/evidence/v3-loki-langfuse.txt]
- **DCGM profiling:** DCGM installs on Kaggle and its basic fields return values, but every profiling field (SM
  active, DRAM active and others) fails with `DCGMError_RequiresRoot`, so GPU metrics are nvidia-smi samples every
  500 ms, replayed, not live. [benchmark/results/raw/kaggle-20261010-v3a/README.md; observability/SLOs.md]
- **CI failure paths:** a failing Trivy scan and the `.trivyignore` path were not exercised; `ci/run.sh` ran only
  inside Jenkins, not directly on a Linux host. [docs/evidence/phase5b-ci.txt]
- **Makefile:** untested (no make on the dev machine). [docs/STATUS.md]
- **Alert delivery:** no Alertmanager, so alerts show in Prometheus and Grafana only. [observability/SLOs.md]
- **Kaggle cap kill:** the session took about 39 min, so whether `/kaggle/working` survives the 4 h cap is untested. [inference/README.md]

## Measurement limits

- **The laptop is not a dedicated machine.** Desktop apps stayed open during the CPU rerun. One burst reached 83% host
  CPU with an unexplained `uv` process (it fell in a discarded warming pass); reported runs averaged 36 to 43% host
  CPU, llama.cpp included. [docs/evidence/phase2-cpu-rerun.txt]
- **Small n.** 3 repeats per benchmark cell; GSM8K on 250 items (both AWQ minus FP16 CIs include 0); RAG recall on 30
  questions; the judge on 12, where one question moves a mean by about 0.08; the TTFT SLO threshold rests on 33 cold
  requests. [benchmark/RESULTS.md; rag/eval/README.md; observability/SLOs.md]
- **Judge reliability.** A 7B Q4 judge, no human calibration. It contradicted itself on 2 of 24 scores and misread
  others (q10, q7). llama.cpp was not fully deterministic at temperature 0 (q11 re-asked gave different text).
  [rag/eval/README.md]
- **Prompt cache effects.** llama.cpp serves repeated prompts from cache. The RAG eval run had 27 of 30 prompts as cache
  hits, so its latency (median 4.9 s) is not a cold number; cold TTFT there was 21.7 to 34.5 s. CPU benchmark rows are
  split warm and cold for this reason. [rag/eval/results/20261008T113955Z/summary.md; benchmark/RESULTS.md]
- **GPU clock confound.** The run v2 FP16 c1 cell ran at about 56% of the SM clock of the other GPU cells (p50 575
  MHz vs median 1030 MHz; cause not recorded: no clock event reasons were sampled during run v2). Every run v2 GPU run
  averaged 64 to 67 W against the 70 W cap. Later sessions sampled clock event reasons: the SW power cap was active in
  91 to 100% of samples of every v3i cell, both engines; in v3i the vLLM c1 cell ran at 642.5 MHz against 1040.0 MHz
  for TensorRT-LLM c1, so part of that c1 gap may be clock. [benchmark/RESULTS.md; docs/evidence/phase2-gpu.txt;
  docs/evidence/v3-trtllm.txt]
- **repetition_penalty in run v2.** vLLM applied the model's `generation_config.json` (repetition_penalty 1.1 for 1.5B,
  1.05 for 7B) to every request, including GSM8K. The FP16 vs AWQ delta is like for like; absolute GSM8K scores are not
  stock settings. strict-match also measures format: 113 (FP16) and 123 (AWQ) of 250 responses had no `#### N` answer.
  [docs/evidence/phase2-gpu.txt; benchmark/RESULTS.md]
- **CPU and GPU are not comparable.** Different hardware, weights, servers, concurrency levels and cache regimes. No
  CPU vs GPU ratio is reported. [benchmark/RESULTS.md]
- **Fixed 128-token decode.** Benchmarks force 128 output tokens with `ignore_eos`; real answer lengths differ.
  [inference/README.md]
- **RAG eval provenance.** The eval run records `git_describe 2e92380-dirty`: the rag tree was uncommitted when it ran.
  Phase 3 and 4 evidence was recorded on the `python:3.12.13` base before the bump to 3.12.15. [rag/eval/results/20261008T113955Z/config.json; docs/VERSIONS.md]

## Platform limits

- **Single replica dependencies.** Postgres, Elasticsearch and Prometheus in `platform-deps` are single instances with no
  HA and no backups. Only llama.cpp autoscales. [deploy/README.md]
- **No session access control by default.** Anyone who can reach rag-api can read any `session_id`, and `user` is
  whatever the caller sends. Acceptable only because compose binds it to 127.0.0.1 and the kind ingress listens on
  127.0.0.1:8090. The optional sign-in overlay (`docker-compose.auth.yml`, Keycloak plus oauth2-proxy) fixes this on
  Docker Compose only: sessions become owner-only and document changes need the `rag-admins` group. Checked with
  real Keycloak tokens and a browser login; not on kind or OKD, not with another identity provider, and not under
  TLS. The document library is shared by all signed-in users. [docker-compose.auth.yml; SECURITY.md]
- **Chat page browsers.** Checked in Playwright's Chromium, Firefox and WebKit on Windows (39 checks, all passed):
  upload, indexing, streamed answer, citation chip, theme, reload, phone-width layout. Not checked in real Safari, on a
  real phone or tablet, in old browsers, or with a screen reader. [docs/evidence/chat-page-browsers.txt]
- **No index during a re-index.** A changed rag image recreates the ingest Job, which deletes the `corpus` index and
  builds a new one. In build #1's rollout the index was deleted at 13:10:55Z and recreated at 13:20:05Z: about 9
  minutes with no index to retrieve from. An alias swap would avoid it but needs a change in `rag/app/ingest.py`; not
  made. Since 2026-10-13 the Job deletes only the demo documents' passages, not the whole index, so documents added
  from the chat page survive; the 9 minute gap above was measured with the old behavior and not re-measured.
  [docs/evidence/phase5b-ci.txt; deploy/README.md]
- **Your own documents (chat page).** Run on compose and, with curl through the ingress, on kind (upload, chat, survives a demo re-index; [docs/evidence/kind-own-documents.txt]). Not run on OKD. Formats are PDF, DOCX, TXT, MD and HTML
  (scanned PDFs only with the opt-in OCR build `docker-compose.ocr.yml`, checked on one generated 2 page English scan, not on
  real scans, handwriting or other languages; DOCX is read as plain text, so headers, footers, comments and images are
  skipped, and it was checked with files made by python-docx, not ones saved by Word). One item is indexed at a time, and the job list lives in the memory of the
  rag-api process that took the request, so it is lost on restart and not shared between replicas. The page stores
  nothing about who added a document, and anyone who can reach rag-api can add, remove or wipe documents (turn it
  off with `DOCS_WRITE=0`). Web pages are fetched by the server: only public addresses are allowed by default, but a
  host name that changes its address between the check and the fetch is not caught. Answer quality on added documents was measured once, on two public
  documents (a 32 page PDF and a web page) with 16 answerable questions: retrieval found the right document and page
  every time, 13 of 16 answers were right, and the model said "I do not know" on 1 of 3 questions whose answer is not
  in the documents. Small, one author, English text only. [rag/eval/README.md, rag/eval/results/uploads-20261014T051604Z]
- **Same digest only through a shared build cache.** kind-up and `ci/run.sh` now build the rag image from
  `git archive`, and build #3 pushed the same digest kind-up built (rag@sha256:6f10a3e7...). That equality relies on the
  shared BuildKit cache of the one Docker engine: the RUN steps (uv sync, model download) are not bit-reproducible on
  their own. Before the fix (build #1) the digests differed because the Windows context gave every file mode 0755 and
  the Linux checkout 0644 (and, in the first archive version, CRLF from Git for Windows; BuildKit's cache key ignores
  mtimes), so every CI deploy after a kind-up deploy rolled rag-api and rebuilt the index. [docs/evidence/phase5b-ci.txt]
- **Ingest Job failures in build #1 not diagnosed.** The rag-ingest Job completed only after 2 failed attempts
  (exit 1, 13:02:44Z to 13:04:14Z). Likely cause, not proven: the corpus download hit NTRS 503 responses (4 retries
  seen). `restartPolicy: OnFailure` had dropped their logs; the Job now uses `restartPolicy: Never` with
  `backoffLimit: 3`, and build #2's ingest ran with 0 failures. [docs/evidence/phase5b-ci.txt]
- **First run on a slow network: untested after the fixes.** kind-up now pre-pulls images on the host and imports
  them into the node, retries downloads, waits past the 600 s progress deadline (30 min per resource), forces CoreDNS
  to TCP, resumes the model download and gives the final ArgoCD wait 3 hours, because a first run downloads the 1.1 GB
  model inside the cluster. Only one clean run happened after the fixes: 28m46s, first attempt, with all images
  already on the host. The 34m25s to 46m28s clean runs were before them, and 4 clean attempts failed before them (the
  first post-commit Phase 5a run, then `kind load`, a Traefik fetch and DNS in Phase 5b). A clean run with a slow first
  pull was not repeated after the fixes; recovery from a ProgressDeadlineExceeded KEDA install was tested (442 s).
  [deploy/README.md; docs/evidence/phase5a-platform.txt, section 8c; docs/evidence/phase5b-ci.txt]
- **No traces in kind.** Grafana, Tempo, the OTel Collector and the SLO rules run only in compose; in kind the
  `trace_id` is all zeros. [deploy/README.md]
- **GitOps stand-in.** ArgoCD reads a local `git daemon` with no auth and sees only what was last published.
  [deploy/README.md]

## Security notes

- **Jenkins has the Docker socket.** `/var/run/docker.sock` is mounted into the controller, so whoever controls Jenkins
  controls the Docker engine. Keep port 8088 on 127.0.0.1. [ci/jenkins/README.md]
- **Alloy has the Docker socket too.** It is mounted `:ro`, but read-only does not restrict Docker API calls, so the
  log shipper has root-equivalent access to the Docker engine. A socket proxy limited to container listing and
  logs would reduce this; not done. [docker-compose.yml]
- **Builds can read the Jenkins admin password.** It is now a compose secret, not in the controller environment, so
  builds do not inherit it (`env` shows nothing). But builds run on the controller as the same user, and a probe
  build could read `/run/secrets/jenkins_admin_password`. Keeping it from builds needs a separate agent; not done.
  [docs/evidence/phase5b-ci.txt; ci/jenkins/README.md]
- **OKD VM reaches the host's loopback.** crc `host-network-access` (needed so ArgoCD in the VM reads the git
  daemon on the host) lets every pod in the OKD VM reach every 127.0.0.1 port on the host, including compose
  Postgres and Elasticsearch. Turning it off stops ArgoCD's git polling; not done. [deploy/README.md;
  docs/evidence/v3-openshift-okd.txt]
- **Plain HTTP and dev passwords.** Elasticsearch is plain HTTP; `.env.example` holds dev placeholder passwords. All
  compose ports bind to 127.0.0.1.
- **Partial supply chain pinning.** kind, kubeconform and the kind node image are checked by sha256 or digest; ArgoCD
  `install.yaml`, the KEDA and Traefik charts, registry, alpine and Trivy are pinned by version or tag only.
  [docs/VERSIONS.md]
- **Trivy results depend on the scan date.** The vulnerability DB downloads at scan time. CI gates on CRITICAL only:
  builds #1 to #3 found 0. The Phase 5a image (sha256:c07aa644..., Trivy DB of 2026-10-08) had 44 HIGH findings, none
  with a fixed version. [docs/evidence/phase5b-ci.txt; docs/evidence/phase5a-platform.txt]

## Future work (stretch list from docs/PLAN.md)

- Triton's native TensorRT and TensorRT-LLM backends (need Docker, so not on Kaggle).
- `bitbucket-pipelines.yml` mirror of the CI.
- OpenShift Developer Sandbox test.
- llama.cpp Vulkan on the local AMD iGPU.
