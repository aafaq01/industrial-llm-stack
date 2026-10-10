# Graph Report - industrial-llm-platform  (2026-10-10)

## Corpus Check
- 147 files · ~156,116 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 41 file(s) not represented in the graph (top: (none) 19, .jsonl 5, .in 5)

## Summary
- 928 nodes · 1756 edges · 77 communities (65 shown, 12 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 180 edges (avg confidence: 0.87)
- Token cost: 640,068 input · 0 output

## Community Hubs (Navigation)
- Vision CPU Baseline and Serving Notes
- Observability Config (Prom/Loki/OTel)
- Decode and Truncation Checks
- Triton TensorRT Vision Serving
- Report Regimes and Clock Reasons
- llama.cpp Helm Chart and KEDA
- Compose Stack Services
- Load Generator Runtime
- Vision Benchmark Runner
- RAG Judge Run (DeepEval)
- Benchmark Results and GSM8K
- Jenkins and GitHub Actions CI
- DCGM and Vision Report Sections
- RAG Evaluation Metrics
- Report Inputs and Host Notes
- OpenTelemetry Instrumentation
- ArgoCD App-of-Apps
- ONNX Export Comparison
- Prompt Set Generation
- Report Tables and Charts
- rag-api Deployment and Jobs
- Design Decisions
- Status and v3 Evidence
- GPU Models and Quantisation
- RAG API Tests
- GPU Latency Chart
- Metric Definitions
- RAG Retrieval and Generation
- Early Phase Decisions and Evidence
- CPU Inference and Caching
- Smoke Test Checks
- README and Raw Results
- Chat History Database
- rag-api Exposure and OKD
- Plan and Phases
- Platform and CI Concepts
- RAG Stack Concepts
- CPU Latency Chart
- COCO Subset Fetch
- Benchmark Methods and Gates
- Tool Bootstrap Script
- Deployment Charts Overview
- FastAPI Chat Endpoint
- RAG Test Fixtures
- kind-up GitOps Bring-up
- CPU Reporting and Limitations
- TensorRT-LLM Run v3i
- Corpus Ingest
- OKD Bring-up Script
- Alembic Migrations
- Vision mAP Evaluation
- Local Judge Model Wrapper
- TRT-LLM vs vLLM Comparison
- TRT-LLM Engine Build
- Paired Accuracy Delta
- Chat Worker Streaming
- Chunking Tests
- Project Rules and Agents
- DCGM Sampler
- Slim Scope and Run v2
- Observability Stack Versions
- Vision Stack
- Elasticsearch Pipelines
- Decode Check Recompute
- Session Server-arg Notes
- CI Stage Runner
- Indexing Pipeline
- App Lifespan and Tracing
- KEDA Demo Script
- Vision Flags
- GSM8K Runner Script
- CPU Vision Baseline Script
- GPU Replay Backfill
- Helm Check Script
- kind Teardown Script
- rag Package

## God Nodes (most connected - your core abstractions)
1. `main()` - 26 edges
2. `main()` - 20 edges
3. `vision_section()` - 18 edges
4. `run_cell()` - 16 edges
5. `run_chat()` - 16 edges
6. `docker-compose.yml (compose stack)` - 16 edges
7. `flags()` - 15 edges
8. `main()` - 13 edges
9. `Inference Serving Paths README` - 13 edges
10. `Observability Image Versions` - 13 edges

## Surprising Connections (you probably didn't know these)
- `benchmark-analyst agent` --semantically_similar_to--> `Never fabricate results rule`  [INFERRED] [semantically similar]
  .claude/agents/benchmark-analyst.md → CLAUDE.md
- `llamacpp Helm chart` --semantically_similar_to--> `model-init (sha256-checked GGUF download)`  [INFERRED] [semantically similar]
  deploy/README.md → docker-compose.yml
- `llamacpp Helm chart` --semantically_similar_to--> `llamacpp service`  [INFERRED] [semantically similar]
  deploy/README.md → docker-compose.yml
- `rag-api Helm chart` --semantically_similar_to--> `rag-api service`  [INFERRED] [semantically similar]
  deploy/README.md → docker-compose.yml
- `GitHub Actions ci.yml` --semantically_similar_to--> `ci/Jenkinsfile`  [INFERRED] [semantically similar]
  .github/workflows/ci.yml → ci/jenkins/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **RAG chat request flow (compose)** — docker_compose_rag_api, readme_haystack_pipeline, docker_compose_elasticsearch, docker_compose_llamacpp, docker_compose_postgres, docker_compose_otel_collector [EXTRACTED 1.00]
- **CI to GitOps delivery (registry, kind-git, ArgoCD)** — ci_jenkins_readme_pipeline_stages, scripts_gitops_publish, readme_local_registry, readme_kind_git, deploy_readme_argocd_root, scripts_kind_up [EXTRACTED 1.00]
- **Results traced to committed raw files** — claude_never_fabricate_results, benchmark_results_raw_files, benchmark_report, benchmark_results, benchmark_results_flag_rules, _claude_agents_reviewer [INFERRED 0.85]
- **KEDA autoscaling on llama.cpp in-flight requests via in-cluster Prometheus** — deploy_helm_llamacpp_templates_workload_deployment, deploy_helm_platform_deps_templates_prometheus_prom_config, deploy_helm_platform_deps_templates_prometheus_service, deploy_helm_llamacpp_templates_scaledobject_scaledobject, deploy_helm_llamacpp_values_server [EXTRACTED 1.00]
- **ArgoCD sync-wave ordered deployment (deps, model server, API)** — deploy_argocd_apps_values_apps, deploy_helm_platform_deps_chart_platform_deps, deploy_helm_llamacpp_chart_llamacpp, deploy_helm_rag_api_chart_rag_api [EXTRACTED 1.00]
- **rag-api backing services (Elasticsearch, Postgres, llama.cpp)** — deploy_helm_rag_api_values_env, deploy_helm_platform_deps_templates_elasticsearch_service, deploy_helm_platform_deps_templates_postgres_service, deploy_helm_llamacpp_templates_workload_service [EXTRACTED 1.00]
- **Reproducible rag image digest drives rollout and re-index** — docs_decisions_build_from_git_archive, docs_decisions_deploy_by_digest, deploy_helm_rag_api_templates_workload_rag_ingest, docs_evidence_phase5b_ci_digest_equality, docs_limitations_no_index_during_reindex [INFERRED 0.85]
- **Like-for-like benchmark discipline** — docs_decisions_fixed_128_token_decode, docs_decisions_cache_aware_cpu_reporting, docs_evidence_v3_trtllm_comparison_gate, docs_evidence_v3_trtllm_power_cap_clock_confound [INFERRED 0.75]
- **OpenShift compatibility path** — deploy_helm_rag_api_templates_expose_route, docs_evidence_phase5a_platform_arbitrary_uid, docs_evidence_v3_openshift_okd_restricted_v2_scc, docs_evidence_v3_openshift_okd_routes_edge_tls, docs_versions_v3_okd [INFERRED 0.85]
- **RAG SLO pipeline: recording rules, burn-rate alerts, promtool tests** — observability_slos, observability_prometheus_rules_recording, observability_prometheus_rules_alerts, observability_prometheus_tests_slo_test, observability_prometheus_tests_slo_windows_test [EXTRACTED 1.00]
- **Trace and log correlation: OTel Collector to Tempo/Langfuse, Loki derived field to Tempo** — observability_otel_collector_config, observability_otel_collector_config_langfuse, observability_tempo_tempo, observability_loki_loki, observability_grafana_provisioning_datasources_datasources_traceid_derived_field [INFERRED 0.85]
- **Pinned GPU environment locks for Kaggle notebooks** — notebooks_vllm_requirements, notebooks_vllm_overrides, notebooks_trtllm_requirements, notebooks_trtllm_build_constraints, notebooks_vision_requirements [INFERRED 0.85]
- **Local LLM-judge evaluation flow** — rag_eval_run_eval, rag_eval_judge_run, rag_eval_readme_llamacpp_judge_service, rag_eval_readme_deepeval, rag_eval_readme_run_artifacts [EXTRACTED 1.00]
- **RAG eval metric suite** — rag_eval_readme_recall_at_5, rag_eval_readme_doc_recall_at_5, rag_eval_readme_faithfulness, rag_eval_readme_answer_relevancy [EXTRACTED 1.00]
- **RAG eval run lineage (superseded run, current run, judge sample)** — rag_eval_results_20261008t095250z_summary, rag_eval_results_20261008t113955z_summary, rag_eval_results_20261008t113955z_judge_20261008t120952z_summary, rag_eval_readme_nist_tn_2148_drop [EXTRACTED 1.00]
- **Fair TensorRT-LLM vs vLLM comparison** — docs_learn_same_session_control, docs_learn_like_for_like_gate, docs_learn_decode_checks, docs_learn_generation_config_vllm, docs_learn_trtllm_vs_vllm_v3i [EXTRACTED 1.00]
- **One chat request flow (retrieve, rerank, generate, trace)** — docs_learn_rag_api, docs_learn_postgresql, docs_learn_bge_small_en, docs_learn_elasticsearch, docs_learn_minilm_reranker, docs_learn_llama_cpp, docs_learn_opentelemetry_tracing [EXTRACTED 1.00]
- **Code change reaching the cluster (CI to GitOps)** — docs_learn_jenkins, docs_learn_trivy, docs_learn_deploy_by_digest, docs_learn_ci_bump_commit, docs_learn_local_git_daemon, docs_learn_argocd, docs_learn_helm_charts [EXTRACTED 1.00]

## Communities (77 total, 12 thin omitted)

### Community 0 - "Vision CPU Baseline and Serving Notes"
Cohesion: 0.09
Nodes (30): v3 Vision CPU Baseline Evidence, ONNX Export Provenance (v3c file), ORT Intra-op Thread Settings (default vs 4), YOLOv8n ONNX Runtime CPU FP32 Baseline, Inference Serving Paths README, DCGM on Kaggle (item 10), llama.cpp CPU Serving Path, OpenAI-Compatible API Contract (+22 more)

### Community 1 - "Observability Config (Prom/Loki/OTel)"
Cohesion: 0.11
Nodes (31): Grafana Dashboard Provider, Grafana Datasources, Loki TraceID Derived Field, Loki Config, OTel Collector Config, OTel Collector Langfuse Overlay, Prometheus Scrape Config, rag-api scrape job (+23 more)

### Community 2 - "Decode and Truncation Checks"
Cohesion: 0.10
Nodes (18): decode_checks(), stream_truncated(), rec(), run(), test_attention_workspace_formula_matches_the_failed_v3h_build(), test_comparison_only_within_a_session_and_generated_from_raw_facts(), test_comparison_refused_on_truncated_streams_and_off_length_outputs(), test_comparison_refused_without_same_session_vllm_or_with_different_prompts() (+10 more)

### Community 3 - "Triton TensorRT Vision Serving"
Cohesion: 0.12
Nodes (14): TrtModel, main(), make_infer(), use_own_libpython(), build(), code(), main(), make_cells() (+6 more)

### Community 4 - "Report Regimes and Clock Reasons"
Cohesion: 0.12
Nodes (17): cache_share(), cell_label(), clock_reasons(), first_days(), flags(), order(), regime(), rec() (+9 more)

### Community 5 - "llama.cpp Helm Chart and KEDA"
Cohesion: 0.09
Nodes (27): llamacpp Ingress, llamacpp OpenShift Route (600s timeout, edge TLS), llamacpp KEDA ScaledObject (prometheus trigger), llamacpp Deployment (llama-server), model init container (resumable download + sha256 verify), llamacpp-models PVC, llamacpp Service :8080, llamacpp KEDA autoscaling values (+19 more)

### Community 6 - "Compose Stack Services"
Cohesion: 0.19
Nodes (23): observability-engineer agent, rag-engineer agent, docker-compose.yml (compose stack), alloy log shipper, elasticsearch service, grafana service, Langfuse v4 stack (profile langfuse), llamacpp service (+15 more)

### Community 7 - "Load Generator Runtime"
Cohesion: 0.17
Nodes (9): closed_loop(), cpu_model(), DcgmSampler, docker_stats(), GpuSampler, main(), run_cell(), scrape() (+1 more)

### Community 8 - "Vision Benchmark Runner"
Cohesion: 0.13
Nodes (10): git_commit(), sh(), letterbox(), load_model(), main(), OrtModel, run_once(), scrape() (+2 more)

### Community 9 - "RAG Judge Run (DeepEval)"
Cohesion: 0.17
Nodes (7): main(), write_summary(), Run artifacts (raw.jsonl, config.json, summary.md), ask(), hit(), main(), run_cmd()

### Community 10 - "Benchmark Results and GSM8K"
Cohesion: 0.15
Nodes (15): benchmark-analyst agent, gsm8k_pinned lm-eval task, lm-eval 0.4.13, CPU llama.cpp table (warm/cold cache), --generation-config vllm (repetition_penalty 1.1 default), GSM8K accuracy table (lm-evaluation-harness), PyTriton Python backend (not native TensorRT backend), T4 SW power cap / low SM clock confound (+7 more)

### Community 11 - "Jenkins and GitHub Actions CI"
Cohesion: 0.18
Nodes (16): GitHub Actions ci.yml, Jenkins casc.yaml (JCasC), industrial-llm-platform pipelineJob (file:///repo, */main), Jenkins plugins.yaml, Jenkins admin password secret file, ci/run.sh (pinned tool containers), Jenkins controller (Docker, 1 executor), ci/Jenkinsfile (+8 more)

### Community 12 - "DCGM and Vision Report Sections"
Cohesion: 0.16
Nodes (14): dcgm_means(), dcgm_section(), dcgm_values(), vision_section(), req(), test_dcgm_means_and_section(), test_letterbox_keeps_aspect_and_pads(), test_load_runs_keeps_only_llm_workload() (+6 more)

### Community 13 - "RAG Evaluation Metrics"
Cohesion: 0.14
Nodes (11): RAG eval README, DeepEval, llamacpp-judge compose service (profile eval), Local judge: Qwen2.5-7B-Instruct GGUF Q4_K_M on llama.cpp, NIST TN 2148 dropped from corpus, pypdf 6.19.0, questions.jsonl (30-question set), rag/data/SOURCES.md (+3 more)

### Community 14 - "Report Inputs and Host Notes"
Cohesion: 0.13
Nodes (10): arg_explanations(), host_note(), host_note_text(), host_samples(), host_window(), is_smoke(), load_accuracy(), load_docs() (+2 more)

### Community 15 - "OpenTelemetry Instrumentation"
Cohesion: 0.13
Nodes (3): healthz(), messages(), metrics()

### Community 16 - "ArgoCD App-of-Apps"
Cohesion: 0.12
Nodes (16): apps chart (app-of-apps), Per-chart ArgoCD Application template, apps list with sync waves, env selector (values-<env>.yaml), ragImage digest override, ArgoCD root Application, llamacpp Helm chart, llamacpp kind arbitrary-UID emulation (+8 more)

### Community 17 - "ONNX Export Comparison"
Cohesion: 0.23
Nodes (5): canonical(), main(), other_fields(), main(), sha256()

### Community 18 - "Prompt Set Generation"
Cohesion: 0.23
Nodes (5): long_prompt(), main(), one_request(), main(), parse_ts()

### Community 19 - "Report Tables and Charts"
Cohesion: 0.22
Nodes (7): charts(), fmt(), gpu_stats(), low_clock_notes(), main(), trtllm_comparison(), vision_backend_notes()

### Community 21 - "rag-api Deployment and Jobs"
Cohesion: 0.18
Nodes (10): rag-api Deployment, /healthz startup/readiness/liveness probes, rag-ingest Job (Force=true,Replace=true, restartPolicy Never), rag-migrate Job (alembic upgrade head, Helm pre-install/pre-upgrade hook), Images run as arbitrary UID with group 0, KEDA ScaledObject on llama.cpp in-flight requests (scale 1 to 3 and back), crc host-network-access exposes host loopback to the VM, SCC restricted-v2 with namespace UID range (+2 more)

### Community 22 - "Design Decisions"
Cohesion: 0.22
Nodes (4): SLO recording/alert rules, promtool tests (34 groups) and mutation check, Digest equality kind-up vs ci/run.sh (rag@sha256:6f10a3e7), Jenkins builds #1 to #3 (SUCCESS), GitHub Actions run 37970874178 (lint, test, image; success)

### Community 23 - "Status and v3 Evidence"
Cohesion: 0.25
Nodes (10): Langfuse receives OTel spans as generic SPAN observations, rag-api logs trace_id linked from Loki to Tempo, v3 additions (items 7 to 12), Accepted deviations (Phase 1 to 4), v3 item 10: DCGM on Kaggle, v3 item 11: Vision YOLOv8n, v3 item 12: TensorRT-LLM FP16, v3 item 7: GitHub repo + Actions (+2 more)

### Community 24 - "GPU Models and Quantisation"
Cohesion: 0.20
Nodes (11): DCGM 4.7.0-2 GPU telemetry, Kaggle run history v1 to v3i, Kaggle free Tesla T4 sessions, Qwen2.5-1.5B-Instruct-AWQ (INT4), Qwen2.5-1.5B-Instruct FP16, Qwen2.5-7B-Instruct-AWQ, Tensor parallel TP=2 over NCCL (no P2P), TensorRT-LLM 0.14.0 (+3 more)

### Community 25 - "RAG API Tests"
Cohesion: 0.18
Nodes (4): events(), test_chat_streams_and_history_persists(), test_unhandled_exception_counts_as_5xx(), test_unknown_session_404()

### Community 26 - "GPU Latency Chart"
Cohesion: 0.26
Nodes (12): GPU Latency/Throughput Chart (latency_throughput_gpu.png), End-to-end Latency p95 vs Concurrency, Insight: 7B AWQ on T4 only at c=8: ~180 tok/s, ~1s TTFT, ~8.3s p95, Insight: vLLM T4 1.5B AWQ leads throughput (~900 tok/s at c=32) and lowest p95 latency, Insight: 2xT4 tensor-parallel FP16 reaches highest throughput (~1100 tok/s) at c=32, single n=1 points, Insight: TensorRT-LLM FP16 (v3i) lower TTFT at c=1 but higher TTFT and p95 at c=32 than same-session vLLM, Kaggle Runs 2026-10-09/10 (vLLM T4 FP16/AWQ/TP2, 7B AWQ, v3d/v3f/v3h/v3i, TRT-LLM v3i), Chart Methodology: mean over repeats, 1-stdev error bars, log2 concurrency, prefix caching and KV block reuse off (+4 more)

### Community 27 - "Metric Definitions"
Cohesion: 0.23
Nodes (4): dist(), percentile(), request_metrics(), summarize()

### Community 28 - "RAG Retrieval and Generation"
Cohesion: 0.20
Nodes (5): build_messages(), cited(), Rag, cb(), source_label()

### Community 29 - "Early Phase Decisions and Evidence"
Cohesion: 0.17
Nodes (5): Notebook preflight cell (run v1 failed: no GPU, no internet), SSE chat with citations and history (ingest 9 docs, 3129 chunks), Pinned container images, Pinned models (Qwen2.5 GGUF, bge-small, MiniLM reranker), Phase 5b CI pins (Jenkins, actionlint, Helm image)

### Community 30 - "CPU Inference and Caching"
Cohesion: 0.21
Nodes (8): CPU llama.cpp results (cpu-20261009), DeepEval 4.2.8, KEDA autoscaling of llama.cpp, llama.cpp llama-server (CPU), Qwen2.5-1.5B-Instruct GGUF Q4_K_M, Qwen2.5-7B-Instruct GGUF Q4_K_M judge, RAG results (recall@5 0.867, faithfulness 0.592), Reviewer agent (independent PASS/FAIL)

### Community 32 - "README and Raw Results"
Cohesion: 0.31
Nodes (8): inference-engineer agent, GPU latency/throughput table (vLLM, TensorRT-LLM), benchmark/results/raw JSON files, vllm_t4_benchmark.ipynb Kaggle notebook, KEDA autoscaling demo (llama.cpp 1 to 3 replicas), llama.cpp server (Qwen2.5-1.5B Q4_K_M), TensorRT-LLM 0.14.0, vLLM 0.31.0 on Kaggle T4

### Community 33 - "Chat History Database"
Cohesion: 0.35
Nodes (6): Base, ChatSession, get_or_create_session(), history(), Message, User

### Community 34 - "rag-api Exposure and OKD"
Cohesion: 0.22
Nodes (8): rag-api Ingress, rag-api OpenShift Route (edge TLS, 600s timeout for SSE), rag-api Service (port 8000), Routes with edge TLS verified on OKD, Compose services and ports (all bound to 127.0.0.1), Repository layout table, No session access control on rag-api, OKD via OpenShift Local pins (crc 2.64.0, OKD 4.22, argocd-operator 0.19.0)

### Community 35 - "Plan and Phases"
Cohesion: 0.40
Nodes (9): Phase 1 smoke test (scripts/smoke.sh, 8 checks pass), Phase 2 benchmark matrix (12 cells + 2 extras), Free GPU options (Kaggle 2x T4 primary), Phase 1: Skeleton + CPU inference, Phase 2: GPU inference + benchmarks, Phase 3: RAG, Phase 4: Observability + SLOs, Phase 5: Platform + CI/CD (+1 more)

### Community 36 - "Platform and CI Concepts"
Cohesion: 0.22
Nodes (10): Air-gapped and on-prem AI (K3s), ArgoCD app-of-apps with sync waves, GitHub Actions CI, gitleaks v8.30.1, Helm charts (platform-deps, llamacpp, rag-api), Jenkins pipeline (ci/run.sh), kind cluster 'llm', kind-git local git daemon (+2 more)

### Community 37 - "RAG Stack Concepts"
Cohesion: 0.20
Nodes (8): bge-small-en-v1.5 embedder, Elasticsearch 9.5.5 (BM25 + vector), Haystack 2.31.0, Learning guide: industrial-llm-platform, ms-marco-MiniLM-L6-v2 cross-encoder reranker, PostgreSQL chat history (SQLAlchemy, Alembic), Private self-hosted AI assistant (Gulf data residency) (+1 more)

### Community 38 - "CPU Latency Chart"
Cohesion: 0.24
Nodes (10): CPU Latency/Throughput Chart (llama.cpp Q4_K_M, cold vs warm cache), Insight: CPU saturates by concurrency 4; extra concurrency only adds queueing latency, End-to-end latency p95: ~15s to ~74s cold, ~6s to ~36s warm, llama.cpp CPU, Qwen2.5-1.5B GGUF Q4_K_M, Output tokens/s (aggregate): ~17-20 cold, ~24-31 warm, flat past concurrency 4, Cold vs Warm Prompt Cache Split (share >= 50% = warm), RESULTS.md (n per point, 3-repeat target), benchmark/results/raw (+2 more)

### Community 39 - "COCO Subset Fetch"
Cohesion: 0.31
Nodes (5): cached(), download(), main(), manifest(), sha256()

### Community 40 - "Benchmark Methods and Gates"
Cohesion: 0.24
Nodes (5): GPU replay dashboard (backfilled, not live), GPU and model sizing service, nvidia-smi 500 ms sampler with clock event reasons, Prometheus v3.15.0 and promtool, benchmark/report.py (tables rebuilt from raw files)

### Community 41 - "Tool Bootstrap Script"
Cohesion: 0.24
Nodes (6): ensure_kind(), ensure_kubeconform(), fetch(), PATH, tools.sh script, SHA256

### Community 42 - "Deployment Charts Overview"
Cohesion: 0.47
Nodes (7): platform-engineer agent, deploy/argocd/root.yaml app-of-apps with sync waves, KEDA ScaledObject, llamacpp Helm chart, OpenShift OKD 4.22 via crc, platform-deps Helm chart, rag-api Helm chart

### Community 43 - "FastAPI Chat Endpoint"
Cohesion: 0.22
Nodes (3): chat(), ChatRequest, metrics_middleware()

### Community 44 - "RAG Test Fixtures"
Cohesion: 0.22
Nodes (3): worker(), client(), FakeRag

### Community 45 - "kind-up GitOps Bring-up"
Cohesion: 0.36
Nodes (7): GIT_SOURCE head/worktree publish modes, kind cluster, k(), retry(), kind-up.sh script, wait_all(), wait_rollout()

### Community 46 - "CPU Reporting and Limitations"
Cohesion: 0.25
Nodes (4): CPU rerun: 3 cold + 3 warm repeats per concurrency (session cpu-20261009), CPU and GPU results not comparable, Jenkins and Alloy have the Docker socket, Small n (3 repeats, 250 GSM8K items, 30 RAG questions)

### Community 47 - "TensorRT-LLM Run v3i"
Cohesion: 0.25
Nodes (6): generation_config repetition_penalty applied in run v2, Like-for-like comparison gate in report.py (trtllm_comparison), T4 70 W power cap / SM clock confound, Kaggle run v3i: TRT-LLM 0.14 FP16 vs same-session vLLM (one T4), GPU clock confound (run v2 FP16 c1, v3i vLLM c1), v3 Kaggle pins (DCGM 4.7.0, TRT-LLM 0.14.0, ultralytics, PyTriton)

### Community 48 - "Corpus Ingest"
Cohesion: 0.31
Nodes (3): fetch(), main(), sha256()

### Community 49 - "OKD Bring-up Script"
Cohesion: 0.31
Nodes (5): has_installplan(), MSYS_NO_PATHCONV, oc(), okd-up.sh script, until_ok()

### Community 51 - "Vision mAP Evaluation"
Cohesion: 0.36
Nodes (4): test_data_yaml_has_train_and_val_for_every_model_format(), data_yaml_dict(), main(), sha256()

### Community 53 - "TRT-LLM vs vLLM Comparison"
Cohesion: 0.36
Nodes (5): --generation-config vllm (repetition_penalty difference), GSM8K paired accuracy (250 items), lm-evaluation-harness 0.4.13 (gsm8k_pinned), Things not fully understood (57 open items), TensorRT-LLM vs vLLM run v3i results

### Community 54 - "TRT-LLM Engine Build"
Cohesion: 0.39
Nodes (4): build(), convert(), main(), untie()

### Community 55 - "Paired Accuracy Delta"
Cohesion: 0.33
Nodes (3): paired_delta(), per_item(), test_paired_delta_from_lm_eval_samples()

### Community 56 - "Chat Worker Streaming"
Cohesion: 0.33
Nodes (3): run_chat(), on_token(), sse()

### Community 57 - "Chunking Tests"
Cohesion: 0.43
Nodes (4): chunker(), run(), test_chunks_are_bounded_overlapping_and_keep_page_numbers(), test_cleaning_drops_nist_footer_and_blank_lines()

### Community 58 - "Project Rules and Agents"
Cohesion: 0.40
Nodes (3): generalist agent, reviewer agent, CLAUDE.md project instructions

### Community 60 - "Slim Scope and Run v2"
Cohesion: 0.47
Nodes (3): Kaggle run v2: vLLM on 2x T4 (39 min 17 s), MarlinLinearKernel for AWQ on T4, Stretch items (TRT-LLM+Triton, YOLOv8n, Bitbucket, Loki, Langfuse, Sandbox, Vulkan iGPU)

### Community 61 - "Observability Stack Versions"
Cohesion: 0.40
Nodes (6): Grafana 13.2.3, Langfuse 4.56.0 (optional profile), Loki 3.7.8 + Alloy log shipping, OpenTelemetry tracing to Tempo, OTel Collector contrib 0.162.0, Tempo 3.1.0

### Community 62 - "Vision Stack"
Cohesion: 0.40
Nodes (5): ONNX Runtime 1.26.0, Triton via PyTriton 0.7.0, TensorRT 10.16 (vision engines), Vision results (TensorRT FP16 295 img/s b1, mAP50-95 0.401), YOLOv8n vision detector

### Community 63 - "Elasticsearch Pipelines"
Cohesion: 0.47
Nodes (3): document_store(), generation_pipeline(), retrieval_pipeline()

### Community 64 - "Decode Check Recompute"
Cohesion: 0.40
Nodes (3): run_decode_checks(), stream_invalid(), facts()

### Community 65 - "Session Server-arg Notes"
Cohesion: 0.40
Nodes (3): server_flags(), session_notes(), test_session_notes_name_server_arg_differences()

### Community 66 - "CI Stage Runner"
Cohesion: 0.90
Nodes (4): d(), rag_image(), run(), run.sh script

### Community 67 - "Indexing Pipeline"
Cohesion: 0.40
Nodes (3): cleaner(), indexing_pipeline(), splitter()

### Community 68 - "App Lifespan and Tracing"
Cohesion: 0.40
Nodes (3): enable_haystack_tracing(), trace(), lifespan()

### Community 69 - "KEDA Demo Script"
Cohesion: 0.70
Nodes (3): k(), keda-demo.sh script, snapshot()

## Knowledge Gaps
- **62 isolated node(s):** `rag`, `MSYS_NO_PATHCONV`, `PATH`, `SHA256`, `generalist agent` (+57 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 262 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GPU Latency/Throughput Chart (latency_throughput_gpu.png)` connect `GPU Latency Chart` to `Report Inputs and Host Notes`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `main()` (e.g. with `cpu_model()` and `git_commit()`) actually correct?**
  _`main()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `rag`, `MSYS_NO_PATHCONV`, `PATH` to the rest of the system?**
  _62 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Vision CPU Baseline and Serving Notes` be split into smaller, more focused modules?**
  _Cohesion score 0.0873440285204991 - nodes in this community are weakly interconnected._
- **Why does `docker-compose.yml (compose stack)` connect `Compose Stack Services` to `README and Raw Results`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Should `Observability Config (Prom/Loki/OTel)` be split into smaller, more focused modules?**
  _Cohesion score 0.10984848484848485 - nodes in this community are weakly interconnected._
- **Why does `Judge sample judge-20261008T120952Z (12 ids)` connect `RAG Evaluation Metrics` to `RAG Judge Run (DeepEval)`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
