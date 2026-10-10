# industrial-llm-platform

An open-source, production-style platform for serving, benchmarking and operating LLMs and vision models
on Kubernetes, with a RAG application, LLMOps observability, SLOs, CI/CD and GitOps.

## Hard rules
- Free and open-source only. No paid APIs, no paid cloud. GPU work runs on free tiers
  (Kaggle notebooks: 2x T4, ~30 GPU-h/week; Google Colab free T4) or on CPU (llama.cpp).
  Everything else runs locally on kind/k3d or Red Hat Developer Sandbox / OKD / MicroShift.
- Never fabricate results. Every number in docs comes from a committed run: config + raw output + script.
  If something was not run, say "not run" and why.
- Pin all versions. Prefer the simplest thing that works; no speculative abstractions.
- Small commits with clear messages. No secrets in git (use .env.example).
- Prose style: short, concrete, no em dashes or en dashes.

## Stack (all OSS). Scope is docs/PLAN.md; items marked (stretch) wait until the core passes.
Inference: llama.cpp (CPU, local), vLLM (T4 on Kaggle); TensorRT-LLM + Triton (stretch)
Models: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct AWQ (Apache-2.0); YOLOv8n vision (stretch)
Quantisation: FP16, AWQ INT4, GGUF Q4_K_M (no FP8 or bf16 on T4)
RAG: Haystack 2.x, Elasticsearch (vector + BM25 hybrid), PostgreSQL, FastAPI
Observability: Prometheus, Grafana, Tempo, OpenTelemetry, DCGM or nvidia-smi capture; Loki, Langfuse (stretch)
Eval: lm-evaluation-harness, RAGAS or DeepEval
Platform: Docker, Kubernetes (kind), OpenShift-compatible manifests, Helm, ArgoCD, KEDA
CI: Jenkins (Jenkinsfile) + GitHub Actions; Trivy; pre-commit; bitbucket-pipelines.yml (stretch)

## Layout
inference/ benchmark/ rag/ deploy/helm/ deploy/argocd/ observability/ ci/ scripts/ docs/ notebooks/

## Knowledge graph (for agents)
`graphify-out/` holds a graphify knowledge graph of the code, configs and written docs (raw evidence data is skipped,
see `.graphifyignore`). Before reading many files, query it: `graphify query "<question>"`, `graphify path "A" "B"`,
`graphify explain "<node>"`, or read `graphify-out/GRAPH_REPORT.md`. After larger changes, refresh with
`/graphify . --update`. `docs/LEARN.md` is the human-oriented guide.
