# Layout, services and ports

## Repository layout

| Path | What |
|------|------|
| `docker-compose.yml`, `scripts/smoke.sh` | the local stack and its smoke test |
| `inference/` | serving notes: llama.cpp flags, vLLM install and T4 limits |
| `benchmark/` | load generator, GSM8K harness, `report.py`, raw results, `RESULTS.md` |
| `notebooks/` | Kaggle notebook and its builder, vLLM lock |
| `rag/` | FastAPI + Haystack app, Alembic migrations, corpus sources, tests, eval |
| `observability/` | Prometheus rules and tests, Grafana dashboards, Tempo, OTel Collector, Loki, Alloy, GPU replay, `SLOs.md` |
| `deploy/helm/`, `deploy/argocd/` | Helm charts and ArgoCD app-of-apps. `values-kind.yaml` is deployed; `values-openshift.yaml` is deployed on OKD 4.22 via OpenShift Local (`scripts/okd-up.sh`; not tested on Red Hat OpenShift) |
| `infra/` | kind cluster config, git daemon image |
| `scripts/` | `kind-up.sh`, `kind-down.sh`, `keda-demo.sh`, `helm-check.sh`, `gitops-publish.sh`, `okd-up.sh` |
| `ci/` | `run.sh` (all stages), `Jenkinsfile`, local Jenkins |
| `.github/workflows/ci.yml` | GitHub Actions workflow: lint, charts, test, build, scan (first run green; see docs/evidence/v3-github-actions.txt) |
| `docs/` | plan, status, versions, decisions, limitations, evidence |

## Compose services

All ports bind to 127.0.0.1.

| Service | Port | Notes |
|---------|------|-------|
| llamacpp | 8080 | `/v1/chat/completions`, `/health`, `/metrics` |
| rag-api | 8000 | `POST /chat` (SSE), `GET /sessions/{id}/messages`, `/healthz`, `/metrics` |
| postgres | 5432 | user `postgres`, password `POSTGRES_PASSWORD` |
| elasticsearch | 9200 | user `elastic`, password `ELASTIC_PASSWORD`, plain HTTP |
| prometheus | 9090 | alerts at `/alerts`; no Alertmanager |
| grafana | 3000 | user `admin`, password `GRAFANA_ADMIN_PASSWORD`; dashboards "RAG API and SLOs", "llama.cpp", "GPU replay" |
| tempo | 3200 | traces per pipeline step |
| otel-collector | 4317, 4318, 8888 | OTLP gRPC, OTLP HTTP, own metrics |
| loki | 3100 | log store; query `{service="<compose service>"}` in Grafana Explore. Use `127.0.0.1:3100`, not `localhost`, if another app also listens on 3100 |
| alloy | 12345 | log shipper UI (Docker socket to Loki) |
| langfuse-web | 3001 | profile `langfuse` only; Langfuse UI, login `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD` |
| llamacpp-judge | 8081 | profile `eval` only |

kind: rag-api at `http://rag-api.localhost:8090`, llama.cpp at `http://127.0.0.1:8090` (Traefik ingress). Jenkins UI:
`http://127.0.0.1:8088`. Details: [deploy/README.md](../deploy/README.md), [ci/jenkins/README.md](../ci/jenkins/README.md).
