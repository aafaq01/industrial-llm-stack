# Observability image versions

Pinned 2026-10-07 to the latest GitHub release of each project. The configs in this folder were validated against these exact tags.

| Component | Image | Config file |
|-----------|-------|-------------|
| Prometheus | `prom/prometheus:v3.15.0` | `prometheus/prometheus.yml` |
| Grafana | `grafana/grafana:13.2.3` | `grafana/provisioning/`, `grafana/dashboards/` |
| Tempo | `grafana/tempo:3.1.0` | `tempo/tempo.yaml` |
| OTel Collector | `otel/opentelemetry-collector-contrib:0.162.0` | `otel-collector/config.yaml` |

Notes:
- Tempo 3.x removed the ingester write path and changed config keys. `tempo.yaml` follows the 3.1.0
  single-binary example. Start it with `-target=all -config.file=/etc/tempo/tempo.yaml`.
- The Tempo and OTel Collector images are distroless (no shell, wget or curl), so a Docker healthcheck
  cannot call `/ready` from inside them.
- llama.cpp metric names were checked against the llama.cpp server README (`--metrics` flag, `llamacpp:` prefix).
- `promtool` for rule checks and unit tests comes from the same `prom/prometheus:v3.15.0` image.

## v3 item 9: logs and Langfuse (pinned 2026-10-09)

| Component | Image | Config file |
|-----------|-------|-------------|
| Loki | `grafana/loki:3.7.8` | `loki/loki.yaml` (single binary, filesystem, tsdb v13, retention 168h) |
| Alloy | `grafana/alloy:v1.20.1` | `alloy/config.alloy` (Docker socket, label `service` = compose service) |
| Langfuse web / worker | `langfuse/langfuse:4.56.0`, `langfuse/langfuse-worker:4.56.0` | `docker-compose.yml`, profile `langfuse` |
| Langfuse ClickHouse | `clickhouse/clickhouse-server:25.12.11.4` | upstream compose uses `25.12` |
| Langfuse Redis | `redis:7.4.11-alpine` | upstream compose uses `redis:7` |
| Langfuse Postgres | `postgres:18.6-alpine3.24` | same image as the RAG Postgres, separate container and volume |
| Langfuse MinIO | `cgr.dev/chainguard/minio@sha256:6476f81b9739912fa9f50ceae2bb049a30336527cccd24beccfbeb4145f8f696` | only `:latest` is published, so pinned by digest (minio `RELEASE.2026-09-22T19-25-18Z`) |

Notes:
- Alloy, not Promtail: Promtail is deprecated in favour of Alloy.
- Loki is distroless (no healthcheck). Alloy has bash but no curl or wget, so its healthcheck uses `/dev/tcp`.
- Langfuse web and worker bind to `$HOSTNAME`, which Docker sets to the container id; `HOSTNAME=0.0.0.0` is needed
  for the 127.0.0.1 healthchecks.
- Langfuse v4 runs in events_only mode: `/api/public/traces` is disabled; read spans with `/api/public/v2/observations`.
- Traces reach Langfuse from the collector, not from rag/: `otel-collector/config-langfuse.yaml` adds a
  `traces/langfuse` pipeline (otlphttp + basicauth). Compose passes it as a second `--config` only when
  `LANGFUSE_OTEL_ENDPOINT` is set; otherwise `config.yaml` is passed twice (a no-op merge).
- rag-api logs `chat session_id=... trace_id=<32 hex>` per chat via `uvicorn.error` (rag/app/main.py `run_chat`);
  the Loki datasource's derived field `trace_id=([0-9a-f]{32})` links it to Tempo.
