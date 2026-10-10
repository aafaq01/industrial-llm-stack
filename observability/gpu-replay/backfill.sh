#!/usr/bin/env bash
# Backfill the Phase 2 Kaggle nvidia-smi capture into the compose Prometheus. Run from the repo root.
# Idempotent: it does nothing if replay data is already present (backfilling twice would duplicate blocks).
# To redo it: docker compose stop prometheus, delete the replay block directory (ULID printed by promtool) from the
# industrial-llm-platform_prometheus volume, then run this again (it starts Prometheus).
set -euo pipefail

RAW=benchmark/results/raw/kaggle-20261009/results/raw
OUT=observability/gpu-replay/out
IMG=prom/prometheus:v3.15.0
VOL=industrial-llm-platform_prometheus

# Refuse to run when unsure: a missing volume would be silently created empty, and an unreachable Prometheus
# would make the "already loaded" check look negative and backfill twice.
docker volume inspect "$VOL" >/dev/null 2>&1 || { echo "no volume $VOL (compose project name differs?)" >&2; exit 1; }
series=$(curl -sf 'http://127.0.0.1:9090/api/v1/series' --data-urlencode 'match[]=gpu_replay_power_watts' \
  --data-urlencode 'start=2026-10-09T05:00:00Z' --data-urlencode 'end=2026-10-09T07:00:00Z') \
  || { echo "Prometheus not reachable on 127.0.0.1:9090; start it first" >&2; exit 1; }
if grep -q '"cell"' <<<"$series"; then
  echo "gpu_replay_* already in Prometheus; nothing to do"
  exit 0
fi

uv run --python 3.12 --no-project observability/gpu-replay/convert.py "$RAW" "$OUT/gpu.om"

# promtool writes 2h-aligned blocks straight into the Prometheus data volume (same image, same uid).
MSYS_NO_PATHCONV=1 docker run --rm --entrypoint promtool \
  -v "$PWD/$OUT:/in:ro" -v "$VOL:/prometheus" \
  "$IMG" tsdb create-blocks-from openmetrics /in/gpu.om /prometheus

# Prometheus loads new block directories on restart (or at its next block reload).
docker compose restart prometheus
