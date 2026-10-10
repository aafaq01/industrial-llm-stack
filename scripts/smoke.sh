#!/usr/bin/env bash
# Phase 1 smoke test. Run from the repo root after `docker compose up -d --wait`.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
set -a; source .env; set +a

fails=0
check() {
  local name=$1; shift
  if "$@"; then echo "PASS $name"; else echo "FAIL $name"; fails=$((fails + 1)); fi
}

chat() {
  local out
  out=$(curl -fsS --max-time 120 http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32,"temperature":0}')
  echo "  reply: $(echo "$out" | grep -o '"message":{[^}]*"content":"[^"]*"' | head -1)"
  echo "$out" | grep -q '"message":{[^}]*"content":"[^"]'
}

pg() {
  docker compose exec -T postgres psql -U postgres -tAc 'SELECT 1' | grep -qx 1
}

es() {
  curl -fsS -u "elastic:$ELASTIC_PASSWORD" http://127.0.0.1:9200/_cluster/health \
    | grep -qE '"status":"(green|yellow)"'
}

grafana() {
  curl -fsS http://127.0.0.1:3000/api/health | grep -q '"database": *"ok"'
}

# up{job=...} == 1 in Prometheus; retries for up to ~60s to allow the first scrape.
prom_up() {
  local job=$1 i
  for i in $(seq 1 12); do
    if curl -fsS -G http://127.0.0.1:9090/api/v1/query --data-urlencode "query=up{job=\"$job\"}" \
      | grep -q '"1"\]'; then
      return 0
    fi
    sleep 5
  done
  return 1
}

tempo() {
  curl -fsS http://127.0.0.1:3200/ready | grep -q ready
}

loki() {
  # Loki reports ready only 15 s after its ingester is up, so retry briefly after a restart.
  for _ in $(seq 1 12); do curl -fsS http://127.0.0.1:3100/ready 2>/dev/null | grep -q ready && return 0; sleep 5; done
  return 1
}

alloy() {
  curl -fsS http://127.0.0.1:12345/-/ready >/dev/null
}

collector() {
  curl -fsS http://127.0.0.1:8888/metrics | grep -q '^otelcol_'
}

check "llamacpp chat completion" chat
check "postgres SELECT 1" pg
check "elasticsearch health green/yellow" es
check "grafana /api/health" grafana
check "prometheus scrapes llamacpp" prom_up llamacpp
check "tempo /ready" tempo
check "loki /ready" loki
check "alloy /-/ready" alloy
check "otel-collector metrics :8888" collector
check "prometheus scrapes otel-collector" prom_up otel-collector

if [ "$fails" -gt 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "all checks passed"
