#!/usr/bin/env bash
# KEDA scale-up/down demo on the kind cluster: runs benchmark/loadgen against the in-cluster llama.cpp through
# the Traefik ingress (load is spread over all replicas) and prints a UTC timeline every 10 s of the
# KEDA/Prometheus in-flight metric, HPA desired/current replicas and llama.cpp pods, until it is back at 1 replica.
#   bash scripts/keda-demo.sh [concurrency] [requests]       (defaults 6 and 120)
set -euo pipefail
source "$(dirname "$0")/tools.sh"
cd "$ROOT"
CONC=${1:-6}
REQS=${2:-120}
OUT=docs/evidence/phase5a-keda
k() { kubectl --context "$CONTEXT" -n llm "$@"; }
PY=benchmark/.venv/Scripts/python
[ -x "$PY" ] || PY=benchmark/.venv/bin/python

inflight() {
  MSYS_NO_PATHCONV=1 kubectl --context "$CONTEXT" get --raw \
    '/api/v1/namespaces/llm/services/prometheus:9090/proxy/api/v1/query?query=sum(llamacpp:requests_processing%2Bllamacpp:requests_deferred)' \
    | sed -n 's/.*"value":\[[^,]*,"\([^"]*\)".*/\1/p'
}
snapshot() {
  local hpa pods
  hpa=$(k get hpa keda-hpa-llamacpp -o jsonpath='{.status.desiredReplicas}/{.status.currentReplicas}')
  pods=$(k get pods -l app=llamacpp --no-headers | awk '{printf "%s:%s:%s ", substr($1, length($1)-4), $2, $3}')
  echo "$(date -u +%H:%M:%SZ) inflight=$(inflight) hpa_desired/current=$hpa pods=[ $pods]"
}

log "ScaledObject and HPA before load"
k get scaledobject,hpa
(
  while :; do snapshot; sleep 10; done
) &
watcher=$!
trap 'kill $watcher 2>/dev/null || true' EXIT

log "load: concurrency $CONC, $REQS requests, 128 tokens each"
"$PY" -I benchmark/loadgen/run.py --base-url http://127.0.0.1:8090/v1 --model qwen2.5-1.5b-instruct-q4_k_m \
  --engine llama.cpp --engine-version server-b11459 --precision Q4_K_M --hardware "kind, CPU, 1..3 replicas" \
  --label kind-keda --concurrency "$CONC" --repeats 1 --requests "$REQS" --smoke \
  --notes "KEDA scale demo through Traefik ingress, not a benchmark" --out-dir "$OUT" 2>&1 | tail -3
log "load finished, waiting for scale-down to 1 replica"
for _ in $(seq 1 60); do
  [ "$(k get deploy llamacpp -o jsonpath='{.status.replicas}')" = 1 ] && break
  sleep 10
done
sleep 10
kill $watcher 2>/dev/null || true
log "after"
k get scaledobject,hpa
k get pods -l app=llamacpp
k get events --field-selector involvedObject.name=keda-hpa-llamacpp -o custom-columns=LAST:.lastTimestamp,MSG:.message --sort-by=.lastTimestamp
