#!/usr/bin/env bash
# Publish a commit to the local git server ArgoCD reads (git://kind-git) and a rag image to the root Application,
# then wait until root, platform-deps, llamacpp and rag-api are Synced and Healthy at that commit.
#   bash scripts/gitops-publish.sh <commit> <rag image, localhost:5001/industrial-llm-platform/rag@sha256:...>
# Needs a cluster from scripts/kind-up.sh. kubectl runs inside the kind node (its own pinned binary and admin
# kubeconfig), so this works from the Jenkins container with only the Docker socket. KIND_GIT_DIR is the directory
# kind-git serves (default .tools/git of this checkout; Jenkins mounts the host's at /repo/.tools/git).
# Called by scripts/kind-up.sh (step 8) and ci/run.sh deploy.
set -euo pipefail
source "$(dirname "$0")/tools.sh"
cd "$ROOT"
rev=$(git rev-parse --verify "$1^{commit}")
RAG_IMAGE=$2
bare="${KIND_GIT_DIR:-$ROOT/.tools/git}/industrial-llm-platform.git"
k() { d exec -i "$CLUSTER-control-plane" kubectl --kubeconfig /etc/kubernetes/admin.conf "$@"; }

[ -d "$bare" ] || { log "$bare not found: run bash scripts/kind-up.sh first"; exit 1; }
git push -q --force "$bare" "$rev:refs/heads/main"
log "git://kind-git/industrial-llm-platform main = $rev"

# The root Application from this commit, with the image digest, as kind-up.sh applies it.
k patch --local -f - --type merge -o yaml \
  -p "{\"spec\":{\"source\":{\"helm\":{\"valuesObject\":{\"ragImage\":\"$RAG_IMAGE\"}}}}}" <deploy/argocd/root.yaml |
  k apply -f - >/dev/null
for app in $(k -n argocd get applications -o name 2>/dev/null); do
  k -n argocd annotate "$app" argocd.argoproj.io/refresh=hard --overwrite >/dev/null
done

log "waiting for root, platform-deps, llamacpp, rag-api to be Synced and Healthy at $rev"
deadline=$((SECONDS + ${KIND_UP_TIMEOUT:-3600}))
last=
while :; do
  st=$(k -n argocd get applications -o jsonpath='{range .items[*]}{.metadata.name}={.status.sync.status}/{.status.health.status} {end}' 2>/dev/null || true)
  [ "$st" != "$last" ] && log "$st" && last=$st
  ok=$(echo "$st" | tr ' ' '\n' | grep -c '=Synced/Healthy$' || true)
  # Also require the just-published revision, so status from before the refresh does not count.
  cur=$(k -n argocd get applications -o jsonpath='{range .items[*]}{.status.sync.revision}{"\n"}{end}' 2>/dev/null | grep -cx "$rev" || true)
  [ "$ok" -eq 4 ] && [ "$cur" -eq 4 ] && break
  if [ "$SECONDS" -gt "$deadline" ]; then log "timed out"; k -n llm get pods; exit 1; fi
  sleep 15
done
k -n argocd get applications
