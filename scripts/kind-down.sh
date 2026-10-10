#!/usr/bin/env bash
# Deletes the kind cluster, the local registry, the git server container and the served bare repo.
# The compose stack is not touched.
set -euo pipefail
source "$(dirname "$0")/tools.sh"
ensure_kind
kind delete cluster --name "$CLUSTER"
d rm -f kind-registry kind-git >/dev/null 2>&1 || true
rm -rf "$ROOT/.tools/git"
log "kind cluster $CLUSTER, kind-registry and kind-git removed"
