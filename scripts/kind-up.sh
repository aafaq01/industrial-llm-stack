#!/usr/bin/env bash
# One command: kind cluster + local registry + local git server + KEDA + Traefik + ArgoCD, then the
# app-of-apps syncs platform-deps, llamacpp and rag-api and this waits until all are Synced and Healthy.
#   bash scripts/kind-up.sh                       # ArgoCD sees the committed HEAD (default)
#   GIT_SOURCE=worktree bash scripts/kind-up.sh   # opt-in: a synthetic commit of the working tree, untracked files included
# Re-running is safe, also after a failed run; it re-publishes the git snapshot. Teardown: bash scripts/kind-down.sh
# A slow network only makes it slower: images are pulled on the host and loaded into the node, and readiness waits
# retry past a Deployment's 600 s progress deadline (wait_rollout).
set -euo pipefail
source "$(dirname "$0")/tools.sh"
cd "$ROOT"
start=$SECONDS
GIT_SOURCE=${GIT_SOURCE:-head}
k() { kubectl --context "$CONTEXT" "$@"; }
# `rollout status` (like helm --wait) gives up when a Deployment passes its progress deadline (600 s), although the
# rollout goes on, e.g. while an image still downloads. Retry it within a 30-minute budget instead.
wait_rollout() { # <namespace> <resource>
  local end=$((SECONDS + 1800))
  until k -n "$1" rollout status "$2" --timeout 15m >/dev/null 2>&1; do
    [ "$SECONDS" -lt "$end" ] || { log "$1/$2 not ready after 30 minutes (up to ~45 with the last 15 m rollout check)"; k -n "$1" get pods; exit 1; }
    sleep 10
  done
}
wait_all() { for r in $(k -n "$1" get deploy,statefulset -o name); do wait_rollout "$1" "$r"; done; }
# A flaky network (connection reset) fails single downloads; retry them. Pinned charts and the ArgoCD manifest are
# downloaded once into .tools/downloads, so re-runs need no network for them.
retry() { local i; for i in 1 2 3 4 5; do "$@" && return; log "attempt $i failed: $*"; sleep 15; done; return 1; }
dl="$ROOT/.tools/downloads"
mkdir -p "$dl"
ARGOCD_MANIFEST="$dl/argocd-install-$ARGOCD_VERSION.yaml"
[ -s "$ARGOCD_MANIFEST" ] || retry curl -fsSL -o "$ARGOCD_MANIFEST"   "https://raw.githubusercontent.com/argoproj/argo-cd/$ARGOCD_VERSION/manifests/install.yaml"
[ -s "$dl/keda-$KEDA_CHART_VERSION.tgz" ] ||
  retry helm pull keda --repo https://kedacore.github.io/charts --version "$KEDA_CHART_VERSION" -d "$dl"
[ -s "$dl/traefik-$TRAEFIK_CHART_VERSION.tgz" ] ||
  retry helm pull traefik --repo https://traefik.github.io/charts --version "$TRAEFIK_CHART_VERSION" -d "$dl"
# IfNotPresent: the chart default (Always) would re-contact ghcr.io at every KEDA pod start despite the preloaded images.
KEDA=(keda "$dl/keda-$KEDA_CHART_VERSION.tgz" -n keda --set image.pullPolicy=IfNotPresent)
# The endpoint IP gives Ingresses a status address; without it ArgoCD keeps them Progressing.
TRAEFIK=(traefik "$dl/traefik-$TRAEFIK_CHART_VERSION.tgz" -n traefik
  --set service.type=NodePort --set ports.web.nodePort=30080 --set ingressClass.isDefaultClass=false
  --set providers.kubernetesIngress.publishedService.enabled=false
  --set 'additionalArguments={--providers.kubernetesingress.ingressendpoint.ip=127.0.0.1}')

ensure_kind
log "kind $KIND_VERSION, context $CONTEXT, git source $GIT_SOURCE"

# 1. Local registry (localhost:5001 on the host, kind-registry:5000 inside the kind network).
if [ "$(d inspect -f '{{.State.Running}}' kind-registry 2>/dev/null || true)" != true ]; then
  d rm -f kind-registry >/dev/null 2>&1 || true
  d run -d --restart=unless-stopped -p 127.0.0.1:5001:5000 --name kind-registry "$REGISTRY_IMAGE" >/dev/null
fi

# 2. Cluster.
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  kind create cluster --config infra/kind/cluster.yaml --wait 120s
fi
for node in $(kind get nodes --name "$CLUSTER"); do
  d exec "$node" mkdir -p /etc/containerd/certs.d/localhost:5001
  echo '[host."http://kind-registry:5000"]' | d exec -i "$node" cp /dev/stdin /etc/containerd/certs.d/localhost:5001/hosts.toml
done
d network connect kind kind-registry 2>/dev/null || true
# CoreDNS forwards to Docker's DNS on the kind network gateway. On the dev machine after a network change
# (2026-10-09) its UDP queries timed out while TCP answered, so no pod could resolve a name. Use TCP.
if ! k -n kube-system get cm coredns -o jsonpath='{.data.Corefile}' | grep -q force_tcp; then
  k -n kube-system get cm coredns -o json | sed 's/max_concurrent 1000/max_concurrent 1000\\n       force_tcp/' |
    k replace -f - >/dev/null
  k -n kube-system rollout restart deploy/coredns >/dev/null
fi

# 2b. Every image the cluster runs (except ours, which comes from the local registry), listed from the same
#     manifests installed below, pulled on the host (docker pull has no deadline) and loaded into the node.
#     Present images are skipped, so re-runs are quick. ArgoCD's pods use imagePullPolicy Always, so they still
#     fetch the manifest from the registry, but the layers are local. KEDA is set to IfNotPresent below.
log "pre-pulling images on the host and loading them into the node"
images=$({
  helm template "${KEDA[@]}"
  helm template "${TRAEFIK[@]}"
  cat "$ARGOCD_MANIFEST"
  for c in platform-deps llamacpp; do
    helm template x "deploy/helm/$c" -f "deploy/helm/$c/values.yaml" -f "deploy/helm/$c/values-kind.yaml"
  done
} | sed -n 's/^[ -]*image: *"\{0,1\}\([^" ]*\)"\{0,1\} *$/\1/p' | grep -v '^localhost:5001/' | sort -u)
# Not `kind load docker-image`: with Docker's containerd image store it exports multi-platform indexes whose other
# platforms are missing and the import fails ("content digest ... not found"). Export linux/amd64 only.
for img in $images; do
  d image inspect "$img" >/dev/null 2>&1 || { log "pulling $img"; retry d pull -q --platform linux/amd64 "$img" >/dev/null; }
  for node in $(kind get nodes --name "$CLUSTER"); do
    d exec "$node" crictl inspecti -q "$img" >/dev/null 2>&1 && continue
    d save --platform linux/amd64 "$img" |
      d exec -i "$node" ctr -n k8s.io images import --platform linux/amd64 --digests --snapshotter=overlayfs - >/dev/null
  done
done

# 3. The revision to publish. head: the committed HEAD. worktree: a synthetic commit of the working tree (tracked +
#    untracked, minus .gitignore) built with a throwaway index: no branch, ref or index of this repo changes, but
#    its blobs, trees and the commit are written to .git/objects and stay there until git gc prunes them.
if [ "$GIT_SOURCE" = head ]; then
  rev=$(git rev-parse HEAD)
else
  idx=$(mktemp)
  cp "$(git rev-parse --git-path index)" "$idx"
  GIT_INDEX_FILE="$idx" git add -A
  tree=$(GIT_INDEX_FILE="$idx" git write-tree)
  rm -f "$idx"
  rev=$(GIT_AUTHOR_NAME=kind-up GIT_AUTHOR_EMAIL=kind-up@localhost GIT_COMMITTER_NAME=kind-up \
    GIT_COMMITTER_EMAIL=kind-up@localhost git commit-tree "$tree" -p HEAD -m "kind-up worktree snapshot")
fi

# Our image, built from rag/ at that revision, -> local registry. The context is a git archive, so modes and
# times do not depend on the checkout (Windows or Linux): the blobs as committed (core.autocrlf=false: Git for
# Windows would otherwise add CRLF, since rag/ alone carries no .gitattributes) and a fixed mtime (an explicit
# date: git reads "@0" as now). ci/run.sh builds the same way, so the same revision gives the same digest. No
# provenance attestation: it carries a timestamp, so even a fully cached build would get a new digest.
log "building and pushing the rag image"
git -c core.autocrlf=false archive --format=tar --mtime=1970-01-01T00:00:00Z "$rev:rag" |
  d build -q --provenance=false -t industrial-llm-platform/rag:0.1.0 - >/dev/null
d tag industrial-llm-platform/rag:0.1.0 localhost:5001/industrial-llm-platform/rag:0.1.0
# The tag is mutable, so ArgoCD gets the pushed digest: a rebuilt image changes the pod template and rolls out.
digest=$(d push localhost:5001/industrial-llm-platform/rag:0.1.0 | sed -n 's/.*digest: \(sha256:[0-9a-f]*\).*/\1/p')
[ -n "$digest" ] || { log "could not read the pushed digest"; exit 1; }
RAG_IMAGE=localhost:5001/industrial-llm-platform/rag@$digest
log "rag image $RAG_IMAGE"

# 4. Local git server for ArgoCD: a bare repo in .tools/git served read-only by git daemon as
#    git://kind-git/industrial-llm-platform. Step 8 publishes $rev to it.
bare="$ROOT/.tools/git/industrial-llm-platform.git"
[ -d "$bare" ] || git init -q --bare "$bare"
d build -q -t "$GIT_SERVER_IMAGE" infra/git-server >/dev/null
if [ "$(d inspect -f '{{.State.Running}}' kind-git 2>/dev/null || true)" != true ]; then
  d rm -f kind-git >/dev/null 2>&1 || true
  d run -d --restart=unless-stopped --network kind --name kind-git \
    -v "$(hostpath "$ROOT/.tools/git"):/srv/git:ro" "$GIT_SERVER_IMAGE" >/dev/null
fi

# 5. KEDA and Traefik (kind-only ingress; OpenShift uses its router and Routes). No helm --wait: it fails at the
#    progress deadline and marks the release failed. wait_all waits instead, and the next upgrade replaces a
#    release that an earlier run left failed.
log "installing KEDA $KEDA_CHART_VERSION and Traefik chart $TRAEFIK_CHART_VERSION"
helm --kube-context "$CONTEXT" upgrade --install "${KEDA[@]}" --create-namespace >/dev/null
helm --kube-context "$CONTEXT" upgrade --install "${TRAEFIK[@]}" --create-namespace >/dev/null
wait_all keda
wait_all traefik

# 6. ArgoCD, with Application health enabled so app-of-apps sync waves wait for each child to be Healthy.
log "installing ArgoCD $ARGOCD_VERSION"
k create namespace argocd --dry-run=client -o yaml | k apply -f - >/dev/null
k apply -n argocd --server-side --force-conflicts -f "$ARGOCD_MANIFEST" >/dev/null
wait_all argocd
k -n argocd patch configmap argocd-cm --type merge -p "$(cat <<'EOF'
data:
  timeout.reconciliation: 60s
  resource.customizations.health.argoproj.io_Application: |
    hs = {status = "Progressing", message = ""}
    if obj.status ~= nil and obj.status.health ~= nil then
      hs.status = obj.status.health.status
      hs.message = obj.status.health.message or ""
    end
    return hs
EOF
)" >/dev/null
k -n argocd rollout restart deploy/argocd-repo-server statefulset/argocd-application-controller >/dev/null
wait_all argocd

# 7. App namespace (Pod Security "restricted" enforced) and dev secrets: from .env if present, else random.
#    Kept across re-runs because Postgres and Elasticsearch store the password on first start.
k apply -f - >/dev/null <<'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: llm
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.36
EOF
if ! k -n llm get secret llm-secrets >/dev/null 2>&1; then
  [ -f .env ] && { set -a; source .env; set +a; }
  rand() { head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
  k -n llm create secret generic llm-secrets \
    --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(rand)}" \
    --from-literal=ELASTIC_PASSWORD="${ELASTIC_PASSWORD:-$(rand)}" >/dev/null
fi

# 8. App-of-apps, then wait. First run downloads the model (1.1 GB) and the corpus and builds the index.
#    gitops-publish.sh pushes $rev to the bare repo, applies the root with ragImage already set (so the very first
#    rag-api sync uses the digest), refreshes the apps and waits for all four at $rev. ci/run.sh deploy uses it too.
#    Its wait fails only at its deadline (not early on Degraded): 3 hours here, because the first run downloads the
#    1.1 GB model inside the cluster (about 85 minutes at 220 KB/s).
KIND_UP_TIMEOUT=${KIND_UP_TIMEOUT:-10800} bash "$ROOT/scripts/gitops-publish.sh" "$rev" "$RAG_IMAGE"

k -n llm get pods
log "done in $(((SECONDS - start) / 60))m$(((SECONDS - start) % 60))s"
cat <<'EOF'
rag-api:  curl -N http://rag-api.localhost:8090/chat -H 'Content-Type: application/json' -d '{"message": "What coverage factor does NIST use for expanded uncertainty?"}'
llamacpp: http://127.0.0.1:8090/v1  (OpenAI API; KEDA scales it 1..3 on in-flight requests)
ArgoCD:   kubectl --context kind-llm -n argocd port-forward svc/argocd-server 8443:443
          password: kubectl --context kind-llm -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d
EOF
