#!/usr/bin/env bash
# OpenShift path on OKD via OpenShift Local (crc 2.64.0, preset okd: no Red Hat account), GitOps as on kind: the
# community ArgoCD operator syncs the same app-of-apps with env=openshift (values-openshift.yaml).
#   bash scripts/okd-up.sh        # from a normal, non-admin Git Bash on Windows (Hyper-V bundle). Re-runnable.
#   crc stop                      # stop; the VM and everything in it is kept
# One-time admin steps (not done here): install the crc MSI, be in the crc-users and Hyper-V Administrators
# groups, reboot. RAM: crc takes 20 GB; free it first if short (docker stop llm-control-plane kind-registry kind-git).
# ArgoCD reads the committed HEAD, pushed to branch okd of the bare repo kind-up.sh uses, served by a git daemon
# container on 127.0.0.1:9418 that the VM reaches as host.crc.testing (crc host-network-access). That option lets
# every pod in the VM reach every port on the host's loopback (compose Postgres, Elasticsearch, Grafana, ...), not
# only 9418. It stays on because ArgoCD keeps polling git; to close it after a sync, accepting that ArgoCD then
# cannot reach git: crc config set host-network-access false, crc stop, Stop-Process -Name crc, crc setup, crc start.
# Evidence: docs/evidence/v3-openshift-okd.txt
set -euo pipefail
source "$(dirname "$0")/tools.sh"
cd "$ROOT"
start=$SECONDS
export MSYS_NO_PATHCONV=1
OKD_VERSION=4.22.0-okd-scos.10
# windows/amd64 layer of quay.io/crcont/okd-bundle:4.22.0-okd-scos.10 (index sha256:58dde1e5...): one gzipped tar
# holding the .crcbundle. Downloaded here because crc's own pull does not resume.
BUNDLE_LAYER=170b7fc908bfbd59deaa6f5b2798359eb17e471e55b198f21a1d033f1ba8bcd2
ARGOCD_OPERATOR_CSV=argocd-operator.v0.19.0
SKOPEO=quay.io/skopeo/stable:v1.22.3@sha256:e5d5d2815b94d74767d40be7111647c3e9cdc84406a638ffe1c7ea1f286f52a3
REGISTRY=default-route-openshift-image-registry.apps-crc.testing
GIT_URL=git://host.crc.testing/industrial-llm-platform
oc() { "$HOME/.crc/bin/oc/oc.exe" --context crc-admin "$@"; }
until_ok() { local end=$((SECONDS + $1)); shift; until "$@" >/dev/null 2>&1; do [ "$SECONDS" -lt "$end" ] || { log "timed out: $*"; exit 1; }; sleep 10; done; }

# 1. Bundle (6.2 GB), resumable, sha256-checked.
dl=.tools/downloads/crc
bundle=$dl/crc_okd_hyperv_${OKD_VERSION}_amd64.crcbundle
mkdir -p "$dl"
if [ ! -f "$bundle" ]; then
  until [ "$(stat -c %s "$dl/okd-bundle.layer" 2>/dev/null || echo 0)" -ge 6194525758 ]; do
    t=$(curl -fsSL "https://quay.io/v2/auth?service=quay.io&scope=repository:crcont/okd-bundle:pull" | sed 's/.*"token": *"\([^"]*\)".*/\1/')
    curl -fL -C - -H "Authorization: Bearer $t" -o "$dl/okd-bundle.layer" "https://quay.io/v2/crcont/okd-bundle/blobs/sha256:$BUNDLE_LAYER" || sleep 20
  done
  echo "$BUNDLE_LAYER  $dl/okd-bundle.layer" | sha256sum -c -
  tar xzf "$dl/okd-bundle.layer" -C "$dl" && rm "$dl/okd-bundle.layer"
fi

# 2. crc. host-network-access is read only when the crc daemon starts. crc setup starts the daemon if it is not
#    running but does not restart a running one, so turning the option on under a running daemon needs a restart.
crc version | grep -q '^CRC version: 2\.64\.0+' || { log "needs crc 2.64.0 (the pinned OKD bundle): $(crc version | head -1)"; exit 1; }
if ! crc config get host-network-access 2>/dev/null | grep -q true && tasklist /FI "IMAGENAME eq crc.exe" /NH | grep -qi crc.exe; then
  crc config set host-network-access true >/dev/null
  log "host-network-access was off and the crc daemon is running: it must restart to apply it. Run:"
  log "  crc stop; powershell -c 'Stop-Process -Name crc'; bash scripts/okd-up.sh"
  log "This check fires once (the option is now set): do the restart steps before re-running."
  exit 1
fi
for kv in preset=okd cpus=6 memory=20480 disk-size=60 consent-telemetry=no host-network-access=true; do
  crc config set "${kv%%=*}" "${kv#*=}" >/dev/null
done
crc setup --bundle "$(cygpath -w "$bundle")"
crc status -o json 2>/dev/null | grep -q '"openshiftStatus": *"Running"' || crc start -b "$(cygpath -w "$bundle")"
oc version --client | grep -q "^Client Version: $OKD_VERSION$" || { log "unexpected oc client: $(oc version --client | head -1)"; exit 1; }
log "$(oc version | tr '\n' ' ')"

# 3. ArgoCD (operator, pinned CSV, manual approval) with the Application health check kind-up.sh also sets, and KEDA
#    from the same pinned Helm chart as kind (the catalog's KEDA is 2.7.1).
oc apply -f - >/dev/null <<EOF
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata: {name: argocd-operator, namespace: openshift-operators}
spec: {channel: alpha, name: argocd-operator, source: community-operators, sourceNamespace: openshift-marketplace,
       installPlanApproval: Manual, startingCSV: $ARGOCD_OPERATOR_CSV}
EOF
has_installplan() { oc -n openshift-operators get installplan -o name | grep -q .; }
csv_ready() { [ "$(oc -n openshift-operators get csv "$ARGOCD_OPERATOR_CSV" -o jsonpath='{.status.phase}')" = Succeeded ]; }
argocd_ready() { [ "$(oc -n argocd get argocd argocd -o jsonpath='{.status.phase}')" = Available ]; }
until_ok 300 has_installplan
for ip in $(oc -n openshift-operators get installplan -o jsonpath="{range .items[?(@.spec.clusterServiceVersionNames[0]=='$ARGOCD_OPERATOR_CSV')]}{.metadata.name} {end}"); do
  oc -n openshift-operators patch installplan "$ip" --type merge -p '{"spec":{"approved":true}}' >/dev/null
done
until_ok 900 csv_ready
oc create namespace argocd --dry-run=client -o yaml | oc apply -f - >/dev/null
oc apply -f - >/dev/null <<'EOF'
apiVersion: argoproj.io/v1beta1
kind: ArgoCD
metadata: {name: argocd, namespace: argocd}
spec:
  extraConfig: {timeout.reconciliation: 60s}
  resourceHealthChecks:
    - group: argoproj.io
      kind: Application
      check: |
        hs = {status = "Progressing", message = ""}
        if obj.status ~= nil and obj.status.health ~= nil then
          hs.status = obj.status.health.status
          hs.message = obj.status.health.message or ""
        end
        return hs
EOF
[ -s ".tools/downloads/keda-$KEDA_CHART_VERSION.tgz" ] ||
  helm pull keda --repo https://kedacore.github.io/charts --version "$KEDA_CHART_VERSION" -d .tools/downloads
helm --kube-context crc-admin upgrade --install keda ".tools/downloads/keda-$KEDA_CHART_VERSION.tgz" -n keda \
  --create-namespace --set image.pullPolicy=IfNotPresent >/dev/null
until_ok 900 argocd_ready

# 4. App namespace (the operator only manages labelled namespaces) and dev secrets (random, kept across runs).
oc create namespace llm --dry-run=client -o yaml | oc apply -f - >/dev/null
oc label namespace llm argocd.argoproj.io/managed-by=argocd --overwrite >/dev/null
if ! oc -n llm get secret llm-secrets >/dev/null 2>&1; then
  rand() { head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
  oc -n llm create secret generic llm-secrets --from-literal=POSTGRES_PASSWORD="$(rand)" \
    --from-literal=ELASTIC_PASSWORD="$(rand)" >/dev/null
fi

# 5. rag image, built as kind-up.sh builds it, pushed by skopeo to the internal registry route (self-signed, so
#    no TLS verify on this push only). The digest goes to rag-api as on kind.
rev=$(git rev-parse HEAD)
git -c core.autocrlf=false archive --format=tar --mtime=1970-01-01T00:00:00Z "$rev:rag" |
  d build -q --provenance=false -t industrial-llm-platform/rag:0.1.0 - >/dev/null
# The token travels in an environment variable (-e NAME without a value), not on any command line.
REG_CREDS="kubeadmin:$(oc whoami -t)"
export REG_CREDS
# shellcheck disable=SC2016  # $REG_CREDS and $0 expand in the container's sh
digest=$(d run --rm -e REG_CREDS -v //var/run/docker.sock:/var/run/docker.sock --add-host "$REGISTRY:host-gateway" \
  --entrypoint sh "$SKOPEO" -c 'exec skopeo copy -q --dest-tls-verify=false --dest-creds "$REG_CREDS" \
  --digestfile /dev/stdout docker-daemon:industrial-llm-platform/rag:0.1.0 "docker://$0/llm/rag:0.1.0"' "$REGISTRY")
unset REG_CREDS
RAG_IMAGE=image-registry.openshift-image-registry.svc:5000/llm/rag@$digest
log "rag image $RAG_IMAGE"

# 6. Git: HEAD -> branch okd of the bare repo (kind's main is untouched), served on 127.0.0.1:9418.
bare=.tools/git/industrial-llm-platform.git
[ -d "$bare" ] || git init -q --bare "$bare"
git push -q --force "$bare" "$rev:refs/heads/okd"
d build -q -t "$GIT_SERVER_IMAGE" infra/git-server >/dev/null
if [ "$(d inspect -f '{{.State.Running}}' okd-git 2>/dev/null || true)" != true ]; then
  d rm -f okd-git >/dev/null 2>&1 || true
  d run -d -p 127.0.0.1:9418:9418 --name okd-git -v "$(hostpath "$ROOT/.tools/git"):/srv/git:ro" "$GIT_SERVER_IMAGE" >/dev/null
fi

# 7. Root Application from deploy/argocd/root.yaml, pointed at the okd branch with env=openshift, then wait.
p="{\"spec\":{\"source\":{\"repoURL\":\"$GIT_URL\",\"targetRevision\":\"okd\",\"helm\":{\"valuesObject\":"
p+="{\"repoURL\":\"$GIT_URL\",\"targetRevision\":\"okd\",\"env\":\"openshift\",\"ragImage\":\"$RAG_IMAGE\"}}}}}"
oc patch --local -f deploy/argocd/root.yaml --type merge -p "$p" -o yaml | oc apply -f - >/dev/null
for app in $(oc -n argocd get applications -o name); do
  oc -n argocd annotate "$app" argocd.argoproj.io/refresh=hard --overwrite >/dev/null
done
log "waiting for root, platform-deps, llamacpp, rag-api to be Synced and Healthy at $rev"
end=$((SECONDS + ${OKD_UP_TIMEOUT:-7200}))
last=
while :; do
  st=$(oc -n argocd get applications -o jsonpath='{range .items[*]}{.metadata.name}={.status.sync.status}/{.status.health.status} {end}' 2>/dev/null || true)
  [ "$st" != "$last" ] && log "$st" && last=$st
  ok=$(echo "$st" | tr ' ' '\n' | grep -c '=Synced/Healthy$' || true)
  cur=$(oc -n argocd get applications -o jsonpath='{range .items[*]}{.status.sync.revision}{"\n"}{end}' | grep -cx "$rev" || true)
  [ "$ok" -eq 4 ] && [ "$cur" -eq 4 ] && break
  [ "$SECONDS" -lt "$end" ] || { log "timed out"; oc -n llm get pods; exit 1; }
  sleep 15
done
oc -n llm get pods,route
log "done in $(((SECONDS - start) / 60))m$(((SECONDS - start) % 60))s"
# Router CA, so clients verify the Routes' edge TLS.
mkdir -p .tools/okd
oc -n openshift-config-managed get cm default-ingress-cert -o jsonpath='{.data.ca-bundle\.crt}' >.tools/okd/ingress-ca.crt
cat <<'EOF'
chat:     curl -N --cacert .tools/okd/ingress-ca.crt https://rag-api-llm.apps-crc.testing/chat \
            -H 'Content-Type: application/json' -d '{"message": "What coverage factor does NIST use for expanded uncertainty?"}'
llamacpp: https://llamacpp-llm.apps-crc.testing/v1  (OpenAI API; KEDA scales it 1..3)
EOF
