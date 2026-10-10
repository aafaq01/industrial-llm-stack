# Sourced by kind-up.sh and helm-check.sh. Pinned tool versions and a sha256-checked download into .tools/bin.
# shellcheck shell=bash

KIND_VERSION=v0.33.0
KUBECONFORM_VERSION=v0.8.0
REGISTRY_IMAGE=registry:3.1.2
ARGOCD_VERSION=v3.5.4
KEDA_CHART_VERSION=2.21.0
TRAEFIK_CHART_VERSION=41.6.1
GIT_SERVER_IMAGE=industrial-llm-platform/git-server:0.1.0

CLUSTER=llm
CONTEXT=kind-$CLUSTER

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_BIN="$ROOT/.tools/bin"
mkdir -p "$TOOLS_BIN"
export PATH="$TOOLS_BIN:$PATH"

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) OS=windows EXE=.exe ;;
  Linux) OS=linux EXE= ;;
  *) echo "unsupported OS $(uname -s): only Windows (Git Bash) and Linux amd64 checksums are pinned" >&2; exit 1 ;;
esac

declare -A SHA256=(
  [kind-windows-amd64]=4b22adaa135368c5a465d56bbd8e520cbea87272a06ca00b6078e7b81515c9fc
  [kind-linux-amd64]=aee6151561422756b764a4ae28e7f44cda5af5a9eead3cc9985112b1de8d8e0d
  [kubeconform-windows-amd64.zip]=e3f56102bcf4f50b034a567e2482a1c5330799983ddd655952310211aef73d93
  [kubeconform-linux-amd64.tar.gz]=9bc2bffbf71f261128533edaf912153948b7ff238f9a531ae6d34466ec287883
)

# fetch <url> <asset name> <dest>: download and verify against the pinned sha256.
fetch() {
  curl -fsSL --retry 3 -o "$3.part" "$1"
  echo "${SHA256[$2]}  $3.part" | sha256sum -c - >/dev/null
  mv "$3.part" "$3"
}

ensure_kind() {
  local bin="$TOOLS_BIN/kind$EXE"
  if [ -x "$bin" ] && "$bin" version | grep -q "$KIND_VERSION "; then return; fi
  fetch "https://github.com/kubernetes-sigs/kind/releases/download/$KIND_VERSION/kind-$OS-amd64" "kind-$OS-amd64" "$bin"
  chmod +x "$bin"
}

ensure_kubeconform() {
  local bin="$TOOLS_BIN/kubeconform$EXE" asset
  if [ -x "$bin" ] && [ "$("$bin" -v)" = "$KUBECONFORM_VERSION" ]; then return; fi
  if [ "$OS" = windows ]; then asset=kubeconform-windows-amd64.zip; else asset=kubeconform-linux-amd64.tar.gz; fi
  fetch "https://github.com/yannh/kubeconform/releases/download/$KUBECONFORM_VERSION/$asset" "$asset" "$TOOLS_BIN/$asset"
  if [ "$OS" = windows ]; then unzip -oq "$TOOLS_BIN/$asset" kubeconform.exe -d "$TOOLS_BIN"; else tar -xzf "$TOOLS_BIN/$asset" -C "$TOOLS_BIN" kubeconform; fi
  rm -f "$TOOLS_BIN/$asset"
}

# docker with Git Bash path conversion off, so container paths like /srv/git stay as written.
d() { MSYS_NO_PATHCONV=1 docker "$@"; }

# Path form Docker accepts for bind mounts (C:/... on Git Bash).
hostpath() { if command -v cygpath >/dev/null; then cygpath -m "$1"; else echo "$1"; fi; }

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
