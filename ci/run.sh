#!/usr/bin/env bash
# CI stages, shared by ci/Jenkinsfile and .github/workflows/ci.yml. Every tool runs in a pinned container.
#   bash ci/run.sh lint|charts|test|build|scan|push|bump|deploy|clean
# Tool containers must see the workspace at the same path. On a Linux host (GitHub runner) that is a bind mount of
# the workspace; inside the Jenkins container it is the jenkins-home volume (CI_MOUNT, set by the Jenkinsfile).
# push, bump and deploy need the kind cluster, registry and git server from scripts/kind-up.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
WS=$PWD

UV_IMAGE=ghcr.io/astral-sh/uv:0.11.32-python3.12-trixie@sha256:5c14184583b8da2956605104e364053c863e4de51f8a83b230a89d50e34e9ab0
POSTGRES_IMAGE=postgres:18.6-alpine3.24
PROM_IMAGE=prom/prometheus:v3.15.0
TRIVY_IMAGE=aquasec/trivy:0.75.0
GITLEAKS_IMAGE=zricethezav/gitleaks:v8.30.1
HADOLINT_IMAGE=hadolint/hadolint:v2.15.1
HELM_IMAGE=alpine/helm:4.2.4@sha256:76c375eed56144c68d6197c55bc5a4552fb42002190b796729901cbab3ae6e51
ACTIONLINT_IMAGE=rhysd/actionlint:1.7.12@sha256:b1934ee5f1c509618f2508e6eb47ee0d3520686341fec936f3b79331f9315667

CI_MOUNT=${CI_MOUNT:--v $WS:$WS}
CI_HOME=${CI_HOME:-/tmp}  # tool caches (uv, pre-commit); the Jenkinsfile points it at the persistent volume
BUILD=${BUILD_NUMBER:-${GITHUB_RUN_NUMBER:-0}}
RAG_REPO=localhost:5001/industrial-llm-platform/rag
RAG_TAG=$RAG_REPO:ci-$BUILD

d() { MSYS_NO_PATHCONV=1 docker "$@"; }
# run [docker run options] <image> [args]: as the calling user, workspace mounted at the same path and as cwd.
# shellcheck disable=SC2086  # CI_MOUNT is a list of options
run() { d run --rm -u "$(id -u):$(id -g)" -e HOME="$CI_HOME" $CI_MOUNT -w "$WS" "$@"; }
# Registry reference of the pushed image, localhost:5001/...@sha256:...
rag_image() { d image inspect -f '{{range .RepoDigests}}{{println .}}{{end}}' "$RAG_TAG" | grep -m1 "^$RAG_REPO@sha256:"; }

case ${1:-} in
  lint)
    # gitleaks and hadolint are docker_image hooks; pre-commit cannot mount the workspace from inside a container,
    # so they run here from the same pinned images. gitleaks scans the whole history, not only staged files.
    run -e SKIP=gitleaks,hadolint "$UV_IMAGE" uvx --python 3.12 pre-commit@4.6.2 run --all-files --show-diff-on-failure
    # shellcheck disable=SC2086
    d run --rm $CI_MOUNT -w "$WS" "$GITLEAKS_IMAGE" git --redact --verbose .
    git ls-files '*Dockerfile' | while read -r f; do echo "hadolint $f"; d run --rm -i "$HADOLINT_IMAGE" hadolint - <"$f"; done
    run "$ACTIONLINT_IMAGE"
    ;;
  charts)
    # helm lint --strict + kubeconform for kind and openshift values; the image has helm, bash and curl.
    run --entrypoint bash "$HELM_IMAGE" scripts/helm-check.sh
    ;;
  test)
    run "$UV_IMAGE" uv run --no-project --python 3.12 --with-requirements benchmark/requirements.txt python -m pytest benchmark/tests
    # Throwaway Postgres on a per-build network; trust auth, so no password exists anywhere.
    net=ci-$BUILD-$$
    trap 'd rm -f "$net-pg" >/dev/null 2>&1; d network rm "$net" >/dev/null 2>&1' EXIT
    d network create "$net" >/dev/null
    d run -d --name "$net-pg" --network "$net" -e POSTGRES_HOST_AUTH_METHOD=trust "$POSTGRES_IMAGE" >/dev/null
    # TCP check: during initdb the server listens on the unix socket only.
    for _ in $(seq 60); do d exec "$net-pg" pg_isready -q -h 127.0.0.1 -U postgres && break; sleep 1; done
    d exec "$net-pg" pg_isready -h 127.0.0.1 -U postgres
    run --network "$net" -e TEST_PG_URL="postgresql+psycopg://postgres@$net-pg:5432" -w "$WS/rag" "$UV_IMAGE" \
      uv run --python 3.12 --frozen pytest
    p=observability/prometheus
    run --entrypoint promtool "$PROM_IMAGE" check config "$p/prometheus.yml"
    run --entrypoint promtool "$PROM_IMAGE" test rules "$p/tests/slo_test.yml" "$p/tests/slo_windows_test.yml"
    ;;
  build)
    # Same as kind-up.sh (see the comment there): a git archive of rag/ as committed with a fixed mtime, and no
    # provenance attestation, so kind-up and CI get the same digest for one revision.
    git -c core.autocrlf=false archive --format=tar --mtime=1970-01-01T00:00:00Z HEAD:rag |
      d build --provenance=false -t "$RAG_TAG" -
    ;;
  scan)
    # Fails on any CRITICAL, fixed or not. Accepted exceptions go in .trivyignore at the repo root (read from the
    # working directory), one CVE id per line with a comment saying why and until when. There is none yet.
    # shellcheck disable=SC2086
    d run --rm -v /var/run/docker.sock:/var/run/docker.sock -v trivy-cache:/root/.cache $CI_MOUNT -w "$WS" "$TRIVY_IMAGE" \
      image --scanners vuln --severity CRITICAL --exit-code 1 --no-progress "$RAG_TAG"
    ;;
  push)
    d inspect kind-registry >/dev/null 2>&1 || { echo "kind-registry not found: run bash scripts/kind-up.sh first"; exit 1; }
    d push -q "$RAG_TAG"
    rag_image
    ;;
  bump)
    # ArgoCD reads git, so the bump is a commit: on top of the checked-out revision, only in this workspace.
    # deploy publishes it to kind-git; it is never pushed to the developer's branch.
    chart=deploy/helm/rag-api/Chart.yaml
    digest=$(rag_image)
    digest=${digest#*@}
    base=$(sed -n 's/^version: //p' "$chart")
    version=${base%%+*}+ci.$BUILD
    sed -i -e "s/^version: .*/version: $version/" -e "s/^appVersion: .*/appVersion: '$digest'/" "$chart"
    git -c user.name=ci -c user.email=ci@localhost commit -q -m "ci: rag-api chart $version, image $digest" -- "$chart"
    git log -1 --format='%H %s'
    ;;
  deploy)
    bash scripts/gitops-publish.sh "$(git rev-parse HEAD)" "$(rag_image)"
    ;;
  clean)
    d rmi "$RAG_TAG" >/dev/null 2>&1 || true
    ;;
  *)
    echo "usage: $0 lint|charts|test|build|scan|push|bump|deploy|clean" >&2
    exit 2
    ;;
esac
