#!/usr/bin/env bash
# helm lint + kubeconform (strict) for every chart with values-kind.yaml and values-openshift.yaml.
# Schemas: Kubernetes from yannh/kubernetes-json-schema, ScaledObject/Application from datreeio/CRDs-catalog,
# Route from melmorabity/openshift-json-schemas (OpenShift 4.20). All pinned to commits; no -ignore-missing-schemas.
set -euo pipefail
source "$(dirname "$0")/tools.sh"
cd "$ROOT"
ensure_kubeconform

K8S_VERSION=1.36.4  # matches infra/kind/cluster.yaml
K8S_SCHEMAS="https://raw.githubusercontent.com/yannh/kubernetes-json-schema/39a87cd29966bc0b3434e8cfbc2061782c8cb69a/{{.NormalizedKubernetesVersion}}-standalone{{.StrictSuffix}}/{{.ResourceKind}}{{.KindSuffix}}.json"
CRD_SCHEMAS="https://raw.githubusercontent.com/datreeio/CRDs-catalog/63669a570e231d4f1f8396d229a1de512bcf0a34/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json"
OCP_SCHEMAS="https://raw.githubusercontent.com/melmorabity/openshift-json-schemas/62ad608b18edcd8ef4b6dd3741fc36115813fd18/v4.20-standalone-strict/{{.ResourceKind}}{{.KindSuffix}}.json"

fails=0
for chart in deploy/helm/*/ deploy/argocd/apps/; do
  chart=${chart%/}
  for env in kind openshift; do
    args=(-f "$chart/values.yaml")
    [ -f "$chart/values-$env.yaml" ] && args+=(-f "$chart/values-$env.yaml")
    [ "$chart" = deploy/argocd/apps ] && args+=(--set "env=$env")
    echo "== $chart ($env)"
    if ! helm lint --strict "$chart" "${args[@]}" | grep -v '^$'; then fails=$((fails + 1)); fi
    if ! helm template check "$chart" "${args[@]}" \
      | kubeconform -strict -summary -kubernetes-version "$K8S_VERSION" \
        -schema-location "$K8S_SCHEMAS" -schema-location "$CRD_SCHEMAS" -schema-location "$OCP_SCHEMAS"; then
      fails=$((fails + 1))
    fi
  done
done
echo "== deploy/argocd/apps (kind, ragImage digest set as kind-up.sh does)"
out=$(helm template check deploy/argocd/apps --set ragImage=localhost:5001/x@sha256:0000000000000000000000000000000000000000000000000000000000000000)
echo "rag-api image parameters rendered: $(grep -c 'name: image, value:' <<<"$out")"
grep -q 'name: image, value:' <<<"$out" || fails=$((fails + 1))
kubeconform -strict -summary -schema-location "$CRD_SCHEMAS" <<<"$out" || fails=$((fails + 1))
echo "== deploy/argocd/root.yaml"
kubeconform -strict -summary -schema-location "$CRD_SCHEMAS" deploy/argocd/root.yaml || fails=$((fails + 1))
if [ "$fails" -gt 0 ]; then echo "$fails check(s) failed"; exit 1; fi
echo "all charts pass helm lint and kubeconform"
