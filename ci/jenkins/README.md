# Local Jenkins

Jenkins LTS in Docker, configured entirely by `casc.yaml` (Configuration as Code) with pinned plugins
(`plugins.yaml`). One pipeline job, `industrial-llm-platform`, runs `ci/Jenkinsfile` from the committed `main`
branch of this checkout. No remote, account or token is involved.

## Run

Needs Docker Desktop (or Docker on Linux) and, for the push, bump and ArgoCD stages, the kind stack from
`bash scripts/kind-up.sh`. From the repo root, in Git Bash:

```bash
cp ci/jenkins/.env.example ci/jenkins/.env      # once; set your own admin password
docker compose -f ci/jenkins/docker-compose.yml --env-file ci/jenkins/.env up -d --build --wait
MSYS_NO_PATHCONV=1 docker exec jenkins sh -c 'curl -fsSo /tmp/cli.jar http://localhost:8080/jnlpJars/jenkins-cli.jar \
  && java -jar /tmp/cli.jar -s http://localhost:8080/ -http -auth "admin:$(cat /run/secrets/jenkins_admin_password)" build industrial-llm-platform -s -v'
```

The last command starts the job, streams its console and exits non-zero if the build fails. `-http`: the default
WebSocket mode is rejected (403, "Unexpected request origin") because the configured URL is 127.0.0.1:8088, not the
localhost:8080 the CLI calls from inside the container. The UI is at
http://127.0.0.1:8088 (user `admin`). Stop: `docker compose -f ci/jenkins/docker-compose.yml down`
(add `-v` to also delete the `jenkins-home` volume with builds and tool caches).

Jenkins builds what is committed on `main`, like `GIT_SOURCE=head` in kind-up.sh: commit first.

## How it works

| Piece | What |
|-------|------|
| Controller | `Dockerfile`: pinned `jenkins/jenkins` LTS plus the Docker CLI and buildx. Builds run on the controller (1 executor); it only runs git, bash and docker. |
| Tools | `ci/run.sh` runs every stage's tool in a pinned container (uv + Python 3.12, postgres, promtool, Trivy, gitleaks, hadolint, actionlint). Tool containers mount the `jenkins-home` volume, so they see the workspace at the same path. |
| Docker access | The Docker Desktop socket `/var/run/docker.sock` is mounted into the controller. It is `root:root 0660` there, so the container gets supplementary group 0 (`group_add`). Whoever controls Jenkins controls the Docker engine: keep port 8088 on 127.0.0.1, as configured. |
| Source | The repo root is mounted read-only at `/repo`; the job clones `file:///repo` (allowed by `-Dhudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true`). |
| GitOps target | `.tools` is mounted read-write at `/repo/.tools`; the deploy stage force-pushes the CI commit into `.tools/git/industrial-llm-platform.git`, which the `kind-git` container serves to ArgoCD. |
| kubectl | Runs inside the kind node container through `docker exec`, with the node's admin kubeconfig. No kubeconfig is copied into Jenkins. |
| Credentials | Only `JENKINS_ADMIN_PASSWORD`, in `ci/jenkins/.env` (git-ignored). Compose passes it as a secret file, `/run/secrets/jenkins_admin_password`, which JCasC reads; it is not in the controller environment, so builds do not inherit it. Builds run on the controller as the same user, though, so a build step can still read that file (checked: it is readable, `env` shows nothing). Keeping it from builds needs a separate agent; not done. The pipeline itself needs no credentials: the test Postgres uses trust auth on a per-build network. |

## Stages

`lint` (pre-commit, gitleaks over the whole history, hadolint, actionlint), `charts` (`scripts/helm-check.sh`:
helm lint and kubeconform, kind and openshift values), `test` (benchmark pytest, rag pytest
against a throwaway Postgres, promtool config check and rule unit tests), `build` (rag image, `--provenance=false`),
`scan` (Trivy, fails on any CRITICAL, fixed or not; an accepted exception goes in `.trivyignore` with a written
reason), `push` (to `localhost:5001`), `bump chart`, `argocd sync`.

`bump chart` commits, in the Jenkins workspace only, `deploy/helm/rag-api/Chart.yaml` with
`version: <version>+ci.<build>` and `appVersion: '<image digest>'`. `argocd sync` runs
`scripts/gitops-publish.sh`: it publishes that commit as `main` on kind-git, applies the root Application with the
pushed image digest (as kind-up.sh does), hard-refreshes the apps and waits until all four are Synced and Healthy at
that commit. `kind-up.sh` uses the same script for its own publish. The CI commit is never pushed to your branch;
the next `kind-up.sh` run publishes your HEAD again.
