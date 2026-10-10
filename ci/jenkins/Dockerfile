# Jenkins controller for the local pipeline: pinned LTS + pinned plugins (plugins.yaml) + Docker CLI and buildx.
# Builds run on this controller; every tool runs in its own pinned container via the mounted Docker socket.
FROM docker:29.6.2-cli@sha256:be132a9f282288de4afaf63379dff75711fda0147c6b72a9df44e51841402144 AS docker

FROM jenkins/jenkins:2.580.1-lts-jdk21@sha256:a660310e39ade10631f774bacd5219de767dfd08947ea5c32a739b5e5bb382c1
USER 0
COPY --from=docker /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker /usr/local/libexec/docker/cli-plugins/docker-buildx /usr/local/libexec/docker/cli-plugins/docker-buildx
# /repo (the host checkout) and the bare repo kind-git serves are bind mounts not owned by the jenkins user.
RUN git config --system --add safe.directory '*'
USER 1000
COPY --chown=1000:1000 plugins.yaml /usr/share/jenkins/ref/plugins.yaml
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.yaml
