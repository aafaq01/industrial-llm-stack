# Read-only git daemon (git://) so in-cluster ArgoCD can read this repo without any remote.
# Built and run by scripts/kind-up.sh on the kind Docker network as "kind-git".
FROM alpine:3.23.0
RUN apk add --no-cache git-daemon=2.52.0-r0 \
    && git config --system safe.directory '*'
USER 65534
EXPOSE 9418
CMD ["git", "daemon", "--reuseaddr", "--export-all", "--base-path=/srv/git", "/srv/git"]
