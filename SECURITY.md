# Security

## This is a development stack

The defaults are for a laptop or a lab cluster, not for production. Before you expose anything:

- By default `rag-api` has no authentication. Any caller can read any chat session if they know its `session_id`.
  Turn sign-in on with the optional overlay (below).
- Elasticsearch runs over plain HTTP with the `elastic` user. Postgres and Grafana use the placeholder passwords from
  `.env.example`.
- Jenkins and Alloy mount the Docker socket, which gives them root on the host.
- The ingress has no TLS.
- Anyone who can reach `rag-api` can add, remove or wipe documents from the chat page. Set `DOCS_WRITE=0` to turn
  that off. Uploads are limited to `MAX_UPLOAD_MB` (25).
- `rag-api` fetches web pages that a caller names. It refuses private, loopback and link-local addresses unless
  `ALLOW_PRIVATE_URLS=1`, and checks every redirect. A host name that changes its address between the check and
  the fetch is not caught. Leave `ALLOW_PRIVATE_URLS` off unless you need internal wiki pages, and then put auth in
  front first.

## Sign-in (optional)

`docker compose -f docker-compose.yml -f docker-compose.auth.yml up -d --wait` adds Keycloak (who you are) and
oauth2-proxy (the browser login) and sets `AUTH_MODE=oidc`. Then:

- Every API call needs a valid OpenID Connect token. `rag-api` checks the signature (RS/PS/ES only, never `none` or
  shared-secret algorithms), issuer, audience and expiry itself; it does not trust a header because it arrived.
  The settings fail closed: a missing issuer or key address stops the server from starting.
- People in the group `rag-admins` can add and remove documents. Anyone else who signs in can chat and read the
  library. Set `AUTH_USER_ROLE` to also require a group for chatting.
- A chat session can only be read or continued by the person who started it. The document library is shared by
  everyone who signs in.
- `rag-api` is no longer published on the host, so the proxy is the only way in. `/healthz` and `/metrics` stay open
  to the Docker network for probes and Prometheus.

The bundled setup is still a development setup: plain HTTP, Keycloak in dev mode (its data lives inside the
container), demo users `admin-demo` and `user-demo` with the placeholder passwords from `.env.example`, and a client
that allows password logins for scripts. For real use, run Keycloak with its own database and TLS (or use your
company's identity provider by changing `AUTH_ISSUER`, `AUTH_JWKS_URL`, `AUTH_AUDIENCE` and the `OAUTH2_PROXY_*`
addresses), set `OAUTH2_PROXY_COOKIE_SECURE=true`, remove the demo users and the password-login client flag, and
change every secret. The overlay is for Docker Compose only; the Helm charts do not have it yet.

All compose ports bind to 127.0.0.1. The full list of limits is in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
For a real deployment, put OIDC (for example oauth2-proxy) in front of the ingress, enable TLS everywhere, use your
own secrets and drop the Docker socket mounts.

## Reporting a problem

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting
("Security" tab, "Report a vulnerability") on this repo. Expect a reply within 7 days.
