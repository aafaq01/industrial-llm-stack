# Security

## This is a development stack

The defaults are for a laptop or a lab cluster, not for production. Before you expose anything:

- `rag-api` has no authentication. Any caller can read any chat session if they know its `session_id`.
- Elasticsearch runs over plain HTTP with the `elastic` user. Postgres and Grafana use the placeholder passwords from
  `.env.example`.
- Jenkins and Alloy mount the Docker socket, which gives them root on the host.
- The ingress has no TLS.

All compose ports bind to 127.0.0.1. The full list of limits is in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
For a real deployment, put OIDC (for example oauth2-proxy) in front of the ingress, enable TLS everywhere, use your
own secrets and drop the Docker socket mounts.

## Reporting a problem

Please do not open a public issue for a security problem. Use GitHub's private vulnerability reporting
("Security" tab, "Report a vulnerability") on this repo. Expect a reply within 7 days.
