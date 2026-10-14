"""API tests against a real Postgres. Set TEST_PG_URL, e.g.
postgresql+psycopg://postgres:<password>@127.0.0.1:5432 ; a throwaway database rag_test is created."""

import json
import os
import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient
from haystack import Document

PG = os.environ.get("TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG, reason="TEST_PG_URL not set")


class FakeRag:
    """Retrieval and generation mocked; records the prompt so history can be checked."""

    def __init__(self):
        self.prompts = []

    def retrieve(self, question):
        return [
            Document(
                content="Bolts are torqued to spec.",
                meta={"doc_id": "a", "title": "Doc A", "page_number": 7, "url": "u"},
            )
        ]

    def generate(self, messages, on_token):
        self.prompts.append(messages)
        for tok in ["Torque ", "to spec ", "[1]."]:
            on_token(tok)
        return "Torque to spec [1].", {"prompt_tokens": 10, "completion_tokens": 3}


@pytest.fixture(scope="module")
def client():
    with psycopg.connect(
        PG.replace("+psycopg", "") + "/postgres", autocommit=True
    ) as c:
        c.execute("DROP DATABASE IF EXISTS rag_test WITH (FORCE)")
        c.execute("CREATE DATABASE rag_test")
    os.environ["DATABASE_URL"] = PG + "/rag_test"

    from alembic import command
    from alembic.config import Config

    command.upgrade(
        Config(os.path.join(os.path.dirname(__file__), "..", "alembic.ini")), "head"
    )

    from app.main import app

    app.state.rag = FakeRag()
    with TestClient(app) as c:
        yield c


def events(resp) -> list[tuple[str, dict]]:
    out = []
    for frame in resp.text.strip().split("\n\n"):
        ev, data = frame.split("\n", 1)
        out.append(
            (ev.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
        )
    return out


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_chat_streams_and_history_persists(client):
    r1 = client.post("/chat", json={"message": "How tight?", "user": "alice"})
    assert r1.headers["content-type"].startswith("text/event-stream")
    ev = events(r1)
    names = [e for e, _ in ev]
    assert names[:2] == ["session", "sources"] and names[-1] == "done"
    assert "".join(d["text"] for e, d in ev if e == "token") == "Torque to spec [1]."
    done = ev[-1][1]
    assert done["citations"] == [
        {"n": 1, "source_id": "a", "title": "Doc A", "page": 7, "url": "u"}
    ]
    sid = ev[0][1]["session_id"]

    r2 = client.post(
        "/chat", json={"message": "And again?", "session_id": sid, "user": "alice"}
    )
    assert events(r2)[0][1]["session_id"] == sid
    # second prompt carries the first turn as history
    texts = [m.text for m in client.app.state.rag.prompts[-1]]
    assert "How tight?" in texts and "Torque to spec [1]." in texts

    msgs = client.get(f"/sessions/{sid}/messages").json()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["citations"][0]["title"] == "Doc A"


def test_unknown_session_404(client):
    assert client.get(f"/sessions/{uuid.uuid4()}/messages").status_code == 404


def test_unhandled_exception_counts_as_5xx(client):
    from app.main import app

    @app.get("/boom")
    def boom():
        raise RuntimeError("boom")

    def count() -> float:
        for line in client.get("/metrics").text.splitlines():
            if line.startswith(
                'rag_http_requests_total{method="GET",path="/boom",status="5xx"}'
            ):
                return float(line.split()[-1])
        return 0.0

    before = count()
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/boom").status_code == 500
    assert count() == before + 1


def test_metrics(client):
    body = client.get("/metrics").text
    assert (
        "rag_stage_duration_seconds_bucket" in body
        and 'rag_llm_tokens_total{kind="completion"}' in body
    )


def test_chat_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Content-Security-Policy" in r.text and "/chat" in r.text


def test_documents_list_and_guards(client, monkeypatch):
    from app import library

    monkeypatch.setattr(
        library,
        "list_documents",
        lambda: [{"doc_id": "a", "title": "A", "url": None, "chunks": 3}],
    )
    body = client.get("/documents").json()
    assert body["chunks"] == 3 and body["writable"] is True

    assert client.delete("/documents").status_code == 400  # needs ?confirm=yes
    seen = []
    monkeypatch.setattr(library, "delete_all", lambda: seen.append("all"))
    assert client.delete("/documents?confirm=yes").status_code == 200 and seen == [
        "all"
    ]

    monkeypatch.setattr(library, "MAX_BYTES", 10)
    assert (
        client.post("/documents/files?filename=a.txt", content=b"x" * 11).status_code
        == 413
    )
    assert (
        client.post("/documents/files?filename=a.txt", content=b"").status_code == 400
    )

    monkeypatch.setattr(library, "WRITABLE", False)
    assert (
        client.post("/documents/files?filename=a.txt", content=b"x").status_code == 403
    )
    assert client.delete("/documents/a").status_code == 403


def test_url_job_rejects_private_address(client):
    r = client.post("/documents/urls", json={"urls": ["http://127.0.0.1:9200/"]})
    assert r.status_code == 202
    job = r.json()["jobs"][0]
    assert job["status"] == "error" and "public internet" in job["error"]


# ---- sign-in (AUTH_MODE=oidc) ----

ISS, AUD = "https://idp.test/realms/x", "rag-web"


@pytest.fixture
def idp(monkeypatch):
    """Switches the API to oidc mode with a locally generated signing key."""
    import types

    from app import auth
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = key.public_key()
    monkeypatch.setattr(auth, "MODE", "oidc")
    monkeypatch.setattr(auth, "ISSUER", ISS)
    monkeypatch.setattr(auth, "AUDIENCE", AUD)
    monkeypatch.setattr(
        auth,
        "_jwks",
        types.SimpleNamespace(
            get_signing_key_from_jwt=lambda token: types.SimpleNamespace(key=pub)
        ),
    )

    def token(user="alice", groups=(), signer=key, **over):
        import time

        import jwt

        claims = {
            "iss": ISS,
            "aud": AUD,
            "exp": int(time.time()) + 300,
            "sub": "id-" + user,
            "preferred_username": user,
            "groups": list(groups),
        } | over
        return jwt.encode(claims, signer, algorithm="RS256")

    pem = pub.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return types.SimpleNamespace(token=token, other=other, pem=pem, auth=auth)


def bearer(t):
    return {"Authorization": f"Bearer {t}"}


def test_oidc_requires_a_token_but_not_for_page_and_health(client, idp):
    for path in ("/me", "/documents", "/sessions/" + str(uuid.uuid4()) + "/messages"):
        assert client.get(path).status_code == 401
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/").status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_oidc_rejects_bad_tokens(client, idp):
    import base64
    import hashlib
    import hmac
    import json

    import jwt

    now = int(__import__("time").time())

    def forged_hs256():  # signed with the public key as a shared secret
        enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=")  # noqa: E731
        head = enc({"alg": "HS256", "typ": "JWT"})
        body = enc(
            {"iss": ISS, "aud": AUD, "exp": now + 300, "preferred_username": "mallory"}
        )
        sig = hmac.new(idp.pem, head + b"." + body, hashlib.sha256).digest()
        return (
            head + b"." + body + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")
        ).decode()

    bad = {
        "expired": idp.token(exp=now - 3600),
        "wrong audience": idp.token(aud="other-app"),
        "wrong issuer": idp.token(iss="https://evil.test/realms/x"),
        "signed by another key": idp.token(signer=idp.other),
        "no signature (alg none)": jwt.encode(
            {"iss": ISS, "aud": AUD, "exp": now + 300, "preferred_username": "m"},
            None,
            algorithm="none",
        ),
        "hs256 with the public key": forged_hs256(),
        "garbage": "not-a-token",
    }
    for name, t in bad.items():
        r = client.get("/me", headers=bearer(t))
        assert r.status_code == 401, name


def test_oidc_roles(client, idp, monkeypatch):
    from app import library

    user, admin = idp.token("alice"), idp.token("root", groups=["/rag-admins"])
    assert client.get("/me", headers=bearer(user)).json() == {
        "auth": "oidc",
        "user": "alice",
        "can_edit": False,
        "logout_url": "/oauth2/sign_out",
    }
    assert client.get("/me", headers=bearer(admin)).json()["can_edit"] is True

    monkeypatch.setattr(library, "delete_all", lambda: None)
    assert (
        client.delete("/documents?confirm=yes", headers=bearer(user)).status_code == 403
    )
    assert (
        client.delete("/documents?confirm=yes", headers=bearer(admin)).status_code
        == 200
    )
    r = client.post(
        "/documents/urls", json={"urls": ["https://example.com/"]}, headers=bearer(user)
    )
    assert r.status_code == 403

    monkeypatch.setattr(
        idp.auth, "USER_ROLE", "rag-users"
    )  # now chatting needs the role too
    assert client.get("/me", headers=bearer(user)).status_code == 403
    assert (
        client.get(
            "/me", headers=bearer(idp.token("bob", groups=["rag-users"]))
        ).status_code
        == 200
    )
    assert (
        client.get("/me", headers=bearer(admin)).status_code == 200
    )  # admins always pass


def test_oidc_sessions_belong_to_their_owner(client, idp):
    alice, bob = bearer(idp.token("alice")), bearer(idp.token("bob"))
    out = client.post(
        "/chat", json={"message": "hello", "user": "ignored"}, headers=alice
    ).text
    sid = json.loads(out.split("event: session\ndata: ")[1].split("\n")[0])[
        "session_id"
    ]

    assert client.get(f"/sessions/{sid}/messages", headers=alice).status_code == 200
    assert client.get(f"/sessions/{sid}/messages", headers=bob).status_code == 404
    assert (
        client.post(
            "/chat", json={"message": "x", "session_id": sid}, headers=bob
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/chat", json={"message": "y", "session_id": sid}, headers=alice
        ).status_code
        == 200
    )


def test_oidc_misconfiguration_fails_closed(monkeypatch):
    import importlib

    from app import auth

    monkeypatch.setenv("AUTH_MODE", "oidc")
    for k in ("AUTH_ISSUER", "AUTH_AUDIENCE", "AUTH_JWKS_URL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="AUTH_ISSUER"):
        importlib.reload(auth)
    monkeypatch.setenv("AUTH_MODE", "maybe")
    with pytest.raises(RuntimeError, match="AUTH_MODE"):
        importlib.reload(auth)
    monkeypatch.delenv("AUTH_MODE")
    importlib.reload(auth)  # back to the default for the other tests
