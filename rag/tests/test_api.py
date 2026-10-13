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
