"""RAG chat API: GET / (chat page), POST /chat (SSE), /documents (add, list, remove), GET /sessions/{id}/messages, GET /healthz, GET /metrics."""

import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from app import db, library
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger("rag")
HISTORY_TURNS = int(os.environ.get("HISTORY_MESSAGES", "4"))

# Exporter reads OTEL_EXPORTER_OTLP_ENDPOINT, resource reads OTEL_SERVICE_NAME.
if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider = TracerProvider(resource=Resource.create())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
tracer = trace.get_tracer("rag")

HTTP_REQUESTS = Counter(
    "rag_http_requests_total", "HTTP requests", ["method", "path", "status"]
)
HTTP_LATENCY = Histogram(
    "rag_http_request_duration_seconds", "Time to response headers", ["method", "path"]
)
STAGE_LATENCY = Histogram(
    "rag_stage_duration_seconds",
    "Chat pipeline stage latency",
    ["stage"],  # retrieval, generation, ttft, total
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
TOKENS = Counter("rag_llm_tokens_total", "LLM tokens reported by llama.cpp", ["kind"])
# status is the class (2xx, 4xx, 5xx). Pre-create /chat series so the first error after a restart shows in rate().
for _status in ("2xx", "4xx", "5xx"):
    HTTP_REQUESTS.labels("POST", "/chat", _status)
CHAT_ERRORS = Counter(
    "rag_chat_errors_total", "Chat requests that failed after the stream started"
)


def enable_haystack_tracing() -> None:
    """Haystack traces every component; name each span after the pipeline step (rag.embed_query, ...)."""
    from haystack import tracing
    from haystack.tracing.opentelemetry import OpenTelemetryTracer

    class StepTracer(OpenTelemetryTracer):
        def trace(self, operation_name, tags=None, parent_span=None):
            name = (tags or {}).get("haystack.component.name")
            return super().trace(
                f"rag.{name}" if name else operation_name, tags, parent_span
            )

    tracing.enable_tracing(StepTracer(trace.get_tracer("haystack")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "rag"):  # tests inject a fake
        from app.rag import Rag

        enable_haystack_tracing()
        app.state.rag = Rag()
    yield


app = FastAPI(
    title="rag",
    lifespan=lifespan,
    # FastAPI's native request spans use the provider above; no probe or scrape spans,
    # and no OTLP metrics/logs because the collector only has a traces pipeline.
    telemetry={
        "auto_configure": False,
        "metrics": False,
        "logs": False,
        "exclude": lambda scope: scope.get("path") in ("/healthz", "/metrics"),
    },
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = 500  # an unhandled exception becomes a 500 outside this middleware
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
        HTTP_REQUESTS.labels(request.method, path, f"{status // 100}xx").inc()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: uuid.UUID | None = None
    user: str = Field(default="anonymous", min_length=1, max_length=200)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def run_chat(
    rag, req: ChatRequest, out: queue.Queue, ctx: otel_context.Context
) -> None:
    """Runs in a worker thread; `ctx` parents rag.chat under the POST /chat request span."""
    from app.rag import build_messages, cited, source_label

    t0 = time.perf_counter()
    try:
        with tracer.start_as_current_span("rag.chat", context=ctx) as root:
            with Session(db.engine) as s:
                sid = db.get_or_create_session(s, req.session_id, req.user).id
                past = [(m.role, m.content) for m in db.history(s, sid, HISTORY_TURNS)]
                # stored at receipt, so its timestamp is the question time
                s.add(db.Message(session_id=sid, role="user", content=req.message))
                s.commit()
            root.set_attribute("rag.session_id", str(sid))
            # uvicorn.error is the logger that prints INFO here; Grafana links trace_id=<32 hex> to Tempo.
            logging.getLogger("uvicorn.error").info(
                "chat session_id=%s trace_id=%032x",
                sid,
                root.get_span_context().trace_id,
            )
            out.put(
                sse(
                    "session",
                    {
                        "session_id": str(sid),
                        "trace_id": format(root.get_span_context().trace_id, "032x"),
                    },
                )
            )

            t = time.perf_counter()
            docs = rag.retrieve(req.message)
            STAGE_LATENCY.labels("retrieval").observe(time.perf_counter() - t)
            out.put(
                sse(
                    "sources",
                    [
                        {"n": i, **source_label(d), "content": d.content}
                        for i, d in enumerate(docs, 1)
                    ],
                )
            )

            with tracer.start_as_current_span("rag.prompt_build"):
                messages = build_messages(req.message, docs, past)

            t = time.perf_counter()
            first = []

            def on_token(tok: str) -> None:
                if not first:
                    first.append(1)
                    STAGE_LATENCY.labels("ttft").observe(time.perf_counter() - t0)
                out.put(sse("token", {"text": tok}))

            answer, usage = rag.generate(messages, on_token)
            STAGE_LATENCY.labels("generation").observe(time.perf_counter() - t)
            TOKENS.labels("prompt").inc(usage.get("prompt_tokens", 0))
            TOKENS.labels("completion").inc(usage.get("completion_tokens", 0))
            citations = cited(answer, docs)

            with tracer.start_as_current_span("rag.db_write"), Session(db.engine) as s:
                s.add(
                    db.Message(
                        session_id=sid,
                        role="assistant",
                        content=answer,
                        citations=citations,
                    )
                )
                s.commit()
            STAGE_LATENCY.labels("total").observe(time.perf_counter() - t0)
            out.put(
                sse("done", {"answer": answer, "citations": citations, "usage": usage})
            )
    except Exception as e:  # stream already started, so report in-band
        log.exception("chat failed")
        CHAT_ERRORS.inc()
        out.put(sse("error", {"detail": str(e)}))
    finally:
        out.put(None)


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


def _writable() -> None:
    if not library.WRITABLE:
        raise HTTPException(
            403, "Adding and removing documents is turned off (DOCS_WRITE=0)."
        )


def _rejected(e: library.Rejected) -> HTTPException:
    return HTTPException(400, str(e))


class UrlsRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=20)


@app.get("/documents")
def documents() -> dict:
    docs = library.list_documents()
    jobs = sorted(library.JOBS.values(), key=lambda j: j["created"], reverse=True)
    return {
        "documents": docs,
        "chunks": sum(d["chunks"] for d in docs),
        "jobs": jobs[:20],
        "writable": library.WRITABLE,
        "max_upload_mb": library.MAX_MB,
        "formats": library.FORMATS,
    }


@app.post("/documents/files", status_code=202)
async def add_file(
    request: Request, filename: str = Query(min_length=1, max_length=255)
) -> dict:
    """The file is the raw request body (no multipart), one file per request."""
    _writable()
    if int(request.headers.get("content-length") or 0) > library.MAX_BYTES:
        raise HTTPException(413, f"Files can be up to {library.MAX_MB} MB.")
    data = await request.body()
    if len(data) > library.MAX_BYTES:
        raise HTTPException(413, f"Files can be up to {library.MAX_MB} MB.")
    if not data:
        raise HTTPException(400, "The file is empty.")
    try:
        return library.submit(
            "file", os.path.basename(filename.replace("\\", "/")), data
        )
    except library.Rejected as e:
        raise _rejected(e) from None


@app.post("/documents/urls", status_code=202)
def add_urls(req: UrlsRequest) -> dict:
    _writable()
    jobs = []
    for u in dict.fromkeys(x.strip() for x in req.urls if x.strip()):
        try:
            library.check_url(
                u
            )  # fail fast on a bad address; redirects are checked again when fetching
            jobs.append(library.submit("url", u))
        except library.Rejected as e:
            failed = {
                "id": uuid.uuid4().hex[:12],
                "kind": "url",
                "name": u,
                "status": "error",
                "error": str(e),
                "created": time.time(),
            }
            library.JOBS[failed["id"]] = failed
            jobs.append(failed)
    return {"jobs": jobs}


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str) -> dict:
    _writable()
    return {"removed_chunks": library.delete_document(doc_id)}


@app.delete("/documents")
def remove_all(confirm: str = "") -> dict:
    _writable()
    if confirm != "yes":
        raise HTTPException(400, "Add ?confirm=yes to remove every document.")
    library.delete_all()
    return {"removed": "all"}


@app.post("/chat")
def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    out: queue.Queue = queue.Queue()
    threading.Thread(
        target=run_chat,
        args=(request.app.state.rag, req, out, otel_context.get_current()),
        daemon=True,
    ).start()
    return StreamingResponse(iter(out.get, None), media_type="text/event-stream")


@app.get("/sessions/{session_id}/messages")
def messages(session_id: uuid.UUID) -> list[dict]:
    with Session(db.engine) as s:
        if s.get(db.ChatSession, session_id) is None:
            raise HTTPException(404, "session not found")
        rows = db.history(s, session_id, limit=1000)
        return [
            {
                "role": m.role,
                "content": m.content,
                "citations": m.citations,
                "created_at": m.created_at.isoformat(),
            }
            for m in rows
        ]


@app.get("/healthz")
def healthz() -> dict:
    with db.engine.connect() as c:
        c.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
