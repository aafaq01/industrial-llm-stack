"""Your own documents: add files or web pages, list, remove, reset.

One background worker indexes one item at a time (embedding runs on CPU and is slow). Jobs live in memory,
so with several rag-api replicas each one only knows its own jobs. The documents themselves are in Elasticsearch.
"""

import hashlib
import ipaddress
import os
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from html.parser import HTMLParser
from urllib.parse import urlsplit

from app.rag import (
    EMBED_MODEL,
    ES_INDEX,
    cleaner,
    document_store,
    splitter,
)
from elasticsearch import NotFoundError
from haystack import Document
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ByteStream
from haystack.document_stores.types import DuplicatePolicy

WRITABLE = os.environ.get("DOCS_WRITE", "1") == "1"
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
MAX_BYTES = MAX_MB * 1024 * 1024
ALLOW_PRIVATE_URLS = os.environ.get("ALLOW_PRIVATE_URLS") == "1"
FORMATS = "PDF, TXT, MD, HTML"

JOBS: dict[str, dict] = {}  # newest last; trimmed to 50
_queue: queue.Queue = queue.Queue(maxsize=20)
_worker_lock = threading.Lock()
_worker: threading.Thread | None = None
_embedder: SentenceTransformersDocumentEmbedder | None = None
_store = None


class Rejected(Exception):
    """A problem the person can fix (wrong type, too big, private address). Shown as is."""


def store():
    global _store
    if _store is None:
        _store = document_store()
    return _store


# --- fetching -----------------------------------------------------------------------------------


def check_url(url: str) -> None:
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise Rejected("Only http and https addresses work.")
    if ALLOW_PRIVATE_URLS:
        return
    try:
        infos = socket.getaddrinfo(
            p.hostname,
            p.port or (443 if p.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError:
        raise Rejected("That host name was not found.") from None
    for info in infos:
        if not ipaddress.ip_address(info[4][0].split("%")[0]).is_global:
            raise Rejected(
                "That address is not on the public internet. The server only fetches public pages "
                "(set ALLOW_PRIVATE_URLS=1 to allow internal hosts)."
            )
    # ponytail: DNS can change between this check and the fetch (rebinding); pin the IP if that matters.


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_url(url: str) -> tuple[bytes, str, str]:
    """Returns (body, content type, final url)."""
    check_url(url)
    opener = urllib.request.build_opener(_CheckedRedirects)
    req = urllib.request.Request(
        url, headers={"User-Agent": "industrial-llm-stack-reader"}
    )
    try:
        with opener.open(req, timeout=30) as r:
            data = r.read(MAX_BYTES + 1)
            ctype, final = r.headers.get_content_type(), r.geturl()
    except urllib.error.HTTPError as e:
        raise Rejected(f"The site answered {e.code}.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise Rejected(f"Could not fetch it: {getattr(e, 'reason', e)}") from None
    if len(data) > MAX_BYTES:
        raise Rejected(f"That page is larger than {MAX_MB} MB.")
    return data, ctype, final


# --- reading ------------------------------------------------------------------------------------


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "template"}
    BREAK = {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "pre",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip += 1
        if tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def extract(name: str, data: bytes, ctype: str = "") -> tuple[str, str, bool]:
    """Returns (title, text, is_pdf). PDF text keeps form feeds so page numbers survive."""
    if data[:5] == b"%PDF-":
        docs = PyPDFToDocument().run(sources=[ByteStream(data=data)])["documents"]
        return name, "\f".join(d.content or "" for d in docs), True
    if b"\x00" in data[:4096] or data[:2] == b"PK":
        raise Rejected(f"That file type is not supported. Use {FORMATS}.")
    text = _decode(data)
    head = text[:1024].lower()
    if (
        "html" in ctype
        or name.lower().endswith((".htm", ".html"))
        or "<html" in head
        or "<!doctype html" in head
    ):
        p = _Text()
        p.feed(text)
        body = "".join(p.parts)
        return " ".join(p.title.split()) or name, body, False
    return name, text, False


def _clean_text(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.splitlines())
    out, blank = [], 0
    for line in lines:
        blank = 0 if line else blank + 1
        if blank < 2:
            out.append(line)
    return "\n".join(out).strip()


# --- indexing -----------------------------------------------------------------------------------


def index_text(
    doc_id: str, title: str, url: str | None, text: str, is_pdf: bool
) -> int:
    """Replaces any earlier version of doc_id, then chunks, embeds and writes. Returns the chunk count."""
    global _embedder
    if not text.replace("\f", "").strip():
        raise Rejected("No text was found in it. A scanned PDF needs OCR first.")
    meta = {"doc_id": doc_id, "title": title, "url": url}
    chunks = splitter().run(
        cleaner().run([Document(content=text, meta=meta)])["documents"]
    )["documents"]
    if (
        not is_pdf
    ):  # the splitter numbers every chunk page 1 when there are no form feeds
        for c in chunks:
            c.meta.pop("page_number", None)
    if _embedder is None:
        _embedder = SentenceTransformersDocumentEmbedder(
            model=EMBED_MODEL, batch_size=64
        )
        _embedder.warm_up()
    embedded = _embedder.run(documents=chunks)["documents"]
    s = store()
    s.delete_by_filter(
        {"field": "meta.doc_id", "operator": "==", "value": doc_id}, refresh=True
    )
    DocumentWriter(s, policy=DuplicatePolicy.OVERWRITE).run(documents=embedded)
    return len(embedded)


def _process(job: dict, kind: str, name: str, data: bytes | None) -> None:
    if kind == "url":
        job["status"] = "fetching"
        data, ctype, final = fetch_url(name)
        doc_id, link = "url-" + hashlib.sha256(final.encode()).hexdigest()[:10], final
        label = final.rstrip("/").rsplit("/", 1)[-1] or final
    else:
        ctype, link, label = "", None, name
        doc_id = "file-" + hashlib.sha256(name.encode()).hexdigest()[:10]
    job["status"] = "reading"
    title, text, is_pdf = extract(label, data, ctype)
    job["status"] = "indexing"
    job["chunks"] = index_text(
        doc_id, title, link, text if is_pdf else _clean_text(text), is_pdf
    )
    job["doc_id"], job["title"] = doc_id, title


def _run_worker() -> None:
    while True:
        job, kind, name, data = _queue.get()
        try:
            _process(job, kind, name, data)
            job["status"] = "done"
        except Rejected as e:
            job.update(status="error", error=str(e))
        except Exception as e:  # keep the worker alive whatever one item does
            job.update(
                status="error", error=f"Unexpected error: {type(e).__name__}: {e}"
            )
        job["finished"] = time.time()


def submit(kind: str, name: str, data: bytes | None = None) -> dict:
    """kind is 'file' or 'url'. Raises Rejected when the queue is full."""
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = threading.Thread(target=_run_worker, daemon=True)
            _worker.start()
    job = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "name": name,
        "status": "queued",
        "created": time.time(),
    }
    try:
        _queue.put_nowait((job, kind, name, data))
    except queue.Full:
        raise Rejected(
            "Too many items are waiting. Try again when some have finished."
        ) from None
    JOBS[job["id"]] = job
    for old in list(JOBS)[:-50]:
        del JOBS[old]
    return job


# --- listing and removing -------------------------------------------------------------------------


def list_documents() -> list[dict]:
    es = store().client
    try:
        res = es.search(
            index=ES_INDEX,
            size=0,
            aggs={
                "d": {
                    "terms": {"field": "doc_id", "size": 1000},
                    "aggs": {
                        "title": {"terms": {"field": "title", "size": 1}},
                        "url": {"terms": {"field": "url", "size": 1}},
                    },
                }
            },
        )
    except NotFoundError:
        return []
    docs = []
    for b in res["aggregations"]["d"]["buckets"]:
        title, url = ((b[k]["buckets"] or [{}])[0].get("key") for k in ("title", "url"))
        docs.append(
            {
                "doc_id": b["key"],
                "title": title or b["key"],
                "url": url,
                "chunks": b["doc_count"],
            }
        )
    return sorted(docs, key=lambda d: d["title"].lower())


def delete_document(doc_id: str) -> int:
    return store().delete_by_filter(
        {"field": "meta.doc_id", "operator": "==", "value": doc_id}, refresh=True
    )


def delete_all() -> None:
    store().delete_all_documents(refresh=True)
