"""Download the corpus (sha256-checked, cached) and rebuild the Elasticsearch index.

Run: docker compose up (first start), or force a rebuild: docker compose run --rm -e INGEST_SKIP_IF_PRESENT=0 rag-ingest
Under `docker compose up` it sets INGEST_SKIP_IF_PRESENT=1 and leaves an existing, non-empty index alone.
"""

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from app.rag import ES_INDEX, ES_URL, document_store, indexing_pipeline
from elasticsearch import Elasticsearch

SOURCES = Path(__file__).resolve().parent.parent / "data" / "sources.json"
CACHE = Path(os.environ.get("CORPUS_DIR", "/data/corpus"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(src: dict) -> Path:
    path = CACHE / f"{src['id']}.pdf"
    if path.exists() and sha256(path) == src["sha256"]:
        return path
    part = path.with_suffix(".part")
    for attempt in range(4):  # NTRS sometimes answers 503
        try:
            req = urllib.request.Request(
                src["url"], headers={"User-Agent": "industrial-llm-platform-ingest"}
            )
            with urllib.request.urlopen(req, timeout=120) as r, part.open("wb") as f:
                f.write(r.read())
            break
        except OSError as e:
            print(f"retry {src['id']}: {e}")
            time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit(f"download failed after 4 attempts: {src['id']} {src['url']}")
    got = sha256(part)
    if got != src["sha256"]:
        raise SystemExit(f"sha256 mismatch for {src['id']}: {got}")
    part.rename(path)
    return path


def main() -> None:
    es = Elasticsearch(ES_URL, basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]))
    if os.environ.get("INGEST_SKIP_IF_PRESENT") == "1" and es.indices.exists(
        index=ES_INDEX
    ):
        n = es.count(index=ES_INDEX)["count"]
        if n > 0:
            print(
                f"index {ES_INDEX} already has {n} chunks, skipping (force: INGEST_SKIP_IF_PRESENT=0)"
            )
            return
    sources = json.loads(SOURCES.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = [fetch(s) for s in sources]
    print(f"corpus ok: {len(paths)} files, sha256 verified")

    es.indices.delete(
        index=ES_INDEX, ignore_unavailable=True
    )  # full rebuild keeps the index consistent
    store = document_store()
    # not "source_id": DocumentSplitter overwrites that key with the parent document id
    meta = [{"doc_id": s["id"], "title": s["title"], "url": s["url"]} for s in sources]
    t = time.perf_counter()
    out = indexing_pipeline(store).run({"convert": {"sources": paths, "meta": meta}})
    print(
        f"indexed {len(sources)} documents, {out['write']['documents_written']} chunks in {time.perf_counter() - t:.0f}s"
    )
    print(f"index count: {store.count_documents()}")


if __name__ == "__main__":
    main()
