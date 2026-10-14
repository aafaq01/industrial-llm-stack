"""Score retrieval and answers on documents that are added through the chat page's own paths.

The demo evaluation (run_eval.py) covers the built-in corpus. This one uploads a PDF as a file and a web page by address,
then asks the questions in eval/uploads/questions.jsonl. Only the standard library is needed.

It needs an API with an EMPTY library, so use a scratch instance with its own index, never your real one:

  docker compose run -d --rm --name rag-scratch -p 127.0.0.1:8002:8000 -e ES_INDEX=scratch rag-api
  python eval/run_uploads_eval.py --api http://127.0.0.1:8002
  docker rm -f rag-scratch

Metrics (all deterministic, no judge model):
  recall_at_5      PDF: a top-5 chunk of the gold document starts on a gold page or the page before it (as in run_eval.py).
                   Web page: a top-5 chunk of the gold document (it has no pages).
  doc_recall_at_5  a top-5 chunk of the gold document.
  answer_has_keys  every key group has at least one of its words in the answer (a crude lower bound: a correct answer
                   in other words fails, a wrong answer that repeats the words passes).
  cites_gold       the answer cites a source number that belongs to the gold document.
  abstained        for questions whose answer is not in the documents: the answer says it does not know.
"""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "industrial-llm-stack-eval"}
ABSTAIN = re.compile(
    r"do(?: not|n't) know|not (?:contain|include|mention|provide|stated|specified|found|available|in the)|"
    r"no (?:information|mention)|cannot (?:be )?(?:determine|find|answer)|unable to",
    re.I,
)


def call(api, method, path, body=None, raw=None, timeout=60):
    data, headers = raw, dict(UA)
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
    req = urllib.request.Request(api + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def download(url):
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=120
    ) as r:
        return r.read()


def ask(api, question):
    out = {"tokens": 0}
    t0 = time.perf_counter()
    req = urllib.request.Request(
        api + "/chat",
        data=json.dumps({"message": question, "user": "eval"}).encode(),
        headers={**UA, "Content-Type": "application/json"},
    )
    event = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "token":
                    out["tokens"] += 1
                    out.setdefault("ttft_s", time.perf_counter() - t0)
                else:
                    out[event] = data
    out["latency_s"] = time.perf_counter() - t0
    return out


def has_keys(answer, keys):
    low = " ".join((answer or "").lower().split())
    return all(any(w.lower() in low for w in group) for group in keys)


def git(*a):
    return subprocess.run(
        ["git", *a], capture_output=True, text=True, cwd=HERE
    ).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8002")
    ap.add_argument(
        "--wipe-library",
        action="store_true",
        help="remove every document first (use on a scratch instance only)",
    )
    args = ap.parse_args()

    manifest = json.loads((HERE / "uploads" / "manifest.json").read_text())
    questions = [
        json.loads(x)
        for x in (HERE / "uploads" / "questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x
    ]

    lib = call(args.api, "GET", "/documents")
    if lib["documents"] and not args.wipe_library:
        raise SystemExit(
            f"The library already has {len(lib['documents'])} documents. Use a scratch instance, or pass --wipe-library."
        )
    if lib["documents"]:
        call(args.api, "DELETE", "/documents?confirm=yes")

    run = HERE / "results" / ("uploads-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    run.mkdir(parents=True)

    # --- add the documents the way a person would
    docs, jobs, hashes = {}, {}, {}
    t_index = time.perf_counter()
    for d in manifest["documents"]:
        body = download(d["url"])
        hashes[d["id"]] = hashlib.sha256(body).hexdigest()
        if hashes[d["id"]] != d["sha256"]:
            print(
                f"WARNING: {d['id']} content differs from the manifest hash ({hashes[d['id']][:12]} vs {d['sha256'][:12]}); results are not comparable."
            )
        if d["kind"] == "file":
            job = call(
                args.api,
                "POST",
                "/documents/files?filename=" + urllib.request.quote(d["filename"]),
                raw=body,
            )
        else:
            job = call(args.api, "POST", "/documents/urls", {"urls": [d["url"]]})[
                "jobs"
            ][0]
        jobs[d["id"]] = job["id"]
    while True:
        state = call(args.api, "GET", "/documents")
        mine = {j["id"]: j for j in state["jobs"]}
        if all(mine[j]["status"] in ("done", "error") for j in jobs.values()):
            break
        time.sleep(3)
    index_s = time.perf_counter() - t_index
    for did, jid in jobs.items():
        j = mine[jid]
        if j["status"] != "done":
            raise SystemExit(f"Adding {did} failed: {j.get('error')}")
        docs[did] = {"doc_id": j["doc_id"], "title": j["title"], "chunks": j["chunks"]}
    print(f"indexed in {index_s:.0f} s: { {k: v['chunks'] for k, v in docs.items()} }")

    # --- ask
    rows = []
    for q in questions:
        r = ask(args.api, q["question"])
        sources = r.get("sources", [])[:5]
        answer = (r.get("done") or {}).get("answer") or ""
        gold = docs[q["doc"]]["doc_id"] if q["doc"] else None
        row = {
            **q,
            "answer": answer,
            "error": r.get("error"),
            "latency_s": round(r["latency_s"], 2),
            "ttft_s": round(r.get("ttft_s", 0), 2),
            "retrieved": [
                {
                    "doc": next(
                        (k for k, v in docs.items() if v["doc_id"] == s["source_id"]),
                        None,
                    ),
                    "page": s["page"],
                }
                for s in sources
            ],
        }
        if q["answerable"]:
            in_doc = [s for s in sources if s["source_id"] == gold]
            row["doc_recall_at_5"] = bool(in_doc)
            row["recall_at_5"] = (
                bool(in_doc)
                if not q["pages"]
                else any(s["page"] in (p, p - 1) for s in in_doc for p in q["pages"])
            )
            row["answer_has_keys"] = has_keys(answer, q["keys"])
            cited = {c["n"] for c in (r.get("done") or {}).get("citations", [])}
            row["cites_gold"] = any(s["n"] in cited for s in in_doc)
        else:
            row["abstained"] = bool(ABSTAIN.search(answer))
        rows.append(row)
        print(
            f"{q['id']:>2} {'ok ' if row.get('answer_has_keys', row.get('abstained')) else 'MISS'} {answer[:90]!r}"
        )

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]

    def rate(rs, k):
        return round(sum(r[k] for r in rs) / len(rs), 3) if rs else None

    summary = {
        "answerable_n": len(ans),
        "recall_at_5": rate(ans, "recall_at_5"),
        "doc_recall_at_5": rate(ans, "doc_recall_at_5"),
        "answer_has_keys": rate(ans, "answer_has_keys"),
        "cites_gold": rate(ans, "cites_gold"),
        "pdf": {
            k: rate([r for r in ans if r["doc"] == "csf"], k)
            for k in ("recall_at_5", "answer_has_keys")
        },
        "web_page": {
            k: rate([r for r in ans if r["doc"] == "rfc8259"], k)
            for k in ("recall_at_5", "answer_has_keys")
        },
        "unanswerable_n": len(unans),
        "abstained": rate(unans, "abstained"),
        "index_seconds": round(index_s, 1),
        "median_latency_s": sorted(r["latency_s"] for r in rows)[len(rows) // 2],
    }
    config = {
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "api": args.api,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", ":/rag")),
        "host": platform.platform(),
        "documents": {k: {**docs[k], "sha256_fetched": hashes[k]} for k in docs},
        "summary": summary,
    }
    (run / "raw.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run / "summary.md").write_text(
        "| Metric | Value |\n|--------|-------|\n"
        + "\n".join(f"| {k} | {v} |" for k, v in summary.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print("written to", run)


if __name__ == "__main__":
    main()
