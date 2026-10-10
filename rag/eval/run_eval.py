"""Score the RAG API on rag/eval/questions.jsonl.

Recall@5 is deterministic. Faithfulness and answer relevancy use DeepEval with a local judge
(Qwen2.5-7B-Instruct Q4_K_M in llama.cpp, compose profile `eval`). See rag/eval/README.md.

uv run --python 3.12 --group eval python eval/run_eval.py [--judge-stride N | --judge-ids 1,2]
"""

import argparse
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from openai import OpenAI

HERE = Path(__file__).resolve().parent


class LocalJudge(DeepEvalBaseLLM):
    """OpenAI-compatible llama.cpp judge. JSON schema constrained decoding keeps a small model parseable."""

    def __init__(self, base_url: str, model: str):
        self.base_url, self.model_id = base_url, model
        super().__init__(model)

    def load_model(self):
        return OpenAI(base_url=self.base_url, api_key="not-needed", timeout=1800)

    def generate(self, prompt: str, schema=None):
        kw = {}
        if schema is not None:
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            }
        r = self.model.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            **kw,
        )
        text = r.choices[0].message.content
        return schema.model_validate_json(text) if schema is not None else text

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return f"local:{self.model_id}"


def run_cmd(*cmd: str) -> str:
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False
    ).stdout.strip()


def ask(api: str, question: str) -> dict:
    """POST /chat and collect the SSE stream."""
    out = {"tokens": 0}
    t0 = time.perf_counter()
    with httpx.stream(
        "POST", f"{api}/chat", json={"message": question, "user": "eval"}, timeout=600
    ) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines():
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


def hit(sources: list[dict], source_id: str, page: int) -> bool:
    # A chunk is tagged with the page it starts on, so a chunk starting one page earlier can hold the answer.
    return any(
        s["source_id"] == source_id and s["page"] in (page, page - 1)
        for s in sources[:5]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--judge-model", default="qwen2.5-7b-instruct-q4_k_m")
    ap.add_argument(
        "--judge-stride",
        type=int,
        default=1,
        help="judge every Nth question (3 = one per document)",
    )
    ap.add_argument(
        "--judge-ids", help="comma-separated ids to judge; overrides --judge-stride"
    )
    args = ap.parse_args()

    questions = [
        json.loads(line)
        for line in (HERE / "questions.jsonl").read_text().splitlines()
        if line
    ]
    run = HERE / "results" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True)
    judge = LocalJudge(args.judge_url, args.judge_model)

    judged = (
        {int(i) for i in args.judge_ids.split(",")}
        if args.judge_ids
        else {q["id"] for q in questions if (q["id"] - 1) % args.judge_stride == 0}
    )
    rows = []
    for q in questions:
        r = ask(args.api, q["question"])
        sources = r.get("sources", [])
        row = {
            **q,
            "answer": r.get("done", {}).get("answer"),
            "error": r.get("error"),
            "retrieved": [{k: s[k] for k in ("source_id", "page")} for s in sources],
            "recall_at_5": hit(sources, q["source_id"], q["page"]),
            "doc_recall_at_5": any(
                s["source_id"] == q["source_id"] for s in sources[:5]
            ),
            "latency_s": round(r["latency_s"], 2),
            "ttft_s": round(r.get("ttft_s", 0), 2),
            "usage": r.get("done", {}).get("usage"),
            "trace_id": r.get("session", {}).get("trace_id"),
        }
        if q["id"] in judged and row["answer"]:
            case = LLMTestCase(
                input=q["question"],
                actual_output=row["answer"],
                retrieval_context=[s["content"] for s in sources],
            )
            for name, metric in (
                ("faithfulness", FaithfulnessMetric(model=judge, async_mode=False)),
                (
                    "answer_relevancy",
                    AnswerRelevancyMetric(model=judge, async_mode=False),
                ),
            ):
                t = time.perf_counter()
                try:
                    metric.measure(case, _show_indicator=False)
                    row[name], row[f"{name}_reason"] = metric.score, metric.reason
                # Record judge failures instead of dropping the question.
                except Exception as e:  # noqa: BLE001
                    row[name], row[f"{name}_error"] = None, repr(e)
                row[f"{name}_judge_s"] = round(time.perf_counter() - t, 1)
        rows.append(row)
        print(
            f"q{q['id']:02d} recall={row['recall_at_5']} faith={row.get('faithfulness')} rel={row.get('answer_relevancy')}"
        )
        with (run / "raw.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    summary = {
        k: mean(k)
        for k in (
            "recall_at_5",
            "doc_recall_at_5",
            "faithfulness",
            "answer_relevancy",
            "latency_s",
        )
    }
    config = {
        "api": args.api,
        "judge_url": args.judge_url,
        "judge_model": args.judge_model,
        "judged_ids": sorted(judged),
        "questions": len(questions),
        "generator": "Qwen2.5-1.5B-Instruct GGUF Q4_K_M via llama.cpp server-b11459 (temperature 0, max_tokens 384)",
        "retrieval": "BM25 top10 + bge-small-en-v1.5 dense top10, RRF join, ms-marco-MiniLM-L6-v2 rerank to top5",
        "judge": "Qwen2.5-7B-Instruct GGUF Q4_K_M via llama.cpp server-b11459, temperature 0, JSON schema output",
        "python": platform.python_version(),
        "packages": {
            p: version(p) for p in ("deepeval", "openai", "httpx", "pydantic")
        },
        # "-dirty" means the code that ran is not exactly the commit
        "git_describe": run_cmd("git", "describe", "--always", "--dirty"),
        "rag_api_image_id": run_cmd(
            "docker",
            "inspect",
            "--format",
            "{{.Image}}",
            "industrial-llm-platform-rag-api-1",
        ),
        "summary": summary,
    }
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n", newline="\n")
    lines = [
        f"# RAG eval {run.name}",
        "",
        "| Metric | Mean | n |",
        "|--------|------|---|",
        *(
            f"| {k} | {'n/a' if v is None else f'{v:.3f}'} | {n} |"
            for k, (v, n) in summary.items()
        ),
        "",
        "recall_at_5: gold doc and page (or the page before) in the top 5 reranked chunks.",
        "doc_recall_at_5: gold doc anywhere in the top 5.",
        "faithfulness, answer_relevancy: DeepEval 0 to 1 with the local judge; see rag/eval/README.md for limits.",
        "Raw per-question rows: raw.jsonl. Config and versions: config.json.",
    ]
    (run / "summary.md").write_text("\n".join(lines) + "\n", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
