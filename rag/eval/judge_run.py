"""Judge selected questions of an existing eval run with the local judge.

raw.jsonl keeps retrieved source ids and pages but not chunk text, so each question is asked again to get the
contexts. The re-asked answer and retrieved list are compared with the stored ones (temperature 0, same index);
the stored answer is judged and any mismatch is flagged.

uv run --python 3.12 --group eval python eval/judge_run.py RUN_ID --ids 1,4,7
Writes results/RUN_ID/judge-<UTC timestamp>/ (raw.jsonl, config.json, summary.md).

Judge self-contradictions (a reason that disagrees with its score) are listed by hand after reading every reason,
because no keyword rule tried was reliable: --contradicted 11:answer_relevancy,22:faithfulness
--summarize JUDGE_DIR rebuilds summary.md and config.json from an existing raw.jsonl without judging again.
"""

import argparse
import json
import time
from datetime import UTC, datetime

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from run_eval import HERE, LocalJudge, ask, run_cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--ids", help="required unless --summarize")
    ap.add_argument(
        "--contradicted", default="", help="verified by reading: id:metric,..."
    )
    ap.add_argument("--summarize", help="existing judge-* folder name; no judging")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--judge-url", default="http://127.0.0.1:8081/v1")
    ap.add_argument("--judge-model", default="qwen2.5-7b-instruct-q4_k_m")
    args = ap.parse_args()

    run = HERE / "results" / args.run_id
    contra = {
        (int(c.split(":")[0]), c.split(":")[1])
        for c in args.contradicted.split(",")
        if c
    }
    if args.summarize:
        out = run / args.summarize
        rows = [
            json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()
        ]
        config = json.loads((out / "config.json").read_text())
        write_summary(out, args.run_id, rows, contra, config)
        return
    ids = [int(i) for i in args.ids.split(",")]
    stored = {
        r["id"]: r
        for r in map(json.loads, (run / "raw.jsonl").read_text().splitlines())
    }
    out = run / f"judge-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir()
    judge = LocalJudge(args.judge_url, args.judge_model)

    rows = []
    for i in ids:
        s = stored[i]
        r = ask(args.api, s["question"])
        sources = r.get("sources", [])
        row = {
            "id": i,
            "question": s["question"],
            "answer": s["answer"],
            "answer_matches_run": r.get("done", {}).get("answer") == s["answer"],
            "retrieval_matches_run": [
                {k: x[k] for k in ("source_id", "page")} for x in sources
            ]
            == s["retrieved"],
            "contexts": [x["content"] for x in sources],
        }
        case = LLMTestCase(
            input=s["question"],
            actual_output=s["answer"],
            retrieval_context=row["contexts"],
        )
        for name, metric in (
            ("faithfulness", FaithfulnessMetric(model=judge, async_mode=False)),
            ("answer_relevancy", AnswerRelevancyMetric(model=judge, async_mode=False)),
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
            f"q{i:02d} faith={row['faithfulness']} rel={row['answer_relevancy']} match={row['answer_matches_run']}"
        )
        with (out / "raw.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")

    config = {
        "run_id": args.run_id,
        "ids": ids,
        "judge_url": args.judge_url,
        "judge": "Qwen2.5-7B-Instruct GGUF Q4_K_M via llama.cpp server-b11459, temperature 0, JSON schema output",
        "git_describe": run_cmd("git", "describe", "--always", "--dirty"),
        "rag_api_image_id": run_cmd(
            "docker",
            "inspect",
            "--format",
            "{{.Image}}",
            "industrial-llm-platform-rag-api-1",
        ),
    }
    write_summary(out, args.run_id, rows, contra, config)


def write_summary(
    out, run_id: str, rows: list[dict], contra: set, config: dict
) -> None:
    def mean(key, skip_contradicted=False):
        vals = [
            r[key]
            for r in rows
            if r.get(key) is not None
            and not (skip_contradicted and (r["id"], key) in contra)
        ]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"

    summary = {
        "faithfulness": mean("faithfulness"),
        "answer_relevancy": mean("answer_relevancy"),
        "faithfulness_excluding_contradicted": mean("faithfulness", True),
        "answer_relevancy_excluding_contradicted": mean("answer_relevancy", True),
    }
    config |= {
        "contradicted": sorted(f"{i}:{m}" for i, m in contra),
        "summary": summary,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", newline="\n")
    ids = [r["id"] for r in rows]
    lines = [
        f"# Judge of run {run_id}: ids {', '.join(map(str, ids))}",
        "",
        "The current judged sample for this run. The judged answers are the ones stored in ../raw.jsonl; the",
        "contexts are the same top-5 chunks (source ids and pages checked against ../raw.jsonl).",
        "",
        "| Metric | Mean | n |",
        "|--------|------|---|",
        *(f"| {k} | {fmt(v)} | {n} |" for k, (v, n) in summary.items()),
        "",
        f"Judge self-contradictions, found by reading every reason: {len(contra)} of {2 * len(rows)} scores"
        + (
            f" ({', '.join(f'q{i} {m}' for i, m in sorted(contra))})."
            if contra
            else "."
        ),
        "The `excluding_contradicted` rows drop only those scores; they are a secondary view, not the headline.",
        "",
        "| id | faithfulness | answer_relevancy | contradicted | same retrieval | re-asked answer identical |",
        "|----|--------------|------------------|--------------|----------------|---------------------------|",
    ]
    for r in rows:
        flags = [
            m for m in ("faithfulness", "answer_relevancy") if (r["id"], m) in contra
        ]
        lines.append(
            f"| {r['id']} | {fmt(r['faithfulness'])} | {fmt(r['answer_relevancy'])} | "
            f"{', '.join(flags) or 'no'} | {r['retrieval_matches_run']} | {r['answer_matches_run']} |"
        )
    lines += [
        "",
        "`re-asked answer identical`: the question was asked again only to fetch chunk text. False means llama.cpp",
        "regenerated different text at temperature 0; the judge still scored the stored answer from ../raw.jsonl.",
        "",
        "## Judge reasons",
        "",
    ]
    for r in rows:
        lines += [
            f"- q{r['id']} faithfulness: {r.get('faithfulness_reason') or r.get('faithfulness_error')}",
            f"- q{r['id']} answer_relevancy: {r.get('answer_relevancy_reason') or r.get('answer_relevancy_error')}",
        ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
