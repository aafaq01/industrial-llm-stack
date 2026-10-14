# RAG eval

## Question set

`questions.jsonl`: 30 questions over the 9 corpus documents (3 each, plus one extra for NASA SE Handbook,
NIST AMS 300-1 and NIST SP 800-82r3). Fields: `question`, `reference` (reference answer),
`source_id` (see `../data/SOURCES.md`) and `page` (1-based physical PDF page, the same numbering the ingest
pipeline stores as `page_number`).

How they were made (2026-10-07): the PDFs were converted to text page by page with pypdf 6.19.0 (the same
converter the ingest uses). For each document a few distinctive facts were found by keyword search
(definitions, numbers, named lists), the page was read, and a question and a short reference answer were written
from that page only. One author (the rag-engineer agent), no second reviewer, so some questions may be ambiguous.
On 2026-10-08 NIST TN 2148 was dropped from the corpus (see `../data/SOURCES.md`) and its questions 10 to 12 were
replaced the same way. Questions are factual lookups; none need multi-hop reasoning or tables that pypdf garbles.

## Metrics

- `recall_at_5` (deterministic): the gold document with page `p` or `p - 1` is among the top 5 reranked chunks.
  `p - 1` counts because a chunk is tagged with the page it starts on.
- `doc_recall_at_5`: the gold document is anywhere in the top 5.
- `faithfulness`, `answer_relevancy`: DeepEval metrics scored by a local judge,
  Qwen2.5-7B-Instruct GGUF Q4_K_M on llama.cpp (compose profile `eval`, port 8081), temperature 0,
  JSON-schema constrained output.

## Run

```bash
docker compose --profile eval up -d --wait llamacpp-judge   # first run downloads 4.7 GB
cd rag && DEEPEVAL_TELEMETRY_OPT_OUT=1 uv run --python 3.12 --group eval python eval/run_eval.py --judge-ids 1,4,7,10,11,12,13,16,19,22,25,28   # the current sample, covering all 9 documents
docker compose --profile eval stop llamacpp-judge            # frees about 5 GB RAM
```

To judge chosen ids of an existing run (asks each question again only to fetch chunk text, then judges the stored
answer), and to rebuild a judge summary after listing the self-contradictions found by reading the reasons:

```bash
uv run --python 3.12 --group eval python eval/judge_run.py RUN_ID --ids 1,4,7,10,11,12,13,16,19,22,25,28
uv run --python 3.12 --group eval python eval/judge_run.py RUN_ID --summarize judge-<ts> --contradicted 11:answer_relevancy
```

## Your own documents

`run_uploads_eval.py` checks the path a person uses: it adds a PDF as a file and a web page by address (both public
documents that are not in the demo corpus, listed with hashes in `uploads/manifest.json`), then asks the 19 questions
in `uploads/questions.jsonl`: 16 with an answer in the documents (10 on the NIST Cybersecurity Framework 2.0 PDF, 6 on
RFC 8259) and 3 whose answer is not there. The questions were written from the text by the same agent that ran the
evaluation, no second reviewer. Standard library only, no judge model. It needs an empty library, so run it against a
scratch instance with its own index (see the docstring), never your real one.

Run `results/uploads-20261014T051604Z/` (Qwen2.5-1.5B Q4_K_M on CPU, rag code at commit 68a4487; see the note in its
`config.json` about two files that were not yet committed):

| Metric | Value | n |
|--------|-------|---|
| recall@5 (PDF: page level; web page: document level) | 1.000 | 16 |
| doc recall@5 | 1.000 | 16 |
| answer contains the key words | 0.812 (13 of 16; 95% interval 0.57 to 0.93) | 16 |
| answer cites a source from the right document | 1.000 | 16 |
| said it did not know, when the answer is not in the documents | 0.333 (1 of 3) | 3 |
| indexing both documents (110 passages) | 78.6 s | |
| median time to a full answer | 45.6 s | 19 |

The keyword rule and a full read of the answers (`manual_review.md` in the run folder) agree on all 16. The three wrong
answers are generation failures, not retrieval: the right page was retrieved each time, and the 1.5B model still
invented six "functions" for question 1, gave the wrong reason in question 7, and named the wrong RFC in question 16.
On the three out-of-scope questions it said it did not know once; one of the other two is a flaw in the question (the
document says its resources are "publicly available"). 16 questions over two documents is small, and the numbers say
nothing about other kinds of documents (tables, scans, long manuals, other languages). A larger model was not tried.

## Current results

Run `results/20261008T113955Z/` (9-document corpus), judge sample `results/20261008T113955Z/judge-20261008T120952Z/`:

| Metric | Value | n |
|--------|-------|---|
| recall@5 (page level) | 0.867 | 30 |
| doc recall@5 | 1.000 | 30 |
| faithfulness | 0.592 | 12 |
| answer relevancy | 0.798 | 12 |
| faithfulness, excluding contradicted scores | 0.645 | 11 |
| answer relevancy, excluding contradicted scores | 0.870 | 11 |

`results/20261008T095250Z/` is superseded (10-document corpus including NIST TN 2148).

Each run writes `results/<UTC timestamp>/` with `raw.jsonl` (one row per question: answer, retrieved
source ids and pages, scores, judge reasons, latency, trace id), `config.json` (settings, versions, summary)
and `summary.md`.

## Judge limitations

- A 7B Q4 model is a much weaker judge than the large hosted models DeepEval is tuned for. Scores are useful for
  comparing runs of this pipeline, not as absolute quality numbers.
- The judge reads the same 5 retrieved chunks the generator saw, so faithfulness says nothing about chunks that
  were never retrieved; recall@5 covers that.
- No human calibration of the judge was done.
- The judge contradicts itself. In the current 12-id sample, 2 of 24 scores disagree with their own reason:
  q11 answer_relevancy 0.0 with a reason saying the answer is fully relevant, and q22 faithfulness 0.0 with a reason
  saying the answer has no contradictions. These were found by reading every reason; a keyword rule tried first
  flagged 9 scores, most of them wrongly, and was dropped. Means are reported with and without these two scores.
- Beyond self-contradictions the judge also misreads: q10 faithfulness 0.0 says the answer contradicts "five cycles",
  but the answer matches page 22 (seven times through Phase F); its real fault is missing the question, yet relevancy
  got 1.0. q7 faithfulness 0.5 treats the arcsecond symbol `"` as stray quotes. The excluding-contradicted view
  removes only self-contradictions, not misreadings like these.
- n = 12 is small: one question moves a mean by about 0.08.
- llama.cpp is not fully deterministic at temperature 0 here: re-asking q11 produced different answer text with
  the same retrieved chunks. The judge always scores the answer stored in the run.
