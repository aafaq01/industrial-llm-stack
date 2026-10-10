# RAG eval 20261008T095250Z (superseded)

Superseded by the run of 2026-10-08 after NIST TN 2148 was dropped from the corpus (licence could not be confirmed)
and questions 10 to 12 were replaced. Kept for the record. Its config.json `git_commit` (5cf3443) does not contain
the rag code that ran: the rag tree was uncommitted at the time.

| Metric | Mean | n |
|--------|------|---|
| recall_at_5 | 0.867 | 30 |
| doc_recall_at_5 | 1.000 | 30 |
| faithfulness | 0.767 | 10 |
| answer_relevancy | 0.683 | 10 |
| latency_s | 28.967 | 30 |

recall_at_5: gold doc and page (or the page before) in the top 5 reranked chunks.
doc_recall_at_5: gold doc anywhere in the top 5.
faithfulness, answer_relevancy: DeepEval 0 to 1 with the local judge; see rag/eval/README.md for limits.
Raw per-question rows: raw.jsonl. Config and versions: config.json.
