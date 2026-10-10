# RAG eval 20261008T113955Z

| Metric | Mean | n |
|--------|------|---|
| recall_at_5 | 0.867 | 30 |
| doc_recall_at_5 | 1.000 | 30 |
| faithfulness | 0.200 | 3 (ids 10-12 only; superseded, see notes) |
| answer_relevancy | 0.400 | 3 (ids 10-12 only; superseded, see notes) |
| latency_s | 7.999 | 30 |

recall_at_5: gold doc and page (or the page before) in the top 5 reranked chunks.
doc_recall_at_5: gold doc anywhere in the top 5.
faithfulness, answer_relevancy: DeepEval 0 to 1 with the local judge; see rag/eval/README.md for limits.
Raw per-question rows: raw.jsonl. Config and versions: config.json.

Run notes (added by hand after the run):
- Corpus: 9 documents (NIST TN 2148 dropped). Questions 10 to 12 are new; the other 27 are unchanged.
- The faithfulness and answer_relevancy rows above cover only ids 10, 11, 12 from the run itself. The current judged
  sample for this run is judge-20261008T120952Z/ (ids 1, 4, 7, 10, 11, 12, 13, 16, 19, 22, 25, 28; one or more per
  document): faithfulness 0.592, answer_relevancy 0.798 (n=12). Use those numbers.
- latency_s here (median 4.9 s, TTFT median 2.6 s) is low because 27 of 30 prompts were llama.cpp prompt-cache hits (cached_tokens = prompt_tokens - 1, repeats of earlier runs), so these are not cold numbers. Cold TTFT here (q10-12) was 21.7 to 34.5 s; the mostly cold superseded run had a TTFT median of 22.8 s.
