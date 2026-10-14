# Manual review of run uploads-20261014T051604Z

Written by the orchestrator agent after reading all 19 answers in `raw.jsonl` in full. One reviewer, who also wrote the
questions, so treat it as a careful read and not an independent judgment.

## Answerable questions (16)

Correct: 13. Wrong: 3 (ids 1, 7, 16). The keyword rule (`answer_has_keys`) and this read agree on every one of the 16.
Wilson 95% interval for 13 of 16: 0.57 to 0.934.

| id | verdict | note |
|----|---------|------|
| 1 | wrong | Lists six invented functions (PROTECT, IDENTIFY, ASSESS, MAINTAIN, AVOID, ELIMINATE) although the right page was retrieved. |
| 7 | wrong | Says Functions are named after outcomes; the document says each is named after a verb. |
| 16 | wrong | Says RFC 4627; the document says RFC 7159. |
| 15 | correct, noisy | Says six, then pastes grammar lines. |
| 6 | correct | The answer lists the four continuous Functions; its citation numbers [1] [2] [3] [4] are unusual. |
| all others | correct | |

## Questions whose answer is not in the documents (3)

| id | automatic | manual | note |
|----|-----------|--------|------|
| 17 | did not abstain | ambiguous, a flaw in the question | The document says the Quick Start Guides are "publicly available", so "available for free download" is a fair reading. It is still more than the text says. The question was not changed after the run. |
| 18 | did not abstain | failure | Answered "RFC 8259 obsoletes RFC 7159", which is true but not what was asked. |
| 19 | abstained | correct | "not specified in the sources provided". |

So on questions outside the documents the model said it did not know once in three.

## What this does and does not show

16 answerable questions over two documents is small. The questions are factual look-ups written from the text by the same
agent that ran the evaluation. Retrieval was perfect on these two documents (recall@5 1.0), so the failures here are in
generation: the Qwen2.5-1.5B Q4_K_M model on CPU. A larger model (the platform can serve 7B with vLLM on a GPU) was not
tried on these questions.
