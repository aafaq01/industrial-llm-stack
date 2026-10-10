#!/usr/bin/env bash
# GSM8K accuracy against any OpenAI-compatible chat endpoint (llama.cpp or vLLM).
# Usage: run_gsm8k.sh BASE_URL MODEL OUT_DIR [LIMIT] [NUM_CONCURRENT]
#   BASE_URL  e.g. http://127.0.0.1:8080/v1/chat/completions
#   MODEL     model name the server expects (vLLM: --served-model-name; llama.cpp: anything)
#   LIMIT     first N test items (default 250, the fixed Phase 2 subset)
# Needs lm_eval on PATH (pip install -r benchmark/requirements.txt).
set -euo pipefail
BASE_URL=$1 MODEL=$2 OUT_DIR=$3 LIMIT=${4:-250} NUM_CONCURRENT=${5:-2}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT_DIR"
lm_eval run \
  --model local-chat-completions \
  --model_args "model=$MODEL,base_url=$BASE_URL,num_concurrent=$NUM_CONCURRENT,max_retries=3,timeout=600,tokenizer_backend=None" \
  --include_path "$HERE/tasks" \
  --tasks gsm8k_pinned \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --limit "$LIMIT" \
  --seed 0,1234,1234,1234 \
  --log_samples \
  --output_path "$OUT_DIR" 2>&1 | tee "$OUT_DIR/lm_eval.log"
