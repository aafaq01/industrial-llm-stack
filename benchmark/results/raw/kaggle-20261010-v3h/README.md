# Kaggle run v3h (2026-10-10 14:20-14:45Z), notebook trtllm_t4 from 414f88e: partial

- Valid: same-session vLLM 0.31.0 control (1.5B FP16 c1/c8/c32 x3, GSM8K 250, `--generation-config vllm`). Subject to review.
- TensorRT-LLM: install, import, download and checkpoint conversion succeeded; trtllm-build FAILED: the GPT attention
  plugin (unfused context path, `--context_fmha disable` because SM75 has no FMHA kernels) needed 21,978,227,456 bytes
  of scratch for max_batch_size 32 / max_input_len 3072 / max_num_tokens 8192, with 15,636,037,632 available on the T4.
  See FAILED-build-engine.txt and logs/trtllm-build.log. No TensorRT-LLM numbers.
