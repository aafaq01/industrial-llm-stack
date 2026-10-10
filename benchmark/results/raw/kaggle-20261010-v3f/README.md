# Kaggle run v3f (2026-10-10, ~12:16-12:41Z), notebook trtllm_t4 from fb5ea24: partial

- Valid: same-session vLLM 0.31.0 control again, 1.5B FP16 c1/c8/c32 x3 and GSM8K 250, `--generation-config vllm`.
  Subject to review.
- TensorRT-LLM: venv install and `import tensorrt_llm` succeeded (pyarrow 20.0.0 and pynvml 11.5.3 pins held), model
  download succeeded, engine build FAILED: the checkpoint conversion could not save to `/tmp/trtllm/ckpt-fp16/`
  (directory not created). See FAILED-build-engine.txt and logs/trtllm-build.log. No TensorRT-LLM numbers.
- NOT_RUN-triton-tensorrtllm-backend.txt: valid record (needs Docker).
