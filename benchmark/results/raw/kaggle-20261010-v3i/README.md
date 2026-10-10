# Kaggle run v3i (2026-10-10, ~15:51-16:2xZ), notebook trtllm_t4 from 693b44b: complete

- Same-session vLLM 0.31.0 control (1.5B FP16, c1/c8/c32 x3, GSM8K 250, `--generation-config vllm`) and
  TensorRT-LLM 0.14.0 FP16 (two engines: `bench` batch 32 / input 1024 for loadgen c1/c8/c32 x3, `gsm8k` batch 8 /
  input 2048 for GSM8K 250). Both on 1x T4 in one session. Subject to review before any number is reported.
- NOT_RUN-triton-tensorrtllm-backend.txt: Triton's TensorRT-LLM backend needs the NGC container (Docker).
- model.cache: TensorRT timing cache written by trtllm-build ("Global timing cache in use"), 154,341 bytes; not needed
  to read any result. Review: docs/evidence/v3-trtllm.txt.
