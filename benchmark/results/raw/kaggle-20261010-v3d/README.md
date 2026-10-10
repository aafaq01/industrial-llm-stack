# Kaggle run v3d (2026-10-10 09:30-09:58Z), notebook trtllm_t4 from 80905ea: partial

- Valid: same-session vLLM 0.31.0 control, Qwen2.5-1.5B-Instruct FP16 on 1x T4 with `--generation-config vllm`,
  c1/c8/c32 x3 (results/raw/vllm-t4-1.5b-fp16/) and GSM8K 250 (results/raw/accuracy/). Subject to review.
- Failed: TensorRT-LLM. The venv installed (mpi4py build fix worked), then `import tensorrt_llm` failed:
  `module 'pyarrow' has no attribute 'PyExtensionType'` (locked datasets vs pyarrow). See FAILED-install.txt.
- NOT_RUN-triton-tensorrtllm-backend.txt: valid record (needs the NGC container, so Docker).
