# Kaggle run v3b (2026-10-10), notebook trtllm_t4 from 87ecf0a: failed before any benchmark

- vLLM control: install failed, `bench/notebooks/vllm-requirements.txt` was not embedded in the notebook.
- TensorRT-LLM: venv install failed building mpi4py 3.1.6 from sdist
  (`TypeError: new_compiler() got an unexpected keyword argument 'dry_run'`, setuptools in uv's build env).
- NOT_RUN-triton-tensorrtllm-backend.txt is a valid record (needs the NGC container, so Docker; Kaggle has none).
- No benchmark numbers. Not a result.
