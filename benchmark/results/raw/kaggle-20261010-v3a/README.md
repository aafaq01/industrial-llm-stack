# Kaggle run v3a (2026-10-10), notebook dcgm_vision_t4 from 87ecf0a: partial

- DCGM probe ran and is a result: basic fields returned values (96, embedded and standalone host engine); every
  profiling field failed with `DCGMError_RequiresRoot: This operation is not supported when the host engine is
  running as non root` (see results/dcgm/ and NOT_RUN-vllm-t4-1.5b-fp16-dcgm.txt). Kaggle notebooks do not run as
  root, so DCGM profiling metrics (SM active, DRAM active, tensor pipe) are not available there.
- The notebook then crashed in DCGM teardown (`nv-hostengine` did not exit within 30 s; uncaught TimeoutExpired),
  so the vision section did not run. Not a vision result.
