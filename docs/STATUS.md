# Status

| Phase | Owner(s) | State | Evidence |
|-------|----------|-------|----------|
| 0 Plan | orchestrator | approved (v2 slim) | [PLAN.md](PLAN.md) |
| 1 Skeleton + CPU inference | generalist, observability-engineer | done (reviewer PASS 2026-10-07) | [phase1-smoke.txt](evidence/phase1-smoke.txt) |
| 2 GPU inference + benchmarks | inference-engineer, benchmark-analyst | CPU done: rerun with 3 cold + 3 warm repeats per concurrency (reviewer PASS 2026-10-09); GPU run v2 done (2x T4, reviewer PASS 2026-10-09) | [phase2-gpu.txt](evidence/phase2-gpu.txt), [phase2-cpu-rerun.txt](evidence/phase2-cpu-rerun.txt), [phase2-cpu.txt](evidence/phase2-cpu.txt), [phase2-prep.txt](evidence/phase2-prep.txt) |
| 3 RAG | rag-engineer | done (reviewer PASS 2026-10-09 after 2 rounds; last doc defect fixed by orchestrator) | [phase3-rag.txt](evidence/phase3-rag.txt) |
| 4 Observability + SLOs | observability-engineer | done incl. GPU replay dashboard (reviewer PASS 2026-10-09) | [phase4-observability.txt](evidence/phase4-observability.txt) |
| 5 Platform + CI/CD | platform-engineer | done: 5a (reviewer PASS, proof run on bea3e53) and 5b (reviewer PASS 2026-10-09; Jenkins builds #1-#3 green; kind-up slow-network fixes, one post-fix clean run 28m46s with images cached) | [phase5a-platform.txt](evidence/phase5a-platform.txt), [phase5b-ci.txt](evidence/phase5b-ci.txt) |
| 6 Docs | generalist | done (reviewer PASS 2026-10-09 after 2 rounds) | [README](../README.md), [DECISIONS](DECISIONS.md), [LIMITATIONS](LIMITATIONS.md) |

## Accepted deviations
- Phase 1: tempo and otel-collector images are distroless, so they have no compose healthcheck. `--wait` only
  waits for them to run; `scripts/smoke.sh` checks their readiness instead.
- Phase 1: hadolint deferred until the first Dockerfile (Phase 3 or 5).
- Phase 1: Makefile untested on the dev machine (no make on Windows).
- Phase 3: answer quality judged by a local 7B Q4 judge on 12 of 30 questions; see rag/eval/README.md for its limits.
  NIST TN 2148 dropped for lack of evidence of federal authorship.
- Phase 4: TTFT SLO threshold is 60 s, a CPU dev target set from cold-prompt measurements.
- Phase 2: GPU sampler fields added after run v2 (clocks.mem, clocks_event_reasons.active): resolved, they returned
  samples on a T4 in runs v3c to v3i (clock event reasons per run in benchmark/RESULTS.md, Raw files and flags).
- Evidence redaction (2026-10-10): the local Windows user name, the Kaggle user name (now `<kaggle-user>`) and one
  LAN IP (now `192.168.x.x`) were replaced in committed evidence and raw logs. No number or other text changed.
- History rewrite (2026-10-10): personal text was removed from docs/LEARN.md in the last commits, which changed their
  ids. Old to new: c3febc1 82a8141, d3acf51 944586c, 5fc13f2 782b343, 2f232e4 53ccd0a, 13662aa 0817da1, 5cd05ee 08a4e3d,
  45477af 93d0324. Commits before c3febc1, which raw run files cite, are unchanged.

## v3 additions

| Item | Owner | State | Evidence |
|------|-------|-------|----------|
| 7 GitHub repo + Actions | orchestrator, platform-engineer | done: private repo, first Actions run green (lint, test, build, scan) | [v3-github-actions.txt](evidence/v3-github-actions.txt) |
| 8 OpenShift (OKD on CRC) | platform-engineer | done: charts on OKD 4.22 via OpenShift Local, restricted-v2, Routes, cited chat (reviewer PASS); single node, KEDA not load-tested, okd-up.sh not run end to end on a fresh machine | [v3-openshift-okd.txt](evidence/v3-openshift-okd.txt) |
| 9 Loki + Langfuse | observability-engineer | done (reviewer: runtime PASS, doc fixes applied) | [v3-loki-langfuse.txt](evidence/v3-loki-langfuse.txt) |
| 10 DCGM on Kaggle | inference-engineer | done, question answered: DCGM installs and basic fields return values; every profiling field fails with `DCGMError_RequiresRoot` (no root on Kaggle), so the DCGM re-run was not run | [kaggle-20261010-v3a/README.md](../benchmark/results/raw/kaggle-20261010-v3a/README.md) |
| 11 Vision (YOLOv8n) | inference-engineer, benchmark-analyst | done (reviewer PASS, see [v3-reviews.txt](evidence/v3-reviews.txt)): ORT CUDA and TensorRT FP16 speed (v3c), Triton via PyTriton Python backend (v3e), mAP on 500 COCO images (v3g), ONNX Runtime CPU baseline; Triton's native TensorRT backend not run (needs Docker) | [RESULTS.md](../benchmark/RESULTS.md) Vision, [v3-vision-cpu.txt](evidence/v3-vision-cpu.txt) |
| 12 TensorRT-LLM FP16 | inference-engineer, benchmark-analyst | done (reviewer PASS, see [v3-reviews.txt](evidence/v3-reviews.txt)): TRT-LLM 0.14.0 FP16 vs same-session vLLM in run v3i, two engines; Triton's TensorRT-LLM backend not run (needs Docker) | [v3-trtllm.txt](evidence/v3-trtllm.txt), [RESULTS.md](../benchmark/RESULTS.md) |
