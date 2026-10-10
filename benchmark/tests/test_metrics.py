import json
import math
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "loadgen"), str(ROOT)]

from metrics import (  # noqa: E402 (needs sys.path above)
    cv,
    percentile,
    request_metrics,
    summarize,
)
from report import (  # noqa: E402
    SUPERSEDED,
    cache_share,
    cell_label,
    clock_reasons,
    first_days,
    flags,
    order,
    paired_delta,
    regime,
    session_notes,
)


def rec(start, token_times, end, n=None, warmup=False, error=None):
    return {
        "start": start,
        "token_times": token_times,
        "end": end,
        "output_tokens": n or len(token_times),
        "warmup": warmup,
        "error": error,
    }


def test_percentile_matches_linear_interpolation():
    xs = list(range(1, 101))  # 1..100
    assert percentile(xs, 50) == pytest.approx(50.5)
    assert percentile(xs, 95) == pytest.approx(95.05)
    assert percentile(xs, 99) == pytest.approx(99.01)
    assert percentile([7], 99) == 7
    assert percentile([], 50) is None
    assert percentile([3, 1, 2], 0) == 1 and percentile([3, 1, 2], 100) == 3


def test_ttft_itl_tpot_from_synthetic_timestamps():
    m = request_metrics(rec(10.0, [10.5, 10.6, 10.8, 11.1], 11.2))
    assert m["ttft"] == pytest.approx(0.5)
    assert m["itl"] == pytest.approx([0.1, 0.2, 0.3])
    assert m["tpot"] == pytest.approx(0.6 / 3)
    assert m["e2e"] == pytest.approx(1.2)
    assert m["tok_s"] == pytest.approx(4 / 1.2)


def test_tpot_uses_token_count_when_chunks_carry_several_tokens():
    m = request_metrics(rec(0.0, [1.0, 2.0], 2.0, n=11))
    assert m["tpot"] == pytest.approx(0.1)


def test_summary_excludes_warmup_and_counts_errors():
    records = [
        rec(
            0.0, [5.0, 6.0], 6.0, warmup=True
        ),  # slow warmup must not leak into measured stats
        rec(10.0, [10.2, 10.3], 10.4),
        rec(10.0, [10.4, 10.5], 10.6),
        rec(10.5, [], 10.7, n=0, error="HTTP 500"),
    ]
    s = summarize(records)
    assert s["requests"] == 3 and s["ok"] == 2 and s["errors"] == 1
    assert s["error_rate"] == pytest.approx(1 / 3)
    assert s["ttft_s"]["p50"] == pytest.approx(0.3)
    assert s["wall_s"] == pytest.approx(0.7)
    assert s["output_tok_s"] == pytest.approx(4 / 0.7)
    assert s["warmup_ttft_mean_s"] == pytest.approx(5.0)


def test_cv_and_flags():
    assert cv([10, 10, 10]) == 0
    assert cv([1]) is None
    assert cv([9, 10, 11]) == pytest.approx(0.1)
    cfg = {
        "max_tokens": 128,
        "ignore_eos": True,
        "requests": 64,
        "warmup_requests": 2,
        "seed": 1234,
        "prompts_count": 64,
    }
    base = {
        "config": cfg,
        "env": {"git_commit": "abc"},
        "engine_version": "x",
        "model_revision": "r",
        "server_args": "s",
    }
    stable = [
        {
            **base,
            "repeat": i,
            "summary": {
                "output_tok_s": v,
                "ttft_s": {"p50": 1.0},
                "e2e_s": {"p95": 2.0},
                "error_rate": 0,
                "warmup_ttft_mean_s": 1.0,
            },
        }
        for i, v in enumerate((100, 101, 99), 1)
    ]
    assert flags(stable) == []
    noisy = [
        {**r, "summary": {**r["summary"], "output_tok_s": v}}
        for r, v in zip(stable, (50, 100, 150), strict=True)
    ]
    assert any("CV" in f for f in flags(noisy))
    errs = [{**r, "summary": {**r["summary"], "error_rate": 0.1}} for r in stable]
    assert any("error" in f for f in flags(errs))
    cold = [
        {**r, "repeat": i, "summary": {**r["summary"], "output_tok_s": v}}
        for i, (r, v) in enumerate(zip(stable, (80, 100, 101), strict=True), 1)
    ]
    assert any("warmup" in f for f in flags(cold))
    assert any("repeat" in f for f in flags(stable[:1]))
    mixed = [
        *stable[:2],
        {
            **stable[2],
            "config": {**cfg, "max_tokens": 64},
            "env": {"git_commit": "def"},
        },
    ]
    assert any("disagree on max_tokens, git_commit" in f for f in flags(mixed))

    def gpu(sm, temp):
        return [
            {"clock_sm_mhz": str(c), "temp_c": str(temp), "memory_used_mib": "1"}
            for c in sm
        ]

    throttled = [{**r, "gpu_samples": gpu((550, 560, 1300), 79)} for r in stable]
    assert sum("SM clock" in f for f in flags(throttled)) == 3
    hot = [{**r, "gpu_samples": gpu((1300, 1250, 1300), 86)} for r in stable]
    assert [f for f in flags(hot) if "temperature" in f] == [
        f"r{i} GPU temperature up to 86 C" for i in (1, 2, 3)
    ]


def test_paired_delta_from_lm_eval_samples(tmp_path):
    def samples(name, hits):
        p = tmp_path / name
        rows = [
            {"doc_id": i, "filter": "strict-match", "exact_match": h}
            for i, h in enumerate(hits)
        ]
        rows.append(
            {"doc_id": 0, "filter": "flexible-extract", "exact_match": 0.0}
        )  # other filter ignored
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return [p]

    fp = samples("fp.jsonl", [1, 1, 0, 1, 0, 1, 1, 0])
    aw = samples("aw.jsonl", [1, 0, 0, 1, 0, 1, 0, 0])
    n, mean, lo, hi = paired_delta(fp, aw)
    assert n == 8 and mean == pytest.approx(-0.25)
    half = 1.96 * statistics.stdev([0, -1, 0, 0, 0, 0, -1, 0]) / math.sqrt(8)
    assert (lo, hi) == pytest.approx((-0.25 - half, -0.25 + half))
    assert paired_delta(fp, samples("one.jsonl", [1])) is None  # 1 shared item: no CI


def test_cache_share_from_llamacpp_metrics_deltas():
    run = {
        "server_metrics_before": [
            "llamacpp:prompt_tokens_total 100",
            "llamacpp:prompt_tokens_cached_total 0",
        ],
        "server_metrics_after": [
            "llamacpp:prompt_tokens_total 125",
            "llamacpp:prompt_tokens_cached_total 75",
        ],
    }
    assert cache_share(run) == 0.75
    assert regime(run) == "warm cache"
    run["server_metrics_after"][1] = "llamacpp:prompt_tokens_cached_total 5"
    assert regime(run) == "cold cache"
    assert regime({}) is None  # vLLM: no cache counters, cell not split
    assert cache_share({"server_metrics_before": None}) is None
    assert (
        cache_share({"server_metrics_before": [], "server_metrics_after": []}) is None
    )


def test_order_numbers_repeats_by_start_when_repeat_numbers_collide():
    cold = [
        {"repeat": 1, "started_utc": "2026-10-09T10:12:10+00:00"},
        {"repeat": 1, "started_utc": "2026-10-09T09:55:44+00:00"},
    ]
    assert [r["_rep"] for r in order(cold)] == ["#1", "#2"]
    assert order(cold)[0]["started_utc"].startswith("2026-10-09T09:55")
    warm = [{"repeat": 2, "started_utc": "b"}, {"repeat": 1, "started_utc": "a"}]
    assert [r["_rep"] for r in order(warm)] == ["r1", "r2"]


def test_superseded_cells_are_tagged_wherever_named():
    old = next(iter(SUPERSEDED))
    runs = [
        {"_cell": old, "started_utc": "2026-10-08T19:22:11+00:00"},
        {"_cell": old, "started_utc": "2026-10-08T20:18:40+00:00"},
        {"_cell": "cpu-x/new", "started_utc": "2026-10-09T09:55:44+00:00"},
    ]
    days = first_days(runs)
    assert cell_label(runs[1], days) == f"{old} (2026-10-08, superseded)"
    assert cell_label(runs[2], days) == "cpu-x/new"


def test_clock_reasons_decode_nvml_bits():
    def run(vals):
        return {"gpu_samples": [{"clocks_event_reasons": v} for v in vals]}

    assert clock_reasons([run([None])]) == "cause not recorded"
    assert (
        clock_reasons([{"gpu_samples": [{"clock_sm_mhz": "500"}]}])
        == "cause not recorded"
    )
    r = run(["0x0000000000000004"] * 3 + ["0x0000000000000024"])
    assert clock_reasons([r]) == (
        "SW power cap active in 100% of samples; SW thermal slowdown active in 25% of samples"
    )
    assert clock_reasons([run(["0x0"])]) == "no clock event reason active"


def test_session_notes_name_server_arg_differences():
    base = "vllm serve m --dtype float16 --seed 0"
    runs = [
        {"label": "cell", "_session": "s-old", "engine": "vllm", "server_args": base},
        {
            "label": "cell",
            "_session": "s-new",
            "engine": "vllm",
            "server_args": base + " --generation-config vllm",
        },
    ]
    notes = session_notes(runs)
    assert (
        "s-new: Server args of cell vs s-old: adds --generation-config vllm"
        in notes["s-new"]
    )
    assert "lacks --generation-config vllm" in notes["s-old"]
    runs.append(
        {"label": "trt", "_session": "s-new", "engine": "trtllm", "server_args": None}
    )
    assert "TensorRT-LLM rows: present" in session_notes(runs)["s-new"]
