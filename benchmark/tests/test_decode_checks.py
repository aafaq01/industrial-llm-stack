"""Decode checks (fixed-length decode, truncated streams), the same-session TensorRT-LLM vs vLLM comparison, and
the stream patch applied to TensorRT-LLM 0.14's openai_server.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "loadgen"),
    str(ROOT),
    str(ROOT.parent / "inference" / "trtllm"),
]

from metrics import decode_checks, stream_truncated  # noqa: E402
from report import flags, trtllm_comparison  # noqa: E402
from serve import STREAM_PATCH, patched_server_source  # noqa: E402


def rec(times, end, n, warmup=False, error=None):
    return {
        "start": 0.0,
        "token_times": times,
        "end": end,
        "output_tokens": n,
        "warmup": warmup,
        "error": error,
    }


STEADY = [0.1 + 0.02 * i for i in range(128)]  # 128 chunks, 20 ms apart


def test_full_stream_is_not_truncated():
    assert not stream_truncated(rec(STEADY, STEADY[-1] + 0.01, 128))


def test_stream_that_goes_quiet_is_truncated():
    # 10 chunks, then silence until the end while usage says 128 tokens (TensorRT-LLM 0.14 empty-diff bug)
    assert stream_truncated(rec(STEADY[:10], STEADY[-1], 128))


def test_multi_token_chunks_without_a_tail_are_fine():
    # fewer chunks than tokens is normal when a chunk carries several tokens; no silent tail
    assert not stream_truncated(rec(STEADY[::4], STEADY[-1] + 0.01, 128))


def test_decode_checks_counts_short_outputs_only_with_ignore_eos():
    recs = [
        rec(STEADY, 3.0, 128),
        rec(STEADY[:50], 1.2, 50),
        rec(STEADY, 3.0, 128, warmup=True),
        rec([], 0.1, 0, error="HTTP 500"),
    ]
    assert decode_checks(recs, 128, True) == {
        "checked": 2,
        "output_tokens_not_max": 1,
        "streams_truncated": 0,
    }
    assert decode_checks(recs, 128, False)["output_tokens_not_max"] is None


def run(
    engine,
    conc,
    rep,
    session="kaggle-x",
    tokens=128,
    times=STEADY,
    sha="ab" * 32,
    tput=500.0,
):
    return {
        "engine": engine,
        "engine_version": "v",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "precision": "fp16",
        "concurrency": conc,
        "repeat": rep,
        "started_utc": f"2026-10-12T10:00:0{rep}+00:00",
        "_session": session,
        "config": {
            "max_tokens": 128,
            "ignore_eos": True,
            "prompts_sha256": sha,
            "requests": 64,
            "warmup_requests": 2,
            "seed": 1234,
            "prompts_count": 64,
        },
        "env": {"git_commit": "c"},
        "model_revision": "r",
        "server_args": "s",
        "summary": {
            "output_tok_s": tput,
            "ttft_s": {"p50": 0.05, "p95": 0.06},
            "e2e_s": {"p95": 3.0},
            "error_rate": 0.0,
        },
        "requests": [
            rec(times, times[-1] + 0.01 if len(times) == 128 else 2.6, tokens)
        ],
    }


def test_flags_report_short_outputs_and_truncated_streams():
    rs = [run("trtllm", 1, r, times=STEADY[:10]) for r in (1, 2, 3)]
    rs += [run("trtllm", 1, 4, tokens=100)]
    fl = "; ".join(flags(rs))
    assert "3 request stream(s) truncated" in fl and "TPOT and ITL invalid" in fl
    assert "1 request(s) decoded a token count other than max_tokens (128)" in fl


def test_comparison_only_within_a_session_and_generated_from_raw_facts():
    runs = [run("trtllm", c, r, tput=600.0) for c in (1, 8) for r in (1, 2, 3)]
    runs += [run("vllm", c, r, tput=500.0) for c in (1, 8) for r in (1, 2, 3)]
    runs += [run("vllm", 1, r, session="kaggle-old", tput=100.0) for r in (1, 2, 3)]
    md = "\n".join(trtllm_comparison(runs))
    assert (
        "prompts sha256 abababababab vs abababababab; max_tokens [128] vs [128]" in md
    )
    assert "| kaggle-x | 1 | 600.0 ± 0.0 | 500.0 ± 0.0 | 1.20 |" in md
    assert "kaggle-old" not in md


def test_comparison_refused_without_same_session_vllm_or_with_different_prompts():
    alone = [run("trtllm", 1, r, session="kaggle-y") for r in (1, 2, 3)]
    other = [run("vllm", 1, r, session="kaggle-z") for r in (1, 2, 3)]
    assert "no vLLM rows in the same session" in "\n".join(
        trtllm_comparison(alone + other)
    )
    diff = [run("trtllm", 1, 1), run("vllm", 1, 1, sha="cd" * 32)]
    assert "Not compared: they differ or are invalid on prompts_sha256" in "\n".join(
        trtllm_comparison(diff)
    )


def test_comparison_refused_on_truncated_streams_and_off_length_outputs():
    cut = [run("trtllm", 1, r, times=STEADY[:10]) for r in (1, 2, 3)]
    cut += [run("vllm", 1, r) for r in (1, 2, 3)]
    assert "Not compared: they differ or are invalid on truncated streams" in "\n".join(
        trtllm_comparison(cut)
    )
    short = [run("trtllm", 1, r, tokens=100) for r in (1, 2, 3)]
    short += [run("vllm", 1, r) for r in (1, 2, 3)]
    assert "invalid on output token counts" in "\n".join(trtllm_comparison(short))
    free = [run("trtllm", 1, 1, tokens=90), run("vllm", 1, 1, tokens=110)]
    for r in free:
        r["config"]["ignore_eos"] = False
    md = "\n".join(trtllm_comparison(free))
    assert "output lengths per prompt (ignore_eos off)" in md
    same = [run("trtllm", 1, 1, tokens=90), run("vllm", 1, 1, tokens=90)]
    for r in same:
        r["config"]["ignore_eos"] = False
    assert "Not compared" not in "\n".join(trtllm_comparison(same))


def test_stream_patch_replaces_the_empty_diff_branch_once():
    old, new = STREAM_PATCH
    src = "if delta_text:\n    yield chunk\n" + old + "tail\n"
    out = patched_server_source(src)
    assert new in out and old not in out
    try:
        patched_server_source("no target here")
    except ValueError as e:
        assert "found 0 times" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_attention_workspace_formula_matches_the_failed_v3h_build():
    from build_engine import PROFILES, workspace_bytes

    # trtllm-build log, Kaggle v3h: "requires 21978227456 bytes of scratch space, but only 15636037632 is available"
    assert workspace_bytes(32, 3072, 8192) == 21978227456
    # each profile keeps the workspace near 2.5 GB, far under a T4's 15 GB minus 3.1 GB of weights and the KV cache
    assert all(workspace_bytes(b, n) < 3e9 for b, n, _ in PROFILES.values())
