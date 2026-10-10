"""Pure metric functions. Times are seconds. Shared by run.py, report.py and the tests."""

import itertools
import math
import statistics


def percentile(values, q):
    """Linear interpolation between closest ranks (same as numpy's default). None for empty input."""
    xs = sorted(values)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(values):
    """mean, p50, p95, p99 of a list (all None if empty)."""
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def request_metrics(r):
    """TTFT, inter-token gaps, TPOT and e2e for one request record.

    r has start, end, token_times (arrival time of each streamed content chunk) and output_tokens.
    ITL is the gap between consecutive chunks. TPOT is (last token - first token) / (output_tokens - 1),
    which stays correct when a chunk carries more than one token.
    """
    tt = r["token_times"]
    ttft = tt[0] - r["start"] if tt else None
    itl = [b - a for a, b in itertools.pairwise(tt)]
    n = r["output_tokens"] or 0
    tpot = (tt[-1] - tt[0]) / (n - 1) if tt and n > 1 else None
    e2e = r["end"] - r["start"]
    return {
        "ttft": ttft,
        "itl": itl,
        "tpot": tpot,
        "e2e": e2e,
        "tok_s": n / e2e if e2e > 0 else None,
    }


def summarize(records):
    """Summary over measured (non-warmup) requests."""
    measured = [r for r in records if not r["warmup"]]
    ok = [r for r in measured if not r["error"]]
    per = [request_metrics(r) for r in ok]

    def pick(k):
        return [m[k] for m in per if m[k] is not None]

    wall = (
        (max(r["end"] for r in measured) - min(r["start"] for r in measured))
        if measured
        else 0
    )
    out_tokens = sum(r["output_tokens"] or 0 for r in ok)
    warm = [
        request_metrics(r)["ttft"]
        for r in records
        if r["warmup"] and not r["error"] and r["token_times"]
    ]
    return {
        "requests": len(measured),
        "ok": len(ok),
        "errors": len(measured) - len(ok),
        "error_rate": (len(measured) - len(ok)) / len(measured) if measured else None,
        "wall_s": wall,
        "output_tokens": out_tokens,
        "output_tok_s": out_tokens / wall if wall > 0 else None,
        "request_s": len(ok) / wall if wall > 0 else None,
        "ttft_s": dist(pick("ttft")),
        "itl_s": dist([g for m in per for g in m["itl"]]),
        "tpot_s": dist(pick("tpot")),
        "e2e_s": dist(pick("e2e")),
        "per_request_tok_s": dist(pick("tok_s")),
        "warmup_ttft_mean_s": statistics.fmean(warm) if warm else None,
    }


TAIL_GAP_S = 0.25  # last streamed chunk this long before the response ended ...
TAIL_GAP_ITL = 10  # ... and this many times the request's median chunk gap = the stream went quiet early


def stream_truncated(r):
    """True if the server stopped streaming text before the request ended: fewer chunks than output tokens and a
    long silent tail. Such a request's TPOT and ITL undercount (its tokens arrived in no streamed chunk)."""
    tt, n = r["token_times"], r.get("output_tokens") or 0
    if not tt or len(tt) >= n:
        return False
    gaps = sorted(b - a for a, b in itertools.pairwise(tt))
    median = gaps[len(gaps) // 2] if gaps else 0
    tail = r["end"] - tt[-1]
    return tail > TAIL_GAP_S and tail > TAIL_GAP_ITL * median


def decode_checks(records, max_tokens, ignore_eos):
    """Over measured successful requests: how many decoded a token count other than max_tokens (only checked with
    ignore_eos, where every request must decode exactly max_tokens), and how many streams were truncated."""
    ok = [r for r in records if not r["warmup"] and not r["error"]]
    return {
        "checked": len(ok),
        "output_tokens_not_max": sum(
            1 for r in ok if r.get("output_tokens") != max_tokens
        )
        if ignore_eos
        else None,
        "streams_truncated": sum(1 for r in ok if stream_truncated(r)),
    }


def cv(values):
    """Coefficient of variation (stdev / mean) across repeats; None if under 2 values or mean 0."""
    xs = [v for v in values if v is not None]
    if len(xs) < 2 or statistics.fmean(xs) == 0:
        return None
    return statistics.stdev(xs) / abs(statistics.fmean(xs))
