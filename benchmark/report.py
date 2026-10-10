"""Build benchmark/RESULTS.md and benchmark/charts/*.png from committed raw files. Offline, no network.

python benchmark/report.py

Reads loadgen runs (benchmark/results/raw/**/*.json with "schema"; "workload" "vision" and "vision-accuracy" files
from benchmark/vision/ get their own tables) and lm-eval results
(benchmark/results/raw/**/accuracy/<label>/**/results_*.json). Anything under a "smoke" directory or marked
"smoke": true is skipped.
"""

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "loadgen"))
from metrics import cv, decode_checks, percentile

HERE = Path(__file__).parent
RAW = HERE / "results" / "raw"
CV_LIMIT = 0.10
WARMUP_DROP = 0.10
WARM_SHARE = (
    0.5  # prompt-cache share at or above this = warm-cache repeat (llama.cpp only)
)
CLOCK_DIP = (
    0.6  # GPU SM clock p50 below this share of the run's max SM clock = low-clock flag
)
TEMP_LIMIT = 85  # GPU temperature (C) at or above this is flagged
# Host conditions not recorded in the raw files, keyed by cell name or prefix; each cites the evidence file it comes from.
HOST_NOTES = {
    "kaggle-": (
        "Integrity checks (attention backend, AWQ kernel, TP, power limit, server warnings) and flag explanations",
        "../docs/evidence/phase2-gpu.txt",
    ),
    "llamacpp-cpu-1.5b-q4km": (
        "not dedicated (desktop apps open, 7 to 43% host CPU before the run, not sampled during runs). "
        "Flag explanations, including the unexplained c8 r3 slowdown",
        "../docs/evidence/phase2-cpu.txt",
    ),
    "cpu-20261009/": (
        "kind cluster and all compose services except llamacpp stopped; desktop apps open but idle. "
        "Idle state, run log, cache shares and flag explanations",
        "../docs/evidence/phase2-cpu-rerun.txt",
    ),
}
HOST_NOTES["cpu-vision-20261010/"] = (
    "kind cluster and the whole compose stack stopped; desktop apps open but idle. Idle state, ONNX file match, "
    "thread settings and run log",
    "../docs/evidence/v3-vision-cpu.txt",
)
HOST_NOTES["kaggle-20261010-v3i/"] = (
    "Integrity checks (decode checks, streams, engine builds, server logs, clock event reasons) and caveats",
    "../docs/evidence/v3-trtllm.txt",
)
# Earlier sessions replaced by a rerun: old cell -> new cell. Kept in RESULTS.md for the record, out of the headline.
SUPERSEDED = {"llamacpp-cpu-1.5b-q4km": "cpu-20261009/llamacpp-cpu-1.5b-q4km"}
# Accuracy caveats per session that the result JSON does not carry; each cites the evidence file.
ACCURACY_NOTES = {
    "kaggle-20261009": (
        "Both servers applied the model's generation_config.json defaults to requests that did not set them "
        "(repetition_penalty 1.1; temperature is set to 0 by lm-eval, so top_k/top_p do not matter), per the "
        "vLLM server logs; the delta compares like with like but absolute scores are not stock GSM8K settings. "
        "strict-match also measures format: 113 (FP16) and 123 (AWQ) of 250 responses had no '#### N' answer and scored [invalid] (samples_*.jsonl). Details",
        "../docs/evidence/phase2-gpu.txt",
    ),
}
# Fixed categorical order (reference palette), assigned by sorted label, never cycled.
COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]

# Repeats of one cell must agree on these, otherwise they are not repeats.
CONFIG_KEYS = {
    "max_tokens": lambda r: r["config"]["max_tokens"],
    "ignore_eos": lambda r: r["config"]["ignore_eos"],
    "requests": lambda r: r["config"]["requests"],
    "warmup_requests": lambda r: r["config"]["warmup_requests"],
    "seed": lambda r: r["config"]["seed"],
    "prompts_count": lambda r: r["config"]["prompts_count"],
    "prompts_sha256": lambda r: r["config"].get(
        "prompts_sha256"
    ),  # absent before 2026-10-09
    "git_commit": lambda r: r["env"]["git_commit"],
    "engine_version": lambda r: r["engine_version"],
    "model_revision": lambda r: r["model_revision"],
    "server_args": lambda r: r["server_args"],
}

# (column, getter on summary)
METRICS = [
    ("out tok/s", lambda s: s["output_tok_s"]),
    ("req/s", lambda s: s["request_s"]),
    ("TTFT p50 s", lambda s: s["ttft_s"]["p50"]),
    ("TTFT p95 s", lambda s: s["ttft_s"]["p95"]),
    ("ITL p50 ms", lambda s: s["itl_s"]["p50"] and s["itl_s"]["p50"] * 1000),
    ("TPOT p50 ms", lambda s: s["tpot_s"]["p50"] and s["tpot_s"]["p50"] * 1000),
    ("E2E p50 s", lambda s: s["e2e_s"]["p50"]),
    ("E2E p95 s", lambda s: s["e2e_s"]["p95"]),
    ("E2E p99 s", lambda s: s["e2e_s"]["p99"]),
    ("tok/s per req p50", lambda s: s["per_request_tok_s"]["p50"]),
    ("err rate", lambda s: s["error_rate"]),
]


def is_smoke(path):
    return "smoke" in path.relative_to(RAW).parts


# DCGM profiling fields shown in RESULTS.md (ratios 0 to 1), from run.py --dcgm-fields.
DCGM_FIELDS = {
    100: "SM clock MHz",
    101: "mem clock MHz",
    155: "power W",
    150: "temp C",
    203: "GPU util %",
    1001: "GR engine active",
    1002: "SM active",
    1003: "SM occupancy",
    1004: "tensor pipe active",
    1005: "DRAM active",
    1008: "FP16 pipe active",
}
VISION_METRICS = [
    ("images/s", lambda s: s["images_s"]),
    ("latency p50 ms", lambda s: s["latency_ms"]["p50"]),
    ("latency p95 ms", lambda s: s["latency_ms"]["p95"]),
    ("latency p99 ms", lambda s: s["latency_ms"]["p99"]),
    ("err rate", lambda s: s["error_rate"]),
]
MAP_KEY = "metrics/mAP50-95(B)"
TRITON_NOTE = "Python backend via PyTriton running the TensorRT engine; not Triton's native TensorRT backend"


def load_docs():
    """Every reportable raw JSON with a "schema" key; "workload" absent means a loadgen (LLM) run."""
    docs = []
    for p in sorted(RAW.rglob("*.json")):
        if is_smoke(p):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and d.get("schema") and not d.get("smoke"):
            d["_path"] = p
            # Cell name includes the session directory (e.g. kaggle-20261010) so reruns never pool.
            parts = p.relative_to(RAW).parts
            d["_session"] = parts[0] if len(parts) > 1 else ""
            d["_cell"] = f"{parts[0]}/{d['label']}" if len(parts) > 1 else d["label"]
            docs.append(d)
    return docs


def load_runs(docs=None):
    return [
        d
        for d in (load_docs() if docs is None else docs)
        if d.get("workload", "llm") == "llm"
    ]


def load_accuracy():
    rows = []
    for p in sorted(RAW.rglob("results_*.json")):
        parts = p.relative_to(RAW).parts
        if is_smoke(p) or "accuracy" not in parts[:-1]:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for task, r in d.get("results", {}).items():
            rows.append(
                {
                    "session": parts[0],
                    "label": parts[parts.index("accuracy") + 1],
                    "task": task,
                    "limit": (d.get("config") or {}).get("limit"),
                    "strict": r.get("exact_match,strict-match"),
                    "strict_se": r.get("exact_match_stderr,strict-match"),
                    "flex": r.get("exact_match,flexible-extract"),
                    "flex_se": r.get("exact_match_stderr,flexible-extract"),
                    "path": p,
                    "samples": sorted(
                        p.parent.glob(
                            f"samples_{task}_{p.stem.removeprefix('results_')}.jsonl"
                        )
                    ),
                }
            )
    return rows


def order(runs):
    """Repeats in start order; each gets _rep: 'rN' from the raw file, or '#k' (start order) if repeat numbers repeat."""
    rs = sorted(runs, key=lambda r: (r.get("started_utc", ""), r["repeat"]))
    unique = len({r["repeat"] for r in rs}) == len(rs)
    for i, r in enumerate(rs, 1):
        r["_rep"] = f"r{r['repeat']}" if unique else f"#{i}"
    return rs


def host_samples(session):
    """(utc, cpu_pct, perf_pct, temp_c, passive_pct) rows from <session>/host-samples.csv, or []."""
    p = RAW / session / "host-samples.csv"
    if not p.exists():
        return []
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines()[1:]:
        f = ln.split(",")
        try:
            rows.append((f[0], float(f[1]), float(f[2]), float(f[4]), float(f[5])))
        except (IndexError, ValueError):
            continue
    return rows


def host_window(run):
    """Host samples inside the run's started..finished window (UTC ISO strings compare in order)."""
    a, b = run.get("started_utc", "")[:19], run.get("finished_utc", "")[:19]
    parts = run["_path"].relative_to(RAW).parts
    rows = host_samples(parts[0]) if len(parts) > 1 else []
    return [x for x in rows if a <= x[0][:19] <= b]


def cell_label(run, first_day):
    """Cell name printed everywhere: session/label, a '(<first day>, superseded)' tag for SUPERSEDED cells, regime."""
    base = run["_cell"]
    tag = (
        f" ({first_day.get(base, 'date unknown')}, superseded)"
        if base in SUPERSEDED
        else ""
    )
    reg = regime(run)
    return base + tag + (f" [{reg}]" if reg else "")


def first_days(runs):
    """_cell -> earliest started_utc date among its runs."""
    out = {}
    for r in runs:
        day = r.get("started_utc", "")[:10]
        if day and (r["_cell"] not in out or day < out[r["_cell"]]):
            out[r["_cell"]] = day
    return out


def flags(runs):
    """Reasons a cell (list of repeats of the same config) is suspicious."""
    runs = order(runs)
    out = []
    if len(runs) < 3:
        out.append(f"only {len(runs)} repeat(s), need 3")
    for name, get in (
        ("out tok/s", lambda s: s["output_tok_s"]),
        ("TTFT p50", lambda s: s["ttft_s"]["p50"]),
        ("E2E p95", lambda s: s["e2e_s"]["p95"]),
    ):
        c = cv([get(r["summary"]) for r in runs])
        if c is not None and c > CV_LIMIT:
            out.append(f"CV {c:.2f} on {name} across repeats (> {CV_LIMIT})")
    differ = [
        k
        for k, get in CONFIG_KEYS.items()
        if len({json.dumps(get(r)) for r in runs}) > 1
    ]
    if differ:
        out.append("repeats disagree on " + ", ".join(differ))
    for r in runs:
        _, _, sm50, sm_max, temp = gpu_stats(r)
        if sm50 and sm_max and sm50 < CLOCK_DIP * sm_max:
            out.append(
                f"{r['_rep']} GPU SM clock p50 {sm50:.0f} MHz vs max {sm_max:.0f} MHz ({clock_reasons([r])})"
            )
        if temp and temp >= TEMP_LIMIT:
            out.append(f"{r['_rep']} GPU temperature up to {temp:.0f} C")
    worst = max((r["summary"]["error_rate"] or 0) for r in runs)
    if worst > 0:
        out.append(f"error rate up to {worst:.3f}")
    checks = [run_decode_checks(r) for r in runs]
    off = sum(c["output_tokens_not_max"] or 0 for c in checks)
    if off:
        out.append(
            f"{off} request(s) decoded a token count other than max_tokens "
            f"({runs[0]['config']['max_tokens']}) despite ignore_eos"
        )
    cut = sum(c["streams_truncated"] for c in checks)
    if cut:
        out.append(
            f"{cut} request stream(s) truncated (fewer chunks than tokens, long silent tail): "
            "TPOT and ITL invalid for this cell"
        )
    # Warmup effect: repeat 1 clearly slower than later repeats means the warmup was too short
    # (or, on llama.cpp, that repeat 1 ran with a colder prompt cache).
    tput = [r["summary"]["output_tok_s"] for r in runs]
    if (
        len(tput) > 1
        and None not in tput
        and tput[0] < (1 - WARMUP_DROP) * statistics.fmean(tput[1:])
    ):
        first = runs[0]["_rep"]
        what = (
            f"{first} slower (llama.cpp: check the prompt cache share)"
            if any(cache_share(r) is not None for r in runs)
            else "warmup effect"
        )
        out.append(
            f"{what}: {first} out tok/s {tput[0]:.1f} vs later repeats mean {statistics.fmean(tput[1:]):.1f}"
        )
    return out


def run_decode_checks(run):
    """decode_checks recomputed from the raw per-request records (works for runs older than the summary field)."""
    cfg = run.get("config") or {}
    return decode_checks(
        run.get("requests") or [], cfg.get("max_tokens"), cfg.get("ignore_eos")
    )


def stream_invalid(runs):
    """True if any repeat had a truncated stream, which makes TPOT and ITL undercount."""
    return any(run_decode_checks(r)["streams_truncated"] for r in runs)


STREAM_METRICS = (
    "ITL p50 ms",
    "TPOT p50 ms",
)  # shown as "invalid" for cells with truncated streams


def trtllm_comparison(runs):
    """TensorRT-LLM vs vLLM lines, only within one session (same model and precision); never across sessions.
    The like-for-like statement is generated from the raw files: prompts sha256, max_tokens, ignore_eos,
    concurrency levels and the decode checks."""
    md = []
    sessions = sorted({r["_session"] for r in runs if r["engine"] == "trtllm"})
    for s in sessions:
        trt = [r for r in runs if r["_session"] == s and r["engine"] == "trtllm"]
        key = {(r["model"], r["precision"]) for r in trt}
        vl = [
            r
            for r in runs
            if r["_session"] == s
            and r["engine"] == "vllm"
            and (r["model"], r["precision"]) in key
        ]
        if not vl:
            md += [
                f"{s}: no vLLM rows in the same session; TensorRT-LLM is not compared with vLLM from other sessions.",
                "",
            ]
            continue

        def facts(rs):
            dc = [run_decode_checks(r) for r in rs]
            return {
                "prompts_sha256": sorted(
                    {r["config"].get("prompts_sha256") or "unknown" for r in rs}
                ),
                "max_tokens": sorted({r["config"]["max_tokens"] for r in rs}),
                "ignore_eos": sorted({r["config"]["ignore_eos"] for r in rs}),
                "conc": sorted({r["concurrency"] for r in rs}),
                "off": sum(d["output_tokens_not_max"] or 0 for d in dc),
                "cut": sum(d["streams_truncated"] for d in dc),
                # output length per prompt (measured, successful requests); compared when ignore_eos is off
                "lens": {
                    q.get("prompt_id", i): q["output_tokens"]
                    for r in rs
                    for i, q in enumerate(
                        x for x in r["requests"] if not x["warmup"] and not x["error"]
                    )
                },
            }

        a, b = facts(trt), facts(vl)
        problems = [
            k
            for k in ("prompts_sha256", "max_tokens", "ignore_eos")
            if a[k] != b[k] or len(a[k]) != 1
        ]
        if a["off"] or b["off"]:
            problems.append("output token counts")
        if a["cut"] or b["cut"]:
            problems.append("truncated streams")
        if a["ignore_eos"] != [True] or b["ignore_eos"] != [True]:
            # without ignore_eos each engine stops where it likes; only equal output lengths keep it like for like
            if a["lens"] != b["lens"]:
                problems.append("output lengths per prompt (ignore_eos off)")
        md.append(
            f"{s}, TensorRT-LLM vs vLLM: prompts sha256 {', '.join(x[:12] for x in a['prompts_sha256'])} vs "
            f"{', '.join(x[:12] for x in b['prompts_sha256'])}; max_tokens {a['max_tokens']} vs {b['max_tokens']}; "
            f"ignore_eos {a['ignore_eos']} vs {b['ignore_eos']}; concurrency {a['conc']} vs {b['conc']}; requests "
            f"whose output tokens differ from max_tokens: {a['off']} vs {b['off']}; truncated streams: "
            f"{a['cut']} vs {b['cut']}."
        )
        if problems:
            md += [
                f"Not compared: they differ or are invalid on {', '.join(problems)}.",
                "",
            ]
            continue
        md += [
            "",
            "| session | conc | TRT-LLM out tok/s | vLLM out tok/s | TRT-LLM / vLLM | TRT-LLM TTFT p50 s | vLLM TTFT p50 s | TRT-LLM E2E p95 s | vLLM E2E p95 s | TRT-LLM SM clock p50 MHz | vLLM SM clock p50 MHz |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for c in sorted(set(a["conc"]) & set(b["conc"])):
            x = [r["summary"] for r in trt if r["concurrency"] == c]
            y = [r["summary"] for r in vl if r["concurrency"] == c]
            tx = [v["output_tok_s"] for v in x if v["output_tok_s"]]
            ty = [v["output_tok_s"] for v in y if v["output_tok_s"]]
            ratio = (
                f"{statistics.fmean(tx) / statistics.fmean(ty):.2f}"
                if tx and ty
                else "n/a"
            )
            md.append(
                f"| {s} | {c} | {fmt(tx)} | {fmt(ty)} | {ratio} | "
                f"{fmt([v['ttft_s']['p50'] for v in x])} | {fmt([v['ttft_s']['p50'] for v in y])} | "
                f"{fmt([v['e2e_s']['p95'] for v in x])} | {fmt([v['e2e_s']['p95'] for v in y])} | "
                f"{fmt([gpu_stats(r)[2] for r in trt if r['concurrency'] == c])} | "
                f"{fmt([gpu_stats(r)[2] for r in vl if r['concurrency'] == c])} |"
            )
        md += [
            "",
            (
                "Both engines ran under the T4's 70 W power cap (clock event reasons per run in Raw files and flags); "
                "where the SM clock columns differ, part of the gap may be clock, not engine."
            ),
            "",
        ]
    return md


def cache_share(run):
    """Share of measured prompt tokens served from llama.cpp's prompt cache, from the /metrics deltas around
    the measured window (prompt_tokens_total counts only processed tokens). None if not scraped."""
    snaps = [run.get("server_metrics_before"), run.get("server_metrics_after")]
    if not all(isinstance(s, list) for s in snaps):
        return None
    b, a = (
        {ln.split()[0]: float(ln.split()[1]) for ln in s if len(ln.split()) == 2}
        for s in snaps
    )
    keys = ("llamacpp:prompt_tokens_total", "llamacpp:prompt_tokens_cached_total")
    if not all(k in a and k in b for k in keys):
        return None
    done, cached = (a[k] - b[k] for k in keys)
    return cached / (done + cached) if done + cached > 0 else None


def regime(run):
    """'warm cache' / 'cold cache' by measured prompt-cache share; None when the server reports no cache counters."""
    c = cache_share(run)
    if c is None:
        return None
    return "warm cache" if c >= WARM_SHARE else "cold cache"


def per_item(paths, filt="strict-match"):
    """doc_id -> exact_match from lm-eval samples_*.jsonl files."""
    out = {}
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d.get("filter") == filt:
                out[d["doc_id"]] = d["exact_match"]
    return out


def paired_delta(a_paths, b_paths, filt="strict-match"):
    """Mean of (b - a) over shared items with a 95% normal CI; None if under 2 shared items."""
    a, b = per_item(a_paths, filt), per_item(b_paths, filt)
    diffs = [b[k] - a[k] for k in sorted(a.keys() & b.keys())]
    if len(diffs) < 2:
        return None
    mean = statistics.fmean(diffs)
    half = 1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs))
    return len(diffs), mean, mean - half, mean + half


def fmt(values):
    xs = [v for v in values if v is not None]
    if not xs:
        return "n/a"
    m = statistics.fmean(xs)
    digits = 3 if abs(m) < 10 else 1
    return f"{m:.{digits}f}" + (
        f" ± {statistics.stdev(xs):.{digits}f}" if len(xs) > 1 else ""
    )


def gpu_stats(run):
    """mem max, util mean, SM clock p50, SM clock max, temp max over all sampled GPUs (None without samples)."""
    rows = run.get("gpu_samples") or []

    def num(k):
        return [float(r[k]) for r in rows if r.get(k, "").replace(".", "").isdigit()]

    mem, util = num("memory_used_mib"), num("util_gpu_pct")
    sm = num("clock_sm_mhz")
    temp = num("temp_c")
    return (
        (max(mem) if mem else None),
        (statistics.fmean(util) if util else None),
        percentile(sm, 50),
        (max(sm) if sm else None),
        (max(temp) if temp else None),
    )


# NVML clock event reason bits (nvmlClocksEventReason* in nvidia-ml-py 13.615.71, same values as nvml.h).
CLOCK_REASONS = {
    0x1: "GPU idle",
    0x2: "applications clocks setting",
    0x4: "SW power cap",
    0x8: "HW slowdown",
    0x10: "sync boost",
    0x20: "SW thermal slowdown",
    0x40: "HW thermal slowdown",
    0x80: "HW power brake",
    0x100: "display clock setting",
}


def clock_reasons(runs):
    """Why the clocks were where they were, from gpu_samples' clocks_event_reasons; 'cause not recorded' without it."""
    vals = []
    for r in runs:
        for row in r.get("gpu_samples") or []:
            try:
                vals.append(int(row["clocks_event_reasons"], 16))
            except (KeyError, TypeError, ValueError):
                continue
    if not vals:
        return "cause not recorded"
    hits = [
        (sum(bool(v & bit) for v in vals), name) for bit, name in CLOCK_REASONS.items()
    ]
    active = [
        f"{name} active in {n / len(vals):.0%} of samples"
        for n, name in sorted(hits, reverse=True)
        if n
    ]
    return "; ".join(active) if active else "no clock event reason active"


# Why a server flag matters, printed next to session differences in server args.
ARG_NOTES = {
    "--generation-config": (
        "without --generation-config vllm, vLLM applies the model's generation_config.json sampling defaults "
        "(Qwen2.5-1.5B: repetition_penalty 1.1, per the kaggle-20261009 server logs)"
    ),
}


def server_flags(args):
    """'--flag value' strings from a server command line (binary and model path dropped)."""
    toks = (args or "").split()
    out = set()
    for i, t in enumerate(toks):
        if t.startswith("--"):
            nxt = (
                toks[i + 1]
                if i + 1 < len(toks) and not toks[i + 1].startswith("--")
                else ""
            )
            out.add(f"{t} {nxt}".strip())
    return out


def session_notes(runs):
    """session -> one line: server-arg differences vs other sessions of the same cells, and TensorRT-LLM status."""
    by_label = defaultdict(lambda: defaultdict(set))
    for r in runs:
        if r.get("server_args") and r["_session"]:
            by_label[r["label"]][r["_session"]] |= server_flags(r["server_args"])
    notes = defaultdict(list)
    for label, sessions in sorted(by_label.items()):
        for s in sorted(sessions):
            diffs = defaultdict(list)
            for o in sorted(sessions):
                if o == s:
                    continue
                add = sorted(sessions[s] - sessions[o])
                miss = sorted(sessions[o] - sessions[s])
                if add or miss:
                    key = "; ".join(
                        [f"adds {', '.join(add)}"] * bool(add)
                        + [f"lacks {', '.join(miss)}"] * bool(miss)
                    )
                    diffs[key].append(o)
            for key, others in diffs.items():
                notes[s].append(f"Server args of {label} vs {', '.join(others)}: {key}")
    out = {}
    for s in sorted({r["_session"] for r in runs if r["_session"]}):
        engines = {r["engine"] for r in runs if r["_session"] == s}
        not_run = sorted(f.name for f in (RAW / s).rglob("NOT_RUN-trtllm-*.txt"))
        failed = sorted(f.name for f in (RAW / s).rglob("FAILED-*.txt"))
        parts = list(notes.get(s, []))
        if "trtllm" in engines:
            parts.append(
                "Same-session vLLM controls for TensorRT-LLM; TensorRT-LLM rows: present"
            )
        elif not_run and "vllm" in engines:
            parts.append(
                "Same-session vLLM controls for TensorRT-LLM; TensorRT-LLM rows: not produced: "
                + ", ".join(failed or not_run)
            )
        if parts:
            out[s] = f"{s}: " + ". ".join(parts) + "."
    return out


def arg_explanations(lines):
    """One sentence explaining every ARG_NOTES flag that the given session lines mention."""
    why = [v for k, v in ARG_NOTES.items() if any(k in ln for ln in lines)]
    return [f"Why it matters: {'; '.join(why)}."] if why else []


def host_note(name):
    """The HOST_NOTES entry for a cell or session name: exact key, else the longest matching prefix key."""
    if name in HOST_NOTES:
        return HOST_NOTES[name]
    keys = [k for k in HOST_NOTES if k.endswith(("/", "-")) and name.startswith(k)]
    return HOST_NOTES[max(keys, key=len)] if keys else None


def host_note_text(name):
    """' Text: [file](link).' for host_note(name), or ''."""
    hn = host_note(name)
    return (
        f" {hn[0][0].upper()}{hn[0][1:]}: [{Path(hn[1]).name}]({hn[1]})." if hn else ""
    )


def low_clock_notes(cells):
    """One neutral line per GPU cell whose mean SM clock p50 is under CLOCK_DIP x the median of the other GPU cells."""
    sm = {}
    for key, rs in cells.items():
        vals = [gpu_stats(r)[2] for r in rs if gpu_stats(r)[2]]
        if vals:
            sm[key] = statistics.fmean(vals)
    out = []
    for (label, conc), v in sorted(sm.items()):
        others = [x for k, x in sm.items() if k != (label, conc)]
        ref = statistics.median(others) if others else None
        if ref and v < CLOCK_DIP * ref:
            out += [
                (
                    f"{label} c{conc} ran at about {v / ref:.0%} of the SM clock of the other GPU cells "
                    f"(p50 {v:.0f} MHz vs median {ref:.0f} MHz; {clock_reasons(cells[(label, conc)])}). "
                    "Read comparisons with this cell "
                    "with that confound in mind."
                ),
                "",
            ]
    return out


def dcgm_values(run):
    """DCGM field id -> parsed values over the run's samples (all sampled GPUs)."""
    vals = defaultdict(list)
    for s in run.get("dcgm_samples") or []:
        try:
            field, value = int(s["field"]), float(s["value"])
        except (KeyError, TypeError, ValueError):
            continue
        vals[field].append(value)
    return vals


def dcgm_means(run):
    """DCGM field id -> mean value over the run's samples; {} without samples."""
    return {f: statistics.fmean(v) for f, v in dcgm_values(run).items()}


def dcgm_section(cells):
    """DCGM profiling table for GPU cells that recorded DCGM samples or a sampler error; [] otherwise."""
    rows, errors = [], []
    for (label, conc), rs in sorted(cells.items()):
        if not any(r.get("dcgm_samples") or r.get("dcgm_error") for r in rs):
            continue
        ms = [dcgm_means(r) for r in rs]
        n = sum(
            len(v) for r in rs for v in dcgm_values(r).values()
        )  # parsed values only
        prof = [v for r in rs for f in (1002, 1005) for v in dcgm_values(r).get(f, [])]
        if prof and not any(prof):
            errors.append(
                f"- {label} c{conc}: SM active and DRAM active are 0 in every sample under load, so the "
                "profiling counters were probably not measuring; do not read them as idle."
            )
        rows.append(
            f"| {label} | {conc} | {len(rs)} | "
            + " | ".join(fmt([m.get(f) for m in ms]) for f in DCGM_FIELDS)
            + f" | {n} |"
        )
        errors += [
            f"- {label} c{conc} {r.get('_rep', r['repeat'])}: sampler stderr ends "
            f"`{r['dcgm_error'].splitlines()[-1][:200]}`"
            for r in rs
            if r.get("dcgm_error")
        ]
    if not rows:
        return []
    head = ["cell", "conc", "repeats", *DCGM_FIELDS.values(), "samples"]
    return [
        "### DCGM",
        "",
        (
            "DCGM fields sampled every 0.5 s during the measured window: clocks, power, temperature and GPU util "
            "(basic fields, units in the headers) and profiling ratios (0 to 1, each averaged by DCGM over its 0.5 s "
            "watch interval). Shown as the mean over samples and sampled GPUs, then mean ± stdev across repeats; "
            "samples = parsed values. DRAM active near 1 means the memory interface was busy nearly all the time "
            "(memory-bound); SM active and the pipe ratios show compute use. Fields a GPU does not support are n/a."
        ),
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *rows,
        "",
        *errors,
        *([""] if errors else []),
    ]


def vision_flags(rs):
    """Reasons a vision cell (repeats of one label and batch) is suspicious."""
    out = []
    if len(rs) < 3:
        out.append(f"only {len(rs)} repeat(s), need 3")
    c = cv([r["summary"]["images_s"] for r in rs])
    if c is not None and c > CV_LIMIT:
        out.append(f"CV {c:.2f} on images/s across repeats (> {CV_LIMIT})")
    worst = max((r["summary"]["error_rate"] or 0) for r in rs)
    if worst > 0:
        out.append(f"error rate up to {worst:.3f}")
    differ = [
        k
        for k in ("model_sha256", "engine_version", "config", "server_args")
        if len({json.dumps(r.get(k), sort_keys=True) for r in rs}) > 1
    ]
    if len({r["env"].get("git_commit") for r in rs}) > 1:
        differ.append("git_commit")
    if differ:
        out.append("repeats disagree on " + ", ".join(differ))
    return out


def vision_backend_notes(cells):
    """Generated notes: TensorRT throughput drop at the largest batch, and Triton rows vs direct TensorRT."""
    ips = {
        k: statistics.fmean([r["summary"]["images_s"] for r in rs])
        for k, rs in cells.items()
    }
    util = {
        k: statistics.fmean(
            [x[1] for x in map(gpu_stats, rs) if x[1] is not None] or [0]
        )
        for k, rs in cells.items()
    }
    eng = {k: rs[0]["engine"] for k, rs in cells.items()}
    out = []
    for label in sorted({k[0] for k in cells if eng[k] == "trt"}):
        bs = sorted(b for (lab, b) in cells if lab == label)
        best = max(bs, key=lambda b: ips[(label, b)])
        big = bs[-1]
        if big != best and ips[(label, big)] < 0.8 * ips[(label, best)]:
            imgsz = cells[(label, big)][0]["config"].get("imgsz", 640)
            mb = big * 3 * imgsz * imgsz * 4 / 1e6
            out += [
                (
                    f"{label}: images/s falls from {ips[(label, best)]:.0f} at b{best} to {ips[(label, big)]:.0f} at "
                    f"b{big}, with GPU util {util[(label, big)]:.0f}%. Likely, not proven: the FP32 input copy "
                    f"({mb:.0f} MB per b{big} batch, from pageable host memory) dominates the batch time."
                ),
                "",
            ]
    trt_best = {}
    for (lab, b), v in ips.items():
        if eng[(lab, b)] == "trt":
            trt_best[b] = max(trt_best.get(b, 0), v)
    for label in sorted({k[0] for k in cells if eng[k] == "triton"}):
        ks = sorted(k for k in cells if k[0] == label)
        ratios = [
            trt_best[b] / ips[(lab, b)]
            for lab, b in ks
            if b in trt_best and ips[(lab, b)]
        ]
        g = [x for k in ks for x in map(gpu_stats, cells[k])]
        us = [x[1] for x in g if x[1] is not None]
        sms = [x[2] for x in g if x[2] is not None]
        line = (
            f"{label}: these rows measure the PyTriton Python backend over HTTP with FP32 payloads"
            + (
                f" (GPU util {min(us):.0f} to {max(us):.0f}%, SM clock p50 about {statistics.median(sms):.0f} MHz)"
                if us and sms
                else ""
            )
            + ", not Triton's native TensorRT backend"
        )
        if ratios:
            line += (
                f"; direct TensorRT FP16 rows are {min(ratios):.0f} to {max(ratios):.0f}x faster at the same batch "
                "sizes (different sessions and engine builds)"
            )
        out += [line + ".", ""]
    return out


def model_provenance(vruns, vacc, docs):
    """Generated provenance sentences: ONNX exports (from onnx-compare files) and TensorRT engine builds."""
    out = []
    for d in docs:
        if d.get("workload") != "onnx-compare":
            continue
        f = d["files"]
        shas = ", ".join(x["sha256"][:8] for x in f)
        same = d["graph_identical"] and d["weights_identical"]
        out += [
            (
                f"ONNX FP32 exports: {len(f)} files with different bytes ({shas}) "
                + (
                    f"have identical canonical graphs ({f[0]['nodes']} nodes) and identical weights "
                    f"({f[0]['initializer_bytes']:,} initializer bytes); they differ only in "
                    f"{', '.join(k for k in d['differing_fields'] if not k.endswith('_sha256'))}"
                    + (
                        f" (metadata keys: {', '.join(d['differing_metadata_keys'])})"
                        if d["differing_metadata_keys"]
                        else ""
                    )
                    if same
                    else "do NOT have identical graphs or weights"
                )
                + f": [{d['_path'].name}]({d['_path'].relative_to(HERE).as_posix()})."
            ),
            "",
        ]
    acc_models = {a["model"] for a in vacc if a["model"].endswith(".engine")}
    for model in sorted(acc_models):
        uses = sorted(
            {
                (r["model_sha256"][:8], r["engine"], r["_session"])
                for r in vruns
                if r["model"] == model and r.get("model_sha256")
            }
            | {
                (a["model_sha256"][:8], "mAP", a["_session"])
                for a in vacc
                if a["model"] == model and a.get("model_sha256")
            }
        )
        if len({u[0] for u in uses}) > 1:
            role = {
                "trt": "timed directly",
                "triton": "behind Triton",
                "mAP": "used for mAP",
            }
            out += [
                (
                    f"TensorRT engines ({model}) are separate builds per session and not bit-reproducible: "
                    + ", ".join(f"{sha} {role.get(e, e)} ({s})" for sha, e, s in uses)
                    + ". The mAP is of a sibling build of the timed engine, not the same file."
                ),
                "",
            ]
    return out


def vision_section(vruns, vacc, docs=()):
    """RESULTS.md lines for vision benchmark runs and vision accuracy files; [] when there are none."""
    md = []
    if vruns:
        cells = defaultdict(list)
        for r in vruns:
            cells[(r["_cell"], r["batch"])].append(r)
        head = ["cell", "hardware", "backend", "precision", "batch", "repeats"]
        head += [m for m, _ in VISION_METRICS] + ["GPU util mean %", "SM clock p50 MHz"]
        md += [
            "## Vision (YOLOv8n, Ultralytics, AGPL-3.0)",
            "",
            (
                "One request is one batch of 640x640 images, sent as soon as the previous one returns. Timed: the "
                "model call including host to device copies and the output copy back; not JPEG decode, letterbox "
                "or NMS. Latency is per batch. CPU and GPU rows are not like for like. Mean ± stdev across repeats."
            ),
            "",
            "| " + " | ".join(head) + " |",
            "|" + "---|" * len(head),
        ]
        notes = []
        for (label, batch), rs in sorted(cells.items()):
            rs[:] = order(rs)
            g = [gpu_stats(r) for r in rs]
            backend = rs[0]["env"].get("backend") or {}
            engine = f"{rs[0]['engine']} {rs[0]['engine_version']}"
            if rs[0]["engine"] == "triton":
                engine += f" ({backend.get('serving') or TRITON_NOTE}; TensorRT {backend.get('tensorrt') or 'version not recorded'})"
            if rs[0]["engine"] == "ort-cpu" and "intra_op_num_threads" in backend:
                t = backend["intra_op_num_threads"]
                engine += f" (intra-op threads {t or 'ORT default'})"
            row = [
                label,
                rs[0]["env"].get("hardware") or "n/a",
                engine,
                rs[0]["precision"],
                str(batch),
                str(len(rs)),
            ]
            row += [fmt([get(r["summary"]) for r in rs]) for _, get in VISION_METRICS]
            row += [fmt([x[1] for x in g]), fmt([x[2] for x in g])]
            md.append("| " + " | ".join(row) + " |")
            links = ", ".join(
                f"[{r['_rep']}]({r['_path'].relative_to(HERE).as_posix()})" for r in rs
            )
            fl = vision_flags(rs)
            notes.append(
                f"- **{label} b{batch}** ({rs[0]['model']}, sha256 {(rs[0].get('model_sha256') or 'not recorded')[:12]}): raw {links}"
                + (
                    f"; GPU clocks: {clock_reasons(rs)}"
                    if any(r.get("gpu_samples") for r in rs)
                    else ""
                )
                + ("; FLAGS: " + "; ".join(fl) if fl else "; no flags")
            )
        md += ["", *notes, ""]
        md += vision_backend_notes(cells)
        for session in sorted({r["_cell"].split("/")[0] for r in vruns}):
            hs = sorted(
                {
                    x
                    for r in vruns
                    if r["_cell"].startswith(session + "/")
                    for x in host_window(r)
                }
            )
            if hs:
                cpu, perf = [x[1] for x in hs], [x[2] for x in hs]
                temp, pas = [x[3] for x in hs], [x[4] for x in hs]
                md += [
                    (
                        f"Host for {session}: samples inside the run windows, warmup included ({len(hs)} unique, "
                        "every ~10 s, "
                        f"[host-samples.csv](results/raw/{session}/host-samples.csv)): total host CPU "
                        f"{statistics.fmean(cpu):.0f}% mean ({min(cpu):.0f} to {max(cpu):.0f}%, the benchmark itself included), "
                        f"processor performance {statistics.fmean(perf):.0f}% of base clock mean "
                        f"({min(perf):.0f} to {max(perf):.0f}%), ACPI thermal zone {min(temp):.0f} to {max(temp):.0f} C, "
                        f"passive limit min {min(pas):.0f}%."
                    )
                    + host_note_text(session + "/"),
                    "",
                ]
    if vacc:
        md += [
            "### Vision accuracy (COCO val2017 subset, Ultralytics validator)",
            "",
            "| session | label | model | model sha256 | precision | images | mAP50-95 | mAP50 | raw |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for a in sorted(vacc, key=lambda a: (a["_session"], a["label"])):
            m = a["metrics"]
            md.append(
                f"| {a['_session']} | {a['label']} | {a['model']} | {(a.get('model_sha256') or 'n/a')[:12]} | "
                f"{a['precision']} | {a['dataset']['images']} | "
                f"{fmt([m.get(MAP_KEY)])} | {fmt([m.get('metrics/mAP50(B)')])} | "
                f"[json]({a['_path'].relative_to(HERE).as_posix()}) |"
            )
        by = {(a["_session"], a["label"]): a for a in vacc}
        for session in sorted({a["_session"] for a in vacc}):
            fp32, fp16 = by.get((session, "onnx-fp32")), by.get((session, "trt-fp16"))
            if (
                fp32
                and fp16
                and MAP_KEY in fp32["metrics"]
                and MAP_KEY in fp16["metrics"]
            ):
                d = fp16["metrics"][MAP_KEY] - fp32["metrics"][MAP_KEY]
                md += [
                    "",
                    f"{session}: TensorRT FP16 minus ONNX FP32 mAP50-95 = {d:+.4f} on the same images.",
                ]
        names = sorted(
            {
                f"{a['dataset'].get('name')} ({a['dataset']['images']} images)"
                for a in vacc
            }
        )
        md += [
            "",
            (
                f"mAP is measured on {', '.join(names)}, not the full 5,000-image val2017, so it is not comparable "
                "to published YOLOv8n figures. COCO annotations are CC BY 4.0; the images keep their Flickr terms of "
                "use; nothing from COCO is committed."
            ),
            "",
            *model_provenance(vruns, vacc, docs),
        ]
        md.append("")
    return md


def charts(cells, out_dir, name, note):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    labels = sorted({label for label, _ in cells})
    panels = [
        ("Output tokens/s (aggregate)", METRICS[0][1]),
        ("TTFT p50 (s)", METRICS[2][1]),
        ("End-to-end latency p95 (s)", METRICS[7][1]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (title, get) in zip(axes, panels, strict=True):
        for i, label in enumerate(labels):
            pts = sorted(
                (c, [get(r["summary"]) for r in runs])
                for (lab, c), runs in cells.items()
                if lab == label
            )
            xs = [c for c, v in pts if any(x is not None for x in v)]
            ys = [
                statistics.fmean([x for x in v if x is not None])
                for c, v in pts
                if any(x is not None for x in v)
            ]
            es = [
                statistics.stdev([x for x in v if x is not None])
                if len([x for x in v if x is not None]) > 1
                else 0
                for c, v in pts
                if any(x is not None for x in v)
            ]
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                marker="o",
                markersize=6,
                linewidth=2,
                capsize=3,
                color=COLORS[i % len(COLORS)],
                label=label,
            )
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("concurrency")
        ax.set_xscale("log", base=2)
        ys_all = [y for line in ax.get_lines() for y in line.get_ydata() if y and y > 0]
        if (
            ys_all and max(ys_all) / min(ys_all) > 20
        ):  # log only when lines span decades
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        else:
            ax.set_ylim(bottom=0)
        ax.set_xticks(sorted({c for _, c in cells}))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        ax.grid(True, color="#e1e0d9", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        (
            "Mean across comparable repeats. n per point in RESULTS.md; points with n < 3 miss the 3-repeat target.\n"
            "Error bars = 1 stdev, none when n = 1. Concurrency axis log2; y axis log when values span decades. "
            + note
        ),
        fontsize=9,
        color="#52514e",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    docs = load_docs()
    runs = load_runs(docs)
    vruns = [d for d in docs if d.get("workload") == "vision"]
    vacc = [d for d in docs if d.get("workload") == "vision-accuracy"]
    acc = load_accuracy()
    md = [
        "# Benchmark results",
        "",
        "Generated by `python benchmark/report.py` from `benchmark/results/raw/`. Do not edit by hand.",
        "",
    ]
    md += [
        "Smoke runs (`results/raw/smoke/`) are excluded. Every number below links to its raw file.",
        "",
        (
            "Every request decodes a fixed number of tokens (`max_tokens` in each raw config, default 128) with `ignore_eos`, so throughput and "
            "latency describe a fixed-length decode, not the model's natural stopping point."
        ),
        "",
    ]
    if not runs and not acc and not vruns and not vacc:
        md += [
            "No results yet. No reportable (non-smoke) benchmark or accuracy runs have been committed.",
            "",
        ]
    if runs:
        cells = defaultdict(list)
        days = first_days(runs)
        for r in runs:
            cells[(cell_label(r, days), r["concurrency"])].append(r)

        def superseded(label):
            return label.endswith(", superseded)") or ", superseded) [" in label

        md += ["## Latency and throughput", ""]
        gpu_engines = sorted(
            {r["engine"] for r in runs if r.get("gpu_samples")},
            key=lambda e: (e != "vllm", e),
        )
        gpu_title = (
            "GPU ("
            + ", ".join(
                {"vllm": "vLLM", "trtllm": "TensorRT-LLM"}.get(e, e)
                for e in gpu_engines
            )
            + ")"
        )
        cache_note = "vLLM prefix caching off." + (
            " TensorRT-LLM KV block reuse off." if "trtllm" in gpu_engines else ""
        )
        # Separate figures: CPU and GPU are not like for like and do not share an axis well.
        for name, title, pick, note in (
            (
                "latency_throughput_gpu.png",
                gpu_title,
                True,
                cache_note,
            ),
            (
                "latency_throughput_cpu.png",
                "CPU (llama.cpp)",
                False,
                f"Split into cold and warm prompt cache (share >= {WARM_SHARE:.0%} = warm).",
            ),
        ):
            sub = {
                k: rs
                for k, rs in cells.items()
                if any(r.get("gpu_samples") for r in rs) == pick
                and not superseded(k[0])
            }
            if sub:
                chart = charts(sub, HERE / "charts", name, note)
                md += [f"![{title}]({chart.relative_to(HERE).as_posix()})", ""]
        md += [
            (
                "Values are mean ± stdev across the repeats in each row (n = repeats column); each repeat's percentile "
                "is computed over its measured requests (warmup excluded)."
            ),
            "",
        ]
        head = ["cell", "hardware", "engine", "precision", "conc", "repeats"] + [
            m for m, _ in METRICS
        ]
        gpu_head = head + [
            "GPU mem max MiB",
            "GPU util mean %",
            "SM clock p50 MHz",
            "GPU temp max C",
        ]

        def table(h):
            return ["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]

        gpu_rows, headline, cold, old = (
            table(gpu_head),
            table(head),
            table(head),
            table(head),
        )
        notes = []
        per_rep = [
            "| cell | conc | repeat | prompt cache | host CPU % | out tok/s | TTFT p50 s | TTFT p95 s | ITL p50 ms | E2E p95 s | raw |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for (label, conc), rs in sorted(cells.items()):
            rs[:] = order(rs)
            g = [gpu_stats(r) for r in rs]
            row = [
                label,
                rs[0]["env"].get("hardware") or "n/a",
                f"{rs[0]['engine']} {rs[0]['engine_version']}",
                rs[0]["precision"],
                str(conc),
                str(len(rs)),
            ]
            bad_stream = stream_invalid(rs)
            row += [
                "invalid"
                if bad_stream and name in STREAM_METRICS
                else fmt([get(r["summary"]) for r in rs])
                for name, get in METRICS
            ]
            if any(r.get("gpu_samples") for r in rs):
                row += [fmt([x[i] for x in g]) for i in (0, 1, 2)]
                row.append(fmt([x[4] for x in g]))
                target = gpu_rows
            elif superseded(label):
                target = old
            else:
                target = cold if label.endswith("[cold cache]") else headline
            target.append("| " + " | ".join(row) + " |")
            links = ", ".join(
                f"[{r['_rep']}]({r['_path'].relative_to(HERE).as_posix()})"
                + (
                    ""
                    if cache_share(r) is None
                    else f" (prompt cache {cache_share(r):.0%})"
                )
                for r in rs
            )
            for r in rs:
                s, c = r["summary"], cache_share(r)
                hw = [x[1] for x in host_window(r)]
                per_rep.append(
                    f"| {label} | {conc} | {r['_rep']} | {'n/a' if c is None else f'{c:.0%}'} | "
                    f"{fmt([statistics.fmean(hw)]) if hw else 'n/a'} | "
                    + " | ".join(
                        "invalid"
                        if name in STREAM_METRICS and stream_invalid([r])
                        else fmt([get(s)])
                        for name, get in (METRICS[i] for i in (0, 2, 3, 4, 7))
                    )
                    + f" | [json]({r['_path'].relative_to(HERE).as_posix()}) |"
                )
            fl = flags(rs)
            notes.append(
                f"- **{label} c{conc}**: raw {links}"
                + ("; FLAGS: " + "; ".join(fl) if fl else "; no flags")
            )
        if len(gpu_rows) > 2:
            md += [
                f"### {gpu_title}",
                "",
                (
                    "Prefix caching is off (`--no-enable-prefix-caching`), so every repeat runs prefill and cells are not "
                    "split by cache regime. GPU mem max mostly reflects vLLM pre-allocating `--gpu-memory-utilization 0.90` "
                    "of each card for weights and KV cache, not the memory the workload needed. GPU columns are over all "
                    "sampled GPUs (both cards for TP=2), from nvidia-smi every 500 ms during the measured window."
                ),
                "",
                *gpu_rows,
                "",
            ]
            if "trtllm" in gpu_engines:
                md += [
                    (
                        "TensorRT-LLM rows: KV cache block reuse (its prefix caching) is off, and its KV cache takes "
                        "0.90 of the GPU memory left free after loading the engine, where vLLM's 0.90 is of the whole "
                        "card. Server and engine build commands: server_args in each raw file and the session's "
                        "server_args/ directory."
                    ),
                    "",
                    *trtllm_comparison(runs),
                ]
            md += low_clock_notes(cells)
            gpu_sessions = {
                r["_session"]
                for rs in cells.values()
                for r in rs
                if r.get("gpu_samples")
            }
            sn = session_notes(runs)
            lines = [sn[x] for x in sorted(sn) if x in gpu_sessions]
            if lines:
                md += [
                    " ".join(
                        ["Session differences:", *lines, *arg_explanations(lines)]
                    ),
                    "",
                ]
            md += dcgm_section(cells)
        if len(headline) > 2 or len(cold) > 2:
            md += ["### CPU (llama.cpp)", ""]
        if len(headline) > 2:
            md += headline + [""]
        if len(cold) > 2:
            md += [
                "Cold prompt cache (llama.cpp repeats with cache share below "
                f"{WARM_SHARE:.0%}, e.g. the first repeat after a server restart):",
                "",
                *cold,
                "",
            ]
        short = sorted(
            f"{label} c{conc} (n={len(rs)})"
            for (label, conc), rs in cells.items()
            if regime(rs[0]) and len(rs) < 3 and not superseded(label)
        )
        if short:
            md += [
                (
                    "Repeats are split by prompt-cache regime because cold and warm repeats are not comparable. "
                    "These regime cells have fewer than 3 comparable repeats and do not meet the 3-repeat criterion: "
                    + ", ".join(short)
                    + "."
                ),
                "",
            ]
        if len(old) > 2:
            md += [
                (
                    "Earlier session, superseded (kept for the record, not in the charts): "
                    + "; ".join(
                        f"{k} replaced by {v}" for k, v in sorted(SUPERSEDED.items())
                    )
                    + ". Its regime cells have fewer than 3 comparable repeats."
                ),
                "",
                *old,
                "",
            ]
        if len(gpu_rows) > 2 and (len(headline) > 2 or len(cold) > 2):
            md += [
                (
                    "GPU and CPU rows are not like for like: different hardware, weights (FP16/AWQ vs GGUF Q4_K_M), "
                    "servers, concurrency levels and cache regimes (vLLM prefix caching off; llama.cpp warm or cold "
                    "prompt cache). Compare within a table, not across."
                ),
                "",
            ]
        for lab in sorted({r["_cell"] for r in runs}):
            env = next(r["env"] for r in runs if r["_cell"] == lab)
            gpus = (env.get("nvidia_smi") or "").splitlines()
            shown = lab + (
                f" ({days.get(lab, 'date unknown')}, superseded)"
                if lab in SUPERSEDED
                else ""
            )
            line = (
                f"Host for {shown}: {env.get('hardware') or 'hardware label not recorded'}; host CPU {env.get('cpu_model')}, "
                f"{env.get('cpu_count')} logical CPUs, {env.get('platform')}"
                + (
                    f"; GPUs in the machine (nvidia-smi): {len(gpus)} x {gpus[0]}"
                    if gpus
                    else ""
                )
                + "."
            )
            hs = sorted({x for r in runs if r["_cell"] == lab for x in host_window(r)})
            if hs:
                cpu, perf = [x[1] for x in hs], [x[2] for x in hs]
                temp, pas = [x[3] for x in hs], [x[4] for x in hs]
                line += (
                    f" Host samples inside the run windows, warmup included ({len(hs)} unique, every ~10 s, "
                    f"[host-samples.csv](results/raw/{lab.split('/')[0]}/host-samples.csv)): total host CPU "
                    f"{statistics.fmean(cpu):.0f}% mean ({min(cpu):.0f} to {max(cpu):.0f}%, llama.cpp's own 4 threads included), "
                    f"processor performance {statistics.fmean(perf):.0f}% of base clock mean "
                    f"({min(perf):.0f} to {max(perf):.0f}%), ACPI thermal zone {min(temp):.0f} to {max(temp):.0f} C, "
                    f"passive limit min {min(pas):.0f}%."
                )
            line += host_note_text(lab)

            md += [line, ""]
        md += ["### Raw files and flags", ""] + notes + [""]
        md += [
            "### Per repeat",
            "",
            "Each repeat on its own, so cold-cache and warm-cache repeats are visible rather than averaged away.",
            "",
            *per_rep,
            "",
        ]
        md += [
            (
                f"Flag rules: CV across repeats > {CV_LIMIT} on out tok/s, TTFT p50 or E2E p95; any error; fewer than 3 repeats; "
                f"first repeat in a row more than {WARMUP_DROP:.0%} below later repeats on out tok/s (warmup effect; on llama.cpp named after the prompt cache instead); "
                "repeats disagree on config, git commit, engine version, model revision or server args. "
                "Repeats show as rN (repeat number in the raw file), or #k (start order) when a cell's raw files share a "
                "repeat number (cold reruns: one run.py call per repeat after a server restart). "
                "Host CPU % = mean of the host samples inside the run's window, where recorded. "
                "Prompt cache % (llama.cpp only) = cached / (cached + processed) prompt tokens over the measured window, "
                "from the `llamacpp:prompt_tokens_cached_total` and `llamacpp:prompt_tokens_total` deltas in each raw file; "
                f"a high share means TTFT reflects cache hits, not prefill. A repeat with share >= {WARM_SHARE:.0%} is "
                "a warm-cache repeat, otherwise cold; cells are split by this, so a row's repeats share one regime. "
                "Runs without cache counters (vLLM, prefix caching off) are not split. "
                f"GPU: a run is flagged when its SM clock p50 is below {CLOCK_DIP:.0%} of that run's max SM clock, or its "
                f"temperature reaches {TEMP_LIMIT} C; a cell whose SM clock is below {CLOCK_DIP:.0%} of the other GPU "
                "cells' median gets the low-clock note above the table."
            ),
            "",
        ]
    md += vision_section(vruns, vacc, docs)
    if acc:
        md += ["## Accuracy (lm-evaluation-harness, GSM8K, exact match)", ""]
        md += [
            "| session | label | task | limit | strict-match | flexible-extract | raw |",
            "|---|---|---|---|---|---|---|",
        ]
        for a in acc:
            cols = [
                "n/a" if a[k] is None else f"{a[k]:.3f} ± {a[k + '_se']:.3f}"
                for k in ("strict", "flex")
            ]
            limit = int(a["limit"]) if a["limit"] else a["limit"]
            md.append(
                f"| {a['session']} | {a['label']} | {a['task']} | {limit} | {cols[0]} | {cols[1]} | "
                f"[json]({a['path'].relative_to(HERE).as_posix()}) |"
            )
        by = {(a["session"], a["label"]): a for a in acc}
        # Paired deltas for known label pairs within one session: (base label, other label, other name, base name).
        pairs = [
            ("gsm8k-1.5b-fp16", "gsm8k-1.5b-awq", "AWQ", "FP16"),
            ("gsm8k-vllm-1.5b-fp16", "gsm8k-trtllm-1.5b-fp16", "TensorRT-LLM", "vLLM"),
        ]
        for session, (base, other, oname, bname) in sorted(
            (x, pr) for x in {a["session"] for a in acc} for pr in pairs
        ):
            fp, aw = by.get((session, base)), by.get((session, other))
            if not (fp and aw and fp["limit"] == aw["limit"]):
                continue
            md += [
                "",
                f"**{session}: {oname} minus {bname}, same {int(fp['limit'])} items**",
                "",
            ]
            for key, filt in (("strict", "strict-match"), ("flex", "flexible-extract")):
                if fp[key] is None or aw[key] is None:
                    continue
                line = (
                    f"- {filt}: {aw[key] - fp[key]:+.3f} ({bname} {fp[key]:.3f} ± {fp[key + '_se']:.3f}, "
                    f"{oname} {aw[key]:.3f} ± {aw[key + '_se']:.3f} stderr)"
                )
                pd = paired_delta(fp["samples"], aw["samples"], filt)
                if pd:
                    n, mean, lo, hi = pd
                    line += f"; paired over {n} items: {mean:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] (normal approximation)"
                md.append(line + ".")
            if session in ACCURACY_NOTES:
                text, link = ACCURACY_NOTES[session]
                md += ["", f"{text}: [{Path(link).name}]({link})."]
        sn = session_notes(runs)
        acc_sessions = sorted({a["session"] for a in acc} & set(sn))
        if acc_sessions:
            md += [
                "",
                " ".join(
                    [
                        "Server settings behind these rows:",
                        *(sn[x] for x in acc_sessions),
                        *arg_explanations([sn[x] for x in acc_sessions]),
                    ]
                ),
            ]
        md.append("")
    (HERE / "RESULTS.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(
        f"wrote {HERE / 'RESULTS.md'}: {len(runs)} runs, {len(acc)} accuracy rows, "
        f"{len(vruns)} vision runs, {len(vacc)} vision accuracy rows"
    )


if __name__ == "__main__":
    main()
