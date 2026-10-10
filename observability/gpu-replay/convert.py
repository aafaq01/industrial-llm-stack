"""Convert nvidia-smi gpu_samples from the Phase 2 Kaggle run into OpenMetrics for promtool backfill.

This replays captured data. Nothing here is live and nothing is simulated: every sample is one row of
`gpu_samples` in a committed raw benchmark file.

Usage (from the repo root, stdlib only):
  uv run --python 3.12 --no-project observability/gpu-replay/convert.py \
      benchmark/results/raw/kaggle-20261009/results/raw observability/gpu-replay/out/gpu.om
"""

import glob
import json
import os
import sys
from datetime import UTC, datetime, timedelta

SOURCE = "kaggle-20261009"
MIB = 1024 * 1024
# metric name -> (gpu_samples key, scale to base unit)
METRICS = {
    "gpu_replay_utilization_percent": ("util_gpu_pct", 1),
    "gpu_replay_memory_used_bytes": ("memory_used_mib", MIB),
    "gpu_replay_memory_total_bytes": ("memory_total_mib", MIB),
    "gpu_replay_power_watts": ("power_w", 1),
    "gpu_replay_sm_clock_hertz": ("clock_sm_mhz", 1_000_000),
    "gpu_replay_temperature_celsius": ("temp_c", 1),
}


def parse_ts(s: str) -> datetime:
    # nvidia-smi prints local time; the Kaggle container runs in UTC (checked below against started_utc).
    return datetime.strptime(s, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)


def main(raw_dir: str, out_path: str) -> None:
    files = sorted(glob.glob(os.path.join(raw_dir, "vllm-*", "*.json")))
    if not files:
        sys.exit(f"no vllm-*/*.json under {raw_dir}")
    series: dict[str, dict[str, list[tuple[float, float]]]] = {m: {} for m in METRICS}
    n_rows = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        if d.get("smoke"):
            continue
        lo = datetime.fromisoformat(d["started_utc"]) - timedelta(seconds=60)
        hi = datetime.fromisoformat(d["finished_utc"]) + timedelta(seconds=60)
        run = os.path.splitext(os.path.basename(f))[0]
        for row in d["gpu_samples"]:
            ts = parse_ts(row["timestamp"])
            # Catches a timezone mismatch instead of shifting data silently.
            if not lo <= ts <= hi:
                sys.exit(f"{f}: sample {row['timestamp']} outside run window")
            labels = (
                f'source="{SOURCE}",cell="{d["label"]}",concurrency="{d["concurrency"]}",'
                f'repeat="{d["repeat"]}",run="{run}",gpu="{row["gpu"]}"'
            )
            for name, (key, scale) in METRICS.items():
                series[name].setdefault(labels, []).append(
                    (ts.timestamp(), float(row[key]) * scale)
                )
            n_rows += 1
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_samples = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for name, by_labels in series.items():
            out.write(f"# TYPE {name} gauge\n")
            for labels, points in sorted(by_labels.items()):
                for ts, v in sorted(points):
                    # repr, not :g, so values keep full precision.
                    out.write(f"{name}{{{labels}}} {v!r} {ts:.3f}\n")
                    n_samples += 1
        out.write("# EOF\n")
    assert n_samples == n_rows * len(METRICS), (n_samples, n_rows)
    print(
        f"{len(files)} files, {n_rows} gpu_samples rows, {n_samples} samples -> {out_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
