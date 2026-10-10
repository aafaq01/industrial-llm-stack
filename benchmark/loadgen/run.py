"""Closed-loop streaming load generator for any OpenAI-compatible chat endpoint (llama.cpp, vLLM).

Writes one raw JSON per (concurrency, repeat) with full config, environment, per-request records and summary.
Example:
  python benchmark/loadgen/run.py --base-url http://127.0.0.1:8080/v1 --model qwen2.5-1.5b-instruct-q4_k_m \
    --engine llama.cpp --engine-version b11459 --precision gguf-q4_k_m --label llamacpp-cpu-1.5b-q4km \
    --concurrency 1 4 8 --repeats 3 --out-dir benchmark/results/raw
"""

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from metrics import decode_checks, summarize

HERE = Path(__file__).parent


def utcnow():
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sh(*cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model():
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def git_commit():
    commit = sh("git", "-C", str(HERE), "rev-parse", "HEAD")
    if not commit:
        return None
    return commit + (
        "-dirty" if sh("git", "-C", str(HERE), "status", "--porcelain") else ""
    )


async def one_request(client, args, prompt, warmup, t0):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt["prompt"]}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": args.seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.ignore_eos:
        body["ignore_eos"] = (
            True  # llama.cpp and vLLM both accept it; fixes output length
        )
    rec = {
        "prompt_id": prompt["id"],
        "kind": prompt["kind"],
        "warmup": warmup,
        "token_times": [],
        "error": None,
    }
    chunks, usage = 0, None
    rec["start"] = round(time.perf_counter() - t0, 6)
    try:
        async with client.stream(
            "POST", f"{args.base_url}/chat/completions", json=body
        ) as resp:
            if resp.status_code != 200:
                rec["error"] = (
                    f"HTTP {resp.status_code}: {(await resp.aread()).decode(errors='replace')[:300]}"
                )
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    msg = json.loads(data)
                    if msg.get("usage"):
                        usage = msg["usage"]
                    for ch in msg.get("choices") or []:
                        if (ch.get("delta") or {}).get("content"):
                            rec["token_times"].append(
                                round(time.perf_counter() - t0, 6)
                            )
                            chunks += 1
    except Exception as e:  # noqa: BLE001 recorded, never raised: a failed request is data
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["end"] = round(time.perf_counter() - t0, 6)
    rec["chunks"] = chunks
    rec["prompt_tokens"] = (usage or {}).get("prompt_tokens")
    rec["output_tokens"] = (usage or {}).get("completion_tokens", chunks)
    rec["token_source"] = (
        "usage" if usage and "completion_tokens" in usage else "chunks"
    )
    if not rec["error"] and chunks == 0:
        rec["error"] = "no tokens streamed"
    return rec


async def closed_loop(client, args, prompts, n, concurrency, warmup, t0):
    """concurrency workers, each sends its next request as soon as its previous one ends."""
    it = iter(range(n))
    out = []

    async def worker():
        for i in it:
            out.append(
                await one_request(client, args, prompts[i % len(prompts)], warmup, t0)
            )

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return out


class GpuSampler:
    """nvidia-smi every 500 ms in the background, written to a file (no pipe to drain); no-op without nvidia-smi."""

    # clocks_event_reasons.active: bitmask (0x4 SW power cap, 0x20 SW thermal, 0x40 HW thermal); needs driver >= 535.
    QUERY = (
        "timestamp,index,memory.used,memory.total,utilization.gpu,clocks.sm,temperature.gpu,power.draw,"
        "clocks.mem,clocks_event_reasons.active"
    )
    KEYS = (
        "timestamp",
        "gpu",
        "memory_used_mib",
        "memory_total_mib",
        "util_gpu_pct",
        "clock_sm_mhz",
        "temp_c",
        "power_w",
        "clock_mem_mhz",
        "clocks_event_reasons",
    )

    def __init__(self):
        self.proc = None
        if shutil.which("nvidia-smi"):
            fd, self.path = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            ids = os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            )  # sample only the GPUs the server uses
            cmd = [
                "nvidia-smi",
                f"--query-gpu={self.QUERY}",
                "--format=csv,noheader,nounits",
                "-lms",
                "500",
            ]
            cmd += ["-i", ids] if ids else []
            self.proc = subprocess.Popen(
                [*cmd, "-f", self.path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def stop(self):
        if not self.proc:
            return None
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        rows = Path(self.path).read_text().splitlines()
        Path(self.path).unlink()
        return [
            dict(zip(self.KEYS, [c.strip() for c in r.split(",")], strict=False))
            for r in rows
            if r.strip()
        ]


class DcgmSampler:
    """dcgm_sampler.py in the background (--dcgm-fields), same lifecycle as GpuSampler; no-op without fields.
    GPU ids are taken from CUDA_VISIBLE_DEVICES, assuming CUDA and DCGM number the GPUs the same way."""

    def __init__(self, fields, host):
        self.proc = None
        if not fields:
            return
        fd, self.path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        fd, self.err = tempfile.mkstemp(suffix=".err")
        os.close(fd)
        ids = os.environ.get("CUDA_VISIBLE_DEVICES")
        cmd = [
            sys.executable,
            str(HERE / "dcgm_sampler.py"),
            "--fields",
            fields,
            "--host",
            host,
        ]
        cmd += ["--out", self.path] + (["--gpus", ids] if ids else [])
        with open(self.err, "w") as err:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)

    def stop(self):
        """(samples, error text or None); (None, None) when off."""
        if not self.proc:
            return None, None
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        lines = Path(self.path).read_text().splitlines()
        keys = lines[0].split(",") if lines else []  # header written by dcgm_sampler.py
        rows = lines[1:]
        err = Path(self.err).read_text().strip() or None
        Path(self.path).unlink()
        Path(self.err).unlink()
        return [
            dict(zip(keys, r.split(","), strict=False)) for r in rows if r.strip()
        ], err


async def docker_stats():
    if not shutil.which("docker"):
        return None
    try:
        p = await asyncio.create_subprocess_exec(
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(p.communicate(), 30)
        return [json.loads(line) for line in out.decode().splitlines() if line.strip()]
    except (TimeoutError, OSError, json.JSONDecodeError):
        return None


async def scrape(client, url):
    if not url:
        return None
    try:
        r = await client.get(url)
        return [
            line for line in r.text.splitlines() if line and not line.startswith("#")
        ]
    except httpx.HTTPError as e:
        return f"scrape failed: {e}"


async def run_cell(args, prompts, concurrency, repeat, env):
    n = args.requests or max(64, 4 * concurrency)
    n_warm = args.warmup if args.warmup is not None else max(2, concurrency)
    limits = httpx.Limits(
        max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4
    )
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(args.timeout), limits=limits
    ) as client:
        t0 = time.perf_counter()
        started = utcnow()
        # Warmup uses prompts from the end of the list so they do not prime the first measured prompts.
        warm = await closed_loop(
            client, args, prompts[::-1], n_warm, concurrency, True, t0
        )
        metrics_before = await scrape(client, args.metrics_url)
        gpu = GpuSampler()
        dcgm = DcgmSampler(args.dcgm_fields, args.dcgm_host)
        stats_task = asyncio.create_task(docker_stats())
        measured = await closed_loop(client, args, prompts, n, concurrency, False, t0)
        stats = await stats_task
        gpu_samples = gpu.stop()
        dcgm_samples, dcgm_error = dcgm.stop()
        metrics_after = await scrape(client, args.metrics_url)
    records = warm + measured
    return {
        "schema": 1,
        "smoke": args.smoke,
        "label": args.label,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "model": args.model,
        "model_revision": args.model_revision,
        "precision": args.precision,
        "concurrency": concurrency,
        "repeat": repeat,
        "started_utc": started,
        "finished_utc": utcnow(),
        "config": {
            "base_url": args.base_url,
            "requests": n,
            "warmup_requests": n_warm,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": args.seed,
            "ignore_eos": args.ignore_eos,
            "prompts_file": str(Path(args.prompts).name),
            "prompts_count": len(prompts),
            "prompts_sha256": args.prompts_sha256,
            "timeout_s": args.timeout,
            "notes": args.notes,
        },
        "env": env,
        "server_args": args.server_args,
        "gpu_samples": gpu_samples,
        "dcgm_samples": dcgm_samples,
        "dcgm_error": dcgm_error,
        "docker_stats": stats,
        "server_metrics_before": metrics_before,
        "server_metrics_after": metrics_after,
        "summary": summarize(records)
        | {"decode_checks": decode_checks(records, args.max_tokens, args.ignore_eos)},
        "requests": records,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--base-url",
        required=True,
        help="OpenAI API root, e.g. http://127.0.0.1:8080/v1",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--model-revision", default=None, help="HF commit or file sha256")
    p.add_argument("--engine", required=True)
    p.add_argument("--engine-version", required=True)
    p.add_argument("--precision", required=True)
    p.add_argument("--label", required=True, help="cell name prefix for output files")
    p.add_argument(
        "--hardware",
        default=None,
        help="free-text hardware label, e.g. '1x T4 (Kaggle)'",
    )
    p.add_argument("--concurrency", type=int, nargs="+", default=[1])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--requests",
        type=int,
        default=0,
        help="measured requests per run (0 = max(64, 4*concurrency))",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="warmup requests per run (default max(2, concurrency))",
    )
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    p.add_argument(
        "--metrics-url",
        default=None,
        help="Prometheus /metrics to snapshot before and after",
    )
    p.add_argument(
        "--server-args", default=None, help="server command line, stored verbatim"
    )
    p.add_argument(
        "--git-commit", default=None, help="override when not run from a git checkout"
    )
    p.add_argument(
        "--dcgm-fields",
        default=None,
        help="comma-separated DCGM field ids to sample during the measured window (dcgm_sampler.py)",
    )
    p.add_argument(
        "--dcgm-host", default="127.0.0.1", help="nv-hostengine address, or 'embedded'"
    )
    p.add_argument("--notes", default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="mark as smoke run (excluded from RESULTS.md)",
    )
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    raw_prompts = Path(args.prompts).read_bytes()
    args.prompts_sha256 = hashlib.sha256(raw_prompts).hexdigest()
    prompts = [
        json.loads(line)
        for line in raw_prompts.decode("utf-8").splitlines()
        if line.strip()
    ]
    env = {
        "hardware": args.hardware,
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "httpx": httpx.__version__,
        "git_commit": args.git_commit or git_commit(),
        "nvidia_smi": sh(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        )
        if shutil.which("nvidia-smi")
        else None,
    }
    with httpx.Client(timeout=30) as c:
        env["server_models"] = c.get(f"{args.base_url}/models").json()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for conc in args.concurrency:
        for rep in range(1, args.repeats + 1):
            res = asyncio.run(run_cell(args, prompts, conc, rep, env))
            stamp = (
                res["started_utc"]
                .replace(":", "")
                .replace("-", "")
                .replace("+0000", "Z")
            )
            path = out_dir / f"{args.label}-c{conc}-r{rep}-{stamp}.json"
            path.write_text(
                json.dumps(res, separators=(",", ":")) + "\n",
                encoding="utf-8",
                newline="\n",  # LF on Windows too
            )  # compact raw data
            s = res["summary"]
            print(
                f"{path.name}: ok {s['ok']}/{s['requests']} out_tok/s {s['output_tok_s']} "
                f"ttft_p50 {s['ttft_s']['p50']} e2e_p95 {s['e2e_s']['p95']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
