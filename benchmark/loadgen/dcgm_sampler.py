"""DCGM field sampler: every --interval s, append the latest value of each field per GPU to a CSV, until SIGTERM
(or --count polls). Started by run.py --dcgm-fields around each measured window, like the nvidia-smi sampler.

Needs libdcgm and the DCGM Python bindings on PYTHONPATH
(/usr/share/datacenter-gpu-manager-4/bindings/python3 from the datacenter-gpu-manager-4-python3 package).
--host 127.0.0.1 connects to a running nv-hostengine; --host embedded starts DCGM inside this process.
The field watch interval equals --interval, so profiling ratios (DCGM_FI_PROF_*) are averages over each interval.
Errors propagate unchanged (the traceback is the evidence of what DCGM refused and why).
"""

import argparse
import datetime as dt
import signal
import time


def field_label(reader, fid):
    """(tag, unit) from DCGM's field metadata, commas removed; ("", "") if DCGM has none."""
    try:
        meta = reader.m_fieldIdToInfo[fid]
        unit = meta.valueFormat.contents.unit.decode() if meta.valueFormat else ""
        return meta.tag.decode().replace(",", " "), unit.replace(",", " ").strip()
    except (AttributeError, KeyError, ValueError):
        return "", ""


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--fields",
        required=True,
        help="comma-separated DCGM field ids, e.g. 100,155,1002,1005",
    )
    p.add_argument(
        "--gpus", default=None, help="comma-separated DCGM GPU ids (default: all)"
    )
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument(
        "--host", default="127.0.0.1", help="nv-hostengine address, or 'embedded'"
    )
    p.add_argument(
        "--count", type=int, default=0, help="stop after N polls (0 = until SIGTERM)"
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from DcgmReader import DcgmReader

    reader = DcgmReader(
        hostname=None if args.host == "embedded" else args.host,
        fieldIds=[int(f) for f in args.fields.split(",")],
        updateFrequency=int(args.interval * 1e6),  # microseconds
        gpuIds=[int(g) for g in args.gpus.split(",")] if args.gpus else None,
    )
    reader.InitWrapped()  # unlike Init(), raises on connection errors instead of logging them
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    labels = {fid: field_label(reader, fid) for fid in reader.m_fieldIdToInfo}
    polls = 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("utc,gpu,field,tag,unit,value\n")
        while not stop:
            now = dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")
            for gpu, values in reader.GetLatestGpuValuesAsFieldIdDict().items():
                for field, value in values.items():
                    tag, unit = labels.get(field, ("", ""))
                    f.write(f"{now},{gpu},{field},{tag},{unit},{value}\n")
            f.flush()
            polls += 1
            if args.count and polls >= args.count:
                break
            time.sleep(args.interval)
    reader.Shutdown()


if __name__ == "__main__":
    main()
