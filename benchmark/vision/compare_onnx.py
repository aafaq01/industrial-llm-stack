"""Compare ONNX exports whose bytes differ: same graph and same weights, or not?

Hashes a canonical form of each model: tensor names are replaced by their order of first appearance, so renamed
but otherwise identical graphs hash the same. Initializers are hashed by canonical name, dtype, shape and values.
Everything else that differs (metadata, value_info, doc strings, producer) is listed field by field.

uvx --python 3.12 --with onnx==1.23.2 python benchmark/vision/compare_onnx.py --out OUT.json A.onnx B.onnx [...]

Writes a JSON with "schema": 1, "workload": "onnx-compare" that benchmark/report.py reads.
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

import onnx
from onnx import numpy_helper


def canonical(model):
    g = model.graph
    names = {}

    def nm(x):
        return names.setdefault(x, f"t{len(names)}") if x else ""

    for i in g.input:
        nm(i.name)
    nodes = []
    for n in g.node:
        attrs = sorted((a.name, a.SerializeToString()) for a in n.attribute)
        nodes.append(
            (
                n.op_type,
                n.domain,
                tuple(nm(x) for x in n.input),
                tuple(nm(x) for x in n.output),
                tuple(attrs),
            )
        )
    outs = tuple(nm(o.name) for o in g.output)
    graph_hash = hashlib.sha256(repr((nodes, outs)).encode()).hexdigest()
    h = hashlib.sha256()
    total = 0
    for cname, t in sorted((nm(t.name), t) for t in g.initializer):
        a = numpy_helper.to_array(t)
        total += a.nbytes
        h.update(f"{cname}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    return {
        "nodes": len(g.node),
        "op_counts": dict(
            sorted(collections.Counter(n.op_type for n in g.node).items())
        ),
        "canonical_graph_sha256": graph_hash,
        "initializers": len(g.initializer),
        "initializer_bytes": total,
        "initializer_sha256": h.hexdigest(),
    }


def other_fields(model):
    """Fields outside the canonical graph and weights, as comparable strings."""
    g = model.graph
    return {
        "metadata_props": {p.key: p.value for p in model.metadata_props},
        "value_info": len(g.value_info),
        "value_info_sha256": hashlib.sha256(
            b"".join(v.SerializeToString() for v in g.value_info)
        ).hexdigest(),
        "producer": f"{model.producer_name} {model.producer_version}",
        "ir_version": model.ir_version,
        "opset_import": [(o.domain, o.version) for o in model.opset_import],
        "doc_string_bytes": len(model.doc_string)
        + sum(len(n.doc_string) for n in g.node),
        "node_names_sha256": hashlib.sha256(
            "|".join(n.name for n in g.node).encode()
        ).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("files", nargs="+")
    args = p.parse_args()
    files = []
    for f in args.files:
        m = onnx.load(f)
        files.append(
            {
                "path": Path(f).as_posix(),
                "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest(),
                **canonical(m),
                "other": other_fields(m),
            }
        )

    def same(key):
        return len({json.dumps(x[key], sort_keys=True) for x in files}) == 1

    differing = sorted(
        k
        for k in files[0]["other"]
        if len({json.dumps(x["other"][k], sort_keys=True) for x in files}) > 1
    )
    meta_keys = sorted(
        k
        for k in set().union(*(x["other"]["metadata_props"] for x in files))
        if len({x["other"]["metadata_props"].get(k) for x in files}) > 1
    )
    out = {
        "schema": 1,
        "workload": "onnx-compare",
        "label": "onnx-compare",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "onnx": onnx.__version__,
        "files": files,
        "bytes_identical": same("sha256"),
        "graph_identical": same("canonical_graph_sha256"),
        "weights_identical": same("initializer_sha256") and same("initializer_bytes"),
        "differing_fields": differing,
        "differing_metadata_keys": meta_keys,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    for x in files:
        print(
            x["sha256"][:12],
            x["nodes"],
            "nodes",
            x["canonical_graph_sha256"][:12],
            x["initializer_sha256"][:12],
            x["initializer_bytes"],
            "init bytes",
        )
    print(
        "graph identical",
        out["graph_identical"],
        "weights identical",
        out["weights_identical"],
        "differing",
        differing,
        "metadata keys",
        meta_keys,
    )


if __name__ == "__main__":
    main()
