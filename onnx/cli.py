"""sneppx-onnx: ONNX import/export CLI (numpy-only).

Subcommands:

    info <model>            print model summary (nodes, ops, shapes)
    check <model>           structural + shape validation
    shapes <model>          run shape inference and dump value_info
    optimize <in> -o <out>  run optimization passes
    convert <in> -o <out>   re-serialize (opset upgrade / external data)
    quantize <in> -o <out>  insert QDQ pairs around float initializers
    run <model> --input NAME:SHAPE:VAL  numpy-execute the model
    save-external <in> -o <out> --dir <dir> [--threshold N]
    opset <in> -o <out> --version N   re-target the opset import

Exit code 0 on success, 1 on failure.
"""

import argparse
import json
import sys
from typing import List, Optional

import numpy as np

from .check import check_model
from .external_data import save_external_data, extract_external_data
from .inference import infer_shapes
from .optimizer import optimize
from .parser import load_model
from .qdq import quantize_model
from .runtime.numpy_executor import Session
from .serializer import save_model

__all__ = ["main"]


def _resolve_paths(args) -> None:
    """Ensure --out defaults next to the input."""
    if not getattr(args, "out", None):
        base = args.model.rsplit(".onnx", 1)[0] if args.model.endswith(".onnx") else args.model
        args.out = f"{base}_out.onnx"


def cmd_info(args) -> int:
    model = load_model(args.model)
    graph = model.graph
    op_types = {}
    for node in graph.nodes:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
    total_params = sum(
        int(np.prod(i.data.shape)) if i.data is not None else 0
        for i in graph.initializers
    )
    print(f"file:            {args.model}")
    print(f"ir_version:      {model.ir_version}")
    print(f"opset:           {model.opset_version}")
    print(f"producer:        {model.producer_name} {model.producer_version}")
    print(f"graph name:      {graph.name}")
    print(f"nodes:           {len(graph.nodes)}")
    print(f"initializers:    {len(graph.initializers)}  ({total_params} params)")
    print(f"inputs:          {len(graph.inputs)}")
    print(f"outputs:         {len(graph.outputs)}")
    print("ops:")
    for op, count in sorted(op_types.items(), key=lambda kv: -kv[1]):
        print(f"  {op:<24} {count}")
    return 0


def cmd_check(args) -> int:
    model = load_model(args.model)
    ok, errors = check_model(model)
    if ok:
        print(f"{args.model}: OK")
        return 0
    print(f"{args.model}: FAILED")
    for err in errors:
        print(f"  - {err}")
    return 1


def cmd_shapes(args) -> int:
    model = load_model(args.model)
    inferred = infer_shapes(model)
    out = {k: {"shape": [repr(d) if isinstance(d, str) else d for d in v[0]], "dtype": v[1]} for k, v in inferred.items()}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for name in sorted(out):
            print(f"  {name:<40} {out[name]['dtype']:<10} {out[name]['shape']}")
    return 0


def cmd_optimize(args) -> int:
    _resolve_paths(args)
    model = load_model(args.model)
    passes: Optional[List[str]] = None
    if args.passes:
        passes = [p.strip() for p in args.passes.split(",")]
    out = optimize(model, passes=passes, num_times=args.times)
    save_model(out, args.out)
    print(f"optimized -> {args.out}")
    return 0


def cmd_convert(args) -> int:
    _resolve_paths(args)
    model = load_model(args.model)
    if getattr(args, "opset_version", None):
        from .model import OpsetImport

        model.opset_imports = [OpsetImport("", args.opset_version)]
    save_model(model, args.out)
    print(f"converted -> {args.out}")
    return 0


def cmd_quantize(args) -> int:
    _resolve_paths(args)
    model = load_model(args.model)
    quantize_model(model, num_bits=args.bits, per_channel=args.per_channel)
    save_model(model, args.out)
    print(f"quantized (QDQ, {args.bits}-bit) -> {args.out}")
    return 0


def cmd_run(args) -> int:
    model = load_model(args.model)
    session = Session(model)
    inputs = {}
    for spec in args.input:
        name, shape, values = spec.split(":", 2)
        shape = [int(v) for v in shape.split(",")] if shape else []
        arr = np.asarray([float(v) for v in values.split(",")], dtype=np.float32)
        if shape:
            arr = arr.reshape(shape)
        inputs[name] = arr
    results = session.run(inputs)
    for idx, arr in enumerate(results):
        name = session.output_names[idx]
        print(f"{name}: shape={list(arr.shape)} dtype={arr.dtype}")
        flat = arr.reshape(-1)
        print(f"  min={float(flat.min()):.6g} max={float(flat.max()):.6g} "
              f"mean={float(flat.mean()):.6g}")
        if args.verbose and flat.size <= 16:
            print(f"  values={arr.tolist()}")
    return 0


def cmd_save_external(args) -> int:
    _resolve_paths(args)
    model = load_model(args.model)
    save_external_data(model, args.dir, size_threshold=args.threshold,
                       location=args.location)
    save_model(model, args.out)
    print(f"external-data -> {args.out} (payloads in {args.dir}/{args.location})")
    return 0


def cmd_load_external(args) -> int:
    model = load_model(args.model)
    extract_external_data(model, args.dir)
    save_model(model, args.out)
    print(f"external-data hydrated -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sneppx-onnx",
        description="SNEPPX ONNX import/export toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="print model summary")
    p.add_argument("model")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("check", help="validate a model")
    p.add_argument("model")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("shapes", help="infer tensor shapes")
    p.add_argument("model")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_shapes)

    p = sub.add_parser("optimize", help="run optimization passes")
    p.add_argument("model")
    p.add_argument("-o", "--out")
    p.add_argument("--passes", help="comma-separated pass names")
    p.add_argument("--times", type=int, default=1)
    p.set_defaults(func=cmd_optimize)

    p = sub.add_parser("convert", help="re-serialize a model")
    p.add_argument("model")
    p.add_argument("-o", "--out")
    p.add_argument("--opset-version", type=int)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("quantize", help="insert QDQ pairs")
    p.add_argument("model")
    p.add_argument("-o", "--out")
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--per-channel", action="store_true")
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("run", help="run a model with the numpy executor")
    p.add_argument("model")
    p.add_argument("--input", action="append", default=[],
                   help="NAME:SHAPE:VALUES (e.g. x:1,3:0.1,0.2,0.3)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("save-external", help="move large initializers to files")
    p.add_argument("model")
    p.add_argument("-o", "--out")
    p.add_argument("--dir", default="weights")
    p.add_argument("--location", default="weights.bin")
    p.add_argument("--threshold", type=int, default=1024)
    p.set_defaults(func=cmd_save_external)

    p = sub.add_parser("load-external", help="hydrate external data")
    p.add_argument("model")
    p.add_argument("-o", "--out")
    p.add_argument("--dir", default="weights")
    p.set_defaults(func=cmd_load_external)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
