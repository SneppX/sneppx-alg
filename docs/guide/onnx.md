# ONNX import/export — Guide

The standalone `onnx` package is a numpy-only ONNX toolkit shipped with SNEPPX-Algo.
It can parse, serialize, validate, shape-infer, optimize, and QDQ-quantize ONNX
models, run them with a pure-numpy executor, and fall back to onnxruntime when
installed.

```
import onnx
```

It is also reachable through the full package as `SneppX_ALG.onnx`.

## What you can do

| Task | API | CLI |
|------|-----|-----|
| Build a graph | `onnx.build_graph(...)` | — |
| Serialize to file | `onnx.save_model(model, path)` | `sneppx-onnx convert` |
| Load / parse | `onnx.load_model(path)` / `onnx.parse_model(bytes)` | `sneppx-onnx info` |
| Validate | `onnx.check_model(model)` → `(ok, errors)` | `sneppx-onnx check` |
| Shape inference | `onnx.infer_shapes(model)` → `{name: (shape, dtype)}` | `sneppx-onnx shapes` |
| Optimize | `onnx.optimize(model, passes=[...])` | `sneppx-onnx optimize` |
| Quantize (QDQ) | `onnx.quantize_model(model, per_channel=...)` | `sneppx-onnx quantize` |
| External data | `onnx.save_external_data(...)` / `onnx.external_data.load_external_data(...)` | `sneppx-onnx save-external` / `load-external` |
| Execute (numpy) | `onnx.Session(model).run({"x": x})` | `sneppx-onnx run` |
| Execute (onnxruntime) | `onnx.OnnxRuntimeSession(model).run({"x": x})` | — |

## Example: build, save, load, validate

```python
import numpy as np
import onnx

W1 = np.random.randn(4, 8).astype(np.float32)
b1 = np.random.randn(8).astype(np.float32)
W2 = np.random.randn(8, 2).astype(np.float32)
b2 = np.random.randn(2).astype(np.float32)

graph = onnx.build_graph(
    name="mlp",
    inputs=[onnx.ValueInfo("x", "float32", ["batch", 4])],
    outputs=[onnx.ValueInfo("y", "float32", ["batch", 2])],
    initializers={"W1": W1, "b1": b1, "W2": W2, "b2": b2},
    nodes=[
        onnx.Node("MatMul", ["x", "W1"], ["mm1"]),
        onnx.Node("Add", ["mm1", "b1"], ["a1"]),
        onnx.Node("Relu", ["a1"], ["r1"]),
        onnx.Node("MatMul", ["r1", "W2"], ["mm2"]),
        onnx.Node("Add", ["mm2", "b2"], ["y"]),
    ],
)
model = onnx.Model(graph, producer_name="SNEPPX", model_version=1)
onnx.save_model(model, "mlp.onnx")

m = onnx.load_model("mlp.onnx")
ok, errors = onnx.check_model(m)          # (True, [])
shapes = onnx.infer_shapes(m)             # {"mm2": ([1, 2], "float32"), ...}
```

Dynamic axes are supported: use a string like `"batch"` in a `ValueInfo` shape.

## Optimizer

`onnx.optimize(model)` runs constant folding, dead-code elimination, and
identity-node elimination. Passes can be selected explicitly:

```python
opt = onnx.optimize(model, passes=["constant_folding", "dead_code_elimination"])
opt = onnx.optimize(model, passes=["identity_elimination"])
```

Folded intermediate constants are materialized as new graph initializers, so the
optimized graph remains valid.

## QDQ quantization

`onnx.quantize_model` inserts `QuantizeLinear`/`DequantizeLinear` pairs around
float weight initializers (per-channel or tensor-wise), skipping tensors whose
name matches the skip pattern (default `"bias"`):

```python
q = onnx.quantize_model(model, per_channel=True)   # per-output-channel scales
```

The original float initializers are kept for the `QuantizeLinear` input, and
per-channel scale/zero-point tensors are stored as separate initializers with
`[C, 1, ...]` broadcast shapes.

## External data

Move large initializers out of the `.onnx` file:

```python
onnx.save_external_data(model, "weights", size_threshold=64, location="w.bin")
onnx.save_model(model, "mlp_ext.onnx")     # ~216-byte model + w.bin payload
loaded = onnx.external_data.load_external_data("mlp_ext.onnx", base_dir="weights")
```

## Runtimes

`onnx.Session` is a pure-numpy, topologically-ordered executor supporting the
SNEPPX op subset (MatMul, Gemm, Conv, pooling, activations, Concat, Reshape,
Transpose, Gather, reductions, Split, Cast, LayerNormalization, BatchNormalization,
QDQ, comparison ops). Unknown ops raise `UnsupportedOpError`.

`onnx.OnnxRuntimeSession` runs through onnxruntime when installed and falls back
to the numpy executor otherwise. `onnx.runtime.ort_adapter.has_onnxruntime()`
reports availability.

## `sneppx-onnx` CLI

```
sneppx-onnx info <model>
sneppx-onnx check <model>
sneppx-onnx shapes <model> [--json]
sneppx-onnx optimize <in> [-o <out>] [--passes a,b] [--times N]
sneppx-onnx convert <in> [-o <out>] [--opset-version N]
sneppx-onnx quantize <in> [-o <out>] [--bits N] [--per-channel]
sneppx-onnx run <model> --input NAME:SHAPE:VALUES
sneppx-onnx save-external <in> -o <out> --dir <dir> [--threshold N]
sneppx-onnx load-external <in> -o <out> --dir <dir>
```

Example:

```
sneppx-onnx check mlp.onnx          # mlp.onnx: OK
sneppx-onnx run mlp.onnx --input "x:1,4:1,2,3,4"
```

## Compatibility with the legacy exporter

The wire format is byte-compatible in both directions with the
`interface_bindings` exporter (`onnx_export.py`):

- Legacy `OnnxExporter.export_binary(path)` output loads with `onnx.load_model`
- New `onnx.save_model` output reads with `protobuf_to_onnx`

`onnx/exporter.py` also exposes `from_sneppx_graph` / `to_sneppx_graph` bridges.

## Tests

```
python -m pytest onnx/tests/
```
