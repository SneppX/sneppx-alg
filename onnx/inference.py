"""ONNX type/shape inference and model checking (numpy-only).

Mirrors ``interface_bindings.onnx_check`` on the standalone :mod:`onnx.model`
data classes: full type/shape inference with symbolic batch propagation,
op-schema arity checks, and structural + connectivity validation. Shapes may
contain ``int`` (concrete), ``None`` (dynamic) or ``str`` (symbolic) entries.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import Graph, Model, Node

__all__ = ["infer_shapes", "check_model", "broadcast_shape", "OnnxShapeError"]


class OnnxShapeError(ValueError):
    """Raised when shape inference fails for a node."""


def _is_symbolic(dim: Any) -> bool:
    return dim is None or isinstance(dim, str)


def broadcast_shape(a: List[Any], b: List[Any]) -> List[Any]:
    """NumPy-style broadcast of two dim lists, preserving symbolic dims."""
    rank = max(len(a), len(b))
    da = [1] * (rank - len(a)) + list(a)
    db = [1] * (rank - len(b)) + list(b)
    out: List[Any] = []
    for x, y in zip(da, db):
        if x == y:
            out.append(x)
        elif x == 1 or x is None:
            out.append(y)
        elif y == 1 or y is None:
            out.append(x)
        elif _is_symbolic(x) and _is_symbolic(y):
            out.append(x)
        else:
            raise OnnxShapeError(f"incompatible broadcast dims {x} vs {y}")
    return out


def _promote_dtype(da: str, db: str) -> str:
    order = [
        "bool", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "float16", "float32", "float64",
    ]
    ia = order.index(da) if da in order else order.index("float32")
    ib = order.index(db) if db in order else order.index("float32")
    return order[max(ia, ib)]


class _InferenceContext:
    def __init__(self, model: Model):
        self.shapes: Dict[str, List[Any]] = {}
        self.dtypes: Dict[str, str] = {}
        self.constants: Dict[str, np.ndarray] = {}
        for inp in model.graph.inputs:
            self.shapes[inp.name] = list(inp.shape)
            self.dtypes[inp.name] = inp.dtype
        for init in model.graph.initializers:
            self.shapes[init.name] = list(init.shape)
            self.dtypes[init.name] = init.dtype
            if init.data is not None:
                self.constants[init.name] = init.data

    def get_shape(self, name: str) -> Optional[List[Any]]:
        return self.shapes.get(name)

    def get_dtype(self, name: str) -> Optional[str]:
        return self.dtypes.get(name)

    def set(self, name: str, shape: List[Any], dtype: str) -> None:
        self.shapes[name] = list(shape)
        self.dtypes[name] = dtype

    def constant(self, name: str) -> Optional[np.ndarray]:
        return self.constants.get(name)


def _attr_int(node: Node, name: str, default: int = 0) -> int:
    val = node.get_attr(name, default)
    return int(val) if val is not None else default


def _attr_ints(node: Node, name: str, default: Optional[List[int]] = None) -> List[int]:
    val = node.get_attr(name, default)
    if val is None:
        return []
    return [int(v) for v in val]


def _infer_identity(ctx: _InferenceContext, node: Node):
    shape = ctx.get_shape(node.inputs[0]) if node.inputs else None
    if shape is None:
        raise OnnxShapeError(f"{node.op_type}: unknown input shape '{node.inputs[0]}'")
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return [list(shape)], [dtype]


def _infer_binary(ctx: _InferenceContext, node: Node):
    sa = ctx.get_shape(node.inputs[0])
    sb = ctx.get_shape(node.inputs[1])
    if sa is None and sb is None:
        raise OnnxShapeError(f"{node.op_type}: no shapes for inputs")
    if sa is None:
        sa = [1] * len(sb)
    if sb is None:
        sb = [1] * len(sa)
    da = ctx.get_dtype(node.inputs[0]) or "float32"
    db = ctx.get_dtype(node.inputs[1]) or "float32"
    return [broadcast_shape(sa, sb)], [_promote_dtype(da, db)]


def _infer_matmul(ctx: _InferenceContext, node: Node):
    sa = ctx.get_shape(node.inputs[0])
    sb = ctx.get_shape(node.inputs[1])
    if sa is None or sb is None:
        raise OnnxShapeError(f"MatMul: missing input shape for {node.inputs}")
    if len(sa) < 1 or len(sb) < 1:
        raise OnnxShapeError("MatMul: inputs must have rank >= 1")
    if sa[-1] != sb[-2] and not (_is_symbolic(sa[-1]) or _is_symbolic(sb[-2])):
        raise OnnxShapeError(f"MatMul: inner dims {sa[-1]} vs {sb[-2]} mismatch")
    batch = broadcast_shape(sa[:-2], sb[:-2])
    shape = batch + [sa[-2], sb[-1]]
    da = ctx.get_dtype(node.inputs[0]) or "float32"
    db = ctx.get_dtype(node.inputs[1]) or "float32"
    return [shape], [_promote_dtype(da, db)]


def _infer_conv(ctx: _InferenceContext, node: Node):
    sx = ctx.get_shape(node.inputs[0])
    sw = ctx.get_shape(node.inputs[1])
    if sx is None or sw is None or len(sx) < 4 or len(sw) < 4:
        raise OnnxShapeError("Conv: X and W must be rank >= 4")
    n, c = sx[0], sx[1]
    out_c = sw[0]
    group = _attr_int(node, "group", 1)
    n_spatial = len(sx) - 2
    strides = _attr_ints(node, "strides", [1] * n_spatial)
    pads = _attr_ints(node, "pads", [0] * (2 * n_spatial))
    dilations = _attr_ints(node, "dilations", [1] * n_spatial)
    if not strides:
        strides = [1] * n_spatial
    if not dilations:
        dilations = [1] * n_spatial
    if not pads:
        pads = [0] * (2 * n_spatial)
    spatial = []
    for i in range(n_spatial):
        size = sx[2 + i]
        kernel = int(sw[2 + i])
        stride = strides[i] if i < len(strides) else 1
        dilation = dilations[i] if i < len(dilations) else 1
        pad = max(
            pads[i] if i < len(pads) else 0,
            pads[i + n_spatial] if i + n_spatial < len(pads) else 0,
        )
        if _is_symbolic(size):
            spatial.append(size)
        else:
            spatial.append(
                max(0, (int(size) + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1)
            )
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return [[n, out_c] + spatial], [dtype]


def _infer_pool(ctx: _InferenceContext, node: Node):
    sx = ctx.get_shape(node.inputs[0])
    if sx is None or len(sx) < 2:
        raise OnnxShapeError(f"{node.op_type}: input must have rank >= 2")
    kernel = _attr_ints(node, "kernel_shape")
    if not kernel:
        raise OnnxShapeError(f"{node.op_type}: missing kernel_shape")
    n_spatial = len(sx) - 2
    strides = _attr_ints(node, "strides", [1] * n_spatial)
    pads = _attr_ints(node, "pads", [0] * (2 * n_spatial))
    if not strides:
        strides = [1] * n_spatial
    if not pads:
        pads = [0] * (2 * n_spatial)
    spatial = []
    for i in range(n_spatial):
        size = sx[2 + i]
        k = int(kernel[i]) if i < len(kernel) else 1
        stride = strides[i] if i < len(strides) else 1
        pad = pads[i] if i < len(pads) else 0
        if _is_symbolic(size):
            spatial.append(size)
        else:
            spatial.append(max(0, (int(size) + 2 * pad - k) // stride + 1))
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return [[sx[0], sx[1]] + spatial], [dtype]


def _infer_concat(ctx: _InferenceContext, node: Node):
    axis = _attr_int(node, "axis", 0)
    shapes = [ctx.get_shape(inp) for inp in node.inputs]
    if not shapes or any(s is None for s in shapes):
        raise OnnxShapeError("Concat: missing input shapes")
    rank = len(shapes[0])
    if axis < 0:
        axis += rank
    out = list(shapes[0])
    total = 0
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    for s in shapes:
        if len(s) != rank:
            raise OnnxShapeError("Concat: input ranks differ")
        for i in range(rank):
            if i != axis and s[i] != out[i]:
                raise OnnxShapeError(f"Concat: dim {i} mismatch {s[i]} vs {out[i]}")
        total += int(s[axis]) if isinstance(s[axis], int) else 0
    if all(_is_symbolic(s[axis]) for s in shapes):
        out[axis] = shapes[0][axis]
    else:
        out[axis] = total if total else shapes[0][axis]
    return [out], [dtype]


def _infer_reshape(ctx: _InferenceContext, node: Node):
    shape_tensor = ctx.constant(node.inputs[1]) if len(node.inputs) > 1 else None
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError("Reshape: missing input shape")
    if shape_tensor is None:
        return [[None] * len(sx)], [ctx.get_dtype(node.inputs[0]) or "float32"]
    target = [int(v) for v in np.asarray(shape_tensor).reshape(-1)]
    resolved = []
    for idx, d in enumerate(target):
        if d == 0:
            resolved.append(sx[idx] if idx < len(sx) else 1)
        else:
            resolved.append(d)
    has_symbolic = any(_is_symbolic(d) for d in sx)
    minus_one = resolved.count(-1)
    if minus_one:
        total = 1
        for d in sx:
            if isinstance(d, int):
                total *= d
        known = 1
        for d in resolved:
            if isinstance(d, int) and d > 0:
                known *= d
        if has_symbolic:
            resolved[target.index(-1)] = None
        elif known and total % known == 0:
            resolved[target.index(-1)] = total // known
        else:
            raise OnnxShapeError(
                f"Reshape: cannot infer -1 dim (total {total}, known {known})"
            )
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return [resolved], [dtype]


def _infer_transpose(ctx: _InferenceContext, node: Node):
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError("Transpose: missing input shape")
    perm = _attr_ints(node, "perm")
    if not perm:
        perm = list(reversed(range(len(sx))))
    return [[sx[p] for p in perm]], [ctx.get_dtype(node.inputs[0]) or "float32"]


def _infer_squeeze(ctx: _InferenceContext, node: Node):
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError("Squeeze: missing input shape")
    axes = _attr_ints(node, "axes")
    if len(node.inputs) > 1 and ctx.constant(node.inputs[1]) is not None:
        axes = [int(v) for v in np.asarray(ctx.constant(node.inputs[1])).reshape(-1)]
    if not axes:
        out = [d for d in sx if d != 1]
    else:
        rank = len(sx)
        drop = {a if a >= 0 else a + rank for a in axes}
        out = [d for i, d in enumerate(sx) if i not in drop]
    return [out], [ctx.get_dtype(node.inputs[0]) or "float32"]


def _infer_unsqueeze(ctx: _InferenceContext, node: Node):
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError("Unsqueeze: missing input shape")
    axes = _attr_ints(node, "axes")
    if len(node.inputs) > 1 and ctx.constant(node.inputs[1]) is not None:
        axes = [int(v) for v in np.asarray(ctx.constant(node.inputs[1])).reshape(-1)]
    rank_out = len(sx) + len(axes)
    positions = {a if a >= 0 else a + rank_out for a in axes}
    out = []
    it = 0
    for i in range(rank_out):
        if i in positions:
            out.append(1)
        else:
            out.append(sx[it])
            it += 1
    return [out], [ctx.get_dtype(node.inputs[0]) or "float32"]


def _infer_gather(ctx: _InferenceContext, node: Node):
    sx = ctx.get_shape(node.inputs[0])
    si = ctx.get_shape(node.inputs[1])
    if sx is None or si is None:
        raise OnnxShapeError("Gather: missing input shapes")
    axis = _attr_int(node, "axis", 0)
    rank = len(sx)
    if axis < 0:
        axis += rank
    return [sx[:axis] + si + sx[axis + 1 :]], [
        ctx.get_dtype(node.inputs[0]) or "float32"
    ]


def _infer_reduce(ctx: _InferenceContext, node: Node, keepdims_default: int = 1):
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError(f"{node.op_type}: missing input shape")
    axes = _attr_ints(node, "axes")
    keepdims = _attr_int(node, "keepdims", keepdims_default)
    if not axes:
        out = [1] * len(sx) if keepdims else []
    else:
        rank = len(sx)
        drop = {a if a >= 0 else a + rank for a in axes}
        if keepdims:
            out = [sx[i] if i not in drop else 1 for i in range(rank)]
        else:
            out = [sx[i] for i in range(rank) if i not in drop]
    return [out], [ctx.get_dtype(node.inputs[0]) or "float32"]


def _infer_split(ctx: _InferenceContext, node: Node):
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError("Split: missing input shape")
    axis = _attr_int(node, "axis", 0)
    rank = len(sx)
    if axis < 0:
        axis += rank
    split = _attr_ints(node, "split")
    if not split:
        size = sx[axis]
        if _is_symbolic(size):
            size = 1
        split = [size] * len(node.outputs)
    shapes = []
    for part in split:
        out = list(sx)
        out[axis] = int(part)
        shapes.append(out)
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return shapes, [dtype] * len(shapes)


def _infer_gemm(ctx: _InferenceContext, node: Node):
    sa = ctx.get_shape(node.inputs[0])
    sb = ctx.get_shape(node.inputs[1])
    if sa is None or sb is None or len(sa) < 2 or len(sb) < 2:
        raise OnnxShapeError("Gemm: inputs must be rank-2")
    trans_a = _attr_int(node, "transA")
    trans_b = _attr_int(node, "transB")
    m, k1 = (sa[1], sa[0]) if trans_a else (sa[0], sa[1])
    k2, n = (sb[1], sb[0]) if trans_b else (sb[0], sb[1])
    if k1 != k2 and not (_is_symbolic(k1) or _is_symbolic(k2)):
        raise OnnxShapeError(f"Gemm: inner dims {k1} vs {k2} mismatch")
    da = ctx.get_dtype(node.inputs[0]) or "float32"
    if len(node.inputs) >= 3:
        db = ctx.get_dtype(node.inputs[2]) or "float32"
        da = _promote_dtype(da, db)
    return [[m, n]], [da]


def _infer_constant(ctx: _InferenceContext, node: Node):
    val = node.get_attr("value")
    if isinstance(val, np.ndarray):
        return [list(val.shape)], ["float32"]
    raise OnnxShapeError("Constant: missing value attribute")


# op -> (handler, expected output count)
_SHAPE_RULES: Dict[str, Tuple[Any, int]] = {
    "Relu": (_infer_identity, 1),
    "Sigmoid": (_infer_identity, 1),
    "Tanh": (_infer_identity, 1),
    "Gelu": (_infer_identity, 1),
    "Softplus": (_infer_identity, 1),
    "Abs": (_infer_identity, 1),
    "Neg": (_infer_identity, 1),
    "Sqrt": (_infer_identity, 1),
    "Exp": (_infer_identity, 1),
    "Log": (_infer_identity, 1),
    "Floor": (_infer_identity, 1),
    "Ceil": (_infer_identity, 1),
    "Erf": (_infer_identity, 1),
    "Softmax": (_infer_identity, 1),
    "HardSwish": (_infer_identity, 1),
    "LeakyRelu": (_infer_identity, 1),
    "Elu": (_infer_identity, 1),
    "Selu": (_infer_identity, 1),
    "Dropout": (_infer_identity, 1),
    "Identity": (_infer_identity, 1),
    "BatchNormalization": (_infer_identity, 1),
    "LayerNormalization": (_infer_identity, 1),
    "Add": (_infer_binary, 1),
    "Sub": (_infer_binary, 1),
    "Mul": (_infer_binary, 1),
    "Div": (_infer_binary, 1),
    "Pow": (_infer_binary, 1),
    "Min": (_infer_binary, 1),
    "Max": (_infer_binary, 1),
    "MatMul": (_infer_matmul, 1),
    "Gemm": (_infer_gemm, 1),
    "Conv": (_infer_conv, 1),
    "MaxPool": (_infer_pool, 1),
    "AveragePool": (_infer_pool, 1),
    "Concat": (_infer_concat, 1),
    "Reshape": (_infer_reshape, 1),
    "Transpose": (_infer_transpose, 1),
    "Squeeze": (_infer_squeeze, 1),
    "Unsqueeze": (_infer_unsqueeze, 1),
    "Gather": (_infer_gather, 1),
    "ReduceSum": (_infer_reduce, 1),
    "ReduceMean": (_infer_reduce, 1),
    "ReduceMax": (_infer_reduce, 1),
    "ReduceMin": (_infer_reduce, 1),
    "ReduceProd": (_infer_reduce, 1),
    "Split": (_infer_split, 0),
    "Constant": (_infer_constant, 1),
    "QuantizeLinear": (_infer_identity, 1),
    "DequantizeLinear": (_infer_identity, 1),
    "Cast": (_infer_identity, 1),
}


def infer_shapes(model: Model) -> Dict[str, Any]:
    """Infer output shapes/dtypes for every node in the graph.

    Walks nodes in topological order, resolving shapes from graph inputs and
    initializer constants, and returns ``{tensor_name: (shape, dtype)}`` for
    every node output. Raises :class:`OnnxShapeError` on the first
    unresolvable or inconsistent shape.
    """
    ctx = _InferenceContext(model)
    result: Dict[str, Any] = {}
    graph: Graph = model.graph

    for node in graph.nodes:
        rule = _SHAPE_RULES.get(node.op_type)
        if rule is None:
            raise OnnxShapeError(
                f"Shape inference not implemented for op '{node.op_type}'"
            )
        handler, nouts = rule
        out_shapes, out_dtypes = handler(ctx, node)

        if nouts and len(node.outputs) != nouts:
            raise OnnxShapeError(
                f"{node.op_type} '{node.name}': expected {nouts} output(s), "
                f"got {len(node.outputs)}"
            )

        for idx, out_name in enumerate(node.outputs):
            out_shape = out_shapes[idx] if idx < len(out_shapes) else out_shapes[0]
            out_dtype = out_dtypes[idx] if idx < len(out_dtypes) else out_dtypes[0]
            ctx.set(out_name, out_shape, out_dtype)
            result[out_name] = (list(out_shape), out_dtype)

    return result


def check_model(model: Model) -> Tuple[bool, List[str]]:
    """Structural + connectivity + arity/type checks for an :class:`Model`.

    Returns ``(ok, errors)`` where errors is a list of human-readable
    messages. Mirrors the C ``SneppX_onnx_validate`` checks and extends them
    with shape inference.
    """
    errors: List[str] = []

    if model.ir_version <= 0:
        errors.append("Model has no valid ir_version")
    if not model.opset_imports:
        errors.append("Model has no opset import")

    graph = model.graph
    if not graph.name:
        errors.append("Graph has no name")
    if not graph.inputs and not graph.initializers:
        errors.append("Model has no inputs")
    if not graph.outputs:
        errors.append("Model has no outputs")
    if not graph.nodes:
        errors.append("Model has no nodes")

    declared = {t.name for t in graph.inputs}
    declared |= {i.name for i in graph.initializers}
    produced: set = set()
    all_outputs: set = set()

    for node in graph.nodes:
        if not node.op_type:
            errors.append("Node has empty op_type")
        for out in node.outputs:
            if out in all_outputs:
                errors.append(f"Duplicate output name: {out}")
            all_outputs.add(out)
            produced.add(out)
        for inp in node.inputs:
            if inp and inp not in declared and inp not in produced:
                errors.append(f"Node input '{inp}' is not declared or produced")

    for out in graph.outputs:
        if out.name not in produced:
            errors.append(f"Graph output '{out.name}' is not produced by any node")

    if not errors:
        try:
            infer_shapes(model)
        except OnnxShapeError as exc:
            errors.append(str(exc))

    return len(errors) == 0, errors
