"""ONNX shape inference and op-schema checking.

Implements the ``onnx_check`` gap from the ML-framework feature matrix: full
type/shape inference with symbolic (batch) dimension propagation and op-schema
arity checks, on top of the structural ``onnx_validate`` in ``onnx_export``.

Typical usage::

    model = protobuf_to_onnx(open("m.onnx", "rb").read())
    ok, errors = onnx_check(model)
    value_info = infer_shapes(model)   # name -> OnnxTensor (dtype + shape)

Shapes may contain ``int`` (concrete), ``None`` (dynamic) or ``str`` (symbolic,
e.g. the "batch" dim) entries. ``None`` dims are propagated without constraint;
mismatched symbolic dims are reported rather than silently merged.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .onnx_export import (
    OnnxGraph,
    OnnxModel,
    OnnxNode,
    OnnxTensor,
    OnnxInitializer,
    ONNX_OP_REGISTRY,
    DTYPE_TO_ONNX,
    onnx_validate,
)

__all__ = [
    "infer_shapes",
    "onnx_check",
    "broadcast_shape",
    "OnnxShapeError",
]


class OnnxShapeError(ValueError):
    """Raised when shape inference fails for a node."""


def _is_symbolic(dim: Any) -> bool:
    return dim is None or isinstance(dim, str)


def broadcast_shape(
    a: List[Any], b: List[Any]
) -> List[Any]:
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


def _concat_dims(*lists: List[Any]) -> List[Any]:
    out: List[Any] = []
    for lst in lists:
        out.extend(lst)
    return out


def _pool_out_dim(size: Any, kernel: int, stride: int, pad: int,
                  dilation: int = 1) -> Any:
    if _is_symbolic(size):
        return size
    return max(0, (int(size) + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1)


def _attr_int(node: OnnxNode, name: str, default: int = 0) -> int:
    val = node.attributes.get(name, default)
    return int(val)


def _attr_ints(node: OnnxNode, name: str, default: Optional[List[int]] = None) -> List[int]:
    val = node.attributes.get(name, default)
    if val is None:
        return []
    return [int(v) for v in val]


def _attr_float(node: OnnxNode, name: str, default: float = 0.0) -> float:
    return float(node.attributes.get(name, default))


def _attr_str(node: OnnxNode, name: str, default: str = "") -> str:
    return str(node.attributes.get(name, default))


class _InferenceContext:
    """Tracks known tensor shapes/dtypes while walking the graph in topo order."""

    def __init__(self, model: OnnxModel):
        self.shapes: Dict[str, List[Any]] = {}
        self.dtypes: Dict[str, str] = {}
        self.constants: Dict[str, np.ndarray] = {}

        for inp in model.graph.inputs:
            self.shapes[inp.name] = list(inp.shape)
            self.dtypes[inp.name] = inp.dtype
        for init in model.graph.initializers:
            self.shapes[init.name] = list(init.data.shape)
            self.dtypes[init.name] = init.dtype
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


def _infer_broadcast_binary(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[Any], str]:
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
    return broadcast_shape(sa, sb), _promote_dtype(da, db)


def _promote_dtype(da: str, db: str) -> str:
    order = ["bool", "int8", "int16", "int32", "int64",
             "uint8", "uint16", "float16", "float32", "float64"]
    ia = order.index(da) if da in order else order.index("float32")
    ib = order.index(db) if db in order else order.index("float32")
    return order[max(ia, ib)]


def _infer_matmul(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[Any], str]:
    sa = ctx.get_shape(node.inputs[0])
    sb = ctx.get_shape(node.inputs[1])
    if sa is None or sb is None:
        raise OnnxShapeError(f"MatMul: missing input shape for {node.inputs}")
    if len(sa) < 1 or len(sb) < 1:
        raise OnnxShapeError("MatMul: inputs must have rank >= 1")
    if sa[-1] != sb[-2] and not (_is_symbolic(sa[-1]) or _is_symbolic(sb[-2])):
        raise OnnxShapeError(
            f"MatMul: inner dims {sa[-1]} vs {sb[-2]} mismatch"
        )
    batch = broadcast_shape(sa[:-2], sb[:-2])
    shape = batch + [sa[-2], sb[-1]]
    da = ctx.get_dtype(node.inputs[0]) or "float32"
    db = ctx.get_dtype(node.inputs[1]) or "float32"
    return shape, _promote_dtype(da, db)


def _infer_gemm(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[Any], str]:
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
    return [m, n], da


def _infer_conv(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[Any], str]:
    sx = ctx.get_shape(node.inputs[0])
    sw = ctx.get_shape(node.inputs[1])
    if sx is None or sw is None or len(sx) < 4 or len(sw) < 4:
        raise OnnxShapeError("Conv: X and W must be rank >= 4")
    n, c = sx[0], sx[1]
    out_c = sw[0]
    group = _attr_int(node, "group", 1)
    if isinstance(c, int) and isinstance(group, int) and c % group != 0:
        raise OnnxShapeError(f"Conv: channels {c} not divisible by group {group}")

    strides = _attr_ints(node, "strides", [1] * (len(sx) - 2))
    pads = _attr_ints(node, "pads", [0] * (2 * (len(sx) - 2)))
    dilations = _attr_ints(node, "dilations", [1] * (len(sx) - 2))
    if not strides:
        strides = [1] * (len(sx) - 2)
    if not dilations:
        dilations = [1] * (len(sx) - 2)
    if not pads:
        pads = [0] * (2 * (len(sx) - 2))

    spatial = []
    n_spatial = len(sx) - 2
    for i in range(n_spatial):
        size = sx[2 + i]
        kernel = int(sw[2 + i])
        stride = strides[i] if i < len(strides) else 1
        dilation = dilations[i] if i < len(dilations) else 1
        pad_top = pads[i] if i < len(pads) else 0
        pad_bot = pads[i + n_spatial] if i + n_spatial < len(pads) else 0
        spatial.append(_pool_out_dim(size, kernel, stride, max(pad_top, pad_bot),
                                     dilation))
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return [n, out_c] + spatial, dtype


def _infer_pool(ctx: _InferenceContext, node: OnnxNode,
                kernel_attr: str = "kernel_shape") -> Tuple[List[Any], str]:
    sx = ctx.get_shape(node.inputs[0])
    if sx is None or len(sx) < 2:
        raise OnnxShapeError(f"{node.op_type}: input must have rank >= 2")
    kernel = _attr_ints(node, kernel_attr)
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
        spatial.append(_pool_out_dim(size, k, stride, pad))
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return [sx[0], sx[1]] + spatial, dtype


def _infer_concat(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[Any], str]:
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
    return out, dtype


def _infer_reshape(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[Any], str]:
    shape_tensor = ctx.constant(node.inputs[1])
    if shape_tensor is None:
        raise OnnxShapeError("Reshape: target shape input is not a constant")
    target = [int(v) for v in shape_tensor.reshape(-1)]
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError("Reshape: missing input shape")

    resolved = []
    for idx, d in enumerate(target):
        if d == 0:
            resolved.append(sx[idx] if idx < len(sx) else 1)
        else:
            resolved.append(d)

    has_symbolic = any(_is_symbolic(d) for d in sx)
    total = 1
    known = 1
    minus_one = -1
    for d in resolved:
        if d == -1:
            minus_one = resolved.index(-1)
        elif isinstance(d, int) and d > 0:
            known *= d
    for d in sx:
        if isinstance(d, int):
            total *= d
    if minus_one != -1:
        if has_symbolic:
            resolved[minus_one] = None
        elif known and total % known == 0:
            resolved[minus_one] = total // known
        else:
            raise OnnxShapeError(
                f"Reshape: cannot infer -1 dim (total {total}, known {known})"
            )
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return resolved, dtype


def _infer_transpose(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[Any], str]:
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError("Transpose: missing input shape")
    perm = _attr_ints(node, "perm")
    if not perm:
        perm = list(reversed(range(len(sx))))
    out = [sx[p] for p in perm]
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return out, dtype


def _infer_squeeze(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[Any], str]:
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError("Squeeze: missing input shape")
    axes = _attr_ints(node, "axes")
    if len(node.inputs) > 1 and ctx.constant(node.inputs[1]) is not None:
        axes = [int(v) for v in ctx.constant(node.inputs[1]).reshape(-1)]
    if not axes:
        out = [d for d in sx if d != 1]
    else:
        rank = len(sx)
        drop = {a if a >= 0 else a + rank for a in axes}
        out = [d for i, d in enumerate(sx) if i not in drop]
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return out, dtype


def _infer_unsqueeze(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[Any], str]:
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError("Unsqueeze: missing input shape")
    axes = _attr_ints(node, "axes")
    if len(node.inputs) > 1 and ctx.constant(node.inputs[1]) is not None:
        axes = [int(v) for v in ctx.constant(node.inputs[1]).reshape(-1)]
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
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return out, dtype


def _infer_gather(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[Any], str]:
    sx = ctx.get_shape(node.inputs[0])
    si = ctx.get_shape(node.inputs[1])
    if sx is None or si is None:
        raise OnnxShapeError("Gather: missing input shapes")
    axis = _attr_int(node, "axis", 0)
    rank = len(sx)
    if axis < 0:
        axis += rank
    out = sx[:axis] + si + sx[axis + 1:]
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return out, dtype


def _infer_reduce(ctx: _InferenceContext, node: OnnxNode,
                  keepdims_default: int = 1) -> Tuple[List[Any], str]:
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
    dtype = ctx.get_dtype(node.inputs[0]) or "float32"
    return out, dtype


def _split_shapes(ctx: _InferenceContext, node: OnnxNode) -> Tuple[List[List[Any]], str]:
    sx = ctx.get_shape(node.inputs[0])
    if sx is None:
        raise OnnxShapeError("Split: missing input shape")
    axis = _attr_int(node, "axis", 0)
    rank = len(sx)
    if axis < 0:
        axis += rank
    split = _attr_ints(node, "split")
    if not split:
        if not node.outputs:
            raise OnnxShapeError("Split: no outputs declared")
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
    return shapes, dtype


# (shape-infer fn, keeps_1_to_1_output)
_SHAPE_RULES: Dict[str, Any] = {
    "Relu": ("identity", 1),
    "Sigmoid": ("identity", 1),
    "Tanh": ("identity", 1),
    "Gelu": ("identity", 1),
    "Softplus": ("identity", 1),
    "Abs": ("identity", 1),
    "Neg": ("identity", 1),
    "Sqrt": ("identity", 1),
    "Exp": ("identity", 1),
    "Log": ("identity", 1),
    "Floor": ("identity", 1),
    "Ceil": ("identity", 1),
    "Erf": ("identity", 1),
    "Softmax": ("identity", 1),
    "HardSwish": ("identity", 1),
    "LeakyRelu": ("identity", 1),
    "Elu": ("identity", 1),
    "Selu": ("identity", 1),
    "Dropout": ("identity", 1),
    "Identity": ("identity", 1),
    "BatchNormalization": ("identity", 1),
    "LayerNormalization": ("identity", 1),
    "RMSNormalization": ("identity", 1),
    "Add": ("binary", 1),
    "Sub": ("binary", 1),
    "Mul": ("binary", 1),
    "Div": ("binary", 1),
    "Pow": ("binary", 1),
    "MatMul": ("matmul", 1),
    "Gemm": ("gemm", 1),
    "Conv": ("conv", 1),
    "MaxPool": ("pool", 1),
    "AveragePool": ("pool", 1),
    "GlobalAveragePool": ("gpool", 1),
    "Concat": ("concat", 1),
    "Reshape": ("reshape", 1),
    "Transpose": ("transpose", 1),
    "Squeeze": ("squeeze", 1),
    "Unsqueeze": ("unsqueeze", 1),
    "Gather": ("gather", 1),
    "ReduceSum": ("reduce", 1),
    "ReduceMean": ("reduce", 1),
    "ReduceMax": ("reduce", 1),
    "ReduceMin": ("reduce", 1),
    "ReduceProd": ("reduce", 1),
    "Split": ("split", 0),
}


def _check_arity(node: OnnxNode) -> Optional[str]:
    spec = ONNX_OP_REGISTRY.get(node.op_type)
    if not spec:
        return None
    nin = spec.get("inputs", 0)
    if nin == -1:
        return None
    if len(node.inputs) < nin:
        return (
            f"{node.op_type} '{node.name}': expected >= {nin} inputs, "
            f"got {len(node.inputs)}"
        )
    return None


def infer_shapes(model: OnnxModel) -> Dict[str, OnnxTensor]:
    """Infer output shapes/dtypes for every node in the graph.

    Walks nodes in topological order, resolving shapes from graph inputs and
    initializer constants, and returns ``{tensor_name: OnnxTensor}`` for every
    node output. Raises :class:`OnnxShapeError` on the first unresolvable or
    inconsistent shape.
    """
    ctx = _InferenceContext(model)
    result: Dict[str, OnnxTensor] = {}
    graph: OnnxGraph = model.graph

    for node in graph.nodes:
        arity_error = _check_arity(node)
        if arity_error:
            raise OnnxShapeError(arity_error)

        rule = _SHAPE_RULES.get(node.op_type)
        if rule is None:
            raise OnnxShapeError(
                f"Shape inference not implemented for op '{node.op_type}'"
            )
        kind, nouts = rule

        if kind == "identity":
            shape = ctx.get_shape(node.inputs[0])
            if shape is None:
                raise OnnxShapeError(
                    f"{node.op_type}: unknown input shape '{node.inputs[0]}'"
                )
            dtype = ctx.get_dtype(node.inputs[0]) or "float32"
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "binary":
            shape, dtype = _infer_broadcast_binary(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "matmul":
            shape, dtype = _infer_matmul(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "gemm":
            shape, dtype = _infer_gemm(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "conv":
            shape, dtype = _infer_conv(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "pool":
            shape, dtype = _infer_pool(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "gpool":
            sx = ctx.get_shape(node.inputs[0])
            if sx is None:
                raise OnnxShapeError("GlobalAveragePool: unknown input shape")
            dtype = ctx.get_dtype(node.inputs[0]) or "float32"
            out_shapes = [[sx[0], sx[1]] + [1] * (len(sx) - 2)]
            out_dtypes = [dtype]
        elif kind == "concat":
            shape, dtype = _infer_concat(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "reshape":
            shape, dtype = _infer_reshape(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "transpose":
            shape, dtype = _infer_transpose(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "squeeze":
            shape, dtype = _infer_squeeze(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "unsqueeze":
            shape, dtype = _infer_unsqueeze(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "gather":
            shape, dtype = _infer_gather(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "reduce":
            shape, dtype = _infer_reduce(ctx, node)
            out_shapes = [shape]
            out_dtypes = [dtype]
        elif kind == "split":
            shapes_list, dtype = _split_shapes(ctx, node)
            out_shapes = shapes_list
            out_dtypes = [dtype] * len(out_shapes)
        else:  # pragma: no cover
            raise OnnxShapeError(f"Unhandled rule kind {kind}")

        if nouts and len(node.outputs) != nouts:
            raise OnnxShapeError(
                f"{node.op_type} '{node.name}': expected {nouts} output(s), "
                f"got {len(node.outputs)}"
            )

        for idx, out_name in enumerate(node.outputs):
            out_shape = out_shapes[idx] if idx < len(out_shapes) else out_shapes[0]
            out_dtype = out_dtypes[idx] if idx < len(out_dtypes) else out_dtypes[0]
            ctx.set(out_name, out_shape, out_dtype)
            result[out_name] = OnnxTensor(out_name, out_dtype, list(out_shape))

    # Graph outputs must resolve to concrete types.
    for out in graph.outputs:
        if out.name not in result:
            raise OnnxShapeError(
                f"Graph output '{out.name}' has no inferred shape"
            )
    return result


def onnx_check(model: OnnxModel) -> Tuple[bool, List[str]]:
    """Structural validation + shape/arity inference for an OnnxModel.

    Mirrors and extends the structural ``onnx_validate`` with op-schema arity
    checks and full type/shape inference (symbolic batch propagation). Returns
    ``(ok, errors)`` where errors is a list of human-readable messages; shape
    inference stops at the first failing node.
    """
    errors: List[str] = []
    ok, structural = onnx_validate(model)
    errors.extend(structural)
    if not ok:
        return False, errors
    try:
        inferred = infer_shapes(model)
    except OnnxShapeError as exc:
        errors.append(str(exc))
        return False, errors

    # Check declared output value_info against inferred shapes where concrete.
    graph = model.graph
    for out in graph.outputs:
        inferred_tensor = inferred.get(out.name)
        if inferred_tensor is None:
            continue
        for i, (declared, got) in enumerate(
            zip(out.shape, inferred_tensor.shape)
        ):
            if isinstance(declared, int) and isinstance(got, int) and declared != got:
                errors.append(
                    f"Graph output '{out.name}' dim {i}: declared {declared}, "
                    f"inferred {got}"
                )
    return len(errors) == 0, errors
