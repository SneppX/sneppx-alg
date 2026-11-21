"""Pure-numpy ONNX executor (numpy-only, no onnxruntime required).

Implements a topologically-ordered interpreter for the operator subset used by
SNEPPX-exported models (MatMul, Gemm, Conv, pool, activations, reductions,
reshape/transpose, QDQ, LayerNormalization, etc.). Unknown ops raise
``UnsupportedOpError``; use :class:`onnxruntime` session for full coverage.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..model import Model, Node

__all__ = ["Session", "UnsupportedOpError", "execute"]


class UnsupportedOpError(NotImplementedError):
    pass


def _attr_int(node: Node, name: str, default: int = 0) -> int:
    val = node.get_attr(name, default)
    return int(val) if val is not None else default


def _attr_ints(node: Node, name: str, default: Optional[List[int]] = None) -> List[int]:
    val = node.get_attr(name, default)
    if val is None:
        return []
    return [int(v) for v in val]


def _attr_float(node: Node, name: str, default: float = 0.0) -> float:
    return float(node.get_attr(name, default))


def _unary(node, tensors, fn):
    x = tensors[node.inputs[0]]
    return fn(x)


def _binary(node, tensors, fn):
    a = tensors[node.inputs[0]]
    b = tensors[node.inputs[1]]
    return fn(a, b)


def _conv(x, w, node):
    strides = _attr_ints(node, "strides", [1, 1]) or [1, 1]
    pads = _attr_ints(node, "pads", [0, 0, 0, 0]) or [0, 0, 0, 0]
    dilations = _attr_ints(node, "dilations", [1, 1]) or [1, 1]
    group = _attr_int(node, "group", 1)

    n, cin, h, wd = x.shape
    cout, wcin, kh, kw = w.shape
    pad_t, pad_l, pad_b, pad_r = (pads + [0, 0, 0, 0])[:4]
    dh, dw = dilations[0], dilations[1]
    sh, sw = strides[0], strides[1]

    hp = h + pad_t + pad_b
    wp = wd + pad_l + pad_r
    xp = np.pad(x, ((0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)))

    group = max(1, group)
    cpg = cin // group          # input channels per group
    opg = cout // group         # output channels per group
    oh = (hp - (kh - 1) * dh - 1) // sh + 1
    ow = (wp - (kw - 1) * dw - 1) // sw + 1
    out = np.zeros((n, cout, oh, ow), dtype=np.float32)
    for ci in range(cout):
        g = ci // opg
        for di in range(cpg):
            src = xp[:, g * cpg + di, :, :]
            kernel = w[ci, di, :, :]
            for i in range(oh):
                for j in range(ow):
                    r0, r1 = i * sh, i * sh + (kh - 1) * dh + 1
                    c0, c1 = j * sw, j * sw + (kw - 1) * dw + 1
                    patch = src[:, r0:r1:dh, c0:c1:dw]
                    out[:, ci, i, j] = np.sum(patch * kernel, axis=(1, 2))
    bias = tensors.get(node.inputs[2]) if len(node.inputs) > 2 else None
    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)
    return out


def _pool(x, node, mode):
    kh, kw = (_attr_ints(node, "kernel_shape") or [2, 2])[:2]
    strides = _attr_ints(node, "strides", [1, 1]) or [1, 1]
    pads = _attr_ints(node, "pads", [0, 0, 0, 0]) or [0, 0, 0, 0]
    sh, sw = strides[0], strides[1]
    pad_t, pad_l, pad_b, pad_r = (pads + [0, 0, 0, 0])[:4]
    n, c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)))
    oh = (h + pad_t + pad_b - kh) // sh + 1
    ow = (wd + pad_l + pad_r - kw) // sw + 1
    out = np.zeros((n, c, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            patch = xp[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw]
            if mode == "max":
                out[:, :, i, j] = patch.max(axis=(2, 3))
            else:
                out[:, :, i, j] = patch.mean(axis=(2, 3))
    return out


def _softmax(x, axis):
    axis = axis if axis >= 0 else axis + x.ndim
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _layer_norm(x, scale, bias, epsilon):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon) * scale + bias


def _execute_node(node: Node, tensors: Dict[str, np.ndarray]) -> List[np.ndarray]:
    op = node.op_type

    if op == "MatMul":
        return [_binary(node, tensors, lambda a, b: a @ b)]
    if op == "Gemm":
        a = tensors[node.inputs[0]]
        b = tensors[node.inputs[1]]
        ta = _attr_int(node, "transA")
        tb = _attr_int(node, "transB")
        if ta:
            a = a.T
        if tb:
            b = b.T
        out = a @ b
        out = out * _attr_float(node, "alpha", 1.0)
        if len(node.inputs) > 2:
            out = out + tensors[node.inputs[2]] * _attr_float(node, "beta", 1.0)
        return [out]
    if op == "Add":
        return [_binary(node, tensors, lambda a, b: a + b)]
    if op == "Sub":
        return [_binary(node, tensors, lambda a, b: a - b)]
    if op == "Mul":
        return [_binary(node, tensors, lambda a, b: a * b)]
    if op == "Div":
        return [_binary(node, tensors, lambda a, b: a / b)]
    if op == "Pow":
        return [_binary(node, tensors, lambda a, b: np.power(a, b))]
    if op == "Relu":
        return [_unary(node, tensors, lambda x: np.maximum(x, 0))]
    if op == "Sigmoid":
        return [_unary(node, tensors, lambda x: 1.0 / (1.0 + np.exp(-x)))]
    if op == "Tanh":
        return [_unary(node, tensors, np.tanh)]
    if op == "Exp":
        return [_unary(node, tensors, np.exp)]
    if op == "Log":
        return [_unary(node, tensors, np.log)]
    if op == "Sqrt":
        return [_unary(node, tensors, np.sqrt)]
    if op == "Abs":
        return [_unary(node, tensors, np.abs)]
    if op == "Neg":
        return [_unary(node, tensors, np.negative)]
    if op == "Identity":
        return [_unary(node, tensors, lambda x: x)]
    if op == "Gelu":
        def _gelu(x):
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        return [_unary(node, tensors, _gelu)]
    if op == "Softmax":
        x = tensors[node.inputs[0]]
        return [_softmax(x, _attr_int(node, "axis", -1))]
    if op == "Softplus":
        x = tensors[node.inputs[0]]
        return [np.log1p(np.exp(x))]
    if op == "LeakyRelu":
        x = tensors[node.inputs[0]]
        alpha = _attr_float(node, "alpha", 0.01)
        return [np.where(x >= 0, x, alpha * x)]
    if op == "Elu":
        x = tensors[node.inputs[0]]
        alpha = _attr_float(node, "alpha", 1.0)
        return [np.where(x >= 0, x, alpha * (np.exp(x) - 1.0))]
    if op == "Conv":
        return [_conv(tensors[node.inputs[0]], tensors[node.inputs[1]], node)]
    if op == "MaxPool":
        return [_pool(tensors[node.inputs[0]], node, "max")]
    if op == "AveragePool":
        return [_pool(tensors[node.inputs[0]], node, "avg")]
    if op == "GlobalAveragePool":
        x = tensors[node.inputs[0]]
        return [x.mean(axis=tuple(range(2, x.ndim)), keepdims=True)]
    if op == "Concat":
        axis = _attr_int(node, "axis", 0)
        return [np.concatenate([tensors[i] for i in node.inputs], axis=axis)]
    if op == "Reshape":
        x = tensors[node.inputs[0]]
        shape = tensors[node.inputs[1]].astype(np.int64).tolist()
        if isinstance(shape, list):
            shape = [int(v) for v in shape]
        return [x.reshape(shape)]
    if op == "Transpose":
        x = tensors[node.inputs[0]]
        perm = _attr_ints(node, "perm")
        if not perm:
            perm = list(reversed(range(x.ndim)))
        return [np.transpose(x, perm)]
    if op == "Squeeze":
        x = tensors[node.inputs[0]]
        axes = _attr_ints(node, "axes")
        if len(node.inputs) > 1:
            axes = [int(v) for v in np.asarray(tensors[node.inputs[1]]).reshape(-1)]
        return [np.squeeze(x, axis=tuple(axes)) if axes else np.squeeze(x)]
    if op == "Unsqueeze":
        x = tensors[node.inputs[0]]
        axes = _attr_ints(node, "axes")
        if len(node.inputs) > 1:
            axes = [int(v) for v in np.asarray(tensors[node.inputs[1]]).reshape(-1)]
        return [np.expand_dims(x, tuple(sorted(axes)))]
    if op == "Gather":
        x = tensors[node.inputs[0]]
        indices = tensors[node.inputs[1]]
        axis = _attr_int(node, "axis", 0)
        return [np.take(x, indices, axis=axis)]
    if op == "ReduceSum":
        x = tensors[node.inputs[0]]
        axes = _attr_ints(node, "axes")
        keepdims = _attr_int(node, "keepdims", 1)
        return [np.sum(x, axis=tuple(axes) if axes else None, keepdims=bool(keepdims))]
    if op == "ReduceMean":
        x = tensors[node.inputs[0]]
        axes = _attr_ints(node, "axes")
        keepdims = _attr_int(node, "keepdims", 1)
        return [np.mean(x, axis=tuple(axes) if axes else None, keepdims=bool(keepdims))]
    if op == "ReduceMax":
        x = tensors[node.inputs[0]]
        axes = _attr_ints(node, "axes")
        keepdims = _attr_int(node, "keepdims", 1)
        return [np.max(x, axis=tuple(axes) if axes else None, keepdims=bool(keepdims))]
    if op == "ReduceMin":
        x = tensors[node.inputs[0]]
        axes = _attr_ints(node, "axes")
        keepdims = _attr_int(node, "keepdims", 1)
        return [np.min(x, axis=tuple(axes) if axes else None, keepdims=bool(keepdims))]
    if op == "Split":
        x = tensors[node.inputs[0]]
        axis = _attr_int(node, "axis", 0)
        split = _attr_ints(node, "split")
        if not split:
            size = x.shape[axis] // len(node.outputs)
            split = [size] * len(node.outputs)
        return np.split(x, np.cumsum(split[:-1]), axis=axis)
    if op == "Flatten":
        x = tensors[node.inputs[0]]
        axis = _attr_int(node, "axis", 1)
        if axis < 0:
            axis += x.ndim
        return [x.reshape(x.shape[:axis], (-1))]
    if op == "Cast":
        x = tensors[node.inputs[0]]
        to = _attr_int(node, "to", 1)
        nptype = {
            1: np.float32, 10: np.float16, 11: np.float64, 3: np.int8,
            2: np.uint8, 6: np.int32, 7: np.int64, 9: np.bool_,
        }.get(to, np.float32)
        return [x.astype(nptype)]
    if op == "Shape":
        x = tensors[node.inputs[0]]
        return [np.asarray(x.shape, dtype=np.int64)]
    if op == "Constant":
        val = node.get_attr("value")
        if isinstance(val, np.ndarray):
            return [val]
        if isinstance(val, (int, float)):
            return [np.asarray(val, dtype=np.float32)]
        raise UnsupportedOpError(f"Constant: unsupported value {val!r}")
    if op == "LayerNormalization":
        x = tensors[node.inputs[0]]
        scale = tensors[node.inputs[1]]
        bias = tensors[node.inputs[2]] if len(node.inputs) > 2 else None
        eps = _attr_float(node, "epsilon", 1e-5)
        out = _layer_norm(x, scale, bias if bias is not None else 0.0, eps)
        return [out]
    if op == "BatchNormalization":
        x = tensors[node.inputs[0]]
        scale = tensors[node.inputs[1]]
        bias = tensors[node.inputs[2]]
        mean = tensors[node.inputs[3]]
        var = tensors[node.inputs[4]]
        eps = _attr_float(node, "epsilon", 1e-5)
        return [(x - mean) / np.sqrt(var + eps) * scale + bias]
    if op == "Dropout":
        x = tensors[node.inputs[0]]
        return [x]
    if op == "QuantizeLinear":
        x = tensors[node.inputs[0]]
        scale = tensors[node.inputs[1]]
        zp = tensors[node.inputs[2]] if len(node.inputs) > 2 else None
        s = scale
        while s.ndim < x.ndim:
            s = s.reshape(s.shape + (1,))
        q = np.clip(np.round(x / s), -128, 127)
        if zp is not None:
            z = zp
            while z.ndim < x.ndim:
                z = z.reshape(z.shape + (1,))
            q = q - z
        return [q.astype(np.int8)]
    if op == "DequantizeLinear":
        x = tensors[node.inputs[0]]
        scale = tensors[node.inputs[1]]
        zp = tensors[node.inputs[2]] if len(node.inputs) > 2 else None
        s = scale
        while s.ndim < x.ndim:
            s = s.reshape(s.shape + (1,))
        if zp is not None:
            z = zp
            while z.ndim < x.ndim:
                z = z.reshape(z.shape + (1,))
            return [(x.astype(np.float32) + z) * s]
        return [x.astype(np.float32) * s]
    if op == "Slice":
        x = tensors[node.inputs[0]]
        if len(node.inputs) < 3:
            raise UnsupportedOpError("Slice with <3 inputs not supported")
        starts = [int(v) for v in np.asarray(tensors[node.inputs[1]]).reshape(-1)]
        ends = [int(v) for v in np.asarray(tensors[node.inputs[2]]).reshape(-1)]
        axes = [int(v) for v in np.asarray(tensors[node.inputs[3]]).reshape(-1)]
        steps = [int(v) for v in np.asarray(tensors[node.inputs[4]]).reshape(-1)]
        sl = [slice(None)] * x.ndim
        for ax, s, e, st in zip(axes, starts, ends, steps):
            sl[ax] = slice(s, e, st)
        return [x[tuple(sl)]]
    if op == "Greater":
        return [_binary(node, tensors, lambda a, b: a > b)]
    if op == "Less":
        return [_binary(node, tensors, lambda a, b: a < b)]
    if op == "Equal":
        return [_binary(node, tensors, lambda a, b: a == b)]

    raise UnsupportedOpError(f"Unsupported ONNX op: {op}")


class Session:
    """Interpret an ONNX :class:`Model` with numpy."""

    def __init__(self, model: Model):
        self.model = model
        self.input_names = [i.name for i in model.graph.inputs]
        self.output_names = [o.name for o in model.graph.outputs]
        self._initializers = {
            i.name: i.data for i in model.graph.initializers if i.data is not None
        }

    def get_inputs(self):
        return [{"name": n} for n in self.input_names]

    def get_outputs(self):
        return [{"name": n} for n in self.output_names]

    def run(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        tensors: Dict[str, np.ndarray] = dict(self._initializers)
        for name, value in inputs.items():
            tensors[name] = np.asarray(value)

        for node in self.model.graph.nodes:
            results = _execute_node(node, tensors)
            for out_name, result in zip(node.outputs, results):
                tensors[out_name] = result

        return [tensors[n] for n in self.output_names]


def execute(model: Model, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Run a model with numpy and return the output tensors."""
    return Session(model).run(inputs)
