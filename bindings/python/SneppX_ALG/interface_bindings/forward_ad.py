"""True single-pass forward-mode (tangent) automatic differentiation.

The public :func:`jvp` in ``autograd`` is an exact forward-mode AD computed via
reverse-of-reverse (one backward pass per output element).  This module provides
:func:`forward_jvp`, a genuine *single forward pass* tangent-mode AD: it evaluates
``func`` once, propagating a tangent alongside the primal through registered
primitive rules.  Functions that touch an op without a registered tangent rule
fall back transparently to the exact :func:`autograd.jvp` (reverse-of-reverse),
so results are always correct -- only the pass count differs.

Only ``autograd_ops.Function.apply`` style functions are intercepted (Tensor
operator shorthands such as ``a @ b`` are not rewritten); for those, prefer
``jvp`` (which is exact regardless).
"""

import numpy as np

from .autograd import Context
from . import autograd_ops as ops
from .tensor import Tensor


class DualTensor:
    """A primal value paired with its tangent (directional derivative)."""

    __slots__ = ("value", "tangent")

    def __init__(self, value, tangent):
        self.value = value
        self.tangent = tangent


def _t(v):
    return np.zeros_like(v)


def _softmax(x, dim):
    z = x - np.max(x, axis=dim, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=dim, keepdims=True)


def _logsoftmax(x, dim):
    z = x - np.max(x, axis=dim, keepdims=True)
    lse = np.log(np.sum(np.exp(z), axis=dim, keepdims=True))
    return z - lse


# Tangent rules: rule(ctx, v, t, kw) -> tangent ndarray of the output.
def _r_add(ctx, v, t, kw):
    return t[0] + t[1]

def _r_sub(ctx, v, t, kw):
    return t[0] - t[1]

def _r_mul(ctx, v, t, kw):
    return t[0] * v[1] + v[0] * t[1]

def _r_div(ctx, v, t, kw):
    return (t[0] * v[1] - v[0] * t[1]) / (v[1] ** 2)

def _r_neg(ctx, v, t, kw):
    return -t[0]

def _r_abs(ctx, v, t, kw):
    return np.sign(v[0]) * t[0]

def _r_matmul(ctx, v, t, kw):
    return t[0] @ v[1] + v[0] @ t[1]

def _r_sum(ctx, v, t, kw):
    return np.sum(t[0], axis=kw.get("axis"), keepdims=kw.get("keepdims", False))

def _r_mean(ctx, v, t, kw):
    return np.mean(t[0], axis=kw.get("axis"), keepdims=kw.get("keepdims", False))

def _r_transpose(ctx, v, t, kw):
    dims = kw.get("dims")
    a = t[0]
    if dims is None:
        return a.T
    inv = [0] * a.ndim
    for i, d in enumerate(dims):
        inv[d] = i
    return np.transpose(a, inv)

def _r_reshape(ctx, v, t, kw):
    return t[0].reshape(v[0].shape)

def _r_squeeze(ctx, v, t, kw):
    return t[0].reshape(v[0].shape)

def _r_unsqueeze(ctx, v, t, kw):
    dim = v[1] if len(v) > 1 else kw.get("dim")
    return np.squeeze(t[0], axis=dim)

def _r_expand(ctx, v, t, kw):
    a = v[0]
    out = t[0]
    while len(out.shape) > len(a.shape) and out.shape[0] == 1:
        out = out.sum(axis=0, keepdims=False)
    while len(out.shape) > len(a.shape) and out.shape[-1] == 1:
        out = out.sum(axis=-1, keepdims=False)
    return out.reshape(a.shape)

def _r_getitem(ctx, v, t, kw):
    key = v[1] if len(v) > 1 else kw.get("key")
    out = np.zeros_like(v[0])
    out[key] = t[0]
    return out

def _r_sigmoid(ctx, v, t, kw):
    s = 1.0 / (1.0 + np.exp(-v[0]))
    return s * (1.0 - s) * t[0]

def _r_tanh(ctx, v, t, kw):
    y = np.tanh(v[0])
    return (1.0 - y * y) * t[0]

def _r_relu(ctx, v, t, kw):
    return (v[0] > 0).astype(t[0].dtype) * t[0]

def _r_sqrt(ctx, v, t, kw):
    return 0.5 / np.sqrt(v[0]) * t[0]

def _r_exp(ctx, v, t, kw):
    return np.exp(v[0]) * t[0]

def _r_log(ctx, v, t, kw):
    return t[0] / v[0]

def _r_pow(ctx, v, t, kw):
    b = v[1]
    z = v[0]
    gz = (b * z ** (b - 1.0)) * t[0]
    if isinstance(b, np.ndarray):
        gz = gz + (z ** b * np.log(z)) * t[1]
    return gz

def _r_softmax(ctx, v, t, kw):
    dim = v[1] if len(v) > 1 else kw.get("dim", -1)
    s = _softmax(v[0], dim)
    sdt = np.sum(s * t[0], axis=dim, keepdims=True)
    return s * (t[0] - sdt)

def _r_logsoftmax(ctx, v, t, kw):
    dim = v[1] if len(v) > 1 else kw.get("dim", -1)
    sm = _logsoftmax(v[0], dim)
    s = np.exp(sm)
    sdt = np.sum(t[0], axis=dim, keepdims=True)
    return t[0] - s * sdt

def _r_mse(ctx, v, t, kw):
    n = float(v[0].size)
    d = v[0] - v[1]
    return 2.0 * np.sum(d * (t[0] - t[1])) / n

def _r_linearfn(ctx, v, t, kw):
    a = v[0]; w = v[1]
    dy = t[0]
    if len(v) > 2 and v[2] is not None:
        da = dy @ w.T
        dw = a.T @ dy
        db = np.sum(dy, axis=0)
        return da, dw, db
    return dy @ w.T, a.T @ dy

def _r_dropoutfn(ctx, v, t, kw):
    rate = v[1] if len(v) > 1 else kw.get("rate", 0.5)
    training = kw.get("training", True)
    if not training:
        return t[0]
    rng = np.random.default_rng(v[2] if len(v) > 2 else 42)
    mask = (rng.random(v[0].shape) > rate).astype(t[0].dtype)
    return t[0] * mask / (1.0 - rate)


def _r_layernorm(ctx, v, t, kw):
    eps = kw.get("eps", 1e-5)
    x = v[0]; g = v[1]; b = v[2]
    axis = -1
    mean = x.mean(axis=axis, keepdims=True)
    xc = x - mean
    var = (xc ** 2).mean(axis=axis, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = xc / std
    dx = t[0]
    dmean = dx.mean(axis=axis, keepdims=True)
    dxc = dx - dmean
    dvar = (2.0 * xc * dx).mean(axis=axis, keepdims=True)
    dxhat = dxc / std - 0.5 * xhat * dvar / (var + eps)
    return dxhat * g + xhat * t[1] + t[2]


def _r_rmsnorm(ctx, v, t, kw):
    eps = kw.get("eps", 1e-6)
    x = v[0]; g = v[1]
    axis = -1
    ms = (x ** 2).mean(axis=axis, keepdims=True)
    rms = np.sqrt(ms + eps)
    xnorm = x / rms
    dx = t[0]
    dms = (2.0 * x * dx).mean(axis=axis, keepdims=True)
    dxnorm = dx / rms - 0.5 * xnorm * dms / (ms + eps)
    return dxnorm * g + xnorm * t[1]


_JVP_RULES = {
    ops.Add: _r_add,
    ops.Sub: _r_sub,
    ops.Mul: _r_mul,
    ops.Div: _r_div,
    ops.Neg: _r_neg,
    ops.Abs: _r_abs,
    ops.MatMul: _r_matmul,
    ops.Sum: _r_sum,
    ops.Mean: _r_mean,
    ops.Transpose: _r_transpose,
    ops.Reshape: _r_reshape,
    ops.Squeeze: _r_squeeze,
    ops.Unsqueeze: _r_unsqueeze,
    ops.Expand: _r_expand,
    ops.GetItem: _r_getitem,
    ops.Sigmoid: _r_sigmoid,
    ops.Tanh: _r_tanh,
    ops.Relu: _r_relu,
    ops.Sqrt: _r_sqrt,
    ops.Exp: _r_exp,
    ops.Log: _r_log,
    ops.Pow: _r_pow,
    ops.Softmax: _r_softmax,
    ops.LogSoftmax: _r_logsoftmax,
    ops.MSELoss: _r_mse,
    ops.LinearFn: _r_linearfn,
    ops.DropoutFn: _r_dropoutfn,
    ops.LayerNorm: _r_layernorm,
    ops.RMSNorm: _r_rmsnorm,
}


def _all_functions():
    out = []
    for name in dir(ops):
        obj = getattr(ops, name)
        if isinstance(obj, type) and issubclass(obj, ops.Function) and obj is not ops.Function:
            out.append(obj)
    return out


def _dual_apply(rule):
    def _apply(cls, *args, **kwargs):
        is_dual = [isinstance(a, DualTensor) for a in args]
        if not any(is_dual):
            return cls._orig_apply(*args, **kwargs)
        primals = []
        tangents = []
        for a in args:
            if isinstance(a, DualTensor):
                primals.append(a.value)
                tangents.append(a.tangent.data.astype(np.float64))
            else:
                primals.append(a)
                tangents.append(None)
        ctx = Context()
        out_val = cls.forward(ctx, *primals, **kwargs)
        v = [p.data.astype(np.float64) if isinstance(p, Tensor) else p for p in primals]
        t = [tt if tt is not None else _t(vi) for tt, vi in zip(tangents, v)]
        out_tan = rule(ctx, v, t, kwargs)
        out_tan = np.asarray(out_tan, dtype=out_val.data.dtype)
        return DualTensor(out_val, Tensor(out_tan))
    return _apply


def forward_jvp(func, inputs, tangents):
    """Single-pass forward-mode JVP.

    Returns ``(output, jvp)`` where ``jvp = J @ tangents`` computed by tangent
    propagation through registered ops.  Falls back to the exact
    :func:`autograd.jvp` (reverse-of-reverse) if ``func`` uses an op without a
    registered tangent rule.
    """
    from .autograd import jvp as _jvp_exact

    patched = []
    # Registered ops get dual-aware apply.
    for cls, rule in _JVP_RULES.items():
        if not hasattr(cls, "_orig_apply"):
            cls._orig_apply = cls.apply
            cls.apply = classmethod(_dual_apply(rule))
            patched.append(cls)
    # Every other Function must abort to fallback if given a DualTensor.
    sentinel = {"hit": False}

    def _fallback_apply(cls):
        def _apply(_cls, *args, **kwargs):
            if any(isinstance(a, DualTensor) for a in args):
                sentinel["hit"] = True
                raise _UnregisteredDual()
            return _cls._orig_apply(*args, **kwargs)
        return _apply

    for cls in _all_functions():
        if cls in _JVP_RULES:
            continue
        if not hasattr(cls, "_orig_apply"):
            cls._orig_apply = cls.apply
            cls.apply = classmethod(_fallback_apply(cls))
            patched.append(cls)

    try:
        dual_inputs = [DualTensor(inp, tan) if not isinstance(inp, DualTensor)
                       else inp for inp, tan in zip(inputs, tangents)]
        out = func(*dual_inputs)
        if sentinel["hit"]:
            raise _UnregisteredDual()
        if isinstance(out, DualTensor):
            return out.value, out.tangent
        return out, Tensor(np.zeros_like(out.data))
    except _UnregisteredDual:
        return _jvp_exact(func, inputs, tangents)
    finally:
        for cls in patched:
            cls.apply = cls._orig_apply
            del cls._orig_apply


class _UnregisteredDual(Exception):
    pass
