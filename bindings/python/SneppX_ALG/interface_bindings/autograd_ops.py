"""Autograd Ops � differentiable operation subclasses of Function.

Each op defines forward(ctx, *args) -> Tensor and the corresponding
backward(ctx, grad_output) -> list of gradients.

Consumed by tensor.py via lazy imports inside the op methods.
"""

import numpy as np

from .tensor import Tensor, _numpy_dtype, _resolve_dtype
from .autograd import Function, Context


def _reduce_to_shape(grad, shape):
    """Sum ``grad`` back to ``shape`` after numpy broadcasting (ndarray)."""
    if isinstance(grad, Tensor):
        grad = grad.data
    if shape is None:
        return grad
    target = tuple(shape)
    grad = np.asarray(grad)
    while grad.ndim > len(target):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(target):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _broadcast_grad(grad, a_shape, b_shape):
    """Return per-input gradients reduced to their broadcast shapes."""
    ga = (
        grad
        if a_shape is None
        else Tensor(_reduce_to_shape(grad, a_shape), dtype=grad.dtype)
    )
    gb = (
        grad
        if b_shape is None
        else Tensor(_reduce_to_shape(grad, b_shape), dtype=grad.dtype)
    )
    return [ga, gb]


def _save_attr(ctx, **attrs):
    if ctx is not None:
        ctx.save_attr(**attrs)


def _get_attr(ctx, name):
    return None if ctx is None else ctx.get_attr(name)


def _get_saved_tensor(ctx, name):
    return None if ctx is None else ctx.get_saved_tensor(name)


# ---------------------------------------------------------------------------
# Higher-order (create_graph) helpers
# ---------------------------------------------------------------------------


def _as_const(arr, dtype):
    """Build a detached constant Tensor from a numpy array / scalar."""
    t = Tensor(np.asarray(arr, dtype=_numpy_dtype(dtype)))
    t.requires_grad = False
    return t


def _cg_broadcast(g, target_shape):
    """Graph-aware broadcast *backward*: map grad `g` (output shape) to `target_shape`.

    Implements numpy broadcast semantics (right-aligned). Works in both
    directions:
      - dims where the input had size 1 and the output was larger are summed
        out (reduce);
      - dims where the output was size 1 and the input was larger are
        broadcast back up (expand).
    The returned tensor has exactly `target_shape`.
    """
    if target_shape is None:
        return g
    tgt = list(target_shape)
    out = list(g.shape)
    n = max(len(tgt), len(out))
    tgt_p = [1] * (n - len(tgt)) + tgt
    out_p = [1] * (n - len(out)) + out
    g2 = g
    if tuple(out_p) != tuple(out):
        g2 = Reshape.apply(g, tuple(out_p))
    for i in range(n):
        if tgt_p[i] == out_p[i]:
            continue
        if tgt_p[i] == 1 and out_p[i] != 1:
            g2 = Sum.apply(g2, i)
            out_p[i] = 1
        elif out_p[i] == 1 and tgt_p[i] != 1:
            ns = list(g2.shape)
            ns[i] = tgt_p[i]
            g2 = Expand.apply(g2, tuple(ns))
            out_p[i] = tgt_p[i]
        else:
            raise ValueError(
                f"broadcast mismatch: target dim {tgt_p[i]}, output dim {out_p[i]}"
            )
    if tuple(g2.shape) != tuple(target_shape):
        g2 = Reshape.apply(g2, tuple(target_shape))
    return g2


def _mean_keepdim(x, axis=-1):
    """Graph-aware mean with keepdims=True (helper for norm ops)."""
    n = x.shape[axis]
    s = Sum.apply(x, axis)
    shp = list(x.shape)
    shp[axis] = 1
    return Reshape.apply(s, tuple(shp))


def _sum_keepdim(x, axis=-1):
    """Graph-aware sum with keepdims=True (helper for norm ops)."""
    s = Sum.apply(x, axis)
    shp = list(x.shape)
    shp[axis] = 1
    return Reshape.apply(s, tuple(shp))


# ===========================================================================
#  Arithmetic Ops
# ===========================================================================


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        if isinstance(b, (int, float)):
            _save_attr(ctx, a_shape=tuple(a.shape))
            return Tensor(a.data + b, dtype=a.dtype)
        if isinstance(a, (int, float)):
            _save_attr(ctx, b_shape=tuple(b.shape))
            return Tensor(a + b.data, dtype=b.dtype)
        _save_attr(ctx, a_shape=tuple(a.shape), b_shape=tuple(b.shape))
        return Tensor(a.data + b.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a_shape = _get_attr(ctx, "a_shape")
        b_shape = _get_attr(ctx, "b_shape")
        if create_graph:
            return [
                _cg_broadcast(grad_output, a_shape),
                _cg_broadcast(grad_output, b_shape),
            ]
        return _broadcast_grad(grad_output, a_shape, b_shape)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        if isinstance(b, (int, float)):
            _save_attr(ctx, a_shape=tuple(a.shape))
            return Tensor(a.data - b, dtype=a.dtype)
        if isinstance(a, (int, float)):
            _save_attr(ctx, b_shape=tuple(b.shape))
            return Tensor(a - b.data, dtype=b.dtype)
        _save_attr(ctx, a_shape=tuple(a.shape), b_shape=tuple(b.shape))
        return Tensor(a.data - b.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a_shape = _get_attr(ctx, "a_shape")
        b_shape = _get_attr(ctx, "b_shape")
        if create_graph:
            ga = _cg_broadcast(grad_output, a_shape)
            gb = _cg_broadcast(grad_output, b_shape)
            return [ga, Mul.apply(gb, _as_const(-1.0, grad_output.dtype))]
        ga, gb = _broadcast_grad(
            grad_output, _get_attr(ctx, "a_shape"), _get_attr(ctx, "b_shape")
        )
        return [ga, -gb]


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if isinstance(b, (int, float)):
            _save_attr(ctx, scalar_side=1, scalar=b, a_shape=tuple(a.shape))
            return Tensor(a.data * b, dtype=a.dtype)
        if isinstance(a, (int, float)):
            _save_attr(ctx, scalar_side=0, scalar=a, b_shape=tuple(b.shape))
            return Tensor(a * b.data, dtype=b.dtype)
        if ctx is not None:
            ctx.save_for_backward(a=a, b=b)
        _save_attr(ctx, a_shape=tuple(a.shape), b_shape=tuple(b.shape))
        return Tensor(a.data * b.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        side = _get_attr(ctx, "scalar_side")
        if side == 1:
            scalar = _get_attr(ctx, "scalar")
            if create_graph:
                g = _cg_broadcast(grad_output, _get_attr(ctx, "a_shape"))
                return [Mul.apply(g, _as_const(scalar, grad_output.dtype)), None]
            return [Tensor(_reduce_to_shape(grad_output, _get_attr(ctx, "a_shape")) * scalar, dtype=grad_output.dtype), None]
        if side == 0:
            scalar = _get_attr(ctx, "scalar")
            if create_graph:
                g = _cg_broadcast(grad_output, _get_attr(ctx, "b_shape"))
                return [None, Mul.apply(g, _as_const(scalar, grad_output.dtype))]
            return [None, Tensor(_reduce_to_shape(grad_output, _get_attr(ctx, "b_shape")) * scalar, dtype=grad_output.dtype)]
        a = _get_saved_tensor(ctx, "a")
        b = _get_saved_tensor(ctx, "b")
        if a is None or b is None:
            return _broadcast_grad(grad_output, None, None)
        if create_graph:
            return [Mul.apply(grad_output, b), Mul.apply(grad_output, a)]
        ga = Tensor(_reduce_to_shape(grad_output.data * b.data, _get_attr(ctx, "a_shape")), dtype=grad_output.dtype)
        gb = Tensor(_reduce_to_shape(grad_output.data * a.data, _get_attr(ctx, "b_shape")), dtype=grad_output.dtype)
        return [ga, gb]


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        if isinstance(b, (int, float)):
            _save_attr(ctx, b_val=b, a_shape=tuple(a.shape))
            return Tensor(a.data / b, dtype=a.dtype)
        if ctx is not None:
            ctx.save_for_backward(a=a, b=b)
        _save_attr(ctx, a_shape=tuple(a.shape), b_shape=tuple(b.shape))
        return Tensor(a.data / b.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        b_val = _get_attr(ctx, "b_val")
        if b_val is not None:
            if create_graph:
                g = _cg_broadcast(grad_output, _get_attr(ctx, "a_shape"))
                return [Div.apply(g, _as_const(b_val, grad_output.dtype)), None]
            ga = Tensor(
                _reduce_to_shape(grad_output, _get_attr(ctx, "a_shape")) / b_val,
                dtype=grad_output.dtype,
            )
            return [ga, None]
        a = _get_saved_tensor(ctx, "a")
        b = _get_saved_tensor(ctx, "b")
        if a is None or b is None:
            return _broadcast_grad(grad_output, None, None)
        if create_graph:
            ga = Div.apply(grad_output, b)
            num = Mul.apply(grad_output, a)
            den = Mul.apply(b, b)
            gb = Mul.apply(_as_const(-1.0, grad_output.dtype), Div.apply(num, den))
            return [ga, gb]
        g = grad_output.data
        ga = Tensor(_reduce_to_shape(g / b.data, _get_attr(ctx, "a_shape")), dtype=a.dtype)
        gb = Tensor(
            _reduce_to_shape(-g * a.data / (b.data ** 2), _get_attr(ctx, "b_shape")),
            dtype=b.dtype,
        )
        return [ga, gb]


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return Tensor(-a.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        if create_graph:
            return [Mul.apply(grad_output, _as_const(-1.0, grad_output.dtype))]
        return [-grad_output]


class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        b_val = b.data if isinstance(b, Tensor) else b
        ctx.save_for_backward(a=a)
        ctx.save_attr(b_val=b_val)
        return Tensor(a.data**b_val, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        b = ctx.get_attr("b_val")
        if create_graph:
            base = Pow.apply(a, _as_const(b - 1, grad_output.dtype))
            return [
                Mul.apply(
                    Mul.apply(grad_output, _as_const(b, grad_output.dtype)), base
                )
            ]
        return [Tensor(b * (a.data ** (b - 1)) * grad_output.data, dtype=a.dtype)]


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a=a, b=b)
        return Tensor(a.data @ b.data, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        b = ctx.get_saved_tensor("b")
        a_data = a.data
        b_data = b.data
        g = grad_output
        if create_graph and a_data.ndim <= 2 and b_data.ndim <= 2:
            # grad_a = g @ b^T ; grad_b = a^T @ g  (graph-aware, 1-D safe)
            g_nd = grad_output.data.ndim
            b_nd = b_data.ndim
            a_nd = a_data.ndim
            # grad_b = a^T @ g
            if a_nd == 1:
                a_mat = Unsqueeze.apply(a, -1)  # (D,1)
                g_col = Unsqueeze.apply(g, -1) if g_nd == 1 else g
                grad_b = MatMul.apply(a_mat, g_col)
                if g_nd == 1:
                    grad_b = Squeeze.apply(grad_b, -1)
            else:
                grad_b = MatMul.apply(Transpose.apply(a, -1, -2), g)
            # grad_a = g @ b^T
            if b_nd == 1:
                g_col = Unsqueeze.apply(g, -1) if g_nd == 1 else g  # (N,1)
                b_row = Unsqueeze.apply(b, 0)  # (1,D)
                grad_a = MatMul.apply(g_col, b_row)
            else:
                g_row = Unsqueeze.apply(g, 0) if g_nd == 1 else g
                grad_a = MatMul.apply(g_row, Transpose.apply(b, -1, -2))
                if g_nd == 1:
                    grad_a = Squeeze.apply(grad_a, 0)
            return [grad_a, grad_b]
        gnp = grad_output.data
        if a_data.ndim == 1:
            grad_a = gnp @ b_data.T
            grad_b = np.outer(a_data, gnp)
        elif b_data.ndim == 1:
            grad_a = np.outer(gnp, b_data)
            grad_b = a_data.T @ gnp
        else:
            grad_a = gnp @ b_data.T
            grad_b = a_data.T @ gnp
        return [Tensor(grad_a, dtype=a.dtype), Tensor(grad_b, dtype=b.dtype)]


# ===========================================================================
#  Reduction Ops
# ===========================================================================


class Sum(Function):
    @staticmethod
    def forward(ctx, a, dim=None):
        ctx.save_attr(shape=a.shape, dim=dim)
        if dim is None:
            return Tensor(float(a.data.sum()), dtype=a.dtype)
        return Tensor(a.data.sum(axis=dim), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        shape = ctx.get_attr("shape")
        dim = ctx.get_attr("dim")
        if create_graph:
            return [_cg_broadcast(grad_output, shape)]
        g = grad_output.data
        if dim is None:
            return [
                Tensor(
                    np.full(
                        shape, g.flatten()[0], dtype=_numpy_dtype(grad_output.dtype)
                    ),
                    dtype=grad_output.dtype,
                )
            ]
        expand_shape = list(shape)
        for d in ([dim] if isinstance(dim, int) else dim):
            expand_shape[d] = 1
        g_br = g.reshape(expand_shape)
        return [
            Tensor(
                np.broadcast_to(g_br, shape).astype(_numpy_dtype(grad_output.dtype)),
                dtype=grad_output.dtype,
            )
        ]


class Mean(Function):
    @staticmethod
    def forward(ctx, a, dim=None):
        ctx.save_attr(shape=a.shape, numel=a.numel, dim=dim)
        if dim is None:
            return Tensor(float(a.data.mean()), dtype=a.dtype)
        return Tensor(a.data.mean(axis=dim), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        shape = ctx.get_attr("shape")
        n = ctx.get_attr("numel")
        dim = ctx.get_attr("dim")
        if create_graph:
            g2 = Div.apply(grad_output, _as_const(n, grad_output.dtype))
            return [_cg_broadcast(g2, shape)]
        g = grad_output.data
        if dim is None:
            return [
                Tensor(
                    np.full(
                        shape, g.flatten()[0] / n, dtype=_numpy_dtype(grad_output.dtype)
                    ),
                    dtype=grad_output.dtype,
                )
            ]
        dims = [dim] if isinstance(dim, int) else dim
        nelem = 1
        for d in dims:
            nelem *= shape[d]
        expand_shape = list(shape)
        for d in dims:
            expand_shape[d] = 1
        g_br = g.reshape(expand_shape)
        return [
            Tensor(
                np.broadcast_to(g_br, shape).astype(_numpy_dtype(grad_output.dtype))
                / nelem,
                dtype=grad_output.dtype,
            )
        ]


# ===========================================================================
#  Activation Ops
# ===========================================================================


class Relu(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a=a)
        return Tensor(np.maximum(0, a.data), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        if create_graph:
            mask = _as_const((a.data > 0), grad_output.dtype)
            return [Mul.apply(grad_output, mask)]
        return [
            Tensor(
                (a.data > 0).astype(grad_output.data.dtype) * grad_output.data,
                dtype=grad_output.dtype,
            )
        ]


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        out = 1.0 / (1.0 + np.exp(-a.data))
        ctx.save_for_backward(a=a)
        ctx.save_attr(out=out)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        if create_graph:
            a = ctx.get_saved_tensor("a")
            out = Sigmoid.apply(a)
            one_minus = Add.apply(_as_const(1.0, grad_output.dtype), Neg.apply(out))
            return [Mul.apply(grad_output, Mul.apply(out, one_minus))]
        out = ctx.get_attr("out")
        return [Tensor(out * (1 - out) * grad_output.data, dtype=grad_output.dtype)]


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.tanh(a.data)
        ctx.save_for_backward(a=a)
        ctx.save_attr(out=out)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        if create_graph:
            a = ctx.get_saved_tensor("a")
            out = Tanh.apply(a)
            out2 = Mul.apply(out, out)
            one_minus = Add.apply(_as_const(1.0, grad_output.dtype), Neg.apply(out2))
            return [Mul.apply(grad_output, one_minus)]
        out = ctx.get_attr("out")
        return [Tensor((1 - out**2) * grad_output.data, dtype=grad_output.dtype)]


class Gelu(Function):
    @staticmethod
    def forward(ctx, a):
        x = a.data
        c = 0.79788456
        tanh_arg = c * (x + 0.044715 * x**3)
        out = 0.5 * x * (1.0 + np.tanh(tanh_arg))
        ctx.save_for_backward(a=a)
        ctx.save_attr(out_val=out, tanh_arg=tanh_arg)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        if create_graph:
            c = 0.79788456
            dt = grad_output.dtype
            a3 = Pow.apply(a, _as_const(3.0, dt))
            ta = Mul.apply(_as_const(c, dt),
                           Add.apply(a, Mul.apply(_as_const(0.044715, dt), a3)))
            tanh_ta = Tanh.apply(ta)
            sech2 = Sub.apply(_as_const(1.0, dt), Mul.apply(tanh_ta, tanh_ta))
            ta_prime = Mul.apply(_as_const(c, dt),
                                 Add.apply(_as_const(1.0, dt),
                                           Mul.apply(_as_const(0.134145, dt),
                                                     Mul.apply(a, a))))
            term1 = Mul.apply(_as_const(0.5, dt),
                              Add.apply(_as_const(1.0, dt), tanh_ta))
            term2 = Mul.apply(Mul.apply(_as_const(0.5, dt), a),
                              Mul.apply(sech2, ta_prime))
            grad_x = Add.apply(term1, term2)
            return [Mul.apply(grad_output, grad_x)]
        x = a.data
        tanh_arg = ctx.get_attr("tanh_arg")
        sech2 = 1 - np.tanh(tanh_arg) ** 2
        c = 0.79788456
        grad_x = 0.5 * (1 + np.tanh(tanh_arg)) + 0.5 * x * sech2 * c * (
            1 + 0.134145 * x**2
        )
        return [Tensor(grad_x * grad_output.data, dtype=grad_output.dtype)]


class Silu(Function):
    @staticmethod
    def forward(ctx, a):
        x = a.data
        sig = 1.0 / (1.0 + np.exp(-x))
        out = x * sig
        ctx.save_attr(sig=sig, out=out)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        if create_graph:
            dt = grad_output.dtype
            a = ctx.get_saved_tensor("a")
            sig = Sigmoid.apply(a)
            out = Mul.apply(a, sig)
            one_minus = Sub.apply(_as_const(1.0, dt), sig)
            grad_x = Add.apply(sig, Mul.apply(out, one_minus))
            return [Mul.apply(grad_output, grad_x)]
        sig = ctx.get_attr("sig")
        out = ctx.get_attr("out")
        return [
            Tensor((sig + out * (1 - sig)) * grad_output.data, dtype=grad_output.dtype)
        ]


# ===========================================================================
#  Unary Math Ops
# ===========================================================================


class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.sqrt(a.data)
        ctx.save_for_backward(a=a)
        ctx.save_attr(out=out)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        if create_graph:
            a = ctx.get_saved_tensor("a")
            out = Sqrt.apply(a)
            den = Add.apply(Mul.apply(_as_const(2.0, grad_output.dtype), out),
                            _as_const(1e-10, grad_output.dtype))
            return [Div.apply(grad_output, den)]
        out = ctx.get_attr("out")
        return [Tensor(grad_output.data / (2 * out + 1e-10), dtype=grad_output.dtype)]


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        out = np.exp(a.data)
        ctx.save_for_backward(a=a)
        ctx.save_attr(out=out)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        if create_graph:
            a = ctx.get_saved_tensor("a")
            return [Mul.apply(grad_output, Exp.apply(a))]
        out = ctx.get_attr("out")
        return [Tensor(out * grad_output.data, dtype=grad_output.dtype)]


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a=a)
        return Tensor(np.log(a.data + 1e-10), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        if create_graph:
            return [Div.apply(grad_output, Add.apply(a, _as_const(1e-10, grad_output.dtype)))]
        return [Tensor(grad_output.data / (a.data + 1e-10), dtype=grad_output.dtype)]


class Abs(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a=a)
        return Tensor(np.abs(a.data), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        if create_graph:
            return [Mul.apply(grad_output, _as_const(np.sign(a.data), grad_output.dtype))]
        return [Tensor(np.sign(a.data) * grad_output.data, dtype=grad_output.dtype)]


# ===========================================================================
#  Softmax / LogSoftmax
# ===========================================================================


class Softmax(Function):
    @staticmethod
    def forward(ctx, a, dim=-1):
        x = a.data - a.data.max(axis=dim, keepdims=True)
        e = np.exp(x)
        out = e / e.sum(axis=dim, keepdims=True)
        ctx.save_attr(out=out, dim=dim)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        dim = ctx.get_attr("dim")
        if create_graph:
            a = ctx.get_saved_tensor("a")
            out = Softmax.apply(a, dim)
            og = Mul.apply(out, grad_output)
            s = Sum.apply(og, dim)
            diff = Sub.apply(grad_output, s)
            return [Mul.apply(out, diff)]
        out = ctx.get_attr("out")
        g = grad_output.data
        s = out * (g - (out * g).sum(axis=dim, keepdims=True))
        return [Tensor(s, dtype=grad_output.dtype)]


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, a, dim=-1):
        x = a.data - a.data.max(axis=dim, keepdims=True)
        e = np.exp(x)
        sm = e / e.sum(axis=dim, keepdims=True)
        out = np.log(sm + 1e-10)
        ctx.save_attr(sm=sm, dim=dim)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        dim = ctx.get_attr("dim")
        if create_graph:
            a = ctx.get_saved_tensor("a")
            sm = Softmax.apply(a, dim)
            s = Sum.apply(grad_output, dim)
            ss = Mul.apply(sm, s)
            return [Sub.apply(grad_output, ss)]
        sm = ctx.get_attr("sm")
        g = grad_output.data
        s = g - sm * g.sum(axis=dim, keepdims=True)
        return [Tensor(s, dtype=grad_output.dtype)]


# ===========================================================================
#  Shape Ops
# ===========================================================================


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_attr(orig_shape=a.shape)
        return Tensor(a.data.reshape(shape), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        orig_shape = ctx.get_attr("orig_shape")
        if create_graph:
            return [Reshape.apply(grad_output, orig_shape)]
        return [Tensor(grad_output.data.reshape(orig_shape), dtype=grad_output.dtype)]


class Transpose(Function):
    @staticmethod
    def forward(ctx, a, dim1=0, dim2=1):
        ctx.save_attr(dim1=dim1, dim2=dim2)
        return Tensor(a.data.swapaxes(dim1, dim2), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        dim1 = ctx.get_attr("dim1")
        dim2 = ctx.get_attr("dim2")
        if create_graph:
            return [Transpose.apply(grad_output, dim1, dim2)]
        return [Tensor(grad_output.data.swapaxes(dim1, dim2), dtype=grad_output.dtype)]


class Expand(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.save_attr(orig_shape=a.shape)
        return Tensor(np.broadcast_to(a.data, shape), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        orig_shape = ctx.get_attr("orig_shape")
        if create_graph:
            return [_cg_broadcast(grad_output, orig_shape)]
        g = grad_output.data
        if len(orig_shape) < g.ndim:
            axes = tuple(range(g.ndim - len(orig_shape)))
            g = g.sum(axis=axes)
        axes = tuple(
            i for i in range(len(orig_shape)) if orig_shape[i] == 1 and g.shape[i] > 1
        )
        if axes:
            g = g.sum(axis=axes, keepdims=True)
        return [Tensor(g.reshape(orig_shape), dtype=grad_output.dtype)]


class Squeeze(Function):
    @staticmethod
    def forward(ctx, a, dim=None):
        ctx.save_attr(orig_shape=a.shape, dim=dim)
        if dim is None:
            return Tensor(np.squeeze(a.data), dtype=a.dtype)
        return Tensor(np.squeeze(a.data, axis=dim), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        orig_shape = ctx.get_attr("orig_shape")
        if create_graph:
            return [Reshape.apply(grad_output, orig_shape)]
        return [Tensor(grad_output.data.reshape(orig_shape), dtype=grad_output.dtype)]


class Unsqueeze(Function):
    @staticmethod
    def forward(ctx, a, dim):
        ctx.save_attr(orig_shape=a.shape)
        return Tensor(np.expand_dims(a.data, dim), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        orig_shape = ctx.get_attr("orig_shape")
        if create_graph:
            return [Reshape.apply(grad_output, orig_shape)]
        return [Tensor(grad_output.data.reshape(orig_shape), dtype=grad_output.dtype)]


# ===========================================================================
#  Indexing
# ===========================================================================


class GetItem(Function):
    @staticmethod
    def forward(ctx, a, key):
        ctx.save_attr(key=key, orig_shape=a.shape)
        return Tensor(a.data[key], dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        key = ctx.get_attr("key")
        if create_graph:
            mask = np.zeros(
                ctx.get_attr("orig_shape"), dtype=_numpy_dtype(grad_output.dtype)
            )
            mask[key] = 1.0
            return [Mul.apply(grad_output, _as_const(mask, grad_output.dtype))]
        g = np.zeros(ctx.get_attr("orig_shape"), dtype=_numpy_dtype(grad_output.dtype))
        g[key] = grad_output.data
        return [Tensor(g, dtype=grad_output.dtype)]


# ===========================================================================
#  Normalization Ops
# ===========================================================================


class LayerNorm(Function):
    @staticmethod
    def forward(ctx, a, gamma, beta, eps=1e-5):
        x = a.data
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        g = gamma.data if isinstance(gamma, Tensor) else gamma
        b = beta.data if isinstance(beta, Tensor) else beta
        out = x_norm * g + b
        ctx.save_attr(
            mean=mean, var=var, x_norm=x_norm, gamma_val=g, eps=eps, shape=x.shape
        )
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        x_norm = ctx.get_attr("x_norm")
        gamma_val = ctx.get_attr("gamma_val")
        shape = ctx.get_attr("shape")
        if create_graph:
            a = ctx.get_saved_tensor("a")
            g = grad_output
            dt = g.dtype
            eps = ctx.get_attr("eps")
            gamma_t = (
                gamma_val
                if isinstance(gamma_val, Tensor)
                else _as_const(gamma_val, dt)
            )
            mean = _mean_keepdim(a, -1)
            xc = Sub.apply(a, mean)
            var = _mean_keepdim(Mul.apply(xc, xc), -1)
            std = Sqrt.apply(Add.apply(var, _as_const(eps, dt)))
            x_norm_g = Div.apply(xc, std)
            dx_norm = Mul.apply(g, gamma_t)
            mean_dxn = _mean_keepdim(dx_norm, -1)
            inner = _mean_keepdim(Mul.apply(dx_norm, x_norm_g), -1)
            dx = Div.apply(
                Sub.apply(Sub.apply(dx_norm, mean_dxn), Mul.apply(x_norm_g, inner)),
                std,
            )
            axes = tuple(range(g.ndim - 1))
            dgamma = Sum.apply(Mul.apply(g, x_norm_g), axes)
            dbeta = Sum.apply(g, axes)
            return [dx, dgamma, dbeta]
        g = grad_output.data
        n = shape[-1]

        dgamma = (g * x_norm).sum(axis=tuple(range(g.ndim - 1)), keepdims=False)
        dbeta = g.sum(axis=tuple(range(g.ndim - 1)), keepdims=False)
        dx_norm = g * gamma_val
        dx = (
            dx_norm
            - dx_norm.mean(axis=-1, keepdims=True)
            - x_norm * (dx_norm * x_norm).mean(axis=-1, keepdims=True)
        )
        dx = dx / np.sqrt(ctx.get_attr("var") + ctx.get_attr("eps"))
        return [
            Tensor(dx, dtype="float32"),
            Tensor(dgamma, dtype="float32"),
            Tensor(dbeta, dtype="float32"),
        ]


class RMSNorm(Function):
    @staticmethod
    def forward(ctx, a, gamma, eps=1e-6):
        x = a.data
        rms = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
        x_norm = x / rms
        g = gamma.data if isinstance(gamma, Tensor) else gamma
        out = x_norm * g
        ctx.save_attr(rms=rms, x_norm=x_norm, gamma_val=g, eps=eps)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        rms = ctx.get_attr("rms")
        x_norm = ctx.get_attr("x_norm")
        gamma_val = ctx.get_attr("gamma_val")
        if create_graph:
            g = grad_output
            dt = g.dtype
            eps = ctx.get_attr("eps")
            n = a.shape[-1]
            gamma_t = (
                gamma_val if isinstance(gamma_val, Tensor) else _as_const(gamma_val, dt)
            )
            rms_g = Sqrt.apply(Add.apply(_mean_keepdim(Mul.apply(a, a), -1),
                                        _as_const(eps, dt)))
            dx_norm = Mul.apply(g, gamma_t)
            inner = _sum_keepdim(Mul.apply(dx_norm, a), -1)
            rms3 = Mul.apply(rms_g, Mul.apply(rms_g, rms_g))
            term1 = Div.apply(dx_norm, rms_g)
            term2 = Div.apply(Mul.apply(inner, a),
                              Mul.apply(_as_const(n, dt), rms3))
            dx = Sub.apply(term1, term2)
            axes = tuple(range(g.ndim - 1))
            dgamma = Sum.apply(Mul.apply(g, x_norm), axes)
            return [dx, dgamma]
        g = grad_output.data
        n = a.shape[-1]

        dgamma = (g * x_norm).sum(axis=tuple(range(g.ndim - 1)), keepdims=False)
        dx_norm = g * gamma_val
        inner = (dx_norm * a.data).sum(axis=-1, keepdims=True)
        dx = dx_norm / rms - inner * a.data / (n * rms * rms * rms)
        return [Tensor(dx, dtype=a.dtype), Tensor(dgamma, dtype=a.dtype)]


class DropoutFn(Function):
    @staticmethod
    def forward(ctx, a, rate=0.5, seed=42):
        rng = np.random.RandomState(seed)
        mask = rng.binomial(1, 1.0 - rate, a.shape).astype(np.float32)
        mask /= 1.0 - rate
        ctx.save_attr(mask=mask)
        return Tensor(a.data * mask, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        mask = ctx.get_attr("mask")
        if create_graph:
            return [Mul.apply(grad_output, _as_const(mask, grad_output.dtype))]
        return [Tensor(grad_output.data * mask, dtype=grad_output.dtype)]


# ===========================================================================
#  Neural Network Ops
# ===========================================================================


class LinearFn(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias=None):
        ctx.save_for_backward(inp=inp, weight=weight)
        out = inp.data @ weight.data.T
        has_b = bias is not None
        ctx.save_attr(has_bias=has_b)
        if has_b:
            b = bias.data if isinstance(bias, Tensor) else bias
            ctx.save_for_backward(bias=bias)
            out = out + b
        return Tensor(out, dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        w = ctx.get_saved_tensor("weight")
        if create_graph and w.data.ndim == 2:
            grads = [
                MatMul.apply(grad_output, Transpose.apply(w, -1, -2)),
                MatMul.apply(Transpose.apply(grad_output, -1, -2), inp),
            ]
            if ctx.get_attr("has_bias"):
                grads.append(Sum.apply(grad_output, 0))
            return grads
        g = grad_output.data
        grads = [
            Tensor(g @ w.data, dtype=inp.dtype),
            Tensor(g.T @ inp.data, dtype=w.dtype),
        ]
        if ctx.get_attr("has_bias"):
            grads.append(Tensor(g.sum(axis=0), dtype="float32"))
        return grads


class EmbeddingFn(Function):
    @staticmethod
    def forward(ctx, weight, indices):
        idx = (
            indices.data.astype(np.int64)
            if isinstance(indices, Tensor)
            else np.array(indices, dtype=np.int64)
        )
        ctx.save_attr(indices=idx)
        ctx.save_for_backward(weight=weight)
        return Tensor(weight.data[idx], dtype=weight.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.get_saved_tensor("weight")
        indices = ctx.get_attr("indices")
        g = grad_output.data
        grad_w = np.zeros(
            (weight.shape[0], weight.shape[1]), dtype=_numpy_dtype(weight.dtype)
        )
        for i in range(grad_w.shape[0]):
            mask = indices == i
            if mask.any():
                grad_w[i] = g[mask].sum(axis=0)
        return [Tensor(grad_w, dtype=weight.dtype), None]


# ===========================================================================
#  Convolution & Pooling Ops
# ===========================================================================


class Conv1d(Function):
    @staticmethod
    def forward(ctx, inp, kernel, stride=1, padding=0):
        from numpy.lib.stride_tricks import sliding_window_view

        arr = inp.data
        k = kernel.data if isinstance(kernel, Tensor) else kernel
        batched = arr.ndim == 3
        if not batched:
            arr = arr[None]
        N, C_in, L = arr.shape
        C_out, C_in_k, K = k.shape
        Lp = L + 2 * padding
        padded = np.pad(arr, [(0, 0), (0, 0), (padding, padding)]) if padding > 0 else arr
        win = sliding_window_view(padded, K, axis=2)[:, :, ::stride, :]
        Lout = win.shape[2]
        out = np.einsum("nclk,ock->nol", win, k)
        ctx.save_for_backward(
            inp=inp, kernel=kernel if isinstance(kernel, Tensor) else Tensor(kernel)
        )
        ctx.save_attr(
            stride=stride,
            padding=padding,
            L=L,
            K=K,
            C_in=C_in,
            C_out=C_out,
            Lout=Lout,
            Lp=Lp,
            batched=batched,
        )
        return Tensor(out if batched else out[0], dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        from numpy.lib.stride_tricks import sliding_window_view

        inp = ctx.get_saved_tensor("inp")
        kernel = ctx.get_saved_tensor("kernel")
        stride = ctx.get_attr("stride")
        padding = ctx.get_attr("padding")
        L = ctx.get_attr("L")
        K = ctx.get_attr("K")
        Lout = ctx.get_attr("Lout")
        Lp = ctx.get_attr("Lp")
        C_in = ctx.get_attr("C_in")
        batched = ctx.get_attr("batched")
        k = kernel.data
        g = grad_output.data
        garr = g if g.ndim == 3 else g[None]

        if not create_graph:
            # grad w.r.t input: scatter-add einsum('nol,ock->nclk') at l*stride+k
            N = garr.shape[0]
            padded = np.zeros((N, C_in, Lp), dtype=g.dtype)
            contrib = np.einsum("nol,ock->nclk", garr, k)
            for l in range(Lout):
                for kk in range(K):
                    padded[:, :, l * stride + kk] += contrib[:, :, l, kk]
            grad_inp_full = padded[:, :, padding : L + padding] if padding > 0 else padded
            grad_inp = grad_inp_full if batched else grad_inp_full[0]

            # grad w.r.t kernel: einsum over batch/spatial of g * window
            arr = inp.data
            arrb = arr if arr.ndim == 3 else arr[None]
            padded_in = (
                np.pad(arrb, [(0, 0), (0, 0), (padding, padding)])
                if padding > 0
                else arrb
            )
            win = sliding_window_view(padded_in, K, axis=2)[:, :, ::stride, :]
            grad_k = np.einsum("nol,nclk->ock", garr, win)
            return [Tensor(grad_inp, dtype=inp.dtype), Tensor(grad_k, dtype=kernel.dtype)]

        # ---- graph-aware backward (higher-order differentiation) ----
        dt = grad_output.dtype
        N = garr.shape[0]
        # build batched graph tensors
        g_graph = grad_output if g.ndim == 3 else Unsqueeze.apply(grad_output, 0)
        if batched:
            inp_graph = inp
        else:
            inp_graph = Unsqueeze.apply(inp, 0)
        if padding > 0:
            zfront = _as_const(
                np.zeros((N, C_in, padding), dtype=_numpy_dtype(dt)), dt
            )
            zback = _as_const(
                np.zeros((N, C_in, padding), dtype=_numpy_dtype(dt)), dt
            )
            inp_padded = Cat.apply(zfront, inp_graph, zback, 2)
        else:
            inp_padded = inp_graph

        # grad w.r.t kernel: windows = inp_padded (N,C_in,Lout,K)
        wins = []
        for l in range(Lout):
            cols = [
                GetItem.apply(inp_padded, (slice(None), slice(None), l * stride + kk))
                for kk in range(K)
            ]
            wins.append(Stack.apply(*cols, -1))  # (N,C_in,K)
        windows = Stack.apply(*wins, 2)  # (N,C_in,Lout,K)
        g_e = Unsqueeze.apply(Unsqueeze.apply(g_graph, 2), -1)  # (N,C_out,1,Lout,1)
        w_e = Unsqueeze.apply(windows, 1)  # (N,1,C_in,Lout,K)
        prod = Mul.apply(g_e, w_e)  # (N,C_out,C_in,Lout,K)
        grad_k = Sum.apply(prod, (0, 3))  # (C_out,C_in,K)

        # grad w.r.t input: scatter-add via constant position masks
        acc = _as_const(np.zeros((N, C_in, Lp), dtype=_numpy_dtype(dt)), dt)
        pos = np.arange(Lp)
        for l in range(Lout):
            gl = GetItem.apply(g_graph, (slice(None), slice(None), l))  # (N,C_out)
            for kk in range(K):
                p = l * stride + kk
                kk_t = GetItem.apply(kernel, (slice(None), slice(None), kk))  # (C_out,C_in)
                Tlk = MatMul.apply(gl, kk_t)  # (N,C_in)
                mask_t = _as_const((pos == p).astype(np.float64), dt)  # (Lp,)
                full_p = Mul.apply(Unsqueeze.apply(Tlk, -1), mask_t)  # (N,C_in,Lp)
                acc = Add.apply(acc, full_p)
        grad_inp_full = acc
        if padding > 0:
            grad_inp_full = GetItem.apply(
                grad_inp_full,
                (slice(None), slice(None), slice(padding, L + padding)),
            )
        grad_inp = grad_inp_full if batched else Squeeze.apply(grad_inp_full, 0)
        return [grad_inp, grad_k]


class MaxPool2d(Function):
    @staticmethod
    def forward(ctx, inp, kernel_h, kernel_w, stride_h=None, stride_w=None):
        stride_h = stride_h or kernel_h
        stride_w = stride_w or kernel_w
        arr = inp.data
        ctx.save_attr(k_h=kernel_h, k_w=kernel_w, s_h=stride_h, s_w=stride_w)
        ctx.save_for_backward(inp=inp)
        out = np.array(
            [
                [
                    arr[..., i : i + kernel_h, j : j + kernel_w].max(axis=(-2, -1))
                    for j in range(0, arr.shape[-1] - kernel_w + 1, stride_w)
                ]
                for i in range(0, arr.shape[-2] - kernel_h + 1, stride_h)
            ]
        )
        out = out.transpose(2, 3, 0, 1) if arr.ndim == 4 else out
        return Tensor(out, dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        inp = ctx.get_saved_tensor("inp")
        k_h = ctx.get_attr("k_h")
        k_w = ctx.get_attr("k_w")
        s_h = ctx.get_attr("s_h")
        s_w = ctx.get_attr("s_w")
        arr = inp.data
        g = grad_output.data
        grad = np.zeros_like(arr)
        for i in range(0, arr.shape[-2] - k_h + 1, s_h):
            for j in range(0, arr.shape[-1] - k_w + 1, s_w):
                patch = arr[..., i : i + k_h, j : j + k_w]
                max_val = patch.max(axis=(-2, -1), keepdims=True)
                max_mask = (patch == max_val).astype(g.dtype)
                gi = i // s_h
                gj = j // s_w
                if g.ndim == 4:
                    g_val = g[..., gi, gj, None, None]
                else:
                    g_val = g
                grad[..., i : i + k_h, j : j + k_w] += max_mask * g_val
        return [Tensor(grad, dtype=inp.dtype)]


class AvgPool2d(Function):
    @staticmethod
    def forward(ctx, inp, kernel_h, kernel_w, stride_h=None, stride_w=None):
        stride_h = stride_h or kernel_h
        stride_w = stride_w or kernel_w
        arr = inp.data
        ctx.save_attr(k_h=kernel_h, k_w=kernel_w, s_h=stride_h, s_w=stride_w)
        ctx.save_for_backward(inp=inp)
        out = np.array(
            [
                [
                    arr[..., i : i + kernel_h, j : j + kernel_w].mean(axis=(-2, -1))
                    for j in range(0, arr.shape[-1] - kernel_w + 1, stride_w)
                ]
                for i in range(0, arr.shape[-2] - kernel_h + 1, stride_h)
            ]
        )
        out = out.transpose(2, 3, 0, 1) if arr.ndim == 4 else out
        return Tensor(out, dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        inp = ctx.get_saved_tensor("inp")
        k_h = ctx.get_attr("k_h")
        k_w = ctx.get_attr("k_w")
        s_h = ctx.get_attr("s_h")
        s_w = ctx.get_attr("s_w")
        arr = inp.data
        g = grad_output.data
        grad = np.zeros_like(arr)
        norm = k_h * k_w
        for i in range(0, arr.shape[-2] - k_h + 1, s_h):
            for j in range(0, arr.shape[-1] - k_w + 1, s_w):
                gi = i // s_h
                gj = j // s_w
                if g.ndim == 4:
                    g_val = g[..., gi, gj, None, None]
                else:
                    g_val = g
                grad[..., i : i + k_h, j : j + k_w] += g_val / norm
        return [Tensor(grad, dtype=inp.dtype)]


# ===========================================================================
#  Loss Ops
# ===========================================================================


class MSELoss(Function):
    @staticmethod
    def forward(ctx, inp, target):
        diff = inp.data - target.data
        ctx.save_for_backward(inp=inp, target=target)
        ctx.save_attr(n=inp.numel)
        return Tensor(np.array([float((diff**2).mean())]), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        target = ctx.get_saved_tensor("target")
        n = ctx.get_attr("n")
        if create_graph:
            dt = grad_output.dtype
            diff = Sub.apply(inp, target)
            grad_inp = Div.apply(
                Mul.apply(_as_const(2.0, dt), diff), _as_const(n, dt)
            )
            grad_inp = Mul.apply(grad_output, grad_inp)
            return [grad_inp, Neg.apply(grad_inp)]
        g = grad_output.data.flat[0]
        grad_inp = Tensor(2.0 * (inp.data - target.data) * g / n, dtype=inp.dtype)
        return [grad_inp, -grad_inp]


class CrossEntropyLoss(Function):
    @staticmethod
    def forward(ctx, inp, target):
        x = inp.data
        x_max = x.max(axis=-1, keepdims=True)
        logits = x - x_max
        e = np.exp(logits)
        sm = e / e.sum(axis=-1, keepdims=True)
        t = target.data if isinstance(target, Tensor) else target
        if t.ndim == sm.ndim:
            t = t.argmax(axis=-1)
        loss = -np.mean(np.log(sm[np.arange(sm.shape[0]), t.astype(np.int64)] + 1e-10))
        ctx.save_attr(sm=sm, t=t.astype(np.int64))
        ctx.save_for_backward(inp=inp)
        return Tensor(np.array([loss]), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        if create_graph:
            inp = ctx.get_saved_tensor("inp")
            t = ctx.get_attr("t")
            dt = grad_output.dtype
            sm_g = Softmax.apply(inp, -1)
            n = sm_g.shape[0]
            oh = np.zeros((n, sm_g.shape[-1]), dtype=_numpy_dtype(dt))
            oh[np.arange(n), t] = 1.0
            grad = Sub.apply(sm_g, _as_const(oh, dt))
            grad = Div.apply(grad, _as_const(n, dt))
            return [Mul.apply(grad_output, grad), None]
        sm = ctx.get_attr("sm")
        t = ctx.get_attr("t")
        g = grad_output.data.flat[0]
        grad = sm.copy()
        grad[np.arange(sm.shape[0]), t] -= 1.0
        grad /= sm.shape[0]
        return [Tensor(grad * g, dtype="float32"), None]


class NLLLoss(Function):
    @staticmethod
    def forward(ctx, inp, target):
        x = inp.data
        t = target.data if isinstance(target, Tensor) else np.asarray(target)
        ctx.save_for_backward(inp=inp)
        ctx.save_attr(t=t, one_hot=(t.ndim == x.ndim))
        n = x.shape[0] if x.ndim > 1 else 1
        ctx.save_attr(n=n)
        if t.ndim == x.ndim:  # one-hot targets
            nll = -np.sum(t * np.log(x + 1e-12), axis=-1)
            loss = float(np.mean(nll))
        else:  # class indices
            t = t.astype(np.int64)
            if x.ndim == 1:
                loss = float(-x[t])
            else:
                loss = float(np.mean(-x[np.arange(n), t]))
        return Tensor(np.array([loss], dtype=x.dtype), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        t = ctx.get_attr("t")
        one_hot = ctx.get_attr("one_hot")
        n = ctx.get_attr("n")
        if create_graph:
            dt = grad_output.dtype
            if one_hot:
                t_np = t.data if isinstance(t, Tensor) else t
                denom = Add.apply(inp, _as_const(1e-12, dt))
                grad_in = Neg.apply(Div.apply(_as_const(t_np, dt), denom))
                grad_in = Div.apply(Mul.apply(grad_in, grad_output), _as_const(n, dt))
            else:
                t_idx = t.astype(np.int64)
                oh = np.zeros_like(inp.data)
                oh[np.arange(oh.shape[0]), t_idx] = -1.0 / n
                grad_in = Mul.apply(grad_output, _as_const(oh, dt))
            return [grad_in, None]
        g = grad_output.data.flat[0]
        x = inp.data
        grad_inp = np.zeros_like(x)
        if one_hot:
            grad_inp = -(t.astype(np.float64) / (x + 1e-12)) * (g / n)
            grad_target = -(np.log(x + 1e-12)) * (g / n)
        else:
            t = t.astype(np.int64)
            if x.ndim == 1:
                grad_inp[t] = -g / n
            else:
                grad_inp[np.arange(n), t] = -g / n
            grad_target = None
        return [Tensor(grad_inp, dtype=inp.dtype), grad_target]


class KLDivLoss(Function):
    @staticmethod
    def forward(ctx, inp, target):
        x = inp.data
        t = target.data if isinstance(target, Tensor) else np.asarray(target)
        ctx.save_for_backward(inp=inp)
        ctx.save_attr(t=t, n=x.size)
        # input `x` is log-probabilities; target `t` is probabilities.
        kl = t * (np.log(t + 1e-10) - x)
        return Tensor(np.array([float(np.mean(kl))], dtype=x.dtype), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        t = ctx.get_attr("t")
        n = ctx.get_attr("n")
        if create_graph:
            t_np = t.data if isinstance(t, Tensor) else t
            return [
                Mul.apply(
                    grad_output,
                    _as_const(
                        -np.asarray(t_np, dtype=_numpy_dtype(grad_output.dtype)) / n,
                        grad_output.dtype,
                    ),
                ),
                None,
            ]
        g = grad_output.data.flat[0]
        # kl = t*(log t - x)  =>  d kl/d x = -t ; target is a probability (not diff)
        grad_inp = Tensor(-(t) * (g / n), dtype=inp.dtype)
        return [grad_inp, None]


class Stack(Function):
    @staticmethod
    def forward(ctx, *args):
        tensors = [a for a in args if isinstance(a, Tensor)]
        dim = args[-1] if isinstance(args[-1], int) else 0
        ctx.save_attr(dim=dim)
        ctx.save_attr(n=len(tensors))
        data = np.stack([t.data for t in tensors], axis=dim)
        return Tensor(data, dtype=tensors[0].dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        dim = ctx.get_attr("dim")
        n = ctx.get_attr("n")
        if create_graph:
            grads = []
            for i in range(n):
                key = [slice(None)] * grad_output.ndim
                key[dim] = slice(i, i + 1)
                grads.append(Squeeze.apply(GetItem.apply(grad_output, tuple(key)), dim))
            return grads
        grads = np.split(grad_output.data, n, axis=dim)
        return [Tensor(np.squeeze(g, axis=dim).copy(), dtype=grad_output.dtype) for g in grads]


class Cat(Function):
    @staticmethod
    def forward(ctx, *args):
        tensors = [a for a in args if isinstance(a, Tensor)]
        dim = args[-1] if isinstance(args[-1], int) else 0
        ctx.save_attr(dim=dim)
        ctx.save_attr(sizes=[t.shape[dim] for t in tensors])
        data = np.concatenate([t.data for t in tensors], axis=dim)
        return Tensor(data, dtype=tensors[0].dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        dim = ctx.get_attr("dim")
        sizes = ctx.get_attr("sizes")
        if create_graph:
            grads = []
            start = 0
            for s in sizes:
                key = [slice(None)] * grad_output.ndim
                key[dim] = slice(start, start + s)
                grads.append(GetItem.apply(grad_output, tuple(key)))
                start += s
            return grads
        split_pts = np.cumsum(sizes)[:-1]
        grads = np.split(grad_output.data, split_pts, axis=dim)
        return [Tensor(g.copy(), dtype=grad_output.dtype) for g in grads]


# ===========================================================================
#  Additional Loss Ops (smooth L1, BCE, focal, ranking, embedding, triplet)
# ===========================================================================


class SmoothL1Loss(Function):
    @staticmethod
    def forward(ctx, inp, target, beta=1.0):
        x = inp.data - target.data
        ctx.save_for_backward(inp=inp, target=target)
        ctx.save_attr(beta=beta, n=x.size)
        absx = np.abs(x)
        loss = np.where(absx < beta, 0.5 * x**2 / beta, absx - 0.5 * beta)
        return Tensor(np.array([float(np.mean(loss))], dtype=x.dtype), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        target = ctx.get_saved_tensor("target")
        beta = ctx.get_attr("beta")
        n = ctx.get_attr("n")
        if create_graph:
            dt = grad_output.dtype
            x = Sub.apply(inp, target)
            absx = Abs.apply(x)
            mask = absx.data < beta  # constant w.r.t. x for 2nd-order
            sgn = np.sign(x.data)
            x_scaled = Mul.apply(x, _as_const(1.0 / beta, dt))
            term_smooth = Mul.apply(_as_const(mask.astype(np.float64), dt), x_scaled)
            term_linear = _as_const((~mask).astype(np.float64) * sgn, dt)
            grad_x = Mul.apply(
                Add.apply(term_smooth, term_linear),
                Div.apply(grad_output, _as_const(n, dt)),
            )
            return [grad_x, Neg.apply(grad_x)]
        g = grad_output.data.flat[0]
        x = inp.data - target.data
        absx = np.abs(x)
        grad_x = np.where(absx < beta, x / beta, np.sign(x)) / n * g
        return [Tensor(grad_x, dtype=inp.dtype), Tensor(-grad_x, dtype=inp.dtype)]


class HuberLoss(Function):
    @staticmethod
    def forward(ctx, inp, target, delta=1.0):
        x = inp.data - target.data
        ctx.save_for_backward(inp=inp, target=target)
        ctx.save_attr(delta=delta, n=x.size)
        absx = np.abs(x)
        loss = np.where(absx <= delta, 0.5 * x**2, delta * (absx - 0.5 * delta))
        return Tensor(np.array([float(np.mean(loss))], dtype=x.dtype), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        target = ctx.get_saved_tensor("target")
        delta = ctx.get_attr("delta")
        n = ctx.get_attr("n")
        if create_graph:
            dt = grad_output.dtype
            x = Sub.apply(inp, target)
            absx = Abs.apply(x)
            mask = absx.data <= delta  # constant w.r.t. x for 2nd-order
            sgn = np.sign(x.data)
            term_quad = Mul.apply(_as_const(mask.astype(np.float64), dt), x)
            term_lin = _as_const((~mask).astype(np.float64) * delta * sgn, dt)
            grad_x = Mul.apply(
                Add.apply(term_quad, term_lin),
                Div.apply(grad_output, _as_const(n, dt)),
            )
            return [grad_x, Neg.apply(grad_x)]
        g = grad_output.data.flat[0]
        x = inp.data - target.data
        absx = np.abs(x)
        grad_x = np.where(absx <= delta, x, delta * np.sign(x)) / n * g
        return [Tensor(grad_x, dtype=inp.dtype), Tensor(-grad_x, dtype=inp.dtype)]


class BCELoss(Function):
    @staticmethod
    def forward(ctx, inp, target):
        x = inp.data
        t = target.data
        ctx.save_for_backward(inp=inp, target=target)
        n = x.size
        ctx.save_attr(n=n)
        loss = -(t * np.log(x + 1e-10) + (1 - t) * np.log(1 - x + 1e-10))
        return Tensor(np.array([float(np.mean(loss))], dtype=x.dtype), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        target = ctx.get_saved_tensor("target")
        n = ctx.get_attr("n")
        if create_graph:
            dt = grad_output.dtype
            denom = Add.apply(
                Mul.apply(inp, Sub.apply(_as_const(1.0, dt), inp)),
                _as_const(1e-10, dt),
            )
            inner = Div.apply(Sub.apply(inp, target), denom)
            grad_in = Div.apply(Mul.apply(grad_output, inner), _as_const(n, dt))
            return [grad_in, None]
        g = grad_output.data.flat[0]
        x = inp.data
        t = target.data
        grad_in = (x - t) / (x * (1 - x) + 1e-10) / n * g
        grad_t = -(np.log(x + 1e-10) - np.log(1 - x + 1e-10)) / n * g
        return [Tensor(grad_in, dtype=inp.dtype), Tensor(grad_t, dtype=inp.dtype)]


class BCEWithLogitsLoss(Function):
    @staticmethod
    def forward(ctx, inp, target):
        z = inp.data
        t = target.data
        ctx.save_for_backward(inp=inp, target=target)
        n = z.size
        ctx.save_attr(n=n)
        z = np.clip(z, -30, 30)
        sig = 1.0 / (1.0 + np.exp(-z))
        loss = -(t * np.log(sig + 1e-10) + (1 - t) * np.log(1 - sig + 1e-10))
        return Tensor(np.array([float(np.mean(loss))], dtype=z.dtype), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        target = ctx.get_saved_tensor("target")
        n = ctx.get_attr("n")
        if create_graph:
            dt = grad_output.dtype
            sig = Sigmoid.apply(inp)
            grad_in = Div.apply(
                Mul.apply(Sub.apply(sig, target), grad_output), _as_const(n, dt)
            )
            return [grad_in, None]
        g = grad_output.data.flat[0]
        z = np.clip(inp.data, -30, 30)
        sig = 1.0 / (1.0 + np.exp(-z))
        grad_in = (sig - target.data) / n * g
        return [Tensor(grad_in, dtype=inp.dtype), None]


class FocalLoss(Function):
    """Focal loss over class logits (target = integer class indices)."""

    @staticmethod
    def forward(ctx, logits, target, gamma=2.0, alpha=1.0):
        z = logits.data.astype(np.float64)
        t = target.data.astype(np.int64).ravel()
        zmax = z.max(axis=-1, keepdims=True)
        e = np.exp(z - zmax)
        p = e / e.sum(axis=-1, keepdims=True)
        n = z.shape[0]
        pt = p[np.arange(n), t]
        loss = -alpha * (1 - pt) ** gamma * np.log(pt + 1e-10)
        ctx.save_for_backward(logits=logits, target=target)
        ctx.save_attr(gamma=gamma, alpha=alpha, n=n, p=p, pt=pt)
        return Tensor(np.array([float(np.mean(loss))], dtype=z.dtype), dtype=logits.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        logits = ctx.get_saved_tensor("logits")
        target = ctx.get_saved_tensor("target")
        gamma = ctx.get_attr("gamma")
        alpha = ctx.get_attr("alpha")
        n = ctx.get_attr("n")
        p = ctx.get_attr("p")
        pt = ctx.get_attr("pt")
        g = grad_output.data.flat[0]
        t = target.data.astype(np.int64).ravel()
        onehot = np.zeros_like(p)
        onehot[np.arange(n), t] = 1.0
        f = gamma * pt * (1 - pt) ** (gamma - 1) * np.log(pt + 1e-10) - (1 - pt) ** gamma
        grad_z = alpha * (onehot - p) * f[:, None] / n * g
        return [Tensor(grad_z.astype(np.float64), dtype=logits.dtype), None]


class CosineEmbeddingLoss(Function):
    @staticmethod
    def forward(ctx, x1, x2, y, margin=0.0):
        a = x1.data
        b = x2.data
        yy = y.data.ravel().astype(np.int64)
        na = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10
        nb = np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10
        cos = np.sum(a * b, axis=-1) / (na.ravel() * nb.ravel())
        loss = np.where(yy == 1, 1 - cos, np.maximum(0.0, cos - margin))
        ctx.save_for_backward(x1=x1, x2=x2, y=y)
        ctx.save_attr(margin=margin, n=a.shape[0], cos=cos, na=na, nb=nb)
        return Tensor(np.array([float(np.mean(loss))], dtype=a.dtype), dtype=x1.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x1 = ctx.get_saved_tensor("x1")
        x2 = ctx.get_saved_tensor("x2")
        y = ctx.get_saved_tensor("y")
        margin = ctx.get_attr("margin")
        n = ctx.get_attr("n")
        cos = ctx.get_attr("cos")
        na = ctx.get_attr("na")
        nb = ctx.get_attr("nb")
        g = grad_output.data.flat[0]
        a = x1.data
        b = x2.data
        yy = y.data.ravel().astype(np.int64)
        da = b / (na * nb) - cos[:, None] * a / na**2
        db = a / (na * nb) - cos[:, None] * b / nb**2
        active = (yy != 1) & (cos > margin)
        ga = np.where(yy[:, None] == 1, -da, np.where(active[:, None], da, 0.0)) / n * g
        gb = np.where(yy[:, None] == 1, -db, np.where(active[:, None], db, 0.0)) / n * g
        return [Tensor(ga, dtype=x1.dtype), Tensor(gb, dtype=x1.dtype), None]


class TripletMarginLoss(Function):
    @staticmethod
    def forward(ctx, anchor, positive, negative, margin=1.0, p=2):
        a = anchor.data
        pos = positive.data
        neg = negative.data
        d_ap = np.linalg.norm(a - pos, axis=-1) + 1e-10
        d_an = np.linalg.norm(a - neg, axis=-1) + 1e-10
        hinge = d_ap - d_an + margin
        loss = np.maximum(0.0, hinge)
        ctx.save_for_backward(anchor=anchor, positive=positive, negative=negative)
        ctx.save_attr(margin=margin, n=a.shape[0], d_ap=d_ap, d_an=d_an)
        return Tensor(np.array([float(np.mean(loss))], dtype=a.dtype), dtype=anchor.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        anchor = ctx.get_saved_tensor("anchor")
        positive = ctx.get_saved_tensor("positive")
        negative = ctx.get_saved_tensor("negative")
        margin = ctx.get_attr("margin")
        n = ctx.get_attr("n")
        d_ap = ctx.get_attr("d_ap")
        d_an = ctx.get_attr("d_an")
        g = grad_output.data.flat[0]
        a = anchor.data
        pos = positive.data
        neg = negative.data
        hinge = d_ap - d_an + margin
        active = (hinge > 0).astype(np.float64)[:, None]
        ga = active * ((a - pos) / d_ap[:, None] - (a - neg) / d_an[:, None]) / n * g
        gp = active * (-(a - pos) / d_ap[:, None]) / n * g
        gn = active * ((a - neg) / d_an[:, None]) / n * g
        return [Tensor(ga, dtype=anchor.dtype), Tensor(gp, dtype=anchor.dtype), Tensor(gn, dtype=anchor.dtype), None]


class ContrastiveLoss(Function):
    @staticmethod
    def forward(ctx, x1, x2, y, margin=1.0):
        a = x1.data
        b = x2.data
        yy = y.data.ravel().astype(np.int64)
        d = np.linalg.norm(a - b, axis=-1) + 1e-10
        sim = (1 - yy) * np.maximum(0.0, margin - d) ** 2
        loss = yy * d**2 + sim
        ctx.save_for_backward(x1=x1, x2=x2, y=y)
        ctx.save_attr(margin=margin, n=a.shape[0], d=d)
        return Tensor(np.array([float(np.mean(loss))], dtype=a.dtype), dtype=x1.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x1 = ctx.get_saved_tensor("x1")
        x2 = ctx.get_saved_tensor("x2")
        y = ctx.get_saved_tensor("y")
        margin = ctx.get_attr("margin")
        n = ctx.get_attr("n")
        d = ctx.get_attr("d")
        g = grad_output.data.flat[0]
        a = x1.data
        b = x2.data
        yy = y.data.ravel().astype(np.float64)
        diff = a - b
        dn = d[:, None]
        dd_ddiff = diff / dn
        grad_d2 = yy[:, None] * 2.0 * diff
        mask = (d < margin)[:, None]
        grad_sim = (1 - yy)[:, None] * np.where(
            mask, -2.0 * (margin - d)[:, None] * dd_ddiff, 0.0
        )
        ga = (grad_d2 + grad_sim) / n * g
        gb = -ga
        return [Tensor(ga, dtype=x1.dtype), Tensor(gb, dtype=x1.dtype), None]


class MarginRankingLoss(Function):
    @staticmethod
    def forward(ctx, x1, x2, y, margin=0.0):
        a = x1.data
        b = x2.data
        yy = y.data.ravel().astype(np.float64)
        loss = np.maximum(0.0, -yy * (a - b) + margin)
        ctx.save_for_backward(x1=x1, x2=x2, y=y)
        ctx.save_attr(margin=margin, n=a.size)
        return Tensor(np.array([float(np.mean(loss))], dtype=a.dtype), dtype=x1.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x1 = ctx.get_saved_tensor("x1")
        x2 = ctx.get_saved_tensor("x2")
        y = ctx.get_saved_tensor("y")
        n = ctx.get_attr("n")
        g = grad_output.data.flat[0]
        a = x1.data
        b = x2.data
        yy = y.data.ravel().astype(np.float64)
        hinge = -yy * (a - b) + ctx.get_attr("margin")
        active = (hinge > 0).astype(np.float64)
        grad_a = (active * (-yy)) / n * g
        grad_b = (active * (yy)) / n * g
        return [Tensor(grad_a, dtype=x1.dtype), Tensor(grad_b, dtype=x1.dtype), None]


def _ctc_logsumexp(a):
    a = list(a)
    m = max(a)
    if m <= -1e29:
        return -1e30
    return m + np.log(sum(np.exp(x - m) for x in a))


class CTCLoss(Function):
    """Connectionist Temporal Classification loss.

    Input ``log_probs`` is log-softmax output of shape ``(T, N, C)`` where T is
    the time dimension, N the batch, C the number of classes (index 0 is the
    blank).  ``targets`` is integer class indices of shape ``(N, S)`` (no blanks,
    no repeated adjacent labels).  ``input_lengths``/``target_lengths`` give the
    valid lengths per sample.  Forward/backward use the log-domain alpha/beta
    recursions; the gradient w.r.t. log-probabilities is ``-posterior``.
    """

    @staticmethod
    def forward(
        ctx, log_probs, targets, blank=0, input_lengths=None, target_lengths=None, reduction="mean"
    ):
        lp = log_probs.data
        T, N, C = lp.shape
        tgt = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
        if input_lengths is None:
            input_lengths = [T] * N
        if target_lengths is None:
            target_lengths = [tgt.shape[1]] * N

        losses = np.zeros(N)
        alphas = []
        exts = []
        for n in range(N):
            tg = tgt[n, : target_lengths[n]].astype(np.int64)
            L = 2 * len(tg) + 1
            ext = np.empty(L, dtype=np.int64)
            ext[0::2] = blank
            ext[1::2] = tg
            Ti = int(input_lengths[n])
            alpha = np.full((Ti, L), -1e30)
            alpha[0, 0] = lp[0, n, ext[0]]
            if L > 1:
                alpha[0, 1] = lp[0, n, ext[1]]
            for t in range(1, Ti):
                for s in range(L):
                    a = alpha[t - 1, s]
                    b = alpha[t - 1, s - 1] if s - 1 >= 0 else -1e30
                    c = (
                        alpha[t - 1, s - 2]
                        if (s - 2 >= 0 and ext[s - 2] != ext[s])
                        else -1e30
                    )
                    alpha[t, s] = _ctc_logsumexp([a, b, c]) + lp[t, n, ext[s]]
            z = _ctc_logsumexp(
                [alpha[Ti - 1, L - 1], alpha[Ti - 1, L - 2] if L >= 2 else -1e30]
            )
            losses[n] = -z
            alphas.append(alpha)
            exts.append(ext)

        loss = float(np.mean(losses)) if reduction == "mean" else float(np.sum(losses))
        ctx.save_for_backward(log_probs=log_probs)
        ctx.save_attr(
            alphas=alphas, exts=exts, N=N, T=T, C=C, blank=blank,
            input_lengths=input_lengths, target_lengths=target_lengths,
            reduction=reduction, lp=lp,
        )
        return Tensor(np.array([loss], dtype=lp.dtype), dtype=log_probs.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        lp = ctx.get_attr("lp")
        alphas = ctx.get_attr("alphas")
        exts = ctx.get_attr("exts")
        N = ctx.get_attr("N")
        T = ctx.get_attr("T")
        C = ctx.get_attr("C")
        blank = ctx.get_attr("blank")
        il = ctx.get_attr("input_lengths")
        tl = ctx.get_attr("target_lengths")
        reduction = ctx.get_attr("reduction")
        g = grad_output.data.flat[0]

        grad = np.zeros((T, N, C))
        for n in range(N):
            ext = exts[n]
            L = len(ext)
            Ti = int(il[n])
            beta = np.full((Ti, L), -1e30)
            beta[Ti - 1, L - 1] = 0.0
            if L >= 2:
                beta[Ti - 1, L - 2] = 0.0
            for t in range(Ti - 2, -1, -1):
                for s in range(L):
                    a = beta[t + 1, s] + lp[t + 1, n, ext[s]]
                    b = (
                        beta[t + 1, s + 1] + lp[t + 1, n, ext[s + 1]]
                        if s + 1 < L
                        else -1e30
                    )
                    c = (
                        beta[t + 1, s + 2] + lp[t + 1, n, ext[s + 2]]
                        if (s + 2 < L and ext[s + 2] != ext[s])
                        else -1e30
                    )
                    beta[t, s] = _ctc_logsumexp([a, b, c])
            z = _ctc_logsumexp(
                [alphas[n][Ti - 1, L - 1], alphas[n][Ti - 1, L - 2] if L >= 2 else -1e30]
            )
            for t in range(Ti):
                for s in range(L):
                    prob = np.exp(alphas[n][t, s] + beta[t, s] - z)
                    grad[t, n, ext[s]] += prob
            grad[:Ti, n, :] = -grad[:Ti, n, :]

        scale = (g / N) if reduction == "mean" else g
        grad = grad * scale
        return [Tensor(grad, dtype=ctx.get_saved_tensor("log_probs").dtype), None]


__all__ = [
    "Add",
    "Stack",
    "Cat",
    "Sub",
    "Mul",
    "Div",
    "Neg",
    "Pow",
    "MatMul",
    "Sum",
    "Mean",
    "Relu",
    "Sigmoid",
    "Tanh",
    "Gelu",
    "Silu",
    "Sqrt",
    "Exp",
    "Log",
    "Abs",
    "Softmax",
    "LogSoftmax",
    "Reshape",
    "Transpose",
    "Expand",
    "Squeeze",
    "Unsqueeze",
    "GetItem",
    "LayerNorm",
    "RMSNorm",
    "DropoutFn",
    "LinearFn",
    "EmbeddingFn",
    "Conv1d",
    "MaxPool2d",
    "AvgPool2d",
    "MSELoss",
    "CrossEntropyLoss",
    "NLLLoss",
    "KLDivLoss",
    "SmoothL1Loss",
    "HuberLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "FocalLoss",
    "CosineEmbeddingLoss",
    "TripletMarginLoss",
    "ContrastiveLoss",
    "MarginRankingLoss",
    "CTCLoss",
]
