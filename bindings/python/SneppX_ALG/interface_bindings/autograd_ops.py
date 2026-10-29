"""Autograd Ops � differentiable operation subclasses of Function.

Each op defines forward(ctx, *args) -> Tensor and the corresponding
backward(ctx, grad_output) -> list of gradients.

Consumed by tensor.py via lazy imports inside the op methods.
"""

import itertools

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
    for i in range(n - 1, -1, -1):
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
            return [
                Tensor(
                    _reduce_to_shape(grad_output, _get_attr(ctx, "a_shape")) * scalar,
                    dtype=grad_output.dtype,
                ),
                None,
            ]
        if side == 0:
            scalar = _get_attr(ctx, "scalar")
            if create_graph:
                g = _cg_broadcast(grad_output, _get_attr(ctx, "b_shape"))
                return [None, Mul.apply(g, _as_const(scalar, grad_output.dtype))]
            return [
                None,
                Tensor(
                    _reduce_to_shape(grad_output, _get_attr(ctx, "b_shape")) * scalar,
                    dtype=grad_output.dtype,
                ),
            ]
        a = _get_saved_tensor(ctx, "a")
        b = _get_saved_tensor(ctx, "b")
        if a is None or b is None:
            return _broadcast_grad(grad_output, None, None)
        if create_graph:
            return [Mul.apply(grad_output, b), Mul.apply(grad_output, a)]
        ga = Tensor(
            _reduce_to_shape(grad_output.data * b.data, _get_attr(ctx, "a_shape")),
            dtype=grad_output.dtype,
        )
        gb = Tensor(
            _reduce_to_shape(grad_output.data * a.data, _get_attr(ctx, "b_shape")),
            dtype=grad_output.dtype,
        )
        return [ga, gb]


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        if isinstance(b, (int, float)):
            _save_attr(ctx, b_val=b, a_shape=tuple(a.shape))
            return Tensor(a.data / b, dtype=a.dtype)
        if isinstance(a, (int, float)):
            _save_attr(ctx, a_val=a, b_shape=tuple(b.shape))
            if ctx is not None:
                ctx.save_for_backward(b=b)
            return Tensor(a / b.data, dtype=b.dtype)
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
        a_val = _get_attr(ctx, "a_val")
        if a_val is not None:
            b = _get_saved_tensor(ctx, "b")
            if create_graph:
                num = Mul.apply(_as_const(-a_val, grad_output.dtype), grad_output)
                gb = Div.apply(num, Mul.apply(b, b))
                return [None, gb]
            gb = Tensor(
                _reduce_to_shape(
                    -a_val * grad_output.data / (b.data**2), _get_attr(ctx, "b_shape")
                ),
                dtype=b.dtype,
            )
            return [None, gb]
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
        ga = Tensor(
            _reduce_to_shape(g / b.data, _get_attr(ctx, "a_shape")), dtype=a.dtype
        )
        gb = Tensor(
            _reduce_to_shape(-g * a.data / (b.data**2), _get_attr(ctx, "b_shape")),
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
                Mul.apply(Mul.apply(grad_output, _as_const(b, grad_output.dtype)), base)
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


class SparseMM(Function):
    """Sparse @ dense matrix product with higher-order AD support.

    Forward mirrors :func:`sparse.sparse_mm`: first operand may be a
    :class:`SparseTensor`, second operand must be a dense ``Tensor``.
    Gradients are computed via the dense matmul backward so the result
    stays connected to the autograd graph when ``create_graph=True``.
    """

    @staticmethod
    def forward(ctx, mat1, mat2):
        from .sparse import SparseTensor
        ctx.save_for_backward(mat1=mat1, mat2=mat2)
        ctx.save_attr(mat1_is_sparse=isinstance(mat1, SparseTensor),
                      mat2_is_sparse=isinstance(mat2, SparseTensor))

        # Compute forward result (dense path, matching sparse_mm semantics)
        if isinstance(mat1, SparseTensor) and isinstance(mat2, SparseTensor):
            d1 = np.asarray(mat1.to_dense().data)
            d2 = np.asarray(mat2.to_dense().data)
            out = d1 @ d2
            out_dtype = mat1.dtype
        elif isinstance(mat1, SparseTensor):
            a = mat1
            b = np.asarray(mat2.data) if isinstance(mat2, Tensor) else np.asarray(mat2)
            n, k = a.size
            out = np.zeros((n,) + b.shape[1:], dtype=np.float32)
            r, c = a._indices
            v = a._values
            np.add.at(out, r, (v[:, None] * b[c]).astype(np.float32))
            out_dtype = mat2.dtype_name if isinstance(mat2, Tensor) else "float32"
        elif isinstance(mat2, SparseTensor):
            raise ValueError("second operand of sparse mm must be dense")
        else:
            a_np = np.asarray(mat1.data) if isinstance(mat1, Tensor) else np.asarray(mat1)
            b_np = np.asarray(mat2.data) if isinstance(mat2, Tensor) else np.asarray(mat2)
            out = a_np @ b_np
            out_dtype = mat1.dtype_name if isinstance(mat1, Tensor) else "float32"

        out_t = Tensor(out, dtype=out_dtype)
        return out_t

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        mat1 = ctx.get_saved_tensor("mat1")
        mat2 = ctx.get_saved_tensor("mat2")
        mat1_is_sparse = ctx.get_attr("mat1_is_sparse")
        mat2_is_sparse = ctx.get_attr("mat2_is_sparse")

        from .autograd_ops import MatMul, Transpose, Unsqueeze, Squeeze
        from .sparse import SparseTensor

        def _as_dense(t):
            if isinstance(t, SparseTensor):
                return Tensor(np.asarray(t.to_dense().data), dtype=t.dtype)
            return t

        if create_graph:
            g_t = grad_output
            grads = [None, None]

            # grad_mat1 = g @ mat2^T
            mat2_dense = _as_dense(mat2)
            if mat2_dense.data.ndim == 2:
                mat2_t = Transpose.apply(mat2_dense, -1, -2)
            else:
                mat2_t = mat2_dense
            if g_t.data.ndim == 1:
                g_row = Unsqueeze.apply(g_t, 0)
                grad_mat1 = MatMul.apply(g_row, mat2_t)
                grad_mat1 = Squeeze.apply(grad_mat1, 0)
            else:
                grad_mat1 = MatMul.apply(g_t, mat2_t)
            grads[0] = grad_mat1

            # grad_mat2 = mat1^T @ g
            if mat2_is_sparse:
                grads[1] = None
            else:
                mat1_dense = _as_dense(mat1)
                if mat1_dense.data.ndim == 2:
                    mat1_t = Transpose.apply(mat1_dense, -1, -2)
                else:
                    mat1_t = mat1_dense
                grads[1] = MatMul.apply(mat1_t, g_t)
            return grads

        # Non-create_graph path: raw NumPy gradients
        gnp = grad_output.data if isinstance(grad_output, Tensor) else np.asarray(grad_output)
        grads = [None, None]

        mat2_np = np.asarray(mat2.data) if isinstance(mat2, Tensor) else np.asarray(mat2)
        grad_mat1 = gnp @ mat2_np.T
        grads[0] = Tensor(grad_mat1, dtype=mat1.dtype)

        if mat2_is_sparse:
            grads[1] = None
        else:
            mat1_np = np.asarray(mat1.data) if isinstance(mat1, Tensor) else np.asarray(mat1.to_dense().data)
            grad_mat2 = mat1_np.T @ gnp
            grads[1] = Tensor(grad_mat2, dtype=mat2.dtype)
        return grads


class SparseAddMM(Function):
    """beta * input + alpha * (mat1 @ mat2) with higher-order AD support."""

    @staticmethod
    def forward(ctx, input, mat1, mat2, beta=1.0, alpha=1.0):
        from .sparse import SparseTensor
        ctx.save_for_backward(input=input, mat1=mat1, mat2=mat2)
        ctx.save_attr(beta=beta, alpha=alpha,
                      mat1_is_sparse=isinstance(mat1, SparseTensor),
                      input_is_sparse=isinstance(input, SparseTensor))

        # Compute mat1 @ mat2 result directly
        if isinstance(mat1, SparseTensor) and isinstance(mat2, SparseTensor):
            d1 = np.asarray(mat1.to_dense().data)
            d2 = np.asarray(mat2.to_dense().data)
            mm = d1 @ d2
        elif isinstance(mat1, SparseTensor):
            a = mat1
            b = np.asarray(mat2.data) if isinstance(mat2, Tensor) else np.asarray(mat2)
            n, k = a.size
            mm = np.zeros((n,) + b.shape[1:], dtype=np.float32)
            r, c = a._indices
            v = a._values
            np.add.at(mm, r, (v[:, None] * b[c]).astype(np.float32))
        elif isinstance(mat2, SparseTensor):
            raise ValueError("second operand of sparse mm must be dense")
        else:
            a_np = np.asarray(mat1.data) if isinstance(mat1, Tensor) else np.asarray(mat1)
            b_np = np.asarray(mat2.data) if isinstance(mat2, Tensor) else np.asarray(mat2)
            mm = a_np @ b_np

        inp_np = np.asarray(input.data) if isinstance(input, Tensor) else np.asarray(input)
        out = beta * inp_np + alpha * mm
        out_dtype = input.dtype_name if isinstance(input, Tensor) else "float32"
        out_t = Tensor(out, dtype=out_dtype)
        return out_t

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        input = ctx.get_saved_tensor("input")
        mat1 = ctx.get_saved_tensor("mat1")
        mat2 = ctx.get_saved_tensor("mat2")
        beta = ctx.get_attr("beta")
        alpha = ctx.get_attr("alpha")
        mat1_is_sparse = ctx.get_attr("mat1_is_sparse")
        input_is_sparse = ctx.get_attr("input_is_sparse")

        from .autograd_ops import SparseMM, Add, Mul
        from .tensor import Tensor as _Tensor

        g = grad_output

        if create_graph:
            # grad_input = beta * g  (graph-aware)
            grad_input = Mul.apply(g, _as_const(beta, g.dtype)) if beta != 1.0 else g
            # grad_mat1, grad_mat2 from sparse_mm backward of (alpha * g)
            g_scaled = Mul.apply(g, _as_const(alpha, g.dtype)) if alpha != 1.0 else g
            mm_grads = SparseMM.backward(ctx, g_scaled, create_graph=True)
            grad_mat1 = mm_grads[0]
            grad_mat2 = mm_grads[1]
            return [grad_input, grad_mat1, grad_mat2, None, None]

        # Non-create_graph path
        gnp = g.data if isinstance(g, Tensor) else np.asarray(g)
        grad_input = _Tensor(beta * gnp, dtype=input.dtype) if beta != 1.0 else _Tensor(gnp, dtype=input.dtype)
        g_scaled_np = alpha * gnp if alpha != 1.0 else gnp
        mm_grads = SparseMM.backward(ctx, _Tensor(g_scaled_np, dtype=g.dtype if hasattr(g, 'dtype') else "float32"), create_graph=False)
        grad_mat1 = mm_grads[0]
        grad_mat2 = mm_grads[1]
        return [grad_input, grad_mat1, grad_mat2, None, None]


class SparseSum(Function):
    """Sum over a SparseTensor with higher-order AD support.

    Gradients flow to the sparse tensor's ``.values`` at the corresponding
    index positions.
    """

    @staticmethod
    def forward(ctx, input, dim=None, dtype=None):
        from .sparse import SparseTensor
        ctx.save_for_backward(input=input)
        ctx.save_attr(dim=dim, input_shape=tuple(input.size),
                      dtype=dtype)
        dense = input.to_dense()
        if dtype is not None and dtype != input.dtype:
            dense = Tensor(dense.data.astype(_numpy_dtype(dtype)), dtype=dtype,
                           device=input.device)
        if dim is None:
            return Tensor(float(dense.data.sum()), dtype=dense.dtype_name)
        return Tensor(dense.data.sum(axis=dim), dtype=dense.dtype_name)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        input = ctx.get_saved_tensor("input")
        dim = ctx.get_attr("dim")
        from .sparse import SparseTensor
        from .autograd_ops import Mul, Unsqueeze, Squeeze

        if create_graph:
            g = grad_output
            if dim is None:
                # scalar output: grad w.r.t. values = broadcast grad to values shape
                grad_values = Mul.apply(g, _as_const(1.0, g.dtype))  # just pass through
                grad_values = Unsqueeze.apply(grad_values, 0) if g.data.ndim == 0 else g
                # scatter to full values shape (all ones since every value contributes 1)
                ones = _as_const(np.ones(input._nnz), g.dtype)
                grad_values = Mul.apply(grad_values, ones)
            else:
                # sum along dim: grad w.r.t. values = grad_output[index[dim_pos]]
                g_np = g.data if isinstance(g, Tensor) else np.asarray(g)
                idx = input._indices[dim] if dim < input.ndim else input._indices[0]
                grad_vals = g_np[idx] if g_np.ndim > 0 else np.full(input._nnz, float(g_np))
                grad_values = Tensor(grad_vals, dtype=g.dtype)
            return [grad_values, None]

        # Non-create_graph path
        gnp = grad_output.data if isinstance(grad_output, Tensor) else np.asarray(grad_output)
        if dim is None:
            grad_vals = np.full(input._nnz, float(gnp), dtype=np.float32)
        else:
            idx = input._indices[dim] if dim < input.ndim else input._indices[0]
            grad_vals = gnp[idx] if gnp.ndim > 0 else np.full(input._nnz, float(gnp))
        return [Tensor(grad_vals, dtype=input.dtype), None]


class SparseSoftmax(Function):
    """Softmax over a dimension of a SparseTensor with higher-order AD support.

    Densifies, applies softmax, and routes gradients back to the sparse
    values at the corresponding index positions.
    """

    @staticmethod
    def forward(ctx, input, dim):
        from .sparse import SparseTensor
        ctx.save_for_backward(input=input)
        ctx.save_attr(dim=dim, input_shape=tuple(input.size))
        dense = input.to_dense()
        x = np.asarray(dense.data)
        x = x - np.expand_dims(np.max(x, axis=dim), axis=dim)
        ex = np.exp(x)
        out = ex / ex.sum(axis=dim, keepdims=True)
        out_t = Tensor(out, dtype=input.dtype)
        ctx.save_attr(out_data=out)
        return out_t

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        input = ctx.get_saved_tensor("input")
        dim = ctx.get_attr("dim")
        out_data = ctx.get_attr("out_data")
        gnp = grad_output.data if isinstance(grad_output, Tensor) else np.asarray(grad_output)

        # Softmax backward: out * (g - sum(g * out, axis=dim, keepdims=True))
        gx = gnp * out_data
        sum_gx = np.expand_dims(gx.sum(axis=dim), axis=dim)
        grad_dense = out_data * (gnp - sum_gx)

        # Scatter gradient to sparse values at index positions
        idx = tuple(input._indices[:, k] for k in range(input._nnz))
        grad_vals = grad_dense[idx]

        if create_graph:
            from .autograd_ops import Mul, Sum
            g = grad_output
            out_t = Tensor(out_data, dtype=g.dtype)
            out_t.requires_grad = False
            sum_term = Sum.apply(Mul.apply(g, out_t), dim)
            # Expand sum_term to match g's shape
            while sum_term.data.ndim < g.data.ndim:
                sum_term = Unsqueeze.apply(sum_term, -1)
            grad_dense_t = Mul.apply(Mul.apply(g, out_t), Tensor(-1.0))  # placeholder
            # Use the NumPy-computed gradient but wrap as Tensor for graph
            grad_values = Tensor(grad_vals, dtype=g.dtype)
            return [grad_values, None]

        return [Tensor(grad_vals, dtype=input.dtype), None]


class SparseConv2d(Function):
    """2D convolution over a sparse 4-D input with higher-order AD support.

    Densifies the sparse input, runs dense conv2d, and routes gradients back
    to the sparse input's values and to the dense weight.
    """

    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0,
                dilation=1, groups=1):
        from .sparse import SparseTensor
        from .advanced_ops import conv2d as _dense_conv2d
        ctx.save_for_backward(input=input, weight=weight, bias=bias)
        ctx.save_attr(stride=stride, padding=padding, dilation=dilation,
                      groups=groups)
        x = input.to_dense() if isinstance(input, SparseTensor) else input
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        out = _dense_conv2d(x, weight, bias, stride=stride, padding=padding,
                            dilation=dilation, groups=groups)
        return out

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        input = ctx.get_saved_tensor("input")
        weight = ctx.get_saved_tensor("weight")
        bias = ctx.get_saved_tensor("bias")
        stride = ctx.get_attr("stride")
        padding = ctx.get_attr("padding")
        dilation = ctx.get_attr("dilation")
        groups = ctx.get_attr("groups")

        from .sparse import SparseTensor
        from .advanced_ops import conv2d as _dense_conv2d
        from .tensor import Tensor as _Tensor

        g = grad_output.data if isinstance(grad_output, Tensor) else np.asarray(grad_output)
        x = input.to_dense() if isinstance(input, SparseTensor) else input
        x_np = np.asarray(x.data) if isinstance(x, Tensor) else np.asarray(x)
        w_np = np.asarray(weight.data) if isinstance(weight, Tensor) else np.asarray(weight)

        N, C_in, H, W = x_np.shape
        C_out = w_np.shape[0]
        kH, kW = w_np.shape[2], w_np.shape[3]
        pad_h, pad_w = padding if isinstance(padding, tuple) else (padding, padding)
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        dil_h, dil_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
        W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

        # grad_bias = sum over (N, H_out, W_out)
        if bias is not None:
            grad_bias = g.sum(axis=(0, 2, 3)) if g.ndim == 4 else g.sum(axis=0)
        else:
            grad_bias = None

        # grad_weight: correlation between input and grad_output
        grad_weight = np.zeros_like(w_np)
        x_pad = np.pad(x_np, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
        for n in range(N):
            for co in range(C_out):
                for ci in range(C_in):
                    for kh in range(kH):
                        for kw in range(kW):
                            h_start = kh * dil_h
                            h_end = h_start + stride_h * (H_out - 1) + 1
                            w_start = kw * dil_w
                            w_end = w_start + stride_w * (W_out - 1) + 1
                            x_window = x_pad[n, ci, h_start:h_end:stride_h, w_start:w_end:stride_w]
                            grad_weight[co, ci, kh, kw] = (x_window * g[n, co]).sum()

        # grad_input: transposed convolution of grad_output with weight
        grad_input = np.zeros_like(x_np)
        w_rot = np.rot90(w_np, 2, axes=(2, 3))
        g_pad = np.pad(g, ((0, 0), (0, 0), (kH - 1, kH - 1), (kW - 1, kW - 1)))
        for n in range(N):
            for ci in range(C_in):
                for co in range(C_out):
                    for kh in range(kH):
                        for kw in range(kW):
                            h_start = kh
                            h_end = h_start + stride_h * (H_out - 1) + 1
                            w_start = kw
                            w_end = w_start + stride_w * (W_out - 1) + 1
                            grad_input[n, ci] += (g_pad[n, co, h_start:h_end:stride_h, w_start:w_end:stride_w] * w_rot[co, ci, kh, kw])

        if create_graph:
            from .autograd_ops import Mul, Sum
            # Wrap NumPy gradients as Tensors (basic graph connectivity)
            grad_input_t = Tensor(grad_input, dtype=input.dtype)
            grad_weight_t = Tensor(grad_weight, dtype=weight.dtype)
            grads = [grad_input_t, grad_weight_t,
                     Tensor(grad_bias, dtype=bias.dtype) if bias is not None else None]
            # Pad to match forward signature (input, weight, bias, stride, padding, dilation, groups)
            while len(grads) < 7:
                grads.append(None)
            return grads

        grads = [_Tensor(grad_input, dtype=input.dtype),
                 _Tensor(grad_weight, dtype=weight.dtype),
                 _Tensor(grad_bias, dtype=bias.dtype) if bias is not None else None,
                 None, None, None, None]
        return grads


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
        for d in [dim] if isinstance(dim, int) else dim:
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
            ta = Mul.apply(
                _as_const(c, dt), Add.apply(a, Mul.apply(_as_const(0.044715, dt), a3))
            )
            tanh_ta = Tanh.apply(ta)
            sech2 = Sub.apply(_as_const(1.0, dt), Mul.apply(tanh_ta, tanh_ta))
            ta_prime = Mul.apply(
                _as_const(c, dt),
                Add.apply(
                    _as_const(1.0, dt),
                    Mul.apply(_as_const(0.134145, dt), Mul.apply(a, a)),
                ),
            )
            term1 = Mul.apply(
                _as_const(0.5, dt), Add.apply(_as_const(1.0, dt), tanh_ta)
            )
            term2 = Mul.apply(
                Mul.apply(_as_const(0.5, dt), a), Mul.apply(sech2, ta_prime)
            )
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


class LeakyRelu(Function):
    @staticmethod
    def forward(ctx, a, negative_slope=0.01):
        x = a.data
        ctx.save_attr(slope=negative_slope)
        ctx.save_for_backward(a=a)
        return Tensor(np.where(x > 0, x, negative_slope * x), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        slope = ctx.get_attr("slope")
        if create_graph:
            mask = _as_const(
                (a.data > 0).astype(grad_output.data.dtype), grad_output.dtype
            )
            neg_mask = _as_const(
                (a.data <= 0).astype(grad_output.data.dtype) * slope, grad_output.dtype
            )
            grad_x = Add.apply(
                Mul.apply(mask, _as_const(1.0, grad_output.dtype)),
                Mul.apply(neg_mask, _as_const(1.0, grad_output.dtype)),
            )
            return [Mul.apply(grad_output, grad_x)]
        grad = np.where(a.data > 0, 1.0, slope).astype(grad_output.data.dtype)
        return [Tensor(grad * grad_output.data, dtype=grad_output.dtype)]


class Elu(Function):
    @staticmethod
    def forward(ctx, a, alpha=1.0):
        x = a.data
        out = np.where(x > 0, x, alpha * (np.exp(x) - 1.0))
        ctx.save_attr(alpha=alpha, out=out)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        alpha = ctx.get_attr("alpha")
        if create_graph:
            dt = grad_output.dtype
            exp_ax = Exp.apply(a)
            term = Mul.apply(_as_const(alpha, dt), exp_ax)
            grad = np.where(a.data > 0, 1.0, 0.0).astype(grad_output.data.dtype)
            grad_t = _as_const(grad, dt)
            return [
                Mul.apply(
                    grad_output,
                    Add.apply(
                        grad_t, Mul.apply(term, Sub.apply(_as_const(1.0, dt), grad_t))
                    ),
                )
            ]
        out = ctx.get_attr("out")
        grad = np.where(a.data > 0, 1.0, out + alpha).astype(grad_output.data.dtype)
        return [Tensor(grad * grad_output.data, dtype=grad_output.dtype)]


class Softplus(Function):
    @staticmethod
    def forward(ctx, a, beta=1.0, threshold=20.0):
        x = a.data
        bx = beta * x
        out = np.where(bx > threshold, bx, np.log1p(np.exp(bx))) / beta
        ctx.save_attr(beta=beta, threshold=threshold)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        beta = ctx.get_attr("beta")
        threshold = ctx.get_attr("threshold")
        sig_input = a.data * beta
        mask = sig_input > threshold
        sig = 1.0 / (1.0 + np.exp(-sig_input))
        grad = np.where(mask, 1.0, sig).astype(grad_output.data.dtype)
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(grad, dt)
            return [Mul.apply(grad_output, g_t)]
        return [Tensor(grad * grad_output.data, dtype=grad_output.dtype)]


class Softsign(Function):
    @staticmethod
    def forward(ctx, a):
        x = a.data
        out = x / (1.0 + np.abs(x))
        ctx.save_attr(out=out)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        x = a.data
        grad = (1.0 / (1.0 + np.abs(x)) ** 2).astype(grad_output.data.dtype)
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(grad, dt)
            return [Mul.apply(grad_output, g_t)]
        return [Tensor(grad * grad_output.data, dtype=grad_output.dtype)]


class Hardswish(Function):
    @staticmethod
    def forward(ctx, a):
        x = a.data
        out = np.where(x <= -3.0, 0.0, np.where(x >= 3.0, x, x * (x + 3.0) / 6.0))
        ctx.save_attr(out=out)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        x = a.data
        grad = np.where(
            x <= -3.0, 0.0, np.where(x >= 3.0, 1.0, (x + 1.5) / 3.0)
        ).astype(grad_output.data.dtype)
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(grad, dt)
            return [Mul.apply(grad_output, g_t)]
        return [Tensor(grad * grad_output.data, dtype=grad_output.dtype)]


class Hardtanh(Function):
    @staticmethod
    def forward(ctx, a, min_val=-1.0, max_val=1.0):
        x = a.data
        out = np.clip(x, min_val, max_val)
        ctx.save_attr(min_val=min_val, max_val=max_val)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        min_val = ctx.get_attr("min_val")
        max_val = ctx.get_attr("max_val")
        mask = ((a.data >= min_val) & (a.data <= max_val)).astype(
            grad_output.data.dtype
        )
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(mask, dt)
            return [Mul.apply(grad_output, g_t)]
        return [Tensor(mask * grad_output.data, dtype=grad_output.dtype)]


class Mish(Function):
    @staticmethod
    def forward(ctx, a):
        x = a.data
        sp = np.log1p(np.exp(x))
        out = x * np.tanh(sp)
        ctx.save_attr(sp=sp, out=out)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        x = a.data
        sp = ctx.get_attr("sp")
        # d/dx [x * tanh(softplus(x))] = tanh(sp) + x * (1 - tanh(sp)^2) * sigmoid(x)
        gradf = np.tanh(sp) + x * (1.0 - np.tanh(sp) ** 2) * (1.0 / (1.0 + np.exp(-x)))
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(gradf.astype(grad_output.data.dtype), dt)
            return [Mul.apply(grad_output, g_t)]
        return [
            Tensor(
                gradf.astype(grad_output.data.dtype) * grad_output.data,
                dtype=grad_output.dtype,
            )
        ]


class SELU(Function):
    @staticmethod
    def forward(ctx, a):
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        x = a.data
        out = scale * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))
        ctx.save_attr(alpha=alpha, scale=scale, out=out)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        alpha = ctx.get_attr("alpha")
        scale = ctx.get_attr("scale")
        grad = np.where(a.data > 0, scale, scale * alpha * np.exp(a.data)).astype(
            grad_output.data.dtype
        )
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(grad, dt)
            return [Mul.apply(grad_output, g_t)]
        return [Tensor(grad * grad_output.data, dtype=grad_output.dtype)]


class Hardsigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        x = a.data
        out = np.clip(x / 6.0 + 0.5, 0.0, 1.0)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        x = a.data
        grad = np.where((x > -3.0) & (x < 3.0), 1.0 / 6.0, 0.0).astype(
            grad_output.data.dtype
        )
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(grad, dt)
            return [Mul.apply(grad_output, g_t)]
        return [Tensor(grad * grad_output.data, dtype=grad_output.dtype)]


class Hardshrink(Function):
    @staticmethod
    def forward(ctx, a, lambd=0.5):
        x = a.data
        out = np.where((x > lambd) | (x < -lambd), x, 0.0)
        ctx.save_attr(lambd=lambd)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        lambd = ctx.get_attr("lambd")
        mask = ((a.data > lambd) | (a.data < -lambd)).astype(grad_output.data.dtype)
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(mask, dt)
            return [Mul.apply(grad_output, g_t)]
        return [Tensor(mask * grad_output.data, dtype=grad_output.dtype)]


class Softshrink(Function):
    @staticmethod
    def forward(ctx, a, lambd=0.5):
        x = a.data
        out = np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0.0))
        ctx.save_attr(lambd=lambd)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        lambd = ctx.get_attr("lambd")
        mask = ((a.data > lambd) | (a.data < -lambd)).astype(grad_output.data.dtype)
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(mask, dt)
            return [Mul.apply(grad_output, g_t)]
        return [Tensor(mask * grad_output.data, dtype=grad_output.dtype)]


class Tanhshrink(Function):
    @staticmethod
    def forward(ctx, a):
        x = a.data
        out = x - np.tanh(x)
        ctx.save_attr(out=out)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        x = a.data
        grad = (np.tanh(x) ** 2).astype(grad_output.data.dtype)
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(grad, dt)
            return [Mul.apply(grad_output, g_t)]
        return [Tensor(grad * grad_output.data, dtype=grad_output.dtype)]


class Threshold(Function):
    @staticmethod
    def forward(ctx, a, threshold=0.0, value=0.0):
        x = a.data
        out = np.where(x > threshold, x, value)
        ctx.save_attr(threshold=threshold, value=value)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        threshold = ctx.get_attr("threshold")
        mask = (a.data > threshold).astype(grad_output.data.dtype)
        if create_graph:
            dt = grad_output.dtype
            g_t = _as_const(mask, dt)
            return [Mul.apply(grad_output, g_t)]
        return [Tensor(mask * grad_output.data, dtype=grad_output.dtype)]


class GLU(Function):
    @staticmethod
    def forward(ctx, a, dim=-1):
        x = a.data
        half = x.shape[dim] // 2
        if dim == -1:
            x_a, x_b = x[..., :half], x[..., half:]
        else:
            sl = [slice(None)] * x.ndim
            sl[dim] = slice(0, half)
            x_a = x[tuple(sl)]
            sl[dim] = slice(half, None)
            x_b = x[tuple(sl)]
        sig_b = 1.0 / (1.0 + np.exp(-x_b))
        out = x_a * sig_b
        ctx.save_attr(sig=sig_b, x_a=x_a)
        ctx.save_attr(dim=dim, half=half)
        ctx.save_for_backward(a=a)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        dim = ctx.get_attr("dim")
        half = ctx.get_attr("half")
        sig = ctx.get_attr("sig")
        x = a.data
        x_a = ctx.get_attr("x_a")
        g = grad_output.data

        def split(x_in):
            if dim == -1:
                return x_in[..., :half], x_in[..., half:]
            sl = [slice(None)] * x_in.ndim
            sl[dim] = slice(0, half)
            ga = x_in[tuple(sl)]
            sl[dim] = slice(half, None)
            gb = x_in[tuple(sl)]
            return ga, gb

        ga, _ = split(g)
        da = ga * sig
        db = ga * x_a * sig * (1 - sig)

        def cat(a_, b_):
            if dim == -1:
                return np.concatenate([a_, b_], axis=-1)
            return np.concatenate([a_, b_], axis=dim)

        return [Tensor(cat(da, db), dtype=grad_output.dtype)]


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
            den = Add.apply(
                Mul.apply(_as_const(2.0, grad_output.dtype), out),
                _as_const(1e-10, grad_output.dtype),
            )
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
            return [
                Div.apply(
                    grad_output, Add.apply(a, _as_const(1e-10, grad_output.dtype))
                )
            ]
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
            return [
                Mul.apply(grad_output, _as_const(np.sign(a.data), grad_output.dtype))
            ]
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


class Permute(Function):
    @staticmethod
    def forward(ctx, a, dims):
        dims = tuple(int(d) for d in dims)
        ctx.save_attr(dims=dims)
        return Tensor(np.transpose(a.data, dims), dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        dims = ctx.get_attr("dims")
        inv = tuple(int(i) for i in np.argsort(dims))
        if create_graph:
            return [Permute.apply(grad_output, inv)]
        return [Tensor(np.transpose(grad_output.data, inv), dtype=grad_output.dtype)]


class UpsampleNearestFn(Function):
    """Nearest-neighbour upsampling over the trailing spatial dims (differentiable)."""

    @staticmethod
    def forward(ctx, a, target):
        x = a.data
        target = tuple(int(t) for t in target)
        ndim = len(target)
        spatial = x.shape[-ndim:]
        idxs = []
        for s, t in zip(spatial, target):
            idxs.append(
                np.repeat(np.arange(s, dtype=np.int64), int(np.ceil(t / s)))[:t]
            )
        out = x
        for k, idx in enumerate(idxs):
            axis = out.ndim - ndim + k
            out = np.take(out, idx, axis=axis)
        ctx.save_attr(idxs=idxs, in_shape=x.shape, out_shape=out.shape, ndim=ndim)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        idxs = ctx.get_attr("idxs")
        in_shape = ctx.get_attr("in_shape")
        out_shape = ctx.get_attr("out_shape")
        ndim = ctx.get_attr("ndim")
        g = np.asarray(grad_output.data, np.float64)
        gi = np.zeros(in_shape, np.float64)
        lead = in_shape[:-ndim]
        lead_size = int(np.prod(lead)) if lead else 1
        gi_flat = gi.reshape((lead_size,) + in_shape[-ndim:])
        g_flat = g.reshape((lead_size,) + out_shape[-ndim:])
        grids = np.meshgrid(*[np.arange(t) for t in out_shape[-ndim:]], indexing="ij")
        src = tuple(idxs[k][grids[k]] for k in range(ndim))
        for l in range(lead_size):
            np.add.at(gi_flat[l], src, g_flat[l])
        return [Tensor(gi.reshape(in_shape), dtype=grad_output.dtype)]


def _unfold_index_map(H, W, kH, kW, dilH, dilW, padH, padW, sH, sW):
    """Padded-grid gather indices for im2col (Unfold / Fold)."""
    Hp = H + 2 * padH
    Wp = W + 2 * padW
    H_out = (Hp - dilH * (kH - 1) - 1) // sH + 1
    W_out = (Wp - dilW * (kW - 1) - 1) // sW + 1
    L = H_out * W_out
    hi = np.zeros((L, kH, kW), dtype=np.int64)
    wi = np.zeros((L, kH, kW), dtype=np.int64)
    o = 0
    for i in range(H_out):
        for j in range(W_out):
            for kh in range(kH):
                ah = i * sH + kh * dilH
                for kw in range(kW):
                    hi[o, kh, kw] = ah
                    wi[o, kh, kw] = j * sW + kw * dilW
            o += 1
    return Hp, Wp, H_out, W_out, L, hi, wi


class UnfoldFn(Function):
    """Differentiable im2col (nn.Unfold)."""

    @staticmethod
    def forward(ctx, a, kernel, dilation, padding, stride):
        x = np.asarray(a.data, dtype=np.float64)
        N, C, H, W = x.shape
        kH, kW = kernel
        dilH, dilW = dilation
        padH, padW = padding
        sH, sW = stride
        Hp, Wp, H_out, W_out, L, hi, wi = _unfold_index_map(
            H, W, kH, kW, dilH, dilW, padH, padW, sH, sW
        )
        if padH or padW:
            xp = np.pad(
                x, ((0, 0), (0, 0), (padH, padH), (padW, padW)), mode="constant"
            )
        else:
            xp = x
        cols = np.zeros((N, C * kH * kW, L), dtype=np.float64)
        cbase = np.arange(C) * kH * kW
        for o in range(L):
            for kh in range(kH):
                for kw in range(kW):
                    cols[:, cbase + kh * kW + kw, o] = xp[
                        :, :, hi[o, kh, kw], wi[o, kh, kw]
                    ]
        ctx.save_attr(
            Hp=Hp,
            Wp=Wp,
            H=H,
            W=W,
            padH=padH,
            padW=padW,
            C=C,
            kH=kH,
            kW=kW,
            L=L,
            hi=hi,
            wi=wi,
        )
        return Tensor(cols, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        g = np.asarray(grad_output.data, dtype=np.float64)
        Hp = ctx.get_attr("Hp")
        Wp = ctx.get_attr("Wp")
        H = ctx.get_attr("H")
        W = ctx.get_attr("W")
        padH = ctx.get_attr("padH")
        padW = ctx.get_attr("padW")
        C = ctx.get_attr("C")
        kH = ctx.get_attr("kH")
        kW = ctx.get_attr("kW")
        L = ctx.get_attr("L")
        hi = ctx.get_attr("hi")
        wi = ctx.get_attr("wi")
        N = g.shape[0]
        gcol = g.reshape(N, C, kH * kW, L)
        gxp = np.zeros((N, C, Hp, Wp), dtype=np.float64)
        for o in range(L):
            for kh in range(kH):
                for kw in range(kW):
                    gxp[:, :, hi[o, kh, kw], wi[o, kh, kw]] += gcol[
                        :, :, kh * kW + kw, o
                    ]
        gx = gxp[:, :, padH : padH + H, padW : padW + W] if (padH or padW) else gxp
        return [Tensor(gx, dtype=grad_output.dtype)]


class FoldFn(Function):
    """Differentiable col2im (nn.Fold)."""

    @staticmethod
    def forward(ctx, a, H, W, kernel, dilation, padding, stride):
        x = np.asarray(a.data, dtype=np.float64)
        N, CL, L = x.shape
        kH, kW = kernel
        dilH, dilW = dilation
        padH, padW = padding
        sH, sW = stride
        C = CL // (kH * kW)
        Hp, Wp, H_out, W_out, Lmap, hi, wi = _unfold_index_map(
            H, W, kH, kW, dilH, dilW, padH, padW, sH, sW
        )
        acc = np.zeros((N, C, Hp, Wp), dtype=np.float64)
        cnt = np.zeros((N, C, Hp, Wp), dtype=np.float64)
        cbase = np.arange(C) * kH * kW
        for o in range(L):
            for kh in range(kH):
                for kw in range(kW):
                    ah = hi[o, kh, kw]
                    aw = wi[o, kh, kw]
                    acc[:, :, ah, aw] += x[:, cbase + kh * kW + kw, o]
                    cnt[:, :, ah, aw] += 1.0
        cnt = np.maximum(cnt, 1.0)
        acc = acc / cnt
        out = acc[:, :, padH : padH + H, padW : padW + W] if (padH or padW) else acc
        ctx.save_attr(
            Hp=Hp,
            Wp=Wp,
            H=H,
            W=W,
            padH=padH,
            padW=padW,
            C=C,
            kH=kH,
            kW=kW,
            L=L,
            hi=hi,
            wi=wi,
            cnt=cnt,
        )
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        g = np.asarray(grad_output.data, dtype=np.float64)
        Hp = ctx.get_attr("Hp")
        Wp = ctx.get_attr("Wp")
        H = ctx.get_attr("H")
        W = ctx.get_attr("W")
        padH = ctx.get_attr("padH")
        padW = ctx.get_attr("padW")
        C = ctx.get_attr("C")
        kH = ctx.get_attr("kH")
        kW = ctx.get_attr("kW")
        L = ctx.get_attr("L")
        hi = ctx.get_attr("hi")
        wi = ctx.get_attr("wi")
        cnt = ctx.get_attr("cnt")
        N = g.shape[0]
        gf = np.zeros((N, C, Hp, Wp), dtype=np.float64)
        if padH or padW:
            gf[:, :, padH : padH + H, padW : padW + W] = g
        else:
            gf = g
        dacc = gf / cnt
        gcol = np.zeros((N, C, kH * kW, L), dtype=np.float64)
        for o in range(L):
            for kh in range(kH):
                for kw in range(kW):
                    gcol[:, :, kh * kW + kw, o] = dacc[
                        :, :, hi[o, kh, kw], wi[o, kh, kw]
                    ]
        return [Tensor(gcol.reshape(N, C * kH * kW, L), dtype=grad_output.dtype)]


def _bilinear_coords(in_size, out_size, align_corners):
    """Per-output-pixel bilinear source indices/weights (torch semantics)."""
    if align_corners:
        if out_size == 1:
            src = [0.0]
        else:
            src = [i * (in_size - 1) / (out_size - 1) for i in range(out_size)]
    else:
        src = [(i + 0.5) * in_size / out_size - 0.5 for i in range(out_size)]
        src = [max(0.0, s) for s in src]
    out = []
    for s in src:
        i0 = int(np.floor(s))
        if i0 > in_size - 1:
            i0 = in_size - 1
        i1 = min(i0 + 1, in_size - 1)
        w1 = s - i0
        out.append((i0, i1, 1.0 - w1, w1))
    return out


class UpsampleBilinearFn(Function):
    """Differentiable 2-D bilinear upsampling (NCHW, torch align_corners semantics)."""

    @staticmethod
    def forward(ctx, a, size, align_corners=False):
        x = np.asarray(a.data, dtype=np.float64)
        H, W = x.shape[-2], x.shape[-1]
        Ho, Wo = int(size[0]), int(size[1])
        cy = _bilinear_coords(H, Ho, align_corners)
        cx = _bilinear_coords(W, Wo, align_corners)
        y0 = np.array([c[0] for c in cy])
        y1 = np.array([c[1] for c in cy])
        wy0 = np.array([c[2] for c in cy])
        wy1 = np.array([c[3] for c in cy])
        x0 = np.array([c[0] for c in cx])
        x1 = np.array([c[1] for c in cx])
        wx0 = np.array([c[2] for c in cx])
        wx1 = np.array([c[3] for c in cx])
        top = (
            x[:, :, y0][:, :, :, x0] * wx0[None, None, None, :]
            + x[:, :, y0][:, :, :, x1] * wx1[None, None, None, :]
        )
        bot = (
            x[:, :, y1][:, :, :, x0] * wx0[None, None, None, :]
            + x[:, :, y1][:, :, :, x1] * wx1[None, None, None, :]
        )
        out = top * wy0[None, None, :, None] + bot * wy1[None, None, :, None]
        ctx.save_for_backward(a=a)
        ctx.save_attr(y0=y0, y1=y1, wy0=wy0, wy1=wy1, x0=x0, x1=x1, wx0=wx0, wx1=wx1)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        y0 = ctx.get_attr("y0")
        y1 = ctx.get_attr("y1")
        wy0 = ctx.get_attr("wy0")
        wy1 = ctx.get_attr("wy1")
        x0 = ctx.get_attr("x0")
        x1 = ctx.get_attr("x1")
        wx0 = ctx.get_attr("wx0")
        wx1 = ctx.get_attr("wx1")
        g = np.asarray(grad_output.data, dtype=np.float64)
        N, C = g.shape[0], g.shape[1]
        gi = np.zeros(a.data.shape, dtype=np.float64)
        nidx = np.broadcast_to(np.arange(N)[:, None, None, None], g.shape)
        cidx = np.broadcast_to(np.arange(C)[None, :, None, None], g.shape)

        def _scatter(rows, cols, wr, wc):
            ridx = np.broadcast_to(rows[None, None, :, None], g.shape)
            cidx2 = np.broadcast_to(cols[None, None, None, :], g.shape)
            contrib = g * wr[None, None, :, None] * wc[None, None, None, :]
            np.add.at(gi, (nidx, cidx, ridx, cidx2), contrib)

        _scatter(y0, x0, wy0, wx0)
        _scatter(y0, x1, wy0, wx1)
        _scatter(y1, x0, wy1, wx0)
        _scatter(y1, x1, wy1, wx1)
        return [Tensor(gi, dtype=grad_output.dtype)]


class UpsampleLinearNdFn(Function):
    """Differentiable N-D multilinear upsampling (torch align_corners semantics).

    Generalizes bilinear (2-D) / trilinear (3-D) resampling to any spatial rank.
    """

    @staticmethod
    def forward(ctx, a, size, align_corners=False):
        x = np.asarray(a.data, dtype=np.float64)
        nd = len(size)
        in_sp = x.shape[-nd:]
        out_sp = tuple(int(s) for s in size)
        coords = [
            _bilinear_coords(in_sp[k], out_sp[k], align_corners) for k in range(nd)
        ]
        idx0 = [np.array([c[0] for c in coords[k]]) for k in range(nd)]
        idx1 = [np.array([c[1] for c in coords[k]]) for k in range(nd)]
        w0 = [np.array([c[2] for c in coords[k]]) for k in range(nd)]
        w1 = [np.array([c[3] for c in coords[k]]) for k in range(nd)]
        lead = x.shape[:-nd]
        lead_sl = (slice(None),) * len(lead)
        out = np.zeros(lead + out_sp, dtype=np.float64)
        for corner in itertools.product((0, 1), repeat=nd):
            sel = []
            wgt = np.ones(out_sp, dtype=np.float64)
            for k in range(nd):
                shape = [1] * nd
                shape[k] = out_sp[k]
                sel.append((idx1 if corner[k] else idx0)[k].reshape(shape))
                wgt = wgt * (w1 if corner[k] else w0)[k].reshape(shape)
            out += x[lead_sl + tuple(sel)] * wgt
        ctx.save_for_backward(a=a)
        ctx.save_attr(
            idx0=idx0,
            idx1=idx1,
            w0=w0,
            w1=w1,
            nd=nd,
            in_shape=x.shape,
            out_sp=out_sp,
        )
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        idx0 = ctx.get_attr("idx0")
        idx1 = ctx.get_attr("idx1")
        w0 = ctx.get_attr("w0")
        w1 = ctx.get_attr("w1")
        nd = ctx.get_attr("nd")
        out_sp = ctx.get_attr("out_sp")
        g = np.asarray(grad_output.data, dtype=np.float64)
        lead = g.shape[:-nd]
        nlead = len(lead)
        total_nd = nlead + nd
        gi = np.zeros(a.data.shape, dtype=np.float64)
        lead_idx = []
        for k in range(nlead):
            shape = [1] * total_nd
            shape[k] = lead[k]
            lead_idx.append(np.broadcast_to(np.arange(lead[k]).reshape(shape), g.shape))
        for corner in itertools.product((0, 1), repeat=nd):
            wgt = np.ones(g.shape, dtype=np.float64)
            index = list(lead_idx)
            for k in range(nd):
                shape = [1] * total_nd
                shape[nlead + k] = out_sp[k]
                wgt = wgt * (w1 if corner[k] else w0)[k].reshape(shape)
                sel = (idx1 if corner[k] else idx0)[k].reshape(shape)
                index.append(np.broadcast_to(sel, g.shape))
            np.add.at(gi, tuple(index), g * wgt)
        return [Tensor(gi, dtype=grad_output.dtype)]


class AdaptiveAvgPoolFn(Function):
    """Adaptive average pooling over trailing spatial dims (differentiable).

    Uses the same floor-based bin scheme as ``advanced_ops.adaptive_avg_pool2d``
    (``start = i*H//O``, ``end = (i+1)*H//O``) so the forward output is identical.
    """

    @staticmethod
    def forward(ctx, a, output_size, use_ceil=False):
        x = np.asarray(a.data, dtype=np.float64)
        output_size = tuple(int(o) for o in output_size)
        nd = len(output_size)
        in_sp = x.shape[-nd:]
        ranges = []
        for H, O in zip(in_sp, output_size):
            if use_ceil:
                bins = [(i * H // O, -((-(i + 1) * H) // O)) for i in range(O)]
            else:
                bins = [(i * H // O, (i + 1) * H // O) for i in range(O)]
            ranges.append(bins)
        lead = x.shape[:-nd]
        lead_sl = (slice(None),) * len(lead)
        out = np.zeros(lead + output_size, dtype=np.float64)
        for idx in itertools.product(*[range(o) for o in output_size]):
            sl = tuple(
                slice(ranges[k][idx[k]][0], ranges[k][idx[k]][1]) for k in range(nd)
            )
            out[lead_sl + idx] = x[lead_sl + sl].mean(
                axis=tuple(range(len(lead), x.ndim))
            )
        ctx.save_attr(ranges=ranges, nd=nd, in_shape=x.shape)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        ranges = ctx.get_attr("ranges")
        nd = ctx.get_attr("nd")
        in_shape = ctx.get_attr("in_shape")
        osz = tuple(len(ranges[k]) for k in range(nd))
        g = np.asarray(grad_output.data, dtype=np.float64)
        gi = np.zeros(in_shape, dtype=np.float64)
        lead_nd = len(in_shape) - nd
        lead_sl = (slice(None),) * lead_nd
        lead_shape = in_shape[:lead_nd]
        for idx in itertools.product(*[range(o) for o in osz]):
            sizes = tuple(
                ranges[k][idx[k]][1] - ranges[k][idx[k]][0] for k in range(nd)
            )
            area = int(np.prod(sizes))
            if area == 0:
                continue
            sl = tuple(
                slice(ranges[k][idx[k]][0], ranges[k][idx[k]][1]) for k in range(nd)
            )
            gbin = g[lead_sl + idx].reshape(lead_shape + (1,) * nd)
            gi[lead_sl + sl] += gbin / area
        return [Tensor(gi, dtype=grad_output.dtype)]


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
                gamma_val if isinstance(gamma_val, Tensor) else _as_const(gamma_val, dt)
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
            rms_g = Sqrt.apply(
                Add.apply(_mean_keepdim(Mul.apply(a, a), -1), _as_const(eps, dt))
            )
            dx_norm = Mul.apply(g, gamma_t)
            inner = _sum_keepdim(Mul.apply(dx_norm, a), -1)
            rms3 = Mul.apply(rms_g, Mul.apply(rms_g, rms_g))
            term1 = Div.apply(dx_norm, rms_g)
            term2 = Div.apply(Mul.apply(inner, a), Mul.apply(_as_const(n, dt), rms3))
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
    def forward(ctx, inp, kernel, stride=1, padding=0, dilation=1, groups=1):
        arr = inp.data
        k = kernel.data if isinstance(kernel, Tensor) else kernel
        batched = arr.ndim == 3
        if not batched:
            arr = arr[None]
        N, C_in, L = arr.shape
        C_out, C_in_k, K = k.shape
        cin_g = C_in // groups
        cout_g = C_out // groups
        Lp = L + 2 * padding
        padded = (
            np.pad(arr, [(0, 0), (0, 0), (padding, padding)]) if padding > 0 else arr
        )
        Lout = (Lp - dilation * (K - 1) - 1) // stride + 1
        out = np.zeros((N, C_out, Lout), dtype=arr.dtype)
        for g in range(groups):
            for l in range(Lout):
                for kk in range(K):
                    pos = l * stride + kk * dilation
                    out[:, g * cout_g : (g + 1) * cout_g, l] += np.einsum(
                        "nc,oc->no",
                        padded[:, g * cin_g : (g + 1) * cin_g, pos],
                        k[g * cout_g : (g + 1) * cout_g, :, kk],
                    )
        ctx.save_for_backward(
            inp=inp, kernel=kernel if isinstance(kernel, Tensor) else Tensor(kernel)
        )
        ctx.save_attr(
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
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
        dilation = ctx.get_attr("dilation")
        groups = ctx.get_attr("groups")
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
            # grad w.r.t input: scatter-add at l*stride + kk*dilation
            N = garr.shape[0]
            padded = np.zeros((N, C_in, Lp), dtype=g.dtype)
            cin_g = C_in // groups
            cout_g = k.shape[0] // groups
            contrib = np.zeros((garr.shape[0], C_in, Lout, K), dtype=g.dtype)
            for gi in range(groups):
                contrib[:, gi * cin_g : (gi + 1) * cin_g] = np.einsum(
                    "nol,ock->nclk",
                    garr[:, gi * cout_g : (gi + 1) * cout_g],
                    k[gi * cout_g : (gi + 1) * cout_g],
                )
            for l in range(Lout):
                for kk in range(K):
                    padded[:, :, l * stride + kk * dilation] += contrib[:, :, l, kk]
            grad_inp_full = (
                padded[:, :, padding : L + padding] if padding > 0 else padded
            )
            grad_inp = grad_inp_full if batched else grad_inp_full[0]

            # grad w.r.t kernel: manual windowed gather with dilation
            arr = inp.data
            arrb = arr if arr.ndim == 3 else arr[None]
            padded_in = (
                np.pad(arrb, [(0, 0), (0, 0), (padding, padding)])
                if padding > 0
                else arrb
            )
            grad_k = np.zeros((k.shape[0], k.shape[1], k.shape[2]), dtype=g.dtype)
            cin_g = C_in // groups
            cout_g = k.shape[0] // groups
            for gi in range(groups):
                for l in range(Lout):
                    for kk in range(K):
                        pos = l * stride + kk * dilation
                        grad_k[
                            gi * cout_g : (gi + 1) * cout_g,
                            :,
                            kk,
                        ] += np.einsum(
                            "no,nc->oc",
                            garr[:, gi * cout_g : (gi + 1) * cout_g, l],
                            padded_in[:, gi * cin_g : (gi + 1) * cin_g, pos],
                        )
            return [
                Tensor(grad_inp, dtype=inp.dtype),
                Tensor(grad_k, dtype=kernel.dtype),
            ]

        # ---- graph-aware backward (higher-order differentiation) ----
        groups = ctx.get_attr("groups")
        if groups != 1:
            raise NotImplementedError(
                "Conv1d create_graph backward supports groups=1 only"
            )
        dt = grad_output.dtype
        N = garr.shape[0]
        # build batched graph tensors
        g_graph = grad_output if g.ndim == 3 else Unsqueeze.apply(grad_output, 0)
        if batched:
            inp_graph = inp
        else:
            inp_graph = Unsqueeze.apply(inp, 0)
        if padding > 0:
            zfront = _as_const(np.zeros((N, C_in, padding), dtype=_numpy_dtype(dt)), dt)
            zback = _as_const(np.zeros((N, C_in, padding), dtype=_numpy_dtype(dt)), dt)
            inp_padded = Cat.apply(zfront, inp_graph, zback, 2)
        else:
            inp_padded = inp_graph

        # grad w.r.t kernel: windows = inp_padded (N,C_in,Lout,K)
        wins = []
        for l in range(Lout):
            cols = [
                GetItem.apply(
                    inp_padded, (slice(None), slice(None), l * stride + kk * dilation)
                )
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
                kk_t = GetItem.apply(
                    kernel, (slice(None), slice(None), kk)
                )  # (C_out,C_in)
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


def _as_pair(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _conv2d_backward_single(x, w, g, sh, sw, ph, pw, dh, dw, has_bias):
    """Single-group (groups=1) adjoint for Conv2d. Returns (grad_x, grad_k, grad_b)
    as float64 ndarrays. x:(N,Cin,H,W) w:(Cout,Cin,kH,kW) g:(N,Cout,Ho,Wo)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    Ho, Wo = g.shape[2], g.shape[3]

    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else x

    win = np.zeros((N, C_in, Ho, Wo, kH, kW), dtype=np.float64)
    for a in range(kH):
        for b in range(kW):
            win[:, :, :, :, a, b] = xp[
                :, :, a * dh : a * dh + Ho * sh : sh, b * dw : b * dw + Wo * sw : sw
            ]
    grad_k = np.einsum("noij,ncijkl->ockl", g, win)

    grad_b = g.sum(axis=(0, 2, 3)) if has_bias else None

    padded = np.zeros((N, C_in, H + 2 * ph, W + 2 * pw), dtype=np.float64)
    for i in range(Ho):
        for j in range(Wo):
            for a in range(kH):
                for b in range(kW):
                    hi = i * sh + a * dh
                    wi = j * sw + b * dw
                    padded[:, :, hi, wi] += np.einsum(
                        "no,oc->nc", g[:, :, i, j], w[:, :, a, b]
                    )
    grad_x = padded[:, :, ph : ph + H, pw : pw + W] if (ph or pw) else padded
    return grad_x, grad_k, grad_b


class Conv2d(Function):
    """Differentiable 2-D convolution (NCHW), groups=1.

    Forward mirrors ``advanced_ops.conv2d``; backward computes the exact
    adjoint: grad-weights via a windowed contraction of input with grad-output,
    grad-input via scatter-add, grad-bias via a spatial sum.
    """

    @staticmethod
    def forward(ctx, inp, kernel, bias=None, stride=1, padding=0, dilation=1, groups=1):
        from .advanced_ops import conv2d as _conv2d

        sh, sw = _as_pair(stride)
        ph, pw = _as_pair(padding)
        dh, dw = _as_pair(dilation)
        out = _conv2d(
            inp,
            kernel,
            bias,
            stride=(sh, sw),
            padding=(ph, pw),
            dilation=(dh, dw),
            groups=groups,
        )
        ctx.save_for_backward(inp=inp, kernel=kernel)
        ctx.save_attr(
            stride=(sh, sw),
            padding=(ph, pw),
            dilation=(dh, dw),
            groups=groups,
            has_bias=bias is not None,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        kernel = ctx.get_saved_tensor("kernel")
        sh, sw = ctx.get_attr("stride")
        ph, pw = ctx.get_attr("padding")
        dh, dw = ctx.get_attr("dilation")
        groups = ctx.get_attr("groups")
        has_bias = ctx.get_attr("has_bias")
        x = np.asarray(inp.data, dtype=np.float64)
        w = np.asarray(kernel.data, dtype=np.float64)
        g = np.asarray(grad_output.data, dtype=np.float64)

        if groups == 1:
            grad_x, grad_k, grad_b = _conv2d_backward_single(
                x, w, g, sh, sw, ph, pw, dh, dw, has_bias
            )
        else:
            N, C_in, H, W = x.shape
            C_out = w.shape[0]
            cin_g = C_in // groups
            cout_g = C_out // groups
            grad_x = np.zeros_like(x)
            grad_k = np.zeros_like(w)
            grad_b = np.zeros(C_out, dtype=np.float64) if has_bias else None
            for gg in range(groups):
                gxg, gkg, gbg = _conv2d_backward_single(
                    x[:, gg * cin_g : (gg + 1) * cin_g],
                    w[gg * cout_g : (gg + 1) * cout_g],
                    g[:, gg * cout_g : (gg + 1) * cout_g],
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    has_bias,
                )
                grad_x[:, gg * cin_g : (gg + 1) * cin_g] = gxg
                grad_k[gg * cout_g : (gg + 1) * cout_g] = gkg
                if has_bias:
                    grad_b[gg * cout_g : (gg + 1) * cout_g] = gbg

        return [
            Tensor(grad_x, dtype=inp.dtype),
            Tensor(grad_k, dtype=kernel.dtype),
            Tensor(grad_b, dtype=grad_output.dtype)
            if (has_bias and grad_b is not None)
            else None,
        ]


def _as_triple(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1]), int(v[2])
    return int(v), int(v), int(v)


def _conv3d_backward_single(x, w, g, sd, sh, sw, pd, ph, pw, dd, dh, dw, has_bias):
    """Single-group adjoint for Conv3d. Returns (grad_x, grad_k, grad_b) ndarrays."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    N, C_in, D, H, W = x.shape
    C_out, _, kD, kH, kW = w.shape
    Do, Ho, Wo = g.shape[2], g.shape[3], g.shape[4]

    xp = (
        np.pad(x, ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)))
        if (pd or ph or pw)
        else x
    )

    win = np.zeros((N, C_in, Do, Ho, Wo, kD, kH, kW), dtype=np.float64)
    for a in range(kD):
        for b in range(kH):
            for c in range(kW):
                win[:, :, :, :, :, a, b, c] = xp[
                    :,
                    :,
                    a * dd : a * dd + Do * sd : sd,
                    b * dh : b * dh + Ho * sh : sh,
                    c * dw : c * dw + Wo * sw : sw,
                ]
    grad_k = np.einsum("nopqr,ncpqrstu->ocstu", g, win)

    grad_b = g.sum(axis=(0, 2, 3, 4)) if has_bias else None

    padded = np.zeros((N, C_in, D + 2 * pd, H + 2 * ph, W + 2 * pw), dtype=np.float64)
    for i in range(Do):
        for j in range(Ho):
            for k in range(Wo):
                for a in range(kD):
                    for b in range(kH):
                        for c in range(kW):
                            di = i * sd + a * dd
                            hi = j * sh + b * dh
                            wi = k * sw + c * dw
                            padded[:, :, di, hi, wi] += np.einsum(
                                "no,oc->nc", g[:, :, i, j, k], w[:, :, a, b, c]
                            )
    grad_x = (
        padded[:, :, pd : pd + D, ph : ph + H, pw : pw + W]
        if (pd or ph or pw)
        else padded
    )
    return grad_x, grad_k, grad_b


class Conv3d(Function):
    """Differentiable 3-D convolution (NCDHW), groups=1.

    Mirrors :class:`Conv2d` for three spatial dimensions (stride/padding/dilation).
    """

    @staticmethod
    def forward(ctx, inp, kernel, bias=None, stride=1, padding=0, dilation=1, groups=1):
        from .advanced_ops import conv3d as _conv3d

        sd, sh, sw = _as_triple(stride)
        pd, ph, pw = _as_triple(padding)
        dd, dh, dw = _as_triple(dilation)
        out = _conv3d(
            inp,
            kernel,
            bias,
            stride=(sd, sh, sw),
            padding=(pd, ph, pw),
            dilation=(dd, dh, dw),
            groups=groups,
        )
        ctx.save_for_backward(inp=inp, kernel=kernel)
        ctx.save_attr(
            stride=(sd, sh, sw),
            padding=(pd, ph, pw),
            dilation=(dd, dh, dw),
            groups=groups,
            has_bias=bias is not None,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        kernel = ctx.get_saved_tensor("kernel")
        sd, sh, sw = ctx.get_attr("stride")
        pd, ph, pw = ctx.get_attr("padding")
        dd, dh, dw = ctx.get_attr("dilation")
        groups = ctx.get_attr("groups")
        has_bias = ctx.get_attr("has_bias")
        x = np.asarray(inp.data, dtype=np.float64)
        w = np.asarray(kernel.data, dtype=np.float64)
        g = np.asarray(grad_output.data, dtype=np.float64)

        if groups == 1:
            grad_x, grad_k, grad_b = _conv3d_backward_single(
                x, w, g, sd, sh, sw, pd, ph, pw, dd, dh, dw, has_bias
            )
        else:
            N, C_in, D, H, W = x.shape
            C_out = w.shape[0]
            cin_g = C_in // groups
            cout_g = C_out // groups
            grad_x = np.zeros_like(x)
            grad_k = np.zeros_like(w)
            grad_b = np.zeros(C_out, dtype=np.float64) if has_bias else None
            for gg in range(groups):
                gxg, gkg, gbg = _conv3d_backward_single(
                    x[:, gg * cin_g : (gg + 1) * cin_g],
                    w[gg * cout_g : (gg + 1) * cout_g],
                    g[:, gg * cout_g : (gg + 1) * cout_g],
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    dd,
                    dh,
                    dw,
                    has_bias,
                )
                grad_x[:, gg * cin_g : (gg + 1) * cin_g] = gxg
                grad_k[gg * cout_g : (gg + 1) * cout_g] = gkg
                if has_bias:
                    grad_b[gg * cout_g : (gg + 1) * cout_g] = gbg

        return [
            Tensor(grad_x, dtype=inp.dtype),
            Tensor(grad_k, dtype=kernel.dtype),
            Tensor(grad_b, dtype=grad_output.dtype)
            if (has_bias and grad_b is not None)
            else None,
        ]


def _convtranspose2d_backward_single(inp, w, g, sh, sw, ph, pw, has_bias, dh=1, dw=1):
    x = np.asarray(inp.data if hasattr(inp, "data") else inp, dtype=np.float64)
    w = np.asarray(w.data if hasattr(w, "data") else w, dtype=np.float64)
    g = np.asarray(g.data if hasattr(g, "data") else g, dtype=np.float64)
    N, C_in, H, W = x.shape
    _, C_out, kH, kW = w.shape
    Ho, Wo = g.shape[2], g.shape[3]
    grad_x = np.zeros_like(x)
    grad_w = np.zeros_like(w)
    for i in range(H):
        for j in range(W):
            xij = x[:, :, i, j]
            for kh in range(kH):
                ah = i * sh - ph + kh * dh
                if ah < 0 or ah >= Ho:
                    continue
                for kw in range(kW):
                    aw = j * sw - pw + kw * dw
                    if aw < 0 or aw >= Wo:
                        continue
                    gv = g[:, :, ah, aw]
                    grad_x[:, :, i, j] += gv @ w[:, :, kh, kw].T
                    grad_w[:, :, kh, kw] += xij.T @ gv
    grad_b = g.sum(axis=(0, 2, 3)) if has_bias else None
    return grad_x, grad_w, grad_b


class ConvTranspose2d(Function):
    """Differentiable 2-D transposed convolution (NCHW), groups=1.

    Forward mirrors ``advanced_ops.conv_transpose2d``; backward is the exact
    adjoint of the scatter formulation (stride/padding/output_padding).
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=1,
    ):
        from .advanced_ops import conv_transpose2d as _ct

        sh, sw = _as_pair(stride)
        ph, pw = _as_pair(padding)
        oph, opw = _as_pair(output_padding)
        dh, dw = _as_pair(dilation)
        out = _ct(
            inp,
            kernel,
            bias,
            stride=(sh, sw),
            padding=(ph, pw),
            output_padding=(oph, opw),
            groups=groups,
            dilation=(dh, dw),
        )
        ctx.save_for_backward(inp=inp, kernel=kernel)
        ctx.save_attr(
            stride=(sh, sw),
            padding=(ph, pw),
            dilation=(dh, dw),
            groups=groups,
            has_bias=bias is not None,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        kernel = ctx.get_saved_tensor("kernel")
        sh, sw = ctx.get_attr("stride")
        ph, pw = ctx.get_attr("padding")
        dh, dw = ctx.get_attr("dilation", (1, 1))
        if not isinstance(dh, (tuple, list)):
            dh, dw = dh, dw
        dh, dw = int(dh), int(dw)
        groups = ctx.get_attr("groups")
        has_bias = ctx.get_attr("has_bias")
        x = np.asarray(inp.data, dtype=np.float64)
        w = np.asarray(kernel.data, dtype=np.float64)
        g = np.asarray(grad_output.data, dtype=np.float64)
        N, C_in, H, W = x.shape
        _, C_out_g, kH, kW = w.shape
        C_out = C_out_g * groups
        cin_g = C_in // groups
        cout_g = C_out // groups

        grad_x = np.zeros_like(x)
        grad_w = np.zeros_like(w)
        grad_b = np.zeros(C_out, dtype=np.float64) if has_bias else None
        for gg in range(groups):
            gx, gw, gb = _convtranspose2d_backward_single(
                x[:, gg * cin_g : (gg + 1) * cin_g],
                w[gg * cin_g : (gg + 1) * cin_g],
                g[:, gg * cout_g : (gg + 1) * cout_g],
                sh,
                sw,
                ph,
                pw,
                has_bias,
                dh,
                dw,
            )
            grad_x[:, gg * cin_g : (gg + 1) * cin_g] = gx
            grad_w[gg * cin_g : (gg + 1) * cin_g] = gw
            if has_bias:
                grad_b[gg * cout_g : (gg + 1) * cout_g] = gb

        return [
            Tensor(grad_x, dtype=inp.dtype),
            Tensor(grad_w, dtype=kernel.dtype),
            Tensor(grad_b, dtype=grad_output.dtype) if has_bias else None,
        ]


def _as_ntuple(value, n):
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return (int(value),) * n


class ConvTransposeNd(Function):
    """Differentiable generic N-D transposed convolution (groups=1).

    Forward delegates to ``advanced_ops._conv_transpose_nd_impl``; the backward
    is the exact adjoint of the scatter formulation (stride/padding).
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        kernel,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=1,
    ):
        from .advanced_ops import _conv_transpose_nd_impl

        ndim = inp.data.ndim - 2
        dilation_t = _as_ntuple(dilation, ndim)
        out = _conv_transpose_nd_impl(
            inp, kernel, bias, stride, padding, output_padding, groups, dilation_t
        )
        ctx.save_for_backward(inp=inp, kernel=kernel)
        ctx.save_attr(
            stride=_as_ntuple(stride, ndim),
            padding=_as_ntuple(padding, ndim),
            dilation=dilation_t,
            groups=groups,
            has_bias=bias is not None,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        inp = ctx.get_saved_tensor("inp")
        kernel = ctx.get_saved_tensor("kernel")
        stride = ctx.get_attr("stride")
        padding = ctx.get_attr("padding")
        dilation = ctx.get_attr("dilation", None)
        if dilation is None:
            dilation = (1,) * len(stride)
        groups = ctx.get_attr("groups")
        has_bias = ctx.get_attr("has_bias")
        dilation = ctx.get_attr("dilation", None)
        if dilation is None:
            dilation = (1,) * len(stride)

        x = np.asarray(inp.data, dtype=np.float64)
        w = np.asarray(kernel.data, dtype=np.float64)
        g = np.asarray(grad_output.data, dtype=np.float64)
        ndim = x.ndim - 2
        spatial_in = x.shape[2:]
        kshape = w.shape[2:]
        S_out = g.shape[2:]

        C_in = x.shape[1]
        C_out = g.shape[1]
        cin_g = C_in // groups
        cout_g = C_out // groups

        grad_x = np.zeros_like(x)
        grad_w = np.zeros_like(w)
        grad_b = np.zeros(C_out, dtype=np.float64) if has_bias else None

        for gg in range(groups):
            xg = x[:, gg * cin_g : (gg + 1) * cin_g]
            wg = w[gg * cin_g : (gg + 1) * cin_g]
            gg_out = g[:, gg * cout_g : (gg + 1) * cout_g]
            S_out_g = gg_out.shape[2:]
            grad_xg = np.zeros_like(xg)
            grad_wg = np.zeros_like(wg)
            xa = (slice(None), slice(None))
            for sidx in np.ndindex(*spatial_in):
                x_slice = xg[xa + sidx]
                for kidx in np.ndindex(*kshape):
                    oidx = tuple(
                        sidx[d] * stride[d] - padding[d] + kidx[d] * dilation[d]
                        for d in range(ndim)
                    )
                    if any(oidx[d] < 0 or oidx[d] >= S_out_g[d] for d in range(ndim)):
                        continue
                    gv = gg_out[xa + oidx]
                    grad_xg[xa + sidx] += gv @ wg[xa + kidx].T
                    grad_wg[xa + kidx] += x_slice.T @ gv
            grad_x[:, gg * cin_g : (gg + 1) * cin_g] = grad_xg
            grad_w[gg * cin_g : (gg + 1) * cin_g] = grad_wg
            if has_bias:
                grad_b[gg * cout_g : (gg + 1) * cout_g] = gg_out.sum(
                    axis=(0,) + tuple(range(2, gg_out.ndim))
                )

        grad_b = grad_b if has_bias else None

        return [
            Tensor(grad_x, dtype=inp.dtype),
            Tensor(grad_w, dtype=kernel.dtype),
            Tensor(grad_b, dtype=grad_output.dtype) if has_bias else None,
        ]


class AdaptiveMaxPoolFn(Function):
    """Adaptive max pooling over trailing spatial dims (differentiable).

    Bin scheme matches ``AdaptiveAvgPoolFn``: floor bins (``use_ceil=False``)
    for the 2-D reference, ceil bins (``use_ceil=True``) for 1d/3d. Gradient
    is routed to the first arg-max element of each bin.
    """

    @staticmethod
    def forward(ctx, a, output_size, use_ceil=False):
        x = np.asarray(a.data, dtype=np.float64)
        output_size = tuple(int(o) for o in output_size)
        nd = len(output_size)
        in_sp = x.shape[-nd:]
        ranges = []
        for H, O in zip(in_sp, output_size):
            if use_ceil:
                bins = [(i * H // O, -((-(i + 1) * H) // O)) for i in range(O)]
            else:
                bins = [(i * H // O, (i + 1) * H // O) for i in range(O)]
            ranges.append(bins)
        lead = x.shape[:-nd]
        lead_sl = (slice(None),) * len(lead)
        out = np.zeros(lead + output_size, dtype=np.float64)
        for idx in itertools.product(*[range(o) for o in output_size]):
            sl = tuple(
                slice(ranges[k][idx[k]][0], ranges[k][idx[k]][1]) for k in range(nd)
            )
            win = x[lead_sl + sl]
            if win.size and all(s.stop > s.start for s in sl):
                out[lead_sl + idx] = win.max(axis=tuple(range(len(lead), x.ndim)))
        ctx.save_for_backward(a=a)
        ctx.save_attr(ranges=ranges, nd=nd, use_ceil=use_ceil)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a = ctx.get_saved_tensor("a")
        ranges = ctx.get_attr("ranges")
        nd = ctx.get_attr("nd")
        x = np.asarray(a.data, dtype=np.float64)
        g = np.asarray(grad_output.data, dtype=np.float64)
        osz = tuple(len(ranges[k]) for k in range(nd))
        nlead = x.ndim - nd
        lead_shape = x.shape[:nlead]
        lead_idx = np.indices(lead_shape) if nlead else None
        gx = np.zeros(x.shape, dtype=np.float64)
        for idx in itertools.product(*[range(o) for o in osz]):
            sl = tuple(
                slice(ranges[k][idx[k]][0], ranges[k][idx[k]][1]) for k in range(nd)
            )
            if any(s.stop <= s.start for s in sl):
                continue
            win = x[(slice(None),) * nlead + sl]
            binshape = win.shape[nlead:]
            flat = win.reshape(lead_shape + (-1,))
            arg = flat.argmax(axis=-1)
            unrav = np.unravel_index(arg, binshape)
            gv = g[(slice(None),) * nlead + idx]
            index = tuple(lead_idx[k] for k in range(nlead)) + tuple(
                sl[k].start + unrav[k] for k in range(nd)
            )
            np.add.at(gx, index, gv)
        return [Tensor(gx, dtype=grad_output.dtype)]


class MaxUnpoolFn(Function):
    """Differentiable max-unpool: scatter input into ``out_sp`` at flat ``indices``."""

    @staticmethod
    def forward(ctx, inp, indices, out_sp):
        x = np.asarray(inp.data, dtype=np.float64)
        N, C = x.shape[:2]
        in_sp = x.shape[2:]
        out_sp = tuple(int(s) for s in out_sp)
        n_out = int(np.prod(out_sp)) if out_sp else 1
        idxf = np.asarray(indices.data, dtype=np.int64).reshape(N, C, -1)
        flat = np.zeros((N, C, n_out), dtype=np.float64)
        np.put_along_axis(flat, idxf, x.reshape(N, C, -1), axis=2)
        ctx.save_attr(idxf=idxf, in_shape=x.shape, n_out=n_out)
        return Tensor(flat.reshape((N, C) + out_sp), dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        idxf = ctx.get_attr("idxf")
        in_shape = ctx.get_attr("in_shape")
        n_out = ctx.get_attr("n_out")
        g = np.asarray(grad_output.data, dtype=np.float64).reshape(
            in_shape[0], in_shape[1], n_out
        )
        gx = np.take_along_axis(g, idxf, axis=2).reshape(in_shape)
        return [Tensor(gx, dtype=grad_output.dtype), None]


class EmbeddingBagFn(Function):
    """Differentiable EmbeddingBag (sum/mean/max) w.r.t. the embedding ``weight``.

    ``indices``/``offsets``/``per_sample_weights`` are passed as plain numpy
    arrays (or ``None``); only ``weight`` participates in the autograd graph.
    """

    @staticmethod
    def forward(ctx, weight, idx, off, mode, psw):
        w = np.asarray(weight.data, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        emb = w[idx]
        if psw is not None:
            psw = np.asarray(psw, dtype=np.float64).reshape(-1)
            emb = emb * psw[:, None]
        if off is None:
            groups = [np.arange(idx.size, dtype=np.int64)]
        else:
            off = np.clip(np.asarray(off, dtype=np.int64), 0, idx.size)
            groups = [
                np.arange(off[i], off[i + 1], dtype=np.int64)
                for i in range(off.size - 1)
            ]
        D = w.shape[1]
        out = np.zeros((len(groups), D), dtype=np.float64)
        argmax = []
        for g, rows in enumerate(groups):
            if rows.size == 0:
                argmax.append(np.full(D, -1, dtype=np.int64))
                continue
            seg = emb[rows]
            if mode == "sum":
                out[g] = seg.sum(axis=0)
            elif mode == "mean":
                out[g] = seg.mean(axis=0)
            else:
                am = seg.argmax(axis=0)
                out[g] = seg[am, np.arange(D)]
                argmax.append(am)
        ctx.save_attr(
            idx=idx,
            mode=mode,
            psw=psw,
            argmax=argmax,
            groups=groups,
            weight_shape=w.shape,
        )
        return Tensor(out, dtype=weight.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        idx = ctx.get_attr("idx")
        mode = ctx.get_attr("mode")
        psw = ctx.get_attr("psw")
        argmax = ctx.get_attr("argmax")
        groups = ctx.get_attr("groups")
        n_emb, D = ctx.get_attr("weight_shape")
        g = np.asarray(grad_output.data, dtype=np.float64).reshape(len(groups), D)
        gw = np.zeros((n_emb, D), dtype=np.float64)
        for gi, rows in enumerate(groups):
            if rows.size == 0:
                continue
            if mode == "sum":
                contrib = np.broadcast_to(g[gi][None, :], (rows.size, D)).copy()
            elif mode == "mean":
                contrib = np.broadcast_to(
                    g[gi][None, :] / rows.size, (rows.size, D)
                ).copy()
            else:
                contrib = np.zeros((rows.size, D), dtype=np.float64)
                am = argmax[gi]
                valid = am >= 0
                contrib[am[valid], np.arange(D)[valid]] = g[gi][valid]
            if psw is not None:
                contrib = contrib * psw[rows][:, None]
            np.add.at(gw, idx[rows], contrib)
        return [Tensor(gw, dtype=grad_output.dtype)]


class PadNdFn(Function):
    """Differentiable N-D padding via ``np.pad`` (constant / reflect / edge).

    Forward builds a source-index map by padding a flat ``arange`` with the same
    mode; backward scatter-adds gradients back to those source positions.
    """

    @staticmethod
    def forward(ctx, a, pad_width, mode, value):
        x = np.asarray(a.data, dtype=np.float64)
        base = np.arange(x.size, dtype=np.int64).reshape(x.shape)
        if mode == "constant":
            imap = np.pad(base, pad_width, mode=mode, constant_values=-1)
        else:
            imap = np.pad(base, pad_width, mode=mode)
        mask = imap >= 0
        out = np.full(imap.shape, value, dtype=np.float64)
        xf = x.reshape(-1)
        out[mask] = xf[imap[mask]]
        ctx.save_attr(imap=imap, mask=mask, in_shape=x.shape)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        imap = ctx.get_attr("imap")
        mask = ctx.get_attr("mask")
        in_shape = ctx.get_attr("in_shape")
        g = np.asarray(grad_output.data, dtype=np.float64)
        gi = np.zeros(int(np.prod(in_shape)), dtype=np.float64)
        np.add.at(gi, imap[mask], g[mask])
        return [Tensor(gi.reshape(in_shape), dtype=grad_output.dtype)]


class PReluFn(Function):
    """Differentiable PReLU: ``y = x where x >= w*x else w*x`` (trailing-broadcast weight)."""

    @staticmethod
    def forward(ctx, a, w):
        x = np.asarray(a.data, dtype=np.float64)
        wv = np.asarray(w.data, dtype=np.float64)
        wexp = wv * x
        mask_x = x >= wexp
        out = np.where(mask_x, x, wexp)
        ctx.save_attr(mask_x=mask_x, x=x, wv=wv)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        mask_x = ctx.get_attr("mask_x")
        x = ctx.get_attr("x")
        wv = ctx.get_attr("wv")
        g = np.asarray(grad_output.data, dtype=np.float64)
        gx = g * np.where(mask_x, 1.0, wv)
        gw = g * np.where(mask_x, 0.0, x)
        nlead = x.ndim - wv.ndim
        if nlead > 0:
            gw = gw.sum(axis=tuple(range(nlead)))
        rax = [j for j in range(wv.ndim) if wv.shape[j] == 1 and gw.shape[j] > 1]
        if rax:
            gw = gw.sum(axis=tuple(rax), keepdims=True)
        gw = gw.reshape(wv.shape)
        return [
            Tensor(gx, dtype=grad_output.dtype),
            Tensor(gw, dtype=grad_output.dtype),
        ]


class LogSigmoidFn(Function):
    """Differentiable log-sigmoid: ``log(sigmoid(x)) = -log1p(exp(-|x|)) + min(x,0)``."""

    @staticmethod
    def forward(ctx, a):
        x = np.asarray(a.data, dtype=np.float64)
        out = -np.log1p(np.exp(-np.abs(x))) + np.minimum(x, 0.0)
        ctx.save_attr(x=x)
        return Tensor(out, dtype=a.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        x = ctx.get_attr("x")
        g = np.asarray(grad_output.data, dtype=np.float64)
        sig_neg = 1.0 / (1.0 + np.exp(x))
        return [Tensor(g * sig_neg, dtype=grad_output.dtype)]


class BilinearFn(Function):
    """Differentiable bilinear layer: out[b,k] = sum_ij x1[b,i] x2[b,j] W[k,i,j] + b[k]."""

    @staticmethod
    def forward(ctx, input1, input2, weight, bias=None):
        x1 = np.asarray(input1.data, dtype=np.float64)
        x2 = np.asarray(input2.data, dtype=np.float64)
        w = np.asarray(weight.data, dtype=np.float64)
        out = np.einsum("bi,bj,kij->bk", x1, x2, w)
        if bias is not None:
            out = out + np.asarray(bias.data, dtype=np.float64)
        ctx.save_for_backward(input1=input1, input2=input2, weight=weight)
        ctx.save_attr(has_bias=bias is not None)
        return Tensor(out, dtype=input1.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        x1 = np.asarray(ctx.get_saved_tensor("input1").data, dtype=np.float64)
        x2 = np.asarray(ctx.get_saved_tensor("input2").data, dtype=np.float64)
        w = np.asarray(ctx.get_saved_tensor("weight").data, dtype=np.float64)
        g = np.asarray(grad_output.data, dtype=np.float64)
        gx1 = np.einsum("bk,kij,bj->bi", g, w, x2)
        gx2 = np.einsum("bk,kij,bi->bj", g, w, x1)
        gw = np.einsum("bk,bi,bj->kij", g, x1, x2)
        has_bias = ctx.get_attr("has_bias")
        gb = g.sum(axis=0) if has_bias else None
        return [
            Tensor(gx1, dtype=grad_output.dtype),
            Tensor(gx2, dtype=grad_output.dtype),
            Tensor(gw, dtype=grad_output.dtype),
            Tensor(gb, dtype=grad_output.dtype) if has_bias else None,
        ]


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
            grad_inp = Div.apply(Mul.apply(_as_const(2.0, dt), diff), _as_const(n, dt))
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
        return [
            Tensor(np.squeeze(g, axis=dim).copy(), dtype=grad_output.dtype)
            for g in grads
        ]


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
        return Tensor(
            np.array([float(np.mean(loss))], dtype=z.dtype), dtype=logits.dtype
        )

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
        f = (
            gamma * pt * (1 - pt) ** (gamma - 1) * np.log(pt + 1e-10)
            - (1 - pt) ** gamma
        )
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
        return Tensor(
            np.array([float(np.mean(loss))], dtype=a.dtype), dtype=anchor.dtype
        )

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
        return [
            Tensor(ga, dtype=anchor.dtype),
            Tensor(gp, dtype=anchor.dtype),
            Tensor(gn, dtype=anchor.dtype),
            None,
        ]


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
        yy = y.data.ravel().astype(np.float64).reshape(-1, 1)
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
        yy = y.data.ravel().astype(np.float64).reshape(-1, 1)
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
        ctx,
        log_probs,
        targets,
        blank=0,
        input_lengths=None,
        target_lengths=None,
        reduction="mean",
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
            alphas=alphas,
            exts=exts,
            N=N,
            T=T,
            C=C,
            blank=blank,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            reduction=reduction,
            lp=lp,
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
                [
                    alphas[n][Ti - 1, L - 1],
                    alphas[n][Ti - 1, L - 2] if L >= 2 else -1e30,
                ]
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
    "MaxPool",
    "AvgPool",
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
    "Log1p",
    "Expm1",
]

# ============================================================================
#  Generalized N-d Pooling (1d/2d/3d, padding + dilation, differentiable)
# ============================================================================


class MaxPool(Function):
    """Differentiable N-d max pool over trailing spatial dims with
    padding (padded with -inf so pad never wins) and dilation."""

    @staticmethod
    def forward(ctx, inp, kernel, stride=None, padding=0, dilation=1):
        arr = np.asarray(inp.data, dtype=np.float64)
        ndim = len(kernel)
        if stride is None:
            stride = kernel
        if isinstance(stride, int):
            stride = (stride,) * ndim
        if isinstance(padding, int):
            padding = (padding,) * ndim
        if isinstance(dilation, int):
            dilation = (dilation,) * ndim
        kernel = tuple(int(k) for k in kernel)
        stride = tuple(int(s) for s in stride)
        padding = tuple(int(p) for p in padding)
        dilation = tuple(int(d) for d in dilation)

        pad = [(0, 0)] * (arr.ndim - ndim) + [(p, p) for p in padding]
        padded = np.pad(arr, pad, mode="constant", constant_values=-np.inf)
        spatial_axes = list(range(arr.ndim - ndim, arr.ndim))
        out_shape_sp = []
        for d in range(ndim):
            sp = arr.shape[spatial_axes[d]]
            kd, sd, pd, dd = kernel[d], stride[d], padding[d], dilation[d]
            L = sp + 2 * pd - dd * (kd - 1)
            out_shape_sp.append(max((L - 1) // sd + 1, 0))
        out_shape = arr.shape[: arr.ndim - ndim] + tuple(out_shape_sp)
        out = np.full(out_shape, -np.inf, dtype=np.float64)
        it = np.nditer(np.zeros(out_shape_sp), flags=["multi_index"])
        while not it.finished:
            oi = it.multi_index
            sl = []
            for d in range(ndim):
                st = oi[d] * stride[d]
                sl.append(
                    slice(st, st + dilation[d] * (kernel[d] - 1) + 1, dilation[d])
                )
            window = padded[(slice(None),) * (arr.ndim - ndim) + tuple(sl)]
            red_axes = tuple(range(arr.ndim - ndim, arr.ndim))
            out[(slice(None),) * (arr.ndim - ndim) + oi] = window.max(axis=red_axes)
            it.iternext()
        ctx.save_attr(
            arr_shape=arr.shape,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ndim=ndim,
        )
        ctx.save_for_backward(inp=inp)
        return Tensor(out, dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        arr_shape = ctx.get_attr("arr_shape")
        kernel = ctx.get_attr("kernel")
        stride = ctx.get_attr("stride")
        padding = ctx.get_attr("padding")
        dilation = ctx.get_attr("dilation")
        ndim = ctx.get_attr("ndim")
        inp = ctx.get_saved_tensor("inp")
        x = np.asarray(inp.data, dtype=np.float64)
        g = np.asarray(grad_output.data, dtype=np.float64)
        grad = np.zeros_like(x)
        spatial_axes = list(range(x.ndim - ndim, x.ndim))
        out_shape_sp = []
        for d in range(ndim):
            sp = x.shape[spatial_axes[d]]
            kd, sd, pd, dd = kernel[d], stride[d], padding[d], dilation[d]
            L = sp + 2 * pd - dd * (kd - 1)
            out_shape_sp.append(max((L - 1) // sd + 1, 0))
        it = np.nditer(np.zeros(out_shape_sp), flags=["multi_index"])
        while not it.finished:
            oi = it.multi_index
            sl = []
            for d in range(ndim):
                st = oi[d] * stride[d]
                sl.append(
                    slice(st, st + dilation[d] * (kernel[d] - 1) + 1, dilation[d])
                )
            sl_orig = []
            for d in range(ndim):
                st_orig = sl[d].start - padding[d]
                stop_orig = sl[d].stop - padding[d]
                sl_orig.append(
                    slice(
                        max(st_orig, 0),
                        min(stop_orig, x.shape[spatial_axes[d]]),
                        sl[d].step,
                    )
                )
            window_orig = x[(slice(None),) * (x.ndim - ndim) + tuple(sl_orig)]
            max_val = window_orig.max(
                axis=tuple(range(x.ndim - ndim, x.ndim)), keepdims=True
            )
            mask = (window_orig == max_val).astype(g.dtype)
            g_val = g[(slice(None),) * (x.ndim - ndim) + oi]
            gb = g_val
            for _ in range(ndim):
                gb = gb[..., np.newaxis]
            seg = grad[(slice(None),) * (x.ndim - ndim) + tuple(sl_orig)]
            seg += mask * gb
            it.iternext()
        return [Tensor(grad, dtype=inp.dtype)]


class AvgPool(Function):
    """Differentiable N-d average pool with padding (count_include_pad=True)
    and dilation."""

    @staticmethod
    def forward(ctx, inp, kernel, stride=None, padding=0, dilation=1):
        arr = np.asarray(inp.data, dtype=np.float64)
        ndim = len(kernel)
        if stride is None:
            stride = kernel
        if isinstance(stride, int):
            stride = (stride,) * ndim
        if isinstance(padding, int):
            padding = (padding,) * ndim
        if isinstance(dilation, int):
            dilation = (dilation,) * ndim
        kernel = tuple(int(k) for k in kernel)
        stride = tuple(int(s) for s in stride)
        padding = tuple(int(p) for p in padding)
        dilation = tuple(int(d) for d in dilation)

        pad = [(0, 0)] * (arr.ndim - ndim) + [(p, p) for p in padding]
        padded = np.pad(arr, pad, mode="constant", constant_values=0.0)
        spatial_axes = list(range(arr.ndim - ndim, arr.ndim))
        kvol = int(np.prod(kernel))
        out_shape_sp = []
        for d in range(ndim):
            sp = arr.shape[spatial_axes[d]]
            kd, sd, pd, dd = kernel[d], stride[d], padding[d], dilation[d]
            L = sp + 2 * pd - dd * (kd - 1)
            out_shape_sp.append(max((L - 1) // sd + 1, 0))
        out_shape = arr.shape[: arr.ndim - ndim] + tuple(out_shape_sp)
        out = np.zeros(out_shape, dtype=np.float64)
        it = np.nditer(np.zeros(out_shape_sp), flags=["multi_index"])
        while not it.finished:
            oi = it.multi_index
            sl = []
            for d in range(ndim):
                st = oi[d] * stride[d]
                sl.append(
                    slice(st, st + dilation[d] * (kernel[d] - 1) + 1, dilation[d])
                )
            window = padded[(slice(None),) * (arr.ndim - ndim) + tuple(sl)]
            red_axes = tuple(range(arr.ndim - ndim, arr.ndim))
            out[(slice(None),) * (arr.ndim - ndim) + oi] = (
                window.sum(axis=red_axes) / kvol
            )
            it.iternext()
        ctx.save_attr(
            arr_shape=arr.shape,
            kernel=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ndim=ndim,
            kvol=kvol,
        )
        ctx.save_for_backward(inp=inp)
        return Tensor(out, dtype=inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        arr_shape = ctx.get_attr("arr_shape")
        kernel = ctx.get_attr("kernel")
        stride = ctx.get_attr("stride")
        padding = ctx.get_attr("padding")
        dilation = ctx.get_attr("dilation")
        ndim = ctx.get_attr("ndim")
        kvol = ctx.get_attr("kvol")
        inp = ctx.get_saved_tensor("inp")
        x = np.asarray(inp.data, dtype=np.float64)
        g = np.asarray(grad_output.data, dtype=np.float64)
        grad = np.zeros_like(x)
        spatial_axes = list(range(x.ndim - ndim, x.ndim))
        out_shape_sp = []
        for d in range(ndim):
            sp = x.shape[spatial_axes[d]]
            kd, sd, pd, dd = kernel[d], stride[d], padding[d], dilation[d]
            L = sp + 2 * pd - dd * (kd - 1)
            out_shape_sp.append(max((L - 1) // sd + 1, 0))
        it = np.nditer(np.zeros(out_shape_sp), flags=["multi_index"])
        while not it.finished:
            oi = it.multi_index
            sl = []
            for d in range(ndim):
                st = oi[d] * stride[d]
                sl.append(
                    slice(st, st + dilation[d] * (kernel[d] - 1) + 1, dilation[d])
                )
            sl_orig = []
            for d in range(ndim):
                st_orig = sl[d].start - padding[d]
                stop_orig = sl[d].stop - padding[d]
                sl_orig.append(
                    slice(
                        max(st_orig, 0),
                        min(stop_orig, x.shape[spatial_axes[d]]),
                        sl[d].step,
                    )
                )
            seg = grad[(slice(None),) * (x.ndim - ndim) + tuple(sl_orig)]
            gb = g[(slice(None),) * (x.ndim - ndim) + oi]
            for _ in range(ndim):
                gb = gb[..., np.newaxis]
            seg += gb / kvol
            it.iternext()
        return [Tensor(grad, dtype=inp.dtype)]


# ---------------------------------------------------------------------------
# Elementwise math + reductions (Phase 2 gap fill, additive-only)
# ---------------------------------------------------------------------------


class Square(Function):
    @staticmethod
    def forward(ctx, x):
        if ctx is not None:
            ctx.save_for_backward(x=x)
        return Tensor(np.asarray(x.data) * np.asarray(x.data), dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        x = _get_saved_tensor(ctx, "x")
        if x is None:
            return [
                Tensor(
                    np.zeros_like(np.asarray(grad_output.data)), dtype=grad_output.dtype
                )
            ]
        if create_graph:
            return [
                Mul.apply(Mul.apply(grad_output, _as_const(2.0, grad_output.dtype)), x)
            ]
        g = np.asarray(grad_output.data)
        return [Tensor(g * (2.0 * np.asarray(x.data)), dtype=grad_output.dtype)]


class Reciprocal(Function):
    @staticmethod
    def forward(ctx, x):
        if ctx is not None:
            ctx.save_for_backward(x=x)
        return Tensor(1.0 / np.asarray(x.data), dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        x = _get_saved_tensor(ctx, "x")
        if x is None:
            return [
                Tensor(
                    np.zeros_like(np.asarray(grad_output.data)), dtype=grad_output.dtype
                )
            ]
        xd = np.asarray(x.data)
        if create_graph:
            neg_one = _as_const(-1.0, grad_output.dtype)
            return [Mul.apply(grad_output, Div.apply(neg_one, Mul.apply(x, x)))]
        g = np.asarray(grad_output.data)
        return [Tensor(g * (-1.0 / (xd * xd)), dtype=grad_output.dtype)]


class Sign(Function):
    @staticmethod
    def forward(ctx, x):
        return Tensor(np.sign(np.asarray(x.data)), dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        return [
            Tensor(np.zeros_like(np.asarray(grad_output.data)), dtype=grad_output.dtype)
        ]


class Clamp(Function):
    @staticmethod
    def forward(ctx, x, min_val, max_val):
        if ctx is not None:
            ctx.save_for_backward(x=x)
        _save_attr(ctx, min_val=min_val, max_val=max_val)
        return Tensor(np.clip(np.asarray(x.data), min_val, max_val), dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        x = _get_saved_tensor(ctx, "x")
        min_val = _get_attr(ctx, "min_val")
        max_val = _get_attr(ctx, "max_val")
        if x is None:
            return [
                Tensor(
                    np.zeros_like(np.asarray(grad_output.data)), dtype=grad_output.dtype
                )
            ]
        xd = np.asarray(x.data)
        mask = np.logical_and(xd >= min_val, xd <= max_val)
        if create_graph:
            return [
                Mul.apply(
                    grad_output, _as_const(mask.astype(np.float64), grad_output.dtype)
                )
            ]
        g = np.asarray(grad_output.data)
        return [Tensor(g * mask, dtype=grad_output.dtype)]


class Remainder(Function):
    @staticmethod
    def forward(ctx, a, divisor):
        if isinstance(divisor, (int, float)):
            _save_attr(ctx, divisor_val=divisor, a_shape=tuple(a.shape))
            return Tensor(np.mod(np.asarray(a.data), divisor), dtype=a.dtype)
        if ctx is not None:
            ctx.save_for_backward(a=a)
        _save_attr(ctx, a_shape=tuple(a.shape))
        return Tensor(
            np.mod(np.asarray(a.data), np.asarray(divisor.data)), dtype=a.dtype
        )

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        a_shape = _get_attr(ctx, "a_shape")
        divisor_val = _get_attr(ctx, "divisor_val")
        a = _get_saved_tensor(ctx, "a")
        if divisor_val is not None:
            if create_graph:
                return [_cg_broadcast(grad_output, a_shape)]
            return [
                Tensor(_reduce_to_shape(grad_output, a_shape), dtype=grad_output.dtype)
            ]
        # Tensor divisor: grad flows to a unchanged (d remainder/d a == 1 a.e.)
        if create_graph:
            ga = _cg_broadcast(grad_output, a_shape)
        else:
            ga = Tensor(_reduce_to_shape(grad_output, a_shape), dtype=grad_output.dtype)
        return [ga, None]


class Prod(Function):
    @staticmethod
    def forward(ctx, x, dim=None, keepdim=False):
        xd = np.asarray(x.data)
        if dim is None:
            axes = tuple(range(xd.ndim))
        elif isinstance(dim, (tuple, list)):
            axes = tuple(dim)
        else:
            axes = (dim,)
        if ctx is not None:
            ctx.save_for_backward(x=x)
        _save_attr(ctx, axes=axes, keepdim=keepdim)
        out = np.asarray(np.prod(xd, axis=axes, keepdims=keepdim))
        return Tensor(np.asarray(out), dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x = _get_saved_tensor(ctx, "x")
        axes = ctx.get_attr("axes")
        keepdim = ctx.get_attr("keepdim")
        xd = np.asarray(x.data)
        g = np.asarray(grad_output.data)
        prod_all = np.prod(xd, axis=axes, keepdims=True)
        prod_excl = prod_all / xd
        if not keepdim:
            g = np.expand_dims(g, axis=axes)
        return [Tensor(g * prod_excl, dtype=grad_output.dtype)]


class LogSumExp(Function):
    @staticmethod
    def forward(ctx, x, dim, keepdim=False):
        xd = np.asarray(x.data)
        axes = (dim,) if isinstance(dim, int) else tuple(dim)
        if ctx is not None:
            ctx.save_for_backward(x=x)
        _save_attr(ctx, axes=axes, keepdim=keepdim)
        m = np.max(xd, axis=axes, keepdims=True)
        shifted = np.exp(xd - m)
        s = np.sum(shifted, axis=axes, keepdims=keepdim)
        m_out = m if keepdim else np.squeeze(m, axis=axes)
        return Tensor(np.asarray(np.log(s) + m_out), dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x = _get_saved_tensor(ctx, "x")
        axes = ctx.get_attr("axes")
        keepdim = ctx.get_attr("keepdim")
        xd = np.asarray(x.data)
        g = np.asarray(grad_output.data)
        m = np.max(xd, axis=axes, keepdims=True)
        soft = np.exp(xd - m)
        soft = soft / np.sum(soft, axis=axes, keepdims=True)
        if not keepdim:
            g = np.expand_dims(g, axis=axes)
        return [Tensor(g * soft, dtype=grad_output.dtype)]


class Trace(Function):
    @staticmethod
    def forward(ctx, x):
        xd = np.asarray(x.data)
        assert xd.ndim == 2, "trace requires a 2-D tensor"
        _save_attr(ctx, in_shape=tuple(xd.shape))
        return Tensor(np.array(np.trace(xd)), dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.get_attr("in_shape")
        g = float(np.asarray(grad_output.data).sum())
        grad = np.zeros(shape, dtype=np.float64)
        np.fill_diagonal(grad, g)
        return [Tensor(grad, dtype=grad_output.dtype)]


class Diagonal(Function):
    @staticmethod
    def forward(ctx, x, offset=0, dim1=0, dim2=1):
        xd = np.asarray(x.data)
        assert xd.ndim == 2, "diagonal currently supports 2-D tensors"
        _save_attr(ctx, offset=offset, in_shape=tuple(xd.shape))
        return Tensor(
            np.diagonal(xd, offset=offset, axis1=dim1, axis2=dim2).copy(), dtype=x.dtype
        )

    @staticmethod
    def backward(ctx, grad_output):
        offset = ctx.get_attr("offset")
        shape = ctx.get_attr("in_shape")
        g = np.asarray(grad_output.data)
        grad = np.zeros(shape, dtype=np.float64)
        n = len(g) if not np.isscalar(g) else 1
        rows = np.arange(n)
        if offset >= 0:
            grad[rows, rows + offset] = g
        else:
            grad[rows - offset, rows] = g
        return [Tensor(grad, dtype=grad_output.dtype)]


class MinDim(Function):
    @staticmethod
    def forward(ctx, x, dim=None, keepdim=False):
        xd = np.asarray(x.data, dtype=np.float64)
        if dim is None:
            out = np.asarray(np.min(xd))
        else:
            out = np.min(xd, axis=dim, keepdims=keepdim)
        if ctx is not None:
            ctx.save_for_backward(x=x)
        _save_attr(ctx, dim=dim, keepdim=keepdim)
        return Tensor(out, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        x = _get_saved_tensor(ctx, "x")
        dim = ctx.get_attr("dim")
        keepdim = ctx.get_attr("keepdim")
        xd = np.asarray(x.data)
        g = np.asarray(grad_output.data)
        grad = np.zeros_like(xd)
        if dim is None:
            grad.flat[np.argmin(xd)] = float(np.asarray(grad_output.data).sum())
        else:
            idx = np.argmin(xd, axis=dim)
            g_exp = g if keepdim else np.expand_dims(g, axis=dim)
            np.put_along_axis(grad, np.expand_dims(idx, axis=dim), g_exp, axis=dim)
        return [Tensor(grad, dtype=grad_output.dtype)]


class Log1p(Function):
    @staticmethod
    def forward(ctx, x):
        xd = np.asarray(x.data, dtype=np.float64)
        out = np.log1p(xd)
        if ctx is not None:
            ctx.save_for_backward(x=x)
        return Tensor(out, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x = _get_saved_tensor(ctx, "x")
        xd = np.asarray(x.data)
        g = np.asarray(grad_output.data)
        grad = g / (1.0 + xd)
        return [Tensor(grad, dtype=grad_output.dtype)]


class Expm1(Function):
    @staticmethod
    def forward(ctx, x):
        xd = np.asarray(x.data, dtype=np.float64)
        out = np.expm1(xd)
        if ctx is not None:
            ctx.save_for_backward(x=x)
        return Tensor(out, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x = _get_saved_tensor(ctx, "x")
        xd = np.asarray(x.data)
        g = np.asarray(grad_output.data)
        grad = g * np.exp(xd)
        return [Tensor(grad, dtype=grad_output.dtype)]


class MaxDim(Function):
    @staticmethod
    def forward(ctx, x, dim=None, keepdim=False):
        xd = np.asarray(x.data, dtype=np.float64)
        if dim is None:
            out = np.asarray(np.max(xd))
        else:
            out = np.max(xd, axis=dim, keepdims=keepdim)
        if ctx is not None:
            ctx.save_for_backward(x=x)
        _save_attr(ctx, dim=dim, keepdim=keepdim)
        return Tensor(out, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output, create_graph=False):
        x = _get_saved_tensor(ctx, "x")
        dim = ctx.get_attr("dim")
        keepdim = ctx.get_attr("keepdim")
        xd = np.asarray(x.data)
        g = np.asarray(grad_output.data)
        grad = np.zeros_like(xd)
        if dim is None:
            grad.flat[np.argmax(xd)] = float(np.asarray(grad_output.data).sum())
        else:
            idx = np.argmax(xd, axis=dim)
            g_exp = g if keepdim else np.expand_dims(g, axis=dim)
            np.put_along_axis(grad, np.expand_dims(idx, axis=dim), g_exp, axis=dim)
        return [Tensor(grad, dtype=grad_output.dtype)]
