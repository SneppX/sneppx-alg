"""Functional differentiation API — JVP / VJP / Jacobians / higher-order grad.

Provides forward- and reverse-mode primitives on top of the tape autograd
engine. ``jvp``/``jacfwd`` are implemented by reverse-mode over the output
coordinates (correct for any op that defines ``backward``); ``vjp``/``jacrev``
use the native reverse tape. This makes forward-mode (JVP) and higher-order
differentiation (grad-of-grad) available for every differentiable op without
requiring a separate forward tape.
"""

import numpy as np

from .tensor import Tensor
from .autograd import _wrap_tensor_backward


def _as_tensor(x):
    if isinstance(x, Tensor):
        return x
    arr = np.asarray(x)
    dt = "float64" if arr.dtype.kind in ("f", "c") else _resolve_dtype(arr.dtype)
    return Tensor(arr, dtype=dt)


def _zeros_grad(primals):
    for p in primals:
        if isinstance(p, Tensor):
            p.grad = None


def vjp(f, primals, v):
    """Reverse mode: returns ``(output, cotangents)`` where ``cotangents``
    are the gradients of ``f`` w.r.t. each primal given the output cotangent
    ``v``."""
    primals = [_as_tensor(p) for p in primals]
    for p in primals:
        p.requires_grad_(True)
    out = f(*primals)
    seed = _as_tensor(v) if not isinstance(v, Tensor) else v
    _zeros_grad(primals)
    _wrap_tensor_backward(out, seed)
    cotangents = [p.grad if p.grad is not None else Tensor(np.zeros_like(p.data)) for p in primals]
    return out, cotangents


def jvp(f, primals, tangents):
    """Forward mode: returns ``(output, tangent_of_output)``. Implemented by
    reverse-mode over each output coordinate (JVP = sum_rows(grad_k * tangent))."""
    primals = [_as_tensor(p) for p in primals]
    tangents = [_as_tensor(t) for t in tangents]
    for p in primals:
        p.requires_grad_(True)
    out = f(*primals)
    out_data = out.data
    jvp_out = np.zeros_like(out_data)
    n = out.numel
    if n == 1:
            _zeros_grad(primals)
            _wrap_tensor_backward(out, Tensor(np.ones_like(out_data)))
            acc = 0.0
            for p, t in zip(primals, tangents):
                if p.grad is None:
                    continue
                acc = acc + float(np.sum(p.grad.data * t.data))
            jvp_out = np.array(acc)
    else:
        flat = jvp_out.reshape(-1)
        for k in range(n):
            seed = np.zeros_like(out_data)
            seed.reshape(-1)[k] = 1.0
            _zeros_grad(primals)
            _wrap_tensor_backward(out, Tensor(seed))
            for p, t in zip(primals, tangents):
                if p.grad is None:
                    continue
                flat[k] = flat[k] + float(np.sum(p.grad.data * t.data))
    return out, Tensor(jvp_out, dtype=out.dtype)


def jacrev(f, primals):
    """Reverse-mode Jacobian. Returns a list (one per primal) of Jacobian
    matrices with shape ``(out_dim, primal_dim)``."""
    primals = [_as_tensor(p) for p in primals]
    for p in primals:
        p.requires_grad_(True)
    out = f(*primals)
    out_flat = out.reshape(-1)
    n = out.numel
    rows = []
    for k in range(n):
        seed = np.zeros_like(out.data)
        seed.reshape(-1)[k] = 1.0
        _zeros_grad(primals)
        _wrap_tensor_backward(out, Tensor(seed))
        rows.append([p.grad.data.reshape(-1).copy() if p.grad is not None else np.zeros(p.numel) for p in primals])
    # rows[k][i] = d out_k / d primal_i  (already flattened)
    jacs = []
    for i in range(len(primals)):
        J = np.stack([rows[k][i] for k in range(n)], axis=0)  # (out_dim, primal_dim)
        jacs.append(J)
    return jacs


def jacfwd(f, primals):
    """Forward-mode Jacobian. Returns a list (one per primal) of Jacobian
    matrices with shape ``(out_dim, primal_dim)``."""
    primals = [_as_tensor(p) for p in primals]
    out = f(*primals)
    out_dim = out.numel
    jacs = []
    for i, p in enumerate(primals):
        e = np.zeros(p.numel)
        J = np.zeros((out_dim, p.numel))
        for j in range(p.numel):
            e[:] = 0.0
            e[j] = 1.0
            tangents = [np.zeros_like(pr.data) for pr in primals]
            tangents[i] = e.reshape(p.shape)
            _, jv = jvp(f, primals, [Tensor(t) for t in tangents])
            J[:, j] = jv.data.reshape(-1)
        jacs.append(J)
    return jacs


def grad(f):
    """Returns a function that computes the gradient of scalar-valued ``f``
    w.r.t. its (tensor) arguments. Higher-order ``grad(grad(f))`` is supported
    because each call builds a fresh tape."""
    def _g(*primals):
        ps = [_as_tensor(p) for p in primals]
        for p in ps:
            p.requires_grad_(True)
        out = f(*ps)
        _zeros_grad(ps)
        out.backward(create_graph=True)
        return [p.grad if p.grad is not None else Tensor(np.zeros_like(p.data)) for p in ps]
    return _g


__all__ = ["vjp", "jvp", "jacrev", "jacfwd", "grad"]
