"""Regression tests for batched higher-order Softmax backward.

The create_graph branch previously used ``Sum.apply(og, dim)`` which drops
the reduced axis, so a (N, C) cotangent failed to broadcast against (N,).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings.autograd import grad
from SneppX_ALG.interface_bindings.tensor import Tensor


def _first_grad(x_np, w_np):
    x = Tensor(x_np.copy(), dtype="float64")
    x.requires_grad_(True)
    sm = ops.Softmax.apply(x, -1)
    loss = (sm * Tensor(w_np.copy(), dtype="float64")).sum()
    return grad(loss, x, create_graph=True)[0], sm.data


def test_softmax_batched_create_graph_first_order():
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    w_np = np.array([[0.3, -0.2, 0.5], [0.1, 0.7, -0.4]], dtype=np.float64)
    g1, s = _first_grad(x_np, w_np)
    expected = s * (w_np - (s * w_np).sum(axis=-1, keepdims=True))
    assert np.allclose(g1.data, expected, atol=1e-6)


def test_softmax_batched_create_graph_second_order():
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    w_np = np.array([[0.3, -0.2, 0.5], [0.1, 0.7, -0.4]], dtype=np.float64)
    v_np = np.array([[0.2, 0.4, -0.1], [0.3, -0.5, 0.6]], dtype=np.float64)

    def scalar_at(arr):
        g1 = _first_grad(arr, w_np)[0]
        return float(np.sum(np.asarray(g1.data) * v_np))

    eps = 1e-5
    fd = np.zeros_like(x_np)
    for i in range(x_np.shape[0]):
        for j in range(x_np.shape[1]):
            xp = x_np.copy()
            xm = x_np.copy()
            xp[i, j] += eps
            xm[i, j] -= eps
            fd[i, j] = (scalar_at(xp) - scalar_at(xm)) / (2.0 * eps)

    x = Tensor(x_np.copy(), dtype="float64")
    x.requires_grad_(True)
    sm = ops.Softmax.apply(x, -1)
    loss = (sm * Tensor(w_np.copy(), dtype="float64")).sum()
    g1 = grad(loss, x, create_graph=True)[0]
    (g1 * Tensor(v_np.copy(), dtype="float64")).sum().backward()
    assert np.allclose(np.asarray(x.grad.data), fd, atol=3e-4)
