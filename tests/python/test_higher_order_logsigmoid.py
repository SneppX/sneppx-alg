"""Value-based second-order regression for LogSigmoidFn.create_graph backward."""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings.autograd import grad
from SneppX_ALG.interface_bindings.tensor import Tensor


def test_logsigmoid_second_derivative_matches_analytic():
    x_np = np.array([-1.2, -0.3, 0.0, 0.8, 1.5], dtype=np.float64)
    w_np = np.array([0.4, -0.6, 1.1, 0.3, -0.9], dtype=np.float64)
    v_np = np.array([-0.7, 0.2, 0.5, -0.4, 0.8], dtype=np.float64)

    x = Tensor(x_np.copy(), dtype="float64")
    x.requires_grad_(True)
    out = ops.LogSigmoidFn.apply(x)
    loss = (out * Tensor(w_np.copy(), dtype="float64")).sum()
    g1 = grad(loss, x, create_graph=True)[0]

    scalar = (g1 * Tensor(v_np.copy(), dtype="float64")).sum()
    scalar.backward()

    sig_neg = 1.0 / (1.0 + np.exp(x_np))
    expected = -v_np * w_np * sig_neg * (1.0 - sig_neg)
    assert np.allclose(np.asarray(x.grad.data), expected, atol=1e-6)
