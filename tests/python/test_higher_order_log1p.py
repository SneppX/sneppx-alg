"""Value-based second-order regression for Log1p.create_graph backward.

The old branch rebuilt ``x`` from ``x.data``, detaching it from the recorded
graph. The first derivative was numerically right but had no dependence on
the input, so a second derivative silently collapsed to zero.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings.autograd import grad
from SneppX_ALG.interface_bindings.tensor import Tensor


def test_log1p_second_derivative_matches_analytic():
    x_np = np.array([0.2, 0.5, 1.0, -0.3], dtype=np.float64)
    w_np = np.array([0.7, -0.2, 0.4, 1.1], dtype=np.float64)
    v_np = np.array([-0.5, 0.3, 0.9, 0.2], dtype=np.float64)

    x = Tensor(x_np.copy(), dtype="float64")
    x.requires_grad_(True)
    out = ops.Log1p.apply(x)
    loss = (out * Tensor(w_np.copy(), dtype="float64")).sum()
    g1 = grad(loss, x, create_graph=True)[0]

    scalar = (g1 * Tensor(v_np.copy(), dtype="float64")).sum()
    scalar.backward()

    expected = -v_np * w_np / (1.0 + x_np) ** 2
    assert np.allclose(np.asarray(x.grad.data), expected, atol=1e-6)
