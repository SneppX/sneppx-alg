"""Regression: CrossEntropyLoss backward must preserve the input dtype."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings.tensor import Tensor


def test_cross_entropy_backward_preserves_float64():
    logits_np = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]], dtype=np.float64)
    target = np.array([2, 1], dtype=np.int64)

    logits = Tensor(logits_np.copy(), dtype="float64")
    logits.requires_grad_(True)
    loss = ops.CrossEntropyLoss.apply(logits, target)
    loss.backward()

    assert logits.grad.dtype == "float64"

    x = logits_np - logits_np.max(axis=-1, keepdims=True)
    e = np.exp(x)
    sm = e / e.sum(axis=-1, keepdims=True)
    expected = sm.copy()
    expected[np.arange(len(target)), target] -= 1.0
    expected /= len(target)
    assert np.allclose(np.asarray(logits.grad.data), expected, atol=1e-12)
