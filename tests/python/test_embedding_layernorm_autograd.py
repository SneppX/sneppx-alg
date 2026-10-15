"""Regression guards: nn.Embedding and nn.LayerNorm must be differentiable.

Both modules previously used raw NumPy forward paths (no autograd graph),
making embedding/layernorm-based models untrainable. They now route through
the autograd ``EmbeddingFn`` / ``LayerNorm`` Functions.
"""

import sys
import os
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import nn


def test_embedding_is_differentiable():
    e = nn.Embedding(10, 4)
    idx = Tensor(np.array([[1, 2], [3, 1]], dtype="int64"))
    y = e(idx)
    assert y.shape == (2, 2, 4)
    assert y.requires_grad
    assert y.grad_fn is not None
    y.sum().backward()
    assert e.weight.grad is not None


def test_embedding_grad_matches_count():
    # gradient of sum over all embeddings = count of each index
    e = nn.Embedding(5, 3)
    idx = Tensor(np.array([0, 0, 2, 4, 2, 2], dtype="int64"))
    e(idx).sum().backward()
    g = np.asarray(e.weight.grad.data, np.float64)
    expected = np.zeros((5, 3), np.float64)
    for i, c in enumerate([2, 0, 3, 0, 1]):
        expected[i] = c
    assert np.allclose(g, expected)


def test_embedding_forward_matches_gather():
    e = nn.Embedding(10, 4)
    idx_np = np.array([[1, 2], [3, 1]], dtype="int64")
    y = e(Tensor(idx_np))
    assert np.allclose(
        np.asarray(y.data, np.float64), np.asarray(e.weight.data, np.float64)[idx_np]
    )


def test_layernorm_is_differentiable():
    ln = nn.LayerNorm(8)
    x = Tensor(np.random.randn(2, 5, 8).astype("float32"))
    x.requires_grad_(True)
    y = ln(x)
    assert y.requires_grad
    y.sum().backward()
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None
    assert x.grad is not None


def test_layernorm_grad_finite_difference():
    np.random.seed(3)
    ln = nn.LayerNorm(6)
    x = Tensor((np.random.randn(2, 4, 6) * 0.5).astype("float32"))
    x.requires_grad_(True)
    (ln(x) * ln(x)).sum().backward()
    ana = np.asarray(x.grad.data, np.float64).copy()

    x0 = np.asarray(x.data, np.float64).copy()
    w = np.asarray(ln.weight.data, np.float64)
    b = np.asarray(ln.bias.data, np.float64)
    eps = 1e-3

    def loss(xv):
        arr = xv.astype(np.float64)
        mean = arr.mean(-1, keepdims=True)
        var = arr.var(-1, keepdims=True)
        out = (arr - mean) / np.sqrt(var + 1e-5) * w + b
        return float((out * out).sum())

    num = np.zeros_like(x0)
    it = np.nditer(x0, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        xp = x0.copy()
        xp[i] += eps
        xm = x0.copy()
        xm[i] -= eps
        num[i] = (loss(xp) - loss(xm)) / (2 * eps)
    assert np.allclose(ana, num, atol=1e-2, rtol=1e-2), (
        f"layernorm input grad mismatch max|diff|={np.max(np.abs(ana - num)):.3e}"
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL EMBEDDING/LAYERNORM AUTOGRAD TESTS PASSED")
