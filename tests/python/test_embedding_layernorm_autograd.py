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
from SneppX_ALG.interface_bindings import autograd_ops as ops
from tests.python.test_autograd_correctness import numerical_grads


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


def test_rmsnorm_is_differentiable():
    rn = nn.RMSNorm(8)
    x = Tensor(np.random.randn(2, 5, 8).astype("float32"))
    x.requires_grad_(True)
    y = rn(x)
    assert y.requires_grad
    y.sum().backward()
    assert rn.weight.grad is not None
    assert x.grad is not None


def test_rmsnorm_grad_finite_difference():
    np.random.seed(9)
    rn = nn.RMSNorm(6, eps=1e-6)
    x = Tensor((np.random.randn(2, 4, 6) * 0.5).astype("float32"))
    x.requires_grad_(True)
    (rn(x) * rn(x)).sum().backward()
    ana = np.asarray(x.grad.data, np.float64).copy()

    x0 = np.asarray(x.data, np.float64).copy()
    w = np.asarray(rn.weight.data, np.float64)
    eps = 1e-3

    def loss(xv):
        arr = xv.astype(np.float64)
        rms = np.sqrt((arr**2).mean(-1, keepdims=True) + 1e-6)
        out = arr / rms * w
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
        f"rmsnorm input grad mismatch max|diff|={np.max(np.abs(ana - num)):.3e}"
    )


def test_permute_is_differentiable():
    x = Tensor(np.random.randn(2, 3, 4).astype("float32"))
    x.requires_grad_(True)
    y = x.permute(2, 0, 1)
    assert y.shape == (4, 2, 3)
    assert y.requires_grad
    (y * y).sum().backward()
    assert np.allclose(x.grad.data, 2 * x.data, atol=1e-4)


def test_pixel_shuffle_forward_matches_reference():
    ps = nn.PixelShuffle(2)
    xd = np.random.randn(2, 8, 4, 4).astype("float32")
    y = ps(Tensor(xd))
    ref = xd.reshape(2, 2, 2, 2, 4, 4).transpose(0, 1, 4, 2, 5, 3).reshape(2, 2, 8, 8)
    assert y.shape == (2, 2, 8, 8)
    assert np.allclose(y.data, ref)


def test_pixel_shuffle_is_differentiable_and_roundtrips():
    ps = nn.PixelShuffle(2)
    pus = nn.PixelUnshuffle(2)
    x = Tensor(np.random.randn(1, 8, 4, 4).astype("float32"))
    x.requires_grad_(True)
    y = ps(x)
    assert y.requires_grad
    y.sum().backward()
    assert np.allclose(x.grad.data, 1.0)
    assert np.allclose(pus(ps(Tensor(x.data))).data, x.data)


def test_upsample_nearest_is_differentiable():
    u = nn.Upsample(scale_factor=2, mode="nearest")
    x = Tensor(np.random.randn(2, 3, 4, 5).astype("float32"))
    x.requires_grad_(True)
    y = u(x)
    assert y.shape == (2, 3, 8, 10)
    assert y.requires_grad
    y.sum().backward()
    # each input element contributes to 2x2 outputs -> gradient 4
    assert np.allclose(x.grad.data, 4.0)


def test_upsample_nearest_forward_matches_reference():
    from SneppX_ALG.interface_bindings.nn import _zoom_nn

    u = nn.Upsample(scale_factor=2, mode="nearest")
    xd = np.random.randn(2, 3, 4, 5).astype("float32")
    y = u(Tensor(xd))
    ref = _zoom_nn(xd, (8, 10))
    assert np.allclose(y.data, ref)


def test_bilinear_is_differentiable():
    np.random.seed(5)
    x1 = Tensor((np.random.randn(2, 4) * 0.3).astype("float32"))
    x2 = Tensor((np.random.randn(2, 4) * 0.3).astype("float32"))
    for t in (x1, x2):
        t.requires_grad_(True)
        t.grad = None
    m = nn.Bilinear(4, 4, 3)
    y = m(x1, x2)
    assert y.shape == (2, 3)
    assert y.requires_grad
    (y * y).sum().backward()
    assert x1.grad is not None
    assert x2.grad is not None
    assert m.weight.grad is not None
    assert m.bias.grad is not None
    ref = np.einsum("bi,bj,kij->bk", x1.data, x2.data, m.weight.data) + m.bias.data
    assert np.allclose(y.data, ref, atol=1e-5)


def test_bilinear_grad_matches_numeric():
    np.random.seed(7)
    x1 = Tensor((np.random.randn(2, 3) * 0.3).astype("float32"))
    x2 = Tensor((np.random.randn(2, 3) * 0.3).astype("float32"))
    w = Tensor((np.random.randn(2, 3, 3) * 0.3).astype("float32"))
    b = Tensor((np.random.randn(2) * 0.3).astype("float32"))
    for t in (x1, x2, w, b):
        t.requires_grad_(True)
        t.grad = None

    def f(I):
        y = ops.BilinearFn.apply(I[0], I[1], I[2], I[3])
        return (y * y).sum()

    f([x1, x2, w, b]).backward()
    analytic = [t.grad.data.copy() for t in (x1, x2, w, b)]
    numeric = numerical_grads(f, [x1, x2, w, b])
    for a, n in zip(analytic, numeric):
        assert np.allclose(a, n, atol=1e-2, rtol=1e-2)


def test_adaptive_avg_pool1d_is_differentiable_and_matches_reference():
    def ref(x, ol):
        n, c, l = x.shape
        out = np.zeros((n, c, ol))
        for i in range(ol):
            s = int(np.floor(i * l / ol))
            e = int(np.ceil((i + 1) * l / ol))
            if e > s:
                out[:, :, i] = x[:, :, s:e].mean(axis=2)
        return out

    for l, ol in [(6, 2), (7, 3), (5, 4), (9, 2)]:
        xd = np.random.randn(2, 3, l).astype("float32")
        x = Tensor(xd)
        x.requires_grad_(True)
        y = nn.AdaptiveAvgPool1d(ol)(x)
        assert np.allclose(y.data, ref(xd, ol), atol=1e-5)
        assert y.requires_grad
        y.sum().backward()
        assert x.grad is not None


def test_adaptive_avg_pool3d_is_differentiable():
    x = Tensor(np.random.randn(1, 2, 5, 6, 7).astype("float32"))
    x.requires_grad_(True)
    y = nn.AdaptiveAvgPool3d((2, 3, 3))(x)
    assert y.shape == (1, 2, 2, 3, 3)
    assert y.requires_grad
    y.sum().backward()
    assert x.grad is not None


def test_adaptive_avg_pool2d_is_differentiable():
    from SneppX_ALG.interface_bindings.advanced_ops import adaptive_avg_pool2d as ref

    m = nn.AdaptiveAvgPool2d((1, 1))
    xd = np.random.randn(2, 3, 7, 7).astype("float32")
    x = Tensor(xd)
    x.requires_grad_(True)
    y = m(x)
    assert y.shape == (2, 3, 1, 1)
    assert y.requires_grad
    assert np.allclose(y.data, ref(Tensor(xd), (1, 1)).data, atol=1e-6)
    y.sum().backward()
    assert np.allclose(x.grad.data, 1.0 / 49)


def test_adaptive_avg_pool2d_forward_matches_advanced_ops():
    from SneppX_ALG.interface_bindings.advanced_ops import adaptive_avg_pool2d as ref

    xd = np.random.randn(2, 3, 5, 5).astype("float32")
    for osz in [(1, 1), (2, 2), (2, 3)]:
        y = nn.AdaptiveAvgPool2d(osz)(Tensor(xd))
        assert np.allclose(y.data, ref(Tensor(xd), osz).data, atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL EMBEDDING/LAYERNORM AUTOGRAD TESTS PASSED")
