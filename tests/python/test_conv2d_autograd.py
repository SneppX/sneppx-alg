"""Autograd correctness for the differentiable Conv2d Function and nn.Conv2d.

Regression guard for the batched (N>1) convolution bug: verifies that
``ops.Conv2d`` computes the exact forward (matching ``advanced_ops.conv2d``)
and finite-difference-correct gradients w.r.t. input, weight and bias.
"""

import sys
import os
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings import nn
from SneppX_ALG.interface_bindings.advanced_ops import conv2d as ref_conv2d
from SneppX_ALG.interface_bindings.advanced_ops import conv3d as ref_conv3d

from tests.python.test_autograd_correctness import numerical_grads


def _scaled(*shape, seed, scale=0.3):
    np.random.seed(seed)
    return Tensor((np.random.randn(*shape) * scale).astype("float32"))


def _conv2d_fd_check(stride, padding, dilation, N=2, C_in=2, C_out=3, k=3, H=5, W=5):
    x = _scaled(N, C_in, H, W, seed=1)
    w = _scaled(C_out, C_in, k, k, seed=2)
    b = _scaled(C_out, seed=3)
    inputs = [x, w, b]
    for t in inputs:
        t.requires_grad_(True)
        t.grad = None

    def f(I):
        y = ops.Conv2d.apply(I[0], I[1], I[2], stride, padding, dilation, 1)
        return (y * y).sum()

    out = f(inputs)
    out.backward()
    analytic = [t.grad.data.copy() for t in inputs]
    numeric = numerical_grads(f, inputs)
    for a, n in zip(analytic, numeric):
        assert np.allclose(a, n, atol=1e-2, rtol=1e-2), (
            f"conv2d(s={stride},p={padding},d={dilation}) grad mismatch "
            f"max|diff|={np.max(np.abs(a - n)):.3e}"
        )


def test_conv2d_forward_matches_advanced_ops():
    x = _scaled(3, 2, 5, 5, seed=4)
    w = _scaled(4, 2, 3, 3, seed=5)
    b = _scaled(4, seed=6)
    out = ops.Conv2d.apply(x, w, b, 2, 1, 1, 1)
    ref = ref_conv2d(x, w, b, stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1)
    assert out.shape == ref.shape
    assert np.allclose(
        np.asarray(out.data, np.float64), np.asarray(ref.data, np.float64), atol=1e-6
    )


def test_conv2d_grad_stride1_pad0():
    _conv2d_fd_check(1, 0, 1)


def test_conv2d_grad_stride2_pad1():
    _conv2d_fd_check(2, 1, 1)


def test_conv2d_grad_dilation2():
    _conv2d_fd_check(1, 1, 2)


def test_conv2d_grad_batch3():
    _conv2d_fd_check(2, 2, 1, N=3)


def test_nn_conv2d_is_differentiable():
    m = nn.Conv2d(2, 3, 3, padding=1)
    x = Tensor(np.random.randn(2, 2, 5, 5).astype("float32"))
    x.requires_grad_(True)
    y = m(x)
    assert y.requires_grad
    assert y.grad_fn is not None
    y.sum().backward()
    assert m.weight.grad is not None
    assert x.grad is not None


def test_nn_conv2d_batch_training_step():
    # one SGD step must change the weights and reduce the loss
    np.random.seed(7)
    m = nn.Conv2d(2, 3, 3, padding=1)
    x = Tensor(np.random.randn(4, 2, 5, 5).astype("float32"))
    target = Tensor(np.zeros((4, 3, 5, 5), dtype="float32"))

    def loss_fn():
        y = m(x)
        d = y - target
        return (d * d).mean()

    l0 = float(loss_fn().data)
    loss = loss_fn()
    loss.backward()
    g = np.asarray(m.weight.grad.data, np.float64).copy()
    # manual SGD step
    m.weight = Tensor(
        (np.asarray(m.weight.data, np.float64) - 0.1 * g).astype("float32")
    )
    l1 = float(loss_fn().data)
    assert l1 < l0


def _conv3d_fd_check(
    stride, padding, dilation, N=2, C_in=2, C_out=3, k=2, D=4, H=4, W=4
):
    x = _scaled(N, C_in, D, H, W, seed=1)
    w = _scaled(C_out, C_in, k, k, k, seed=2)
    b = _scaled(C_out, seed=3)
    inputs = [x, w, b]
    for t in inputs:
        t.requires_grad_(True)
        t.grad = None

    def f(I):
        y = ops.Conv3d.apply(I[0], I[1], I[2], stride, padding, dilation, 1)
        return (y * y).sum()

    out = f(inputs)
    out.backward()
    analytic = [t.grad.data.copy() for t in inputs]
    numeric = numerical_grads(f, inputs)
    for a, n in zip(analytic, numeric):
        assert np.allclose(a, n, atol=1e-2, rtol=1e-2), (
            f"conv3d(s={stride},p={padding},d={dilation}) grad mismatch "
            f"max|diff|={np.max(np.abs(a - n)):.3e}"
        )


def test_conv3d_forward_matches_advanced_ops():
    x = _scaled(2, 2, 4, 4, 4, seed=4)
    w = _scaled(3, 2, 2, 2, 2, seed=5)
    b = _scaled(3, seed=6)
    out = ops.Conv3d.apply(x, w, b, 1, 0, 1, 1)
    ref = ref_conv3d(
        x, w, b, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1
    )
    assert out.shape == ref.shape
    assert np.allclose(
        np.asarray(out.data, np.float64), np.asarray(ref.data, np.float64), atol=1e-6
    )


def test_conv3d_grad_stride1_pad0():
    _conv3d_fd_check(1, 0, 1)


def test_conv3d_grad_stride2_pad1():
    _conv3d_fd_check(2, 1, 1)


def test_nn_conv3d_is_differentiable():
    m = nn.Conv3d(2, 3, 2)
    x = Tensor(np.random.randn(2, 2, 4, 4, 4).astype("float32"))
    x.requires_grad_(True)
    y = m(x)
    assert y.requires_grad
    y.sum().backward()
    assert m.weight.grad is not None
    assert x.grad is not None


def test_nn_conv1d_is_differentiable():
    m = nn.Conv1d(2, 3, 3, padding=1)
    x = Tensor(np.random.randn(2, 2, 7).astype("float32"))
    x.requires_grad_(True)
    y = m(x)
    assert y.requires_grad
    y.sum().backward()
    assert m.weight.grad is not None
    assert m.bias.grad is not None
    assert x.grad is not None


def test_nn_conv1d_matches_advanced_ops():
    from SneppX_ALG.interface_bindings.advanced_ops import conv1d as ref_conv1d

    m = nn.Conv1d(2, 3, 3, padding=1)
    x = Tensor(np.random.randn(3, 2, 7).astype("float32"))
    y = m(x)
    ref = ref_conv1d(x, m.weight, m.bias, stride=1, padding=1, dilation=1, groups=1)
    assert y.shape == ref.shape
    assert np.allclose(
        np.asarray(y.data, np.float64), np.asarray(ref.data, np.float64), atol=1e-6
    )


def test_nn_conv1d_bias_grad_finite_difference():
    from SneppX_ALG.interface_bindings.advanced_ops import conv1d as ref_conv1d

    np.random.seed(21)
    m = nn.Conv1d(2, 3, 3, padding=1)
    x = Tensor((np.random.randn(2, 2, 6) * 0.3).astype("float32"))
    x.requires_grad_(True)
    y = m(x)
    (y * y).sum().backward()
    ana_b = np.asarray(m.bias.grad.data, np.float64).copy()

    b0 = np.asarray(m.bias.data, np.float64).copy()
    eps = 1e-3
    num_b = np.zeros_like(b0)

    def l_of(b):
        out = ref_conv1d(x, m.weight, Tensor(b.astype("float32")), stride=1, padding=1)
        return float((np.asarray(out.data, np.float64) ** 2).sum())

    for i in range(b0.size):
        bp = b0.copy()
        bp[i] += eps
        bm = b0.copy()
        bm[i] -= eps
        num_b[i] = (l_of(bp) - l_of(bm)) / (2 * eps)
    assert np.allclose(ana_b, num_b, atol=1e-2, rtol=1e-2), (
        f"conv1d bias grad mismatch max|diff|={np.max(np.abs(ana_b - num_b)):.3e}"
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL CONV2D AUTOGRAD TESTS PASSED")
