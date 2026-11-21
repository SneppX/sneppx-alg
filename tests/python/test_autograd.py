"""Tests for the autograd engine and differentiable ops."""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

import numpy as np
from SneppX_ALG.interface_bindings.tensor import Tensor


def _finite_diff(fn, inp, eps=1e-4):
    """Numerical gradient for a function that takes and returns a Tensor."""
    x0 = inp.data.copy()
    grad = np.zeros_like(x0)
    it = np.nditer(x0, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        xp = x0.copy()
        xp[idx] += eps
        xn = x0.copy()
        xn[idx] -= eps
        fp = fn(Tensor(xp, dtype=inp.dtype)).data
        fn_ = fn(Tensor(xn, dtype=inp.dtype)).data
        if fp.ndim == fn_.ndim == 0:
            grad[idx] = (float(fp) - float(fn_)) / (2 * eps)
        else:
            grad[idx] = (float(fp.sum()) - float(fn_.sum())) / (2 * eps)
        it.iternext()
    return grad


def _check_grad(op_fn, *inputs, atol=1.5e-2):
    """Compare analytical vs numerical gradient for a single-input op."""
    inp = inputs[0]
    inp.requires_grad = True
    out = op_fn(inp)
    loss = out.sum()
    loss.backward()
    num_g = _finite_diff(op_fn, inp)
    err = np.abs(inp.grad.data - num_g).max()
    assert err < atol, f"Gradient mismatch: max err={err:.6f} > {atol}"
    return err


def _check_grad_binary(op_fn, a, b, atol=1e-2):
    """Compare gradients for binary ops."""
    a.requires_grad = True
    b.requires_grad = True
    out = op_fn(a, b)
    loss = out.sum()
    loss.backward()

    def fn_a(x):
        return op_fn(x, b.detach())

    def fn_b(x):
        return op_fn(a.detach(), x)

    num_ga = _finite_diff(fn_a, a)
    num_gb = _finite_diff(fn_b, b)
    err_a = np.abs(a.grad.data - num_ga).max()
    err_b = np.abs(b.grad.data - num_gb).max()
    assert err_a < atol, f"Grad a mismatch: {err_a:.6f}"
    assert err_b < atol, f"Grad b mismatch: {err_b:.6f}"
    return max(err_a, err_b)


class TestAutogradArithmetic:
    def test_add(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        b = Tensor(np.random.randn(3, 4), dtype="float32")
        _check_grad_binary(lambda x, y: x + y, a, b)

    def test_add_scalar(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        _check_grad(lambda x: x + 2.0, a)

    def test_sub(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        b = Tensor(np.random.randn(3, 4), dtype="float32")
        _check_grad_binary(lambda x, y: x - y, a, b)

    def test_mul(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        b = Tensor(np.random.randn(3, 4), dtype="float32")
        _check_grad_binary(lambda x, y: x * y, a, b)

    def test_div(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32") + 5.0
        b = Tensor(np.random.randn(3, 4), dtype="float32") + 5.0
        _check_grad_binary(lambda x, y: x / y, a, b)

    def test_neg(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        _check_grad(lambda x: -x, a)

    def test_pow(self):
        a = Tensor(np.abs(np.random.randn(3, 4)) + 0.5, dtype="float32")
        _check_grad(lambda x: x**3, a, atol=5e-2)

    def test_matmul(self):
        a = Tensor(np.random.randn(2, 3), dtype="float32")
        b = Tensor(np.random.randn(3, 4), dtype="float32")
        _check_grad_binary(lambda x, y: x @ y, a, b)

    def test_matmul_2d(self):
        a = Tensor(np.random.randn(5, 8), dtype="float32")
        b = Tensor(np.random.randn(8, 3), dtype="float32")
        a.requires_grad = True
        b.requires_grad = True
        out = a @ b
        loss = out.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape


class TestAutogradReductions:
    def test_sum(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        a.requires_grad = True
        out = a.sum()
        out.backward()
        assert a.grad is not None
        assert np.allclose(a.grad.data, np.ones_like(a.data))

    def test_sum_dim(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        a.requires_grad = True
        out = a.sum(dim=0)
        out.backward(Tensor(np.ones_like(out.data), dtype="float32"))
        assert a.grad is not None
        assert a.grad.shape == a.shape

    def test_mean(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        a.requires_grad = True
        out = a.mean()
        out.backward()
        assert a.grad is not None
        assert np.allclose(a.grad.data, np.ones_like(a.data) / a.numel)


class TestAutogradActivations:
    def test_relu(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        _check_grad(lambda x: x.relu(), a)

    def test_sigmoid(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32") * 0.5
        _check_grad(lambda x: x.sigmoid(), a)

    def test_tanh(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32") * 0.5
        _check_grad(lambda x: x.tanh(), a)

    def test_gelu(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32") * 0.5
        _check_grad(lambda x: x.gelu(), a, atol=5e-3)

    def test_silu(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32") * 0.5
        _check_grad(lambda x: x.silu(), a, atol=5e-3)

    def test_softmax(self):
        a = Tensor(np.random.randn(2, 3), dtype="float32")
        _check_grad(lambda x: x.softmax(dim=-1), a, atol=5e-3)

    def test_log_softmax(self):
        a = Tensor(np.random.randn(2, 3), dtype="float32")
        _check_grad(lambda x: x.log_softmax(dim=-1), a, atol=1e-2)


class TestAutogradUnary:
    def test_sqrt(self):
        a = Tensor(np.abs(np.random.randn(3, 4)) + 0.5, dtype="float32")
        _check_grad(lambda x: x.sqrt(), a)

    def test_exp(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32") * 0.5
        _check_grad(lambda x: x.exp(), a)

    def test_log(self):
        a = Tensor(np.abs(np.random.randn(3, 4)) + 0.5, dtype="float32")
        _check_grad(lambda x: x.log(), a)

    def test_abs(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        _check_grad(lambda x: x.abs(), a)


class TestAutogradShape:
    def test_reshape(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        a.requires_grad = True
        out = a.reshape(12)
        loss = out.sum()
        loss.backward()
        assert a.grad is not None
        assert a.grad.shape == a.shape

    def test_transpose(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        a.requires_grad = True
        out = a.transpose(0, 1)
        loss = out.sum()
        loss.backward()
        assert a.grad is not None
        assert a.grad.shape == a.shape

    def test_squeeze(self):
        a = Tensor(np.random.randn(1, 3, 1, 4), dtype="float32")
        a.requires_grad = True
        out = a.squeeze()
        loss = out.sum()
        loss.backward()
        assert a.grad is not None
        assert a.grad.shape == a.shape

    def test_unsqueeze(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        a.requires_grad = True
        out = a.unsqueeze(0)
        loss = out.sum()
        loss.backward()
        assert a.grad is not None
        assert a.grad.shape == a.shape

    def test_getitem(self):
        a = Tensor(np.random.randn(3, 4), dtype="float32")
        a.requires_grad = True
        out = a[1]
        loss = out.sum()
        loss.backward()
        assert a.grad is not None
        assert a.grad.shape == a.shape
        assert np.allclose(a.grad.data[1], np.ones(4))


class TestAutogradGraph:
    def test_multi_chain(self):
        a = Tensor(np.random.randn(5), dtype="float32")
        b = Tensor(np.random.randn(5), dtype="float32")
        a.requires_grad = True
        b.requires_grad = True
        x = a * b
        y = x + a
        z = y.relu()
        loss = z.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape

    def test_grad_accumulation(self):
        a = Tensor(np.random.randn(5), dtype="float32")
        a.requires_grad = True
        out1 = a.sum()
        out1.backward()
        g1 = a.grad.data.copy()
        a.zero_grad_()
        out2 = (a * 2).sum()
        out2.backward()
        g2 = a.grad.data.copy()
        assert not np.allclose(g1, g2)

    def test_detach(self):
        a = Tensor(np.random.randn(5), dtype="float32")
        a.requires_grad = True
        b = a.detach()
        assert not b.requires_grad
        out = b.sum()
        assert out.grad_fn is None

    def test_is_leaf(self):
        a = Tensor(np.random.randn(5), dtype="float32", requires_grad=True)
        assert a.is_leaf
        b = a.relu()
        assert not b.is_leaf

    def test_grad_fn_type(self):
        a = Tensor(np.ones(3), dtype="float32", requires_grad=True)
        b = a.relu()
        assert b.grad_fn is not None
        assert hasattr(b.grad_fn, "op_cls")
        assert hasattr(b.grad_fn, "ctx")
        assert hasattr(b.grad_fn, "inputs")

    def test_no_grad_ops(self):
        a = Tensor(np.random.randn(5), dtype="float32", requires_grad=False)
        b = a.relu()
        assert b.grad_fn is None
        out = b.sum()
        out.backward()
        assert a.grad is None


class TestAutogradLosses:
    def test_mse_loss(self):
        inp = Tensor(np.random.randn(4), dtype="float32")
        target = Tensor(np.random.randn(4), dtype="float32")
        inp.requires_grad = True
        loss = inp.mse_loss(target)
        loss.backward()
        assert inp.grad is not None
        assert inp.grad.shape == inp.shape

    def test_cross_entropy(self):
        inp = Tensor(np.random.randn(2, 5), dtype="float32")
        target = Tensor(np.array([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]], dtype="float32"))
        inp.requires_grad = True
        loss = inp.cross_entropy(target)
        loss.backward()
        assert inp.grad is not None
        assert inp.grad.shape == inp.shape


class TestAutogradGradientFlow:
    def test_linear_chain_grad(self):
        a = Tensor(np.random.randn(10), dtype="float32", requires_grad=True)
        b = a * 2.0
        c = b + 1.0
        d = c.relu()
        e = d.sum()
        e.backward()
        expected = np.where((a.data * 2 + 1) > 0, 2.0, 0.0)
        assert np.allclose(
            a.grad.data, expected, atol=1e-5
        ), f"Got {a.grad.data}, expected {expected}"

    def test_multi_output_backward(self):
        a = Tensor(np.random.randn(5), dtype="float32", requires_grad=True)
        b = a * a
        c = a + a
        d = b + c
        loss = d.sum()
        loss.backward()
        expected = 2 * a.data + 2
        assert np.allclose(a.grad.data, expected, atol=1e-5)

    def test_broadcast_in_add(self):
        a = Tensor(np.random.randn(3, 1), dtype="float32", requires_grad=True)
        b = Tensor(np.random.randn(1, 4), dtype="float32", requires_grad=True)
        out = a + b
        loss = out.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
