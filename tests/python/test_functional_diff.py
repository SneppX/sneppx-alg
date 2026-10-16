import numpy as np

from SneppX_ALG.interface_bindings import nn_functional as F, Tensor


def _t(*shape):
    return Tensor(np.random.randn(*shape).astype("float32"), requires_grad=True)


def _grads_flow(*tensors):
    return all(t.grad is not None and np.isfinite(t.grad.data).all() for t in tensors)


def test_functional_linear_is_differentiable():
    x = _t(5, 4)
    w = _t(3, 4)
    b = _t(3)
    out = F.linear(x, w, b)
    assert out.requires_grad
    out.sum().backward()
    assert _grads_flow(x, w, b)
    # analytic: for out = x @ w.T, sum -> grad_w each row = sum_n x_n
    expected_w = np.broadcast_to(x.data.sum(axis=0), w.shape)
    np.testing.assert_allclose(w.grad.data, expected_w, atol=1e-4)


def test_functional_prelu_is_differentiable():
    x = _t(2, 4)
    w = _t(4)
    out = F.prelu(x, w)
    assert out.requires_grad
    out.sum().backward()
    assert _grads_flow(x, w)


def test_functional_embedding_is_differentiable():
    idx = Tensor(np.array([[0, 2, 1]], dtype="int64"))
    w = _t(5, 3)
    out = F.embedding(idx, w)
    assert out.requires_grad
    out.sum().backward()
    assert _grads_flow(w)
    # rows 0,1,2 each used once -> grad rows = 1
    np.testing.assert_allclose(w.grad.data[0], 1.0, atol=1e-5)
    np.testing.assert_allclose(w.grad.data[3], 0.0, atol=1e-5)


def test_functional_embedding_bag_is_differentiable():
    idx = Tensor(np.array([0, 1, 2, 0], dtype="int64"))
    w = _t(4, 3)
    off = Tensor(np.array([0, 2, 4], dtype="int64"))
    out = F.embedding_bag(idx, w, off, mode="mean")
    assert out.requires_grad
    out.sum().backward()
    assert _grads_flow(w)
    # bag0 = idx[0:2] = [0,1], bag1 = idx[2:4] = [2,0]; mean -> 1/2 each use
    # row0 used once per bag -> 1/2 + 1/2 = 1.0
    np.testing.assert_allclose(w.grad.data[0], 1.0, atol=1e-5)
    np.testing.assert_allclose(w.grad.data[1], 0.5, atol=1e-5)
    np.testing.assert_allclose(w.grad.data[2], 0.5, atol=1e-5)
    np.testing.assert_allclose(w.grad.data[3], 0.0, atol=1e-5)
