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


def test_functional_conv1d_is_differentiable():
    x = _t(2, 3, 8)
    w = _t(4, 3, 3)
    b = _t(4)
    out = F.conv1d(x, w, b)
    assert out.requires_grad
    out.sum().backward()
    assert _grads_flow(x, w, b)


def test_functional_conv2d_is_differentiable():
    x = _t(2, 3, 8, 8)
    w = _t(4, 3, 3, 3)
    b = _t(4)
    out = F.conv2d(x, w, b, padding=1)
    assert out.requires_grad
    out.sum().backward()
    assert _grads_flow(x, w, b)


def test_functional_conv3d_is_differentiable():
    x = _t(1, 3, 4, 4, 4)
    w = _t(4, 3, 3, 3, 3)
    b = _t(4)
    out = F.conv3d(x, w, b, padding=1)
    assert out.requires_grad
    out.sum().backward()
    assert _grads_flow(x, w, b)


def test_functional_conv_transpose2d_is_differentiable():
    x = _t(2, 3, 6, 6)
    w = _t(3, 4, 3, 3)
    b = _t(4)
    out = F.conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=1)
    assert out.requires_grad
    out.sum().backward()
    assert _grads_flow(x, w, b)


def test_functional_conv_transpose1d_is_differentiable():
    x = _t(2, 3, 6)
    w = _t(3, 4, 3)
    out = F.conv_transpose1d(x, w, stride=2, padding=1, output_padding=1)
    assert out.requires_grad
    out.sum().backward()
    assert _grads_flow(x, w)


def test_functional_conv2d_matches_nn_module_value():
    from SneppX_ALG.interface_bindings import nn

    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    x = _t(2, 3, 8, 8)
    y_f = F.conv2d(x, conv.weight, conv.bias, padding=1)
    y_m = conv(x)
    np.testing.assert_allclose(y_f.data, y_m.data, atol=1e-5)


def test_functional_pools_are_differentiable():
    from SneppX_ALG.interface_bindings import nn

    cases = [
        ("max_pool1d", F.max_pool1d, nn.MaxPool1d(2), (2, 3, 8)),
        ("max_pool2d", F.max_pool2d, nn.MaxPool2d(2), (2, 3, 8, 8)),
        ("max_pool3d", F.max_pool3d, nn.MaxPool3d(2), (1, 3, 4, 4, 4)),
        ("avg_pool1d", F.avg_pool1d, nn.AvgPool1d(2), (2, 3, 8)),
        ("avg_pool2d", F.avg_pool2d, nn.AvgPool2d(2), (2, 3, 8, 8)),
        ("avg_pool3d", F.avg_pool3d, nn.AvgPool3d(2), (1, 3, 4, 4, 4)),
    ]
    for name, fn, mod, shape in cases:
        x = Tensor(np.random.randn(*shape).astype("float32"), requires_grad=True)
        out = fn(x, 2)
        assert out.requires_grad, f"{name} not differentiable"
        out.sum().backward()
        assert x.grad is not None and np.isfinite(x.grad.data).all(), name
        # value must match the nn module
        np.testing.assert_allclose(out.data, mod(x).data, atol=1e-5, err_msg=name)
