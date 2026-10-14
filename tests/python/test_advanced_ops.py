"""Tests for Advanced Tensor Operations."""

import numpy as np
from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings.advanced_ops import (
    conv2d,
    conv1d,
    max_pool2d,
    avg_pool2d,
    adaptive_avg_pool2d,
    rnn_cell,
    lstm_cell,
    gru_cell,
    multi_head_attention,
    softmax,
    silu,
    layernorm,
    linear,
    dropout,
    embedding,
    cat,
    stack,
    split,
    flatten,
    sum,
    mean,
    max,
    min,
    norm,
    clamp,
    pad,
    permute,
    transpose,
    relu,
    leaky_relu,
    sigmoid,
)


def test_conv2d_basic():
    x = Tensor.ones((1, 1, 4, 4))
    w = Tensor.ones((1, 1, 3, 3))
    out = conv2d(x, w, stride=(1, 1), padding=(0, 0))
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out.data, 9.0)


def test_conv2d_with_padding():
    x = Tensor.ones((1, 1, 4, 4))
    w = Tensor.ones((1, 1, 3, 3))
    out = conv2d(x, w, padding=(1, 1))
    assert out.shape == (1, 1, 4, 4)


def test_conv2d_with_bias():
    x = Tensor.ones((1, 1, 4, 4))
    w = Tensor.ones((1, 1, 3, 3))
    b = Tensor.ones((1,)) * 2.0
    out = conv2d(x, w, bias=b, stride=(1, 1), padding=(0, 0))
    assert np.allclose(out.data, 11.0)


def test_conv2d_stride2():
    x = Tensor.ones((1, 1, 6, 6))
    w = Tensor.ones((1, 1, 2, 2))
    out = conv2d(x, w, stride=(2, 2))
    assert out.shape == (1, 1, 3, 3)


def test_conv1d_basic():
    x = Tensor.ones((1, 1, 8))
    w = Tensor.ones((1, 1, 3))
    out = conv1d(x, w, stride=1, padding=0)
    assert out.shape == (1, 1, 6)
    assert np.allclose(out.data, 3.0)


def _direct_conv1d(x, wf, stride, padding):
    N, C_in, L = x.shape
    C_out, _, kL = wf.shape
    L_out = (L + 2 * padding - (kL - 1) - 1) // stride + 1
    y = np.zeros((N, C_out, L_out))
    for n in range(N):
        for oc in range(C_out):
            for i in range(L_out):
                for k in range(kL):
                    t = i * stride - padding + k
                    if 0 <= t < L:
                        y[n, oc, i] += np.dot(x[n, :, t], wf[oc, :, k])
    return y


def test_conv1d_batch_greater_than_one():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 2, 8))
    wf = rng.standard_normal((4, 2, 3))
    out = conv1d(Tensor(x), Tensor(wf), stride=2, padding=1)
    assert out.shape == (3, 4, 4)
    assert np.allclose(
        np.asarray(out.data, np.float64), _direct_conv1d(x, wf, 2, 1), atol=1e-5
    )


def _direct_conv2d(x, wf, stride, padding):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = wf.shape
    Ho = (H + 2 * padding[0] - (kH - 1) - 1) // stride[0] + 1
    Wo = (W + 2 * padding[1] - (kW - 1) - 1) // stride[1] + 1
    y = np.zeros((N, C_out, Ho, Wo))
    for n in range(N):
        for oc in range(C_out):
            for i in range(Ho):
                for j in range(Wo):
                    for a in range(kH):
                        for b in range(kW):
                            hh = i * stride[0] - padding[0] + a
                            ww = j * stride[1] - padding[1] + b
                            if 0 <= hh < H and 0 <= ww < W:
                                y[n, oc, i, j] += np.dot(
                                    x[n, :, hh, ww], wf[oc, :, a, b]
                                )
    return y


def test_conv2d_batch_greater_than_one():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((2, 3, 6, 6))
    wf = rng.standard_normal((5, 3, 3, 3))
    out = conv2d(Tensor(x), Tensor(wf), stride=(1, 1), padding=(1, 1))
    assert out.shape == (2, 5, 6, 6)
    assert np.allclose(
        np.asarray(out.data, np.float64),
        _direct_conv2d(x, wf, (1, 1), (1, 1)),
        atol=1e-5,
    )


def test_conv3d_batch_greater_than_one():
    from SneppX_ALG.interface_bindings.advanced_ops import conv3d

    rng = np.random.default_rng(9)
    x = rng.standard_normal((2, 2, 4, 4, 4))
    wf = rng.standard_normal((3, 2, 2, 2, 2))
    out = conv3d(Tensor(x), Tensor(wf), stride=(1, 1, 1), padding=(1, 1, 1))
    assert out.shape == (2, 3, 5, 5, 5)
    # batch elements must not be permuted/mixed: each sample matches its own
    # direct conv computed independently.
    for n in range(2):
        ref = _direct_conv3d(x[n : n + 1], wf, (1, 1, 1), (1, 1, 1))
        assert np.allclose(np.asarray(out.data, np.float64)[n : n + 1], ref, atol=1e-5)


def _direct_conv3d(x, wf, stride, padding):
    N, C_in, D, H, W = x.shape
    C_out, _, kD, kH, kW = wf.shape
    Do = (D + 2 * padding[0] - (kD - 1) - 1) // stride[0] + 1
    Ho = (H + 2 * padding[1] - (kH - 1) - 1) // stride[1] + 1
    Wo = (W + 2 * padding[2] - (kW - 1) - 1) // stride[2] + 1
    y = np.zeros((N, C_out, Do, Ho, Wo))
    for n in range(N):
        for oc in range(C_out):
            for i in range(Do):
                for j in range(Ho):
                    for k in range(Wo):
                        for a in range(kD):
                            for b in range(kH):
                                for c in range(kW):
                                    dd = i * stride[0] - padding[0] + a
                                    hh = j * stride[1] - padding[1] + b
                                    ww = k * stride[2] - padding[2] + c
                                    if 0 <= dd < D and 0 <= hh < H and 0 <= ww < W:
                                        y[n, oc, i, j, k] += np.dot(
                                            x[n, :, dd, hh, ww], wf[oc, :, a, b, c]
                                        )
    return y


def test_max_pool2d():
    x = Tensor.from_numpy(
        np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
    )
    out = max_pool2d(x, kernel_size=2, stride=1)
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out.data, [[[[5, 6], [8, 9]]]])


def test_avg_pool2d():
    x = Tensor.ones((1, 1, 4, 4))
    out = avg_pool2d(x, kernel_size=2)
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out.data, 1.0)


def test_adaptive_avg_pool2d():
    x = Tensor.ones((1, 1, 8, 8))
    out = adaptive_avg_pool2d(x, (4, 4))
    assert out.shape == (1, 1, 4, 4)


def test_rnn_cell():
    x = Tensor.ones((2, 4))
    w_ih = Tensor.ones((8, 4))
    w_hh = Tensor.ones((8, 8))
    h = rnn_cell(x, None, w_ih, w_hh, nonlinearity="tanh")
    assert h.shape == (2, 8)


def test_lstm_cell():
    x = Tensor.ones((2, 4))
    w_ih = Tensor.ones((16, 4))
    w_hh = Tensor.ones((16, 4))
    h, c = lstm_cell(x, None, None, w_ih, w_hh)
    assert h.shape == (2, 4)
    assert c.shape == (2, 4)


def test_gru_cell():
    x = Tensor.ones((2, 4))
    w_ih = Tensor.ones((12, 4))
    w_hh = Tensor.ones((12, 4))
    h = gru_cell(x, None, w_ih, w_hh)
    assert h.shape == (2, 4)


def test_multi_head_attention():
    q = Tensor.ones((1, 4, 8))
    k = Tensor.ones((1, 4, 8))
    v = Tensor.ones((1, 4, 8))
    out = multi_head_attention(q, k, v, num_heads=2)
    assert out.shape == (1, 4, 8)


def test_softmax():
    x = Tensor.from_numpy(np.array([[1.0, 2.0, 3.0]]))
    out = softmax(x.data)
    assert np.allclose(out.sum(), 1.0, atol=1e-6)


def test_linear():
    x = Tensor.ones((2, 4))
    w = Tensor.ones((3, 4))
    out = linear(x, w)
    assert out.shape == (2, 3)
    assert np.allclose(out.data, 4.0)


def test_dropout():
    x = Tensor.ones((100, 100))
    out = dropout(x, p=0.5, training=True)
    assert out.shape == (100, 100)
    assert abs(out.data.mean() - 1.0) < 0.1


def test_embedding():
    indices = Tensor.from_numpy(np.array([[0, 1], [2, 0]]))
    w = Tensor.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    out = embedding(indices, w)
    assert out.shape == (2, 2, 2)


def test_cat():
    a = Tensor.ones((2, 3))
    b = Tensor.ones((2, 3)) * 2
    out = cat([a, b], dim=1)
    assert out.shape == (2, 6)


def test_stack():
    a = Tensor.ones((2, 3))
    b = Tensor.ones((2, 3)) * 2
    out = stack([a, b], dim=0)
    assert out.shape == (2, 2, 3)


def test_flatten():
    x = Tensor.ones((2, 3, 4))
    out = flatten(x, start_dim=1)
    assert out.shape == (2, 12)


def test_clamp():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0, 2.0]))
    out = clamp(x, min=0.0, max=1.0)
    assert np.allclose(out.data, [0.0, 0.0, 1.0, 1.0])


def test_gelu_nn():
    from SneppX_ALG.interface_bindings.nn import GELU

    x = Tensor.ones((2, 3))
    out = GELU()(x)
    assert out.shape == (2, 3)
    assert np.allclose(out.data, 0.8413, atol=1e-3)


def test_silu():
    x = Tensor.ones((2, 3))
    out = silu(x)
    assert out.shape == (2, 3)


def test_relu():
    x = Tensor.from_numpy(np.array([-1.0, 0.0, 1.0]))
    out = relu(x)
    assert np.allclose(out.data, [0.0, 0.0, 1.0])


def test_leaky_relu():
    x = Tensor.from_numpy(np.array([-1.0, 1.0]))
    out = leaky_relu(x)
    assert out.data[0] == -0.01


def test_sum():
    x = Tensor.ones((2, 3))
    out = sum(x)
    assert np.allclose(out.data, 6.0)


def test_mean():
    x = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    out = mean(x, dim=1)
    assert np.allclose(out.data, [1.5, 3.5])


def test_max():
    x = Tensor.from_numpy(np.array([[1.0, 5.0], [3.0, 2.0]]))
    values, indices = max(x, dim=1)
    assert np.allclose(values.data, [5.0, 3.0])


def test_norm():
    x = Tensor.from_numpy(np.array([3.0, 4.0]))
    out = norm(x, p=2)
    assert np.allclose(out.data, 5.0)


def test_pad():
    x = Tensor.ones((2, 2))
    out = pad(x, (1, 1, 1, 1))
    assert out.shape == (4, 4)


def test_permute():
    x = Tensor.ones((2, 3, 4))
    out = permute(x, (2, 0, 1))
    assert out.shape == (4, 2, 3)


def test_transpose():
    x = Tensor.ones((2, 3))
    out = transpose(x, 0, 1)
    assert out.shape == (3, 2)


if __name__ == "__main__":
    import sys

    locals_ = locals().copy()
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed += 1
    print(f"\n{'=' * 50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
