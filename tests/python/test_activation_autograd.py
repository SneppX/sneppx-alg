import numpy as np
import pytest

from SneppX_ALG.interface_bindings import Tensor, nn


def _fd_grad(build, x0, eps=1e-3, step=3):
    """Central-difference gradient of sum(build(x)**2) sampled along the array."""
    x = Tensor(x0.copy())
    x.requires_grad_(True)
    y = build(x)
    (y * y).sum().backward()
    errs = []
    for i in range(0, x0.size, step):
        idx = np.unravel_index(i, x0.shape)
        a = x0.copy()
        a[idx] += eps
        fp = float(np.asarray((build(Tensor(a)) ** 2).sum().data).sum())
        b = x0.copy()
        b[idx] -= eps
        fm = float(np.asarray((build(Tensor(b)) ** 2).sum().data).sum())
        num = (fp - fm) / (2 * eps)
        ana = float(x.grad.data[idx])
        errs.append(abs(num - ana))
    return max(errs)


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (nn.LeakyReLU, {"negative_slope": 0.2}),
        (nn.ELU, {"alpha": 1.0}),
        (nn.LogSigmoid, {}),
        (nn.PReLU, {"num_parameters": 1, "init": 0.25}),
        (nn.RReLU, {"lower": 0.1, "upper": 0.3}),
    ],
)
def test_activation_is_differentiable(cls, kwargs):
    np.random.seed(1)
    m = cls(**kwargs)
    if hasattr(m, "_training"):
        m._training = False
    x0 = (np.random.randn(2, 5)).astype("float32")
    assert _fd_grad(lambda t: m(t), x0) < 2e-3


def test_leaky_relu_matches_reference_forward():
    np.random.seed(3)
    x0 = np.random.randn(4, 6).astype("float32")
    ref = np.maximum(x0 * 0.05, x0)
    out = nn.LeakyReLU(0.05)(Tensor(x0))
    assert np.allclose(out.data, ref, atol=1e-6)


def test_elu_matches_reference_forward():
    np.random.seed(4)
    x0 = np.random.randn(4, 6).astype("float32")
    ref = np.where(x0 > 0, x0, 1.0 * (np.exp(x0) - 1))
    out = nn.ELU(1.0)(Tensor(x0))
    assert np.allclose(out.data, ref, atol=1e-6)


def test_log_sigmoid_backward_derivative_is_sigmoid_neg():
    np.random.seed(5)
    x0 = np.array([[2.0, -2.0, 0.0]], dtype="float32")
    x = Tensor(x0)
    x.requires_grad_(True)
    y = nn.LogSigmoid()(x)
    y.backward(np.ones_like(x0))
    expected = 1.0 / (1.0 + np.exp(x0.astype(np.float64)))
    assert np.allclose(x.grad.data, expected, atol=1e-6)


@pytest.mark.parametrize(
    "cls,kwargs,shape",
    [
        (nn.ConstantPad2d, {"padding": 2, "value": 1.5}, (1, 3, 5, 5)),
        (nn.ZeroPad1d, {"padding": 1}, (1, 3, 6)),
        (nn.ReflectionPad2d, {"padding": 2}, (1, 3, 5, 5)),
        (nn.ReplicationPad2d, {"padding": 2}, (1, 3, 5, 5)),
        (
            nn.ConstantPad3d,
            {"padding": (1, 1, 0, 1, 0, 0), "value": 0.5},
            (1, 2, 4, 4, 4),
        ),
    ],
)
def test_pad_is_differentiable(cls, kwargs, shape):
    np.random.seed(7)
    m = cls(**kwargs)
    x0 = (np.random.randn(*shape)).astype("float32")
    x = Tensor(x0)
    x.requires_grad_(True)
    y = m(x)
    assert y.requires_grad
    (y * y).sum().backward()
    assert x.grad is not None
    assert np.isfinite(x.grad.data).all()


def test_constant_pad_interior_grad_is_2x():
    np.random.seed(9)
    x0 = (np.random.randn(1, 3, 5, 5)).astype("float32")
    x = Tensor(x0)
    x.requires_grad_(True)
    y = nn.ConstantPad2d(2, 1.5)(x)
    (y * y).sum().backward()
    assert np.allclose(x.grad.data, 2 * x0, atol=1e-5)


def test_dropout_is_differentiable():
    np.random.seed(11)
    m = nn.Dropout(0.5)
    m._training = True
    x0 = (np.random.randn(2, 8) + 3.0).astype("float32")
    x = Tensor(x0)
    x.requires_grad_(True)
    y = m(x)
    assert y.requires_grad
    y.backward(np.ones_like(y.data))
    mask = np.asarray(y.data) / np.asarray(x0)
    assert np.allclose(x.grad.data, mask, atol=1e-5)


def test_dropout_eval_is_identity_differentiable():
    m = nn.Dropout(0.5)
    m._training = False
    x0 = (np.random.randn(3, 4)).astype("float32")
    x = Tensor(x0)
    x.requires_grad_(True)
    y = m(x)
    assert y.requires_grad
    assert np.allclose(y.data, x0)


@pytest.mark.parametrize(
    "cls,shape,args,ptype",
    [
        (nn.Pooling1d, (2, 3, 10), (2, 2, 0, 1), "max"),
        (nn.Pooling1d, (2, 3, 10), (3, 1, 1, 1), "mean"),
        (nn.Pooling2d, (2, 3, 8, 8), (2, 2, 0, 1), "max"),
        (nn.Pooling2d, (2, 2, 6, 6), (3, 2, 1, 1), "mean"),
        (nn.Pooling3d, (1, 2, 5, 5, 5), (2, 2, 0, 1), "max"),
    ],
)
def test_generic_pooling_is_differentiable(cls, shape, args, ptype):
    np.random.seed(13)
    x0 = np.random.randn(*shape).astype("float32")
    m = cls(*args, pool_type=ptype)
    x = Tensor(x0)
    x.requires_grad_(True)
    y = m(x)
    assert y.requires_grad
    (y * y).sum().backward()
    assert np.isfinite(x.grad.data).all()


def test_embedding_bag_sum_grad():
    np.random.seed(15)
    m = nn.EmbeddingBag(6, 3, mode="sum")
    idx = Tensor(np.array([1, 1, 4], dtype="int64"))
    y = m(idx)
    assert y.requires_grad
    y.backward(np.ones_like(y.data))
    gw = m.weight.grad.data
    assert np.allclose(gw[1], 2.0)
    assert np.allclose(gw[4], 1.0)
    assert np.allclose(gw[0], 0.0)


def test_embedding_bag_mean_grad_with_offsets():
    np.random.seed(16)
    m = nn.EmbeddingBag(6, 3, mode="mean")
    idx = Tensor(np.array([0, 2, 4, 1, 3], dtype="int64"))
    off = Tensor(np.array([0, 2, 5], dtype="int64"))
    y = m(idx, off)
    assert y.shape == (2, 3)
    y.backward(np.ones_like(y.data))
    gw = m.weight.grad.data
    assert np.allclose(gw[0], 0.5)
    assert np.allclose(gw[2], 0.5)
    assert np.allclose(gw[1], 1.0 / 3)
    assert np.allclose(gw[4], 1.0 / 3)
    assert np.allclose(gw[3], 1.0 / 3)


def test_embedding_bag_max_grad_routes_to_argmax():
    np.random.seed(17)
    m = nn.EmbeddingBag(4, 2, mode="max")
    m.weight = Tensor(
        np.array([[0.1, 0.1], [0.2, 0.9], [0.5, 0.3], [0.0, 0.0]], dtype="float32")
    )
    idx = Tensor(np.array([0, 1, 2], dtype="int64"))
    y = m(idx)
    assert np.allclose(y.data, [[0.5, 0.9]], atol=1e-6)
    y.backward(np.ones_like(y.data))
    gw = m.weight.grad.data
    assert np.allclose(gw[2], [1.0, 0.0])
    assert np.allclose(gw[1], [0.0, 1.0])
    assert np.allclose(gw[0], [0.0, 0.0])
