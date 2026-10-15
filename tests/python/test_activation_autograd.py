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
