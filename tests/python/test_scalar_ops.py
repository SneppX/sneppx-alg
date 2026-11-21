import numpy as np

from SneppX_ALG.interface_bindings import Tensor


def _t(shape, requires_grad=True):
    return Tensor(
        np.random.randn(*shape).astype("float32"), requires_grad=requires_grad
    )


def test_scalar_first_mul_backprops():
    x = _t((4, 3))
    y = (0.5 * x).sum()
    y.backward()
    assert x.grad is not None
    assert np.allclose(x.grad.data, 0.5, atol=1e-6)


def test_scalar_last_mul_backprops():
    x = _t((4, 3))
    y = (x * 0.5).sum()
    y.backward()
    assert x.grad is not None
    assert np.allclose(x.grad.data, 0.5, atol=1e-6)


def test_scalar_first_add_backprops():
    x = _t((4, 3))
    y = (1.0 + x).sum()
    y.backward()
    assert x.grad is not None
    assert np.allclose(x.grad.data, 1.0, atol=1e-6)


def test_scalar_first_sub_backprops():
    x = _t((4, 3))
    y = (1.0 - x).sum()
    y.backward()
    assert x.grad is not None
    assert np.allclose(x.grad.data, -1.0, atol=1e-6)


def test_scalar_first_div_backprops():
    x = _t((4, 3))
    y = (2.0 / x).sum()
    y.backward()
    assert x.grad is not None
    assert np.allclose(x.grad.data, -2.0 / (x.data**2), atol=1e-4)
