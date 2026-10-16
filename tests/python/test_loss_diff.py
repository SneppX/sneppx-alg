import numpy as np

from SneppX_ALG.interface_bindings import nn, Tensor


def _t(*shape):
    return Tensor(np.random.randn(*shape).astype("float32"), requires_grad=True)


def test_soft_margin_loss_is_differentiable_and_matches_reference():
    x = _t(4, 3)
    y = Tensor(np.random.choice([-1.0, 1.0], size=(4, 3)).astype("float32"))
    loss = nn.SoftMarginLoss()(x, y)
    assert loss.requires_grad
    loss.backward()
    assert x.grad is not None and np.isfinite(x.grad.data).all()
    ref = np.log1p(np.exp(-y.data * x.data)).mean()
    np.testing.assert_allclose(
        float(np.asarray(loss.data).reshape(-1)[0]), ref, atol=1e-4
    )


def test_poisson_nll_loss_is_differentiable_and_matches_reference():
    x = _t(5, 3)
    y = Tensor(np.random.rand(5, 3).astype("float32"))
    loss = nn.PoissonNLLLoss()(x, y)
    assert loss.requires_grad
    loss.backward()
    assert x.grad is not None and np.isfinite(x.grad.data).all()
    ref = (np.exp(x.data) - y.data * x.data).mean()
    np.testing.assert_allclose(
        float(np.asarray(loss.data).reshape(-1)[0]), ref, atol=1e-4
    )


def test_poisson_nll_loss_full_is_differentiable():
    x = _t(4, 2)
    y = Tensor(np.random.rand(4, 2).astype("float32") + 0.1)
    loss = nn.PoissonNLLLoss(full=True)(x, y)
    loss.backward()
    assert np.isfinite(x.grad.data).all()
