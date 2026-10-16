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


def test_gaussian_nll_loss_is_differentiable_and_matches_reference():
    x = _t(4, 3)
    y = Tensor(np.random.randn(4, 3).astype("float32"))
    var = Tensor((np.random.rand(4, 3).astype("float32") + 0.5))
    loss = nn.GaussianNLLLoss()(x, y, var)
    assert loss.requires_grad
    loss.backward()
    assert x.grad is not None and np.isfinite(x.grad.data).all()
    v = var.data
    ref = (
        0.5
        * (np.log(2 * np.pi) + np.log(v + 1e-6) + (x.data - y.data) ** 2 / (v + 1e-6))
    ).mean()
    np.testing.assert_allclose(
        float(np.asarray(loss.data).reshape(-1)[0]), ref, atol=1e-4
    )


def test_hinge_embedding_loss_is_differentiable_and_matches_reference():
    x = _t(4, 3)
    y = Tensor(np.random.choice([-1.0, 1.0], size=(4, 3)).astype("float32"))
    loss = nn.HingeEmbeddingLoss()(x, y)
    assert loss.requires_grad
    loss.backward()
    assert x.grad is not None and np.isfinite(x.grad.data).all()
    ref = np.where(y.data >= 0, x.data, np.maximum(0.0, 1.0 - x.data)).mean()
    np.testing.assert_allclose(
        float(np.asarray(loss.data).reshape(-1)[0]), ref, atol=1e-4
    )


def test_multi_label_soft_margin_loss_is_differentiable_and_matches_reference():
    x = _t(4, 3)
    y = Tensor((np.random.rand(4, 3) > 0.5).astype("float32"))
    loss = nn.MultiLabelSoftMarginLoss()(x, y)
    assert loss.requires_grad
    loss.backward()
    assert x.grad is not None and np.isfinite(x.grad.data).all()
    xc = x.data
    ref = -(
        y.data * np.log(np.clip(xc, 1e-7, 1.0))
        + (1 - y.data) * np.log(np.clip(1 - xc, 1e-7, 1.0))
    ).mean()
    np.testing.assert_allclose(
        float(np.asarray(loss.data).reshape(-1)[0]), ref, atol=1e-3
    )


def test_multi_margin_loss_is_differentiable_and_matches_reference():
    x = _t(4, 5)
    y = Tensor(np.random.randint(0, 5, (4,)).astype("int64"))
    loss = nn.MultiMarginLoss()(x, y)
    assert loss.requires_grad
    loss.backward()
    assert x.grad is not None and np.isfinite(x.grad.data).all()
    n, c = x.data.shape
    yi = y.data.astype(np.int64)
    tgt = x.data[np.arange(n), yi]
    mask = np.arange(c) != yi[:, None]
    md = 1.0 - (tgt[:, None] - x.data)
    md = np.where(mask, md, 0)
    ref = np.maximum(0, np.maximum(md, 0)).sum(axis=1).mean()
    np.testing.assert_allclose(
        float(np.asarray(loss.data).reshape(-1)[0]), ref, atol=1e-4
    )
