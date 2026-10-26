"""L1Loss / Tensor.mae_loss must be differentiable (was raw NumPy)."""

import numpy as np

from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn


def test_mae_loss_builds_graph():
    pred = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)
    tgt = Tensor(np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float32))

    out = pred.mae_loss(tgt)
    assert out.requires_grad
    assert out.grad_fn is not None

    out.backward()
    grad = np.asarray(pred.grad.data).reshape(-1)
    # d/dy mean|y - t| = sign(y - t) / N; only the last element is nonzero
    np.testing.assert_allclose(grad, [0.0, 0.0, 0.0, -0.25], atol=1e-6)


def test_l1_loss_module_backward():
    pred = Tensor(np.zeros(4, dtype=np.float32), requires_grad=True)
    tgt = Tensor(np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float32))

    loss = nn.L1Loss()(pred, tgt)
    loss.backward()
    np.testing.assert_allclose(
        np.asarray(pred.grad.data).reshape(-1), [0.0, 0.0, 0.0, -0.25], atol=1e-6
    )


def test_mae_loss_forward_value_unchanged():
    pred = Tensor(np.array([1.0, 2.0], dtype=np.float32))
    tgt = Tensor(np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(
        np.asarray(pred.mae_loss(tgt).data).reshape(-1), [1.5], atol=1e-6
    )
