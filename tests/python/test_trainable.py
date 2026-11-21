import numpy as np

from SneppX_ALG.interface_bindings import Tensor, Linear, MSELoss, SGD, nn


def _loss_scalar(loss):
    return float(np.asarray(loss.data).reshape(-1)[0])


def test_mlp_training_decreases_loss():
    """End-to-end: a small MLP must learn (loss decreases) via autograd + SGD.

    This validates the differentiability campaign — every module in the graph
    must produce a working gradient for backprop to reduce the loss.
    """
    np.random.seed(0)
    fc1 = Linear(4, 16)
    fc2 = Linear(16, 1)
    act = nn.ReLU()
    params = list(fc1.parameters()) + list(fc2.parameters())
    opt = SGD(params, lr=0.05)
    crit = MSELoss()

    x = Tensor(np.random.randn(64, 4).astype("float32"))
    y = Tensor((np.sum(x.data, axis=1, keepdims=True)).astype("float32"))

    losses = []
    for _ in range(80):
        opt.zero_grad()
        h = act(fc1(x))
        out = fc2(h)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        losses.append(_loss_scalar(loss))

    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not decrease: first={losses[0]:.4f} last={losses[-1]:.4f}"
    )


def test_conv_net_training_decreases_loss():
    """CNN with Conv2d + MaxPool + Flatten + Linear trains end-to-end."""
    np.random.seed(1)
    conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
    pool = nn.MaxPool2d(2)
    flatten = nn.Flatten()
    # after 8x8 -> pool -> 4x4 -> 4 channels = 64
    fc = Linear(64, 1)
    act = nn.ReLU()
    params = list(conv.parameters()) + list(fc.parameters())
    opt = SGD(params, lr=0.01)
    crit = MSELoss()

    x = Tensor(np.random.randn(8, 1, 8, 8).astype("float32"))
    y = Tensor(np.random.randn(8, 1).astype("float32"))

    losses = []
    for _ in range(40):
        opt.zero_grad()
        h = pool(conv(x))
        out = fc(act(flatten(h)))
        loss = crit(out, y)
        loss.backward()
        opt.step()
        losses.append(_loss_scalar(loss))

    assert losses[-1] < losses[0], (
        f"loss did not decrease: first={losses[0]:.4f} last={losses[-1]:.4f}"
    )


def test_loss_is_finite_and_grads_flow_to_all_params():
    np.random.seed(2)
    fc = Linear(4, 3)
    y = Tensor(np.random.randn(5, 3).astype("float32"))
    x = Tensor(np.random.randn(5, 4).astype("float32"))
    out = fc(x)
    loss = MSELoss()(out, y)
    loss.backward()
    for name, p in fc.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert np.isfinite(p.grad.data).all(), f"{name} grad not finite"
