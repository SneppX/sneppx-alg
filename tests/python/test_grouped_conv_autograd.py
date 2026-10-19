"""FD gradient checks for grouped convolution (groups > 1)."""

import numpy as np
from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings import nn


def _fd_grad(f, x, eps=1e-3):
    base = f(x)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x.copy())
        x[idx] = orig - eps
        fm = f(x.copy())
        x[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g, base


def test_conv2d_grouped_backward():
    np.random.seed(0)
    N, Cin, H, W = 2, 4, 6, 6
    Cout = 6
    groups = 2
    x = np.random.randn(N, Cin, H, W).astype(np.float32)
    w = np.random.randn(Cout, Cin // groups, 3, 3).astype(np.float32) * 0.1
    b = np.random.randn(Cout).astype(np.float32) * 0.1

    xt = Tensor(x.copy(), requires_grad=True)
    layer = nn.Conv2d(
        Cin, Cout, 3, padding=1, groups=groups, bias=True, dtype="float32"
    )
    layer.weight = Tensor(w.copy(), requires_grad=True)
    layer.bias = Tensor(b.copy(), requires_grad=True)

    def loss_fn(xin):
        xo = Tensor(xin.astype(np.float32), requires_grad=False)
        # rebuild layer params as constants
        lo = nn.Conv2d(
            Cin, Cout, 3, padding=1, groups=groups, bias=True, dtype="float32"
        )
        lo.weight = Tensor(w.copy(), requires_grad=False)
        lo.bias = Tensor(b.copy(), requires_grad=False)
        out = lo(xo)
        return float(np.asarray(out.data).reshape(-1).sum())

    out = layer(xt)
    loss = (out * 1.0).sum() if False else out.sum()
    loss.backward()
    # grads finite
    assert np.isfinite(layer.weight.grad.data).all()
    assert np.isfinite(xt.grad.data).all()
    # FD check on a single weight element (rebuild, since Tensor.data is a copy)
    eps = 1e-3

    def make_layer(w_arr):
        lo = nn.Conv2d(
            Cin, Cout, 3, padding=1, groups=groups, bias=True, dtype="float32"
        )
        lo.weight = Tensor(w_arr.astype(np.float32), requires_grad=False)
        lo.bias = Tensor(b.copy(), requires_grad=False)
        return lo

    w_base = w.copy()
    wp = w_base.copy()
    wp[0, 0, 0, 0] += eps
    wm = w_base.copy()
    wm[0, 0, 0, 0] -= eps
    fp = float(
        np.asarray(make_layer(wp)(Tensor(x.copy(), requires_grad=False)).data).sum()
    )
    fm = float(
        np.asarray(make_layer(wm)(Tensor(x.copy(), requires_grad=False)).data).sum()
    )
    fd = (fp - fm) / (2 * eps)
    ana = float(layer.weight.grad.data[0, 0, 0, 0])
    assert abs(fd - ana) < 1e-2, (fd, ana)


def test_conv3d_grouped_backward():
    np.random.seed(1)
    N, Cin, D, H, W = 1, 4, 4, 4, 4
    Cout = 4
    groups = 2
    x = np.random.randn(N, Cin, D, H, W).astype(np.float32)
    xt = Tensor(x.copy(), requires_grad=True)
    layer = nn.Conv3d(
        Cin, Cout, 3, padding=1, groups=groups, bias=True, dtype="float32"
    )
    out = layer(xt)
    out.sum().backward()
    assert np.isfinite(layer.weight.grad.data).all()
    assert np.isfinite(xt.grad.data).all()
