import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings.optim_extra import Adam8bit, SGD8bit


def _ref_adam(params, data, target, steps=300, lr=1e-2):
    ps = [Tensor(p.data.copy(), dtype="float64") for p in params]
    for p in ps:
        p.requires_grad_(True)
    for _ in range(steps):
        pred = data @ ps[0] + ps[1]
        ((pred - target) ** 2).mean().backward()
        for p in ps:
            g = p.grad.data
            if not hasattr(p, "_m"):
                p._m = np.zeros_like(g)
                p._v = np.zeros_like(g)
                p._t = 0
            p._t += 1
            p._m = 0.9 * p._m + 0.1 * g
            p._v = 0.999 * p._v + 0.001 * (g * g)
            mhat = p._m / (1 - 0.9 ** p._t)
            vhat = p._v / (1 - 0.999 ** p._t)
            upd = mhat / (np.sqrt(vhat) + 1e-8)
            p.data = (p.data - lr * upd).astype(p.data.dtype)
            p.grad = None
    return ps


def _ref_sgd(params, data, target, steps=400, lr=1e-2, momentum=0.9):
    ps = [Tensor(p.data.copy(), dtype="float64") for p in params]
    for p in ps:
        p.requires_grad_(True)
    buf = [np.zeros_like(p.data) for p in ps]
    for _ in range(steps):
        pred = data @ ps[0] + ps[1]
        ((pred - target) ** 2).mean().backward()
        for k, p in enumerate(ps):
            g = p.grad.data
            if momentum != 0:
                buf[k] = momentum * buf[k] + g
                g = buf[k]
            p.data = (p.data - lr * g).astype(p.data.dtype)
            p.grad = None
    return ps


def test_adam8bit_tracks_fp32():
    rng = np.random.RandomState(1)
    W_true = rng.randn(3, 2)
    b_true = rng.randn(2)
    X = Tensor(rng.randn(64, 3), dtype="float64")
    Y = Tensor(X.data @ W_true + b_true, dtype="float64")

    W = Tensor(rng.randn(3, 2), dtype="float64")
    b = Tensor(rng.randn(2), dtype="float64")
    W.requires_grad_(True); b.requires_grad_(True)
    opt = Adam8bit([W, b], lr=1e-2)
    loss_fn = lambda: ((X @ W + b - Y) ** 2).mean()

    for _ in range(1500):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    l8 = float(loss_fn().data)

    ref = _ref_adam([Tensor(rng.randn(3, 2), dtype="float64"),
                     Tensor(rng.randn(2), dtype="float64")], X, Y, steps=1500, lr=1e-2)
    lref = float(((X.data @ ref[0].data + ref[1].data - Y.data) ** 2).mean())

    assert l8 < 1e-4, f"8bit adam did not converge: loss={l8}"
    assert abs(l8 - lref) < 1e-6, f"8bit {l8} vs fp32 {lref} (should track closely)"
    print(f"adam8bit loss {l8:.2e} vs fp32 {lref:.2e}: OK")


def test_sgd8bit_tracks_fp32():
    rng = np.random.RandomState(2)
    W_true = rng.randn(3, 2)
    b_true = rng.randn(2)
    X = Tensor(rng.randn(64, 3), dtype="float64")
    Y = Tensor(X.data @ W_true + b_true, dtype="float64")

    W = Tensor(rng.randn(3, 2), dtype="float64")
    b = Tensor(rng.randn(2), dtype="float64")
    W.requires_grad_(True); b.requires_grad_(True)
    opt = SGD8bit([W, b], lr=1e-2, momentum=0.9)
    loss_fn = lambda: ((X @ W + b - Y) ** 2).mean()

    for _ in range(400):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    l8 = float(loss_fn().data)

    ref = _ref_sgd([Tensor(rng.randn(3, 2), dtype="float64"),
                    Tensor(rng.randn(2), dtype="float64")], X, Y, steps=400, lr=1e-2, momentum=0.9)
    lref = float(((X.data @ ref[0].data + ref[1].data - Y.data) ** 2).mean())

    assert l8 < 1e-2, f"8bit sgd did not converge: loss={l8}"
    assert abs(l8 - lref) / (lref + 1e-12) < 0.1, f"8bit {l8} vs fp32 {lref}"
    print(f"sgd8bit loss {l8:.2e} vs fp32 {lref:.2e}: OK")


if __name__ == "__main__":
    test_adam8bit_tracks_fp32()
    test_sgd8bit_tracks_fp32()
    print("ALL 8BIT OK")
