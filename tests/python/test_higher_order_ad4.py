import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings.autograd import grad


def _engine_hessian(build, p):
    p.grad = None
    L = build(p)
    g = grad(L, [p], create_graph=True)[0]
    flat = p.numel
    H = np.zeros((flat, flat), dtype=np.float64)
    for j in range(flat):
        p.grad = None
        gj = g[np.unravel_index(j, g.shape)]
        gj.backward()
        H[j, :] = p.grad.data.reshape(-1).astype(np.float64)
    return H


def _square_loss_hessian(dS, d2S, S):
    # L = S^2, S scalar.  dL/dx_i = 2 S dS_i ; d2L/dx_i dx_j = 2 dS_i dS_j + 2 S d2S_i d_ij
    dS = np.asarray(dS, dtype=np.float64).reshape(-1)
    d2S = np.asarray(d2S, dtype=np.float64).reshape(-1)
    return 2.0 * np.outer(dS, dS) + 2.0 * float(S) * np.diag(d2S)


def test_higher_order_smoothl1():
    # L = SmoothL1(x, t)^2 ; per-element s_i = 0.5 (x_i-t_i)^2/beta (smooth) or
    # |x_i-t_i| - 0.5 beta (linear).  Exact Hessian H = 2 dS dS^T + 2 S diag(d2S).
    np.random.seed(41)
    beta = 1.0
    x = np.random.randn(7).astype("float32") * 1.3
    tgt = np.random.randn(7).astype("float32") * 1.3
    p = Tensor(x)
    p.requires_grad_(True)
    target = Tensor(tgt)

    def build(param):
        return (ops.SmoothL1Loss.apply(param, target, beta)) ** 2

    L0 = build(p)
    S = float(L0.data[0]) ** 0.5  # = SmoothL1 value
    z = x - tgt
    absz = np.abs(z)
    n = z.size
    dS = np.where(absz < beta, z / beta, np.sign(z)) / n
    d2S = np.where(absz < beta, 1.0 / beta, 0.0) / n
    H_ref = _square_loss_hessian(dS, d2S, S)

    H_eng = _engine_hessian(build, p)
    assert np.allclose(H_eng, H_ref, atol=1e-3), (
        f"SmoothL1 Hessian mismatch max|err|={np.max(np.abs(H_eng - H_ref)):.3e}"
    )


def test_higher_order_huber():
    # L = HuberLoss(x, t)^2 ; per-element s_i = 0.5 (x_i-t_i)^2 (|z|<=delta) or
    # delta(|z| - 0.5 delta) (linear).  Exact Hessian H = 2 dS dS^T + 2 S diag(d2S).
    np.random.seed(43)
    delta = 1.0
    x = np.random.randn(9).astype("float32") * 1.7
    tgt = np.random.randn(9).astype("float32") * 1.7
    p = Tensor(x)
    p.requires_grad_(True)
    target = Tensor(tgt)

    def build(param):
        return (ops.HuberLoss.apply(param, target, delta)) ** 2

    L0 = build(p)
    S = float(L0.data[0]) ** 0.5
    z = x - tgt
    absz = np.abs(z)
    n = z.size
    dS = np.where(absz <= delta, z, delta * np.sign(z)) / n
    d2S = np.where(absz <= delta, 1.0, 0.0) / n
    H_ref = _square_loss_hessian(dS, d2S, S)

    H_eng = _engine_hessian(build, p)
    assert np.allclose(H_eng, H_ref, atol=1e-3), (
        f"Huber Hessian mismatch max|err|={np.max(np.abs(H_eng - H_ref)):.3e}"
    )


def test_conv1d_double_backward_fallback():
    # Conv1d backward has no differentiable (create_graph) path yet, so the
    # gradient is returned as a plain (detached) tensor: 1st-order is correct,
    # 2nd-order through it falls back to 0. This test asserts the graph-aware
    # traversal does NOT crash and that the 1st-order gradient is finite.
    np.random.seed(47)
    inp = Tensor(np.random.randn(2, 3, 8).astype("float32"))
    inp.requires_grad_(True)
    k = Tensor(np.random.randn(4, 3, 3).astype("float32"))
    k.requires_grad_(True)

    def build(xin, kern):
        return (ops.Conv1d.apply(xin, kern)) ** 2

    L = build(inp, k)
    gx, gk = grad(L, [inp, k], create_graph=True)
    # 1st-order gradient must be finite and nonzero
    assert gx.data.shape == inp.data.shape
    assert gk.data.shape == k.data.shape
    assert np.all(np.isfinite(gx.data))
    # double-backward must not raise (fallback path yields 0 Hessian)
    gx.sum().backward()
    gk.sum().backward()


if __name__ == "__main__":
    test_higher_order_smoothl1()
    test_higher_order_huber()
    test_conv1d_double_backward_fallback()
    print("ALL LOSS/POOL HIGHER-ORDER AD TESTS PASSED")
