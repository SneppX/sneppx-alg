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


def test_higher_order_kldiv():
    # L = KLDiv(p, t)^2  ->  K = (1/n) sum t_i(log t_i - p_i)
    #   dK/dp_j = -t_j/n  =>  Hessian = 2 t t^T / n^2  (exact).
    np.random.seed(21)
    t = np.abs(np.random.randn(6).astype("float32")) + 0.1
    t = t / t.sum()
    target = Tensor(t)
    p = Tensor(np.random.randn(6).astype("float32"))
    p.requires_grad_(True)
    n = t.size

    def build(param):
        return (ops.KLDivLoss.apply(param, target)) ** 2

    H_eng = _engine_hessian(build, p)
    H_ref = 2.0 * np.outer(t, t) / (n * n)
    assert np.allclose(H_eng, H_ref, atol=1e-3), (
        f"KLDiv Hessian mismatch max|err|={np.max(np.abs(H_eng - H_ref)):.3e}"
    )


def test_higher_order_nll_index():
    # L = NLLLoss(z, idx)^2  ->  dNLL/dz_{i,k} = -1/n if k=idx_i else 0
    #   Hessian = (2/n^2) * diag(1_{k=idx_i}) (exact, since L is the square).
    np.random.seed(23)
    n = 4
    c = 5
    idx = np.array([2, 0, 4, 1], dtype=np.int64)
    z = Tensor(np.random.randn(n, c).astype("float32"))
    z.requires_grad_(True)
    target = Tensor(idx)

    def build(param):
        return (ops.NLLLoss.apply(param, target)) ** 2

    H_eng = _engine_hessian(build, z)
    # NLL = -(1/n) sum_i z[i, idx_i] is LINEAR in z, so L = NLL^2 has
    # Hessian 2 * c c^T with c = (-1/n) at each target position:
    cvec = np.zeros(n * c)
    for i in range(n):
        cvec[i * c + idx[i]] = -1.0 / n
    H_ref = 2.0 * np.outer(cvec, cvec)
    assert np.allclose(H_eng, H_ref, atol=1e-3), (
        f"NLL(index) Hessian mismatch max|err|={np.max(np.abs(H_eng - H_ref)):.3e}"
    )


def test_higher_order_bcewithlogits():
    # z_i = x_i . w ;  L = (1/N) sum BCEWithLogits(z_i, t_i)
    #   dL/dw_j = (1/N) sum_i x_{i,j}(s_i - t_i)
    #   Hessian[j,k] = (1/N) sum_i x_{i,j} x_{i,k} s_i (1 - s_i)  (exact)
    np.random.seed(29)
    N, D = 5, 4
    x = np.random.randn(N, D).astype("float32")
    tgt = (np.random.rand(N) > 0.5).astype("float32")
    xT = Tensor(x)
    target = Tensor(tgt)
    w = Tensor(np.random.randn(D).astype("float32"))
    w.requires_grad_(True)

    def build(param):
        z = xT @ param
        return ops.BCEWithLogitsLoss.apply(z, target)

    H_eng = _engine_hessian(build, w)
    z = x @ w.data
    s = 1.0 / (1.0 + np.exp(-z))
    H_ref = (x.T * (s * (1 - s))) @ x / N
    assert np.allclose(H_eng, H_ref, atol=1e-3), (
        f"BCEWithLogits Hessian mismatch max|err|={np.max(np.abs(H_eng - H_ref)):.3e}"
    )


if __name__ == "__main__":
    test_higher_order_kldiv()
    test_higher_order_nll_index()
    test_higher_order_bcewithlogits()
    print("ALL LOSS HIGHER-ORDER AD TESTS PASSED")
