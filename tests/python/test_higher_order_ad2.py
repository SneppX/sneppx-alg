import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings.autograd import grad


def _engine_hessian(build, p):
    """Full Hessian of L(p) w.r.t p via grad(create_graph=True) + per-component backward."""
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


def test_higher_order_chain_mlp():
    """L = ||Gelu(LayerNorm(x@W1)) @ p||^2  ->  exact 2*J^T J Hessian.

    Validates the LayerNorm -> Gelu -> MatMul create_graph chain end-to-end.
    """
    np.random.seed(11)
    x = Tensor(np.random.randn(3, 4).astype("float32"))
    W1 = Tensor(np.random.randn(4, 4).astype("float32"))
    target = Tensor(np.array([0, 1, 0], dtype=np.int64))
    gamma = Tensor(np.ones(4, dtype="float32"))
    beta = Tensor(np.zeros(4, dtype="float32"))
    p = Tensor(np.random.randn(4, 2).astype("float32"))
    p.requires_grad_(True)

    def build(param):
        h = ops.Gelu.apply(ops.LayerNorm.apply(x @ W1, gamma, beta))
        logits = h @ param
        return (logits * logits).sum()  # quadratic in p

    H_eng = _engine_hessian(build, p)
    # analytic: logits = h @ p  => J[(n,m),(k,m')] = h[n,k] * delta_{m,m'}
    h = (x.data @ W1.data)
    # LayerNorm then Gelu (recompute numerically using the op's forward)
    ln = ops.LayerNorm.apply(Tensor(h), gamma, beta).data
    h_act = ops.Gelu.apply(Tensor(ln)).data  # (3,4)
    D, M = p.shape
    S = h_act.T @ h_act  # (D,D)
    H_ref = np.zeros((D * M, D * M))
    for k in range(D):
        for kk in range(D):
            for m in range(M):
                H_ref[k * M + m, kk * M + m] = 2.0 * S[k, kk]
    assert np.allclose(H_eng, H_ref, atol=1e-3), (
        f"Hessian mismatch (MLP chain) max|err|={np.max(np.abs(H_eng - H_ref)):.3e}"
    )


def test_higher_order_chain_rmsnorm_mse():
    """L = ||RMSNorm(Cat(a,b)@W, g) - target||^2  ->  exact 2*J^T J Hessian.

    Validates Cat -> RMSNorm -> MSELoss(-style) create_graph branches.
    """
    np.random.seed(13)
    a = Tensor(np.random.randn(2, 3).astype("float32"))
    b = Tensor(np.random.randn(2, 1).astype("float32"))
    W = Tensor(np.random.randn(4, 5).astype("float32"))
    target = Tensor(np.random.randn(2, 5).astype("float32"))
    p = Tensor(np.random.randn(5).astype("float32"))
    p.requires_grad_(True)

    lin = (ops.Cat.apply(a, b, 1) @ W).data  # fixed (2,5)
    rms = np.sqrt((lin**2).mean(axis=1, keepdims=True) + 1e-6)  # (2,1)

    def build(param):
        out = ops.RMSNorm.apply(Tensor(lin), param)  # linear in param
        diff = out - target
        return (diff * diff).sum()

    H_eng = _engine_hessian(build, p)
    # J[(n,d_out), d'] = lin[n,d_out]/rms[n] * delta_{d_out,d'}
    #   => J^T J is DIAGONAL: H[d,d'] = 2*delta_{d,d'} * sum_n (lin[n,d]/rms[n])^2
    scale = (lin / rms)  # (2,5)
    H_ref = 2.0 * np.diag((scale**2).sum(axis=0))  # (5,5) diagonal
    assert np.allclose(H_eng, H_ref, atol=1e-3), (
        f"Hessian mismatch (RMSNorm/MSE) max|err|={np.max(np.abs(H_eng - H_ref)):.3e}"
    )


if __name__ == "__main__":
    test_higher_order_chain_mlp()
    test_higher_order_chain_rmsnorm_mse()
    print("ALL DEEPER HIGHER-ORDER AD TESTS PASSED")
