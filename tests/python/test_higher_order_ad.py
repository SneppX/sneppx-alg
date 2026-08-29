import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings.autograd import grad, is_grad_enabled


def test_double_backward_mlp():
    """Real second-order AD: d^2L/dx^2 etc. via grad(create_graph=True).

    L = ||x @ W^T + b||^2  (a pure quadratic, so all second derivatives are
    nonzero and have exact closed forms used as the reference). This validates
    the full create_graph backward chain: Sum -> Mul -> Add -> MatMul -> Transpose.
    """
    np.random.seed(7)
    x = Tensor(np.random.randn(5, 4).astype("float32"))
    x.requires_grad_(True)
    W = Tensor(np.random.randn(6, 4).astype("float32"))
    W.requires_grad_(True)
    b = Tensor(np.random.randn(6).astype("float32"))
    b.requires_grad_(True)

    h = x @ ops.Transpose.apply(W, -1, -2) + b
    L = (h * h).sum()

    gx, gW, gb = grad(L, [x, W, b], create_graph=True)
    assert gx.grad_fn is not None, "first grad must be graph-connected"

    # --- first-order sanity (must equal the analytic first gradients) ---
    first_x = 2.0 * (h.data @ W.data)                 # (5,4)
    first_W = 2.0 * (h.data.T @ x.data)               # (6,4)
    first_b = 2.0 * h.data.sum(axis=0)                # (6,)
    assert np.allclose(gx.data, first_x, atol=1e-4), "first grad x mismatch"
    assert np.allclose(gW.data, first_W, atol=1e-4), "first grad W mismatch"
    assert np.allclose(gb.data, first_b, atol=1e-4), "first grad b mismatch"

    # --- second-order via .backward() on each gradient tensor. NOTE: gW and gb
    #     also depend on x, so their .backward() would write into x.grad too;
    #     we therefore check each leaf's .grad immediately after its own pass. ---
    gx.backward()
    WTW = W.data.T @ W.data                            # (4,4)
    expected_d2x = 2.0 * WTW.sum(axis=1)[None, :]      # (5,4): same per row
    assert np.allclose(x.grad.data, expected_d2x, atol=1e-3), (
        f"d2L/dx2 mismatch max|diff|={np.max(np.abs(x.grad.data - expected_d2x)):.3e}"
    )

    W.grad = None
    b.grad = None
    gW.backward()
    XTX = x.data.T @ x.data                            # (4,4)
    expected_d2W = 2.0 * XTX.sum(axis=0)[None, :]      # (6,4)
    assert np.allclose(W.grad.data, expected_d2W, atol=1e-3), (
        f"d2L/dW2 mismatch max|diff|={np.max(np.abs(W.grad.data - expected_d2W)):.3e}"
    )

    b.grad = None
    gb.backward()
    expected_d2b = 2.0 * np.float32(x.data.shape[0])   # (6,): 2*N constant
    assert np.allclose(b.grad.data, expected_d2b, atol=1e-3), (
        f"d2L/db2 mismatch max|diff|={np.max(np.abs(b.grad.data - expected_d2b)):.3e}"
    )


def test_grad_functional_matches_backward():
    """grad() (return_grads) reproduces Tensor.backward() values."""
    np.random.seed(3)
    a = Tensor(np.random.randn(4, 3).astype("float32"))
    a.requires_grad_(True)
    b = Tensor(np.random.randn(3).astype("float32"))
    b.requires_grad_(True)
    out = (a * b).sum()
    g_a, g_b = grad(out, [a, b])
    a2 = Tensor(a.data.copy())
    a2.requires_grad_(True)
    b2 = Tensor(b.data.copy())
    b2.requires_grad_(True)
    out2 = (a2 * b2).sum()
    out2.backward()
    assert np.allclose(g_a.data, a2.grad.data, atol=1e-5)
    assert np.allclose(g_b.data, b2.grad.data, atol=1e-5)


if __name__ == "__main__":
    test_double_backward_mlp()
    test_grad_functional_matches_backward()
    print("ALL HIGHER-ORDER AD TESTS PASSED")
