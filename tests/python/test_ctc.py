"""Tests for CTCLoss: forward values and backward gradients."""

import numpy as np
import sys

sys.path.insert(0, "bindings/python")

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import nn


def _fd_grad(f, x, eps=1e-5):
    """Central finite-difference gradient of scalar f(x) at x (ndarray)."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        x[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return g


def test_ctc_loss_grad():
    rng = np.random.default_rng(9)
    T, N, C = 6, 1, 3  # T >= 2*len(tgt)+1 = 5
    raw = rng.standard_normal((T, N, C))
    logp = raw - np.log(np.exp(raw).sum(-1, keepdims=True))  # Normalize to log-probs
    targets = np.array([[1, 2]], dtype=np.int64)
    lp_tensor = Tensor(logp.copy(), requires_grad=True, dtype="float64")
    tgt_tensor = Tensor(targets.copy(), dtype="float64")

    # Smoke check: loss decreases / is valid
    loss = nn.CTCLoss(blank=0)(lp_tensor, tgt_tensor)
    print("Loss:", float(loss.data))
    assert float(loss.data) > 0.0

    # FD gradient check
    def f(v):
        v_tensor = Tensor(v, dtype="float64")
        return float(nn.CTCLoss(blank=0)(v_tensor, tgt_tensor).data)

    g_fd = _fd_grad(f, logp.copy())
    loss.backward()
    print("FD Max Diff:", np.max(np.abs(lp_tensor.grad.data - g_fd)))
    assert np.max(np.abs(lp_tensor.grad.data - g_fd)) < 1e-4


if __name__ == "__main__":
    test_ctc_loss_grad()
    print("test_ctc_loss_grad: OK")
