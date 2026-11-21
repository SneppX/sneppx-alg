"""Tests for loss functions: correctness of forward values and gradients.

Covers the audit-flagged NLLLoss/KLDivLoss math fixes plus the new
torch.nn-compatible loss Module wrappers.
"""

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


def test_nll_loss_index_targets():
    rng = np.random.default_rng(0)
    N, C = 5, 4
    logits = rng.standard_normal((N, C))
    # log-probabilities as PyTorch expects
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    t = rng.integers(0, C, size=N)

    pred = Tensor(logp.copy(), requires_grad=True, dtype="float64")
    tgt = Tensor(t.copy())
    loss = nn.NLLLoss()(pred, tgt)
    ref = float(-np.mean(logp[np.arange(N), t]))
    assert abs(float(loss.data) - ref) < 1e-10

    loss.backward()
    ref_grad = np.zeros((N, C))
    ref_grad[np.arange(N), t] = -1.0 / N
    assert np.max(np.abs(pred.grad.data - ref_grad)) < 1e-10


def test_kl_div_loss():
    rng = np.random.default_rng(1)
    N, C = 6, 3
    lp = rng.standard_normal((N, C))
    logp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))  # log-prob input
    tp = rng.standard_normal((N, C))
    prob = tp - np.log(np.exp(tp).sum(-1, keepdims=True))
    prob = np.exp(prob)

    pred = Tensor(logp.copy(), requires_grad=True, dtype="float64")
    tgt = Tensor(prob.copy(), dtype="float64")
    loss = nn.KLDivLoss()(pred, tgt)
    ref = float(np.mean(prob * (np.log(prob + 1e-10) - logp)))
    assert abs(float(loss.data) - ref) < 1e-9

    loss.backward()
    ref_grad = -prob / (N * C)
    assert np.max(np.abs(pred.grad.data - ref_grad)) < 1e-7


def test_smooth_l1_and_huber():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 3))
    y = rng.standard_normal((4, 3))
    inp = Tensor(x.copy(), requires_grad=True, dtype="float64")
    tgt = Tensor(y.copy(), dtype="float64")

    for beta in (0.5, 1.0):
        loss = nn.SmoothL1Loss(beta=beta)(inp, tgt)
        diff = x - y
        ref = np.where(
            np.abs(diff) < beta,
            0.5 * diff**2 / beta,
            np.abs(diff) - 0.5 * beta,
        ).mean()
        assert abs(float(loss.data) - ref) < 1e-10

    inp2 = Tensor(x.copy(), requires_grad=True, dtype="float64")
    hl = nn.HuberLoss(delta=1.0)(inp2, tgt)
    refh = np.where(
        np.abs(diff) <= 1.0, 0.5 * diff**2, 1.0 * (np.abs(diff) - 0.5)
    ).mean()
    assert abs(float(hl.data) - refh) < 1e-10

    # FD gradient check on SmoothL1
    def f(v):
        t_ = Tensor(v, dtype="float64")
        return float(nn.SmoothL1Loss(beta=0.5)(t_, tgt).data)

    g = _fd_grad(f, x.copy())
    inp3 = Tensor(x.copy(), requires_grad=True, dtype="float64")
    nn.SmoothL1Loss(beta=0.5)(inp3, tgt).backward()
    assert np.max(np.abs(inp3.grad.data - g)) < 1e-5


def test_bce_with_logits():
    rng = np.random.default_rng(3)
    z = rng.standard_normal((5, 2))
    t = rng.integers(0, 2, size=(5, 2)).astype(float)
    inp = Tensor(z.copy(), requires_grad=True, dtype="float64")
    tgt = Tensor(t.copy(), dtype="float64")
    loss = nn.BCEWithLogitsLoss()(inp, tgt)
    sig = 1.0 / (1.0 + np.exp(-z))
    ref = -(t * np.log(sig + 1e-10) + (1 - t) * np.log(1 - sig + 1e-10)).mean()
    assert abs(float(loss.data) - ref) < 1e-9

    def f(v):
        return float(nn.BCEWithLogitsLoss()(Tensor(v, dtype="float64"), tgt).data)

    g = _fd_grad(f, z.copy())
    inp2 = Tensor(z.copy(), requires_grad=True, dtype="float64")
    nn.BCEWithLogitsLoss()(inp2, tgt).backward()
    assert np.max(np.abs(inp2.grad.data - g)) < 1e-5


def test_loss_module_wrappers_exist():
    assert callable(nn.MSELoss())
    assert callable(nn.L1Loss())
    assert callable(nn.CrossEntropyLoss())
    assert callable(nn.NLLLoss())
    assert callable(nn.KLDivLoss())
    assert callable(nn.BCELoss())
    assert callable(nn.BCEWithLogitsLoss())
    assert callable(nn.SmoothL1Loss())
    assert callable(nn.HuberLoss())
    assert callable(nn.FocalLoss())
    assert callable(nn.TripletMarginLoss())
    assert callable(nn.ContrastiveLoss())
    assert callable(nn.MarginRankingLoss())
    assert callable(nn.CosineEmbeddingLoss())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
