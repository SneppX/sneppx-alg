"""Verification tests for optimizers and param_groups.

Confirms the classic optimizers (SGD w/ momentum & Nesterov, AdamW, RMSprop,
Adagrad, Adadelta, Adamax) actually minimise a convex objective and that
per-group learning rates via param_groups are honoured.
"""

import numpy as np
import sys

sys.path.insert(0, "bindings/python")

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import optim


def _make_param(arr):
    return Tensor(arr.copy(), requires_grad=True, dtype="float64")


def test_classic_optimizers_minimize():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((4, 3))
    # minimise  f(W) = ||W - A||^2  (grad = 2*(W - A))
    configs = [
        ("SGD", dict(lr=0.1, momentum=0.0)),
        ("SGD", dict(lr=0.05, momentum=0.9)),
        ("NesterovSGD", dict(lr=0.01, momentum=0.9)),
        ("AdamW", dict(lr=0.05)),
        ("RMSprop", dict(lr=0.05)),
        ("Adagrad", dict(lr=0.5)),
        ("Adadelta", dict(lr=1.0)),
        ("Adamax", dict(lr=0.02)),
    ]
    for name, kw in configs:
        cls = getattr(optim, name)
        W = _make_param(rng.standard_normal((4, 3)))
        opt = cls([W], **kw)
        init = np.linalg.norm(W.data - A)
        for _ in range(2000):
            opt.zero_grad()
            grad = 2.0 * (W.data - A)
            W.grad = Tensor(grad, dtype="float64")
            opt.step()
        final = np.linalg.norm(W.data - A)
        # Adadelta is intrinsically slow on a flat quadratic; allow a looser bar.
        tol = 0.15 * init if name == "Adadelta" else 0.05 * init
        assert final < tol, f"{name} failed: {init:.3f} -> {final:.3f}"


def test_param_groups_apply_distinct_lrs():
    rng = np.random.default_rng(8)
    p_fast = _make_param(rng.standard_normal((3,)))
    p_slow = _make_param(rng.standard_normal((3,)))
    # both minimise ||p - A||^2 with different per-group lrs
    A = rng.standard_normal((3,))
    groups = [
        {"params": [p_fast], "lr": 0.5},
        {"params": [p_slow], "lr": 0.001},
    ]
    opt = optim.SGD(groups, momentum=0.0)
    for _ in range(5):
        opt.zero_grad()
        p_fast.grad = Tensor(2.0 * (p_fast.data - A), dtype="float64")
        p_slow.grad = Tensor(2.0 * (p_slow.data - A), dtype="float64")
        opt.step()

    # fast group should have moved much further toward A than slow group
    moved_fast = np.linalg.norm(p_fast.data - A)
    moved_slow = np.linalg.norm(p_slow.data - A)
    # initial distance identical; fast lr => smaller residual
    assert moved_fast < moved_slow, f"param_groups lr not honoured: {moved_fast} vs {moved_slow}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
