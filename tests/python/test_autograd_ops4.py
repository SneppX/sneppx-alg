import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops


def _fd_loss_grad(make_tensors, loss_fn, arrs, idx, eps=1e-3):
    """Central-difference numeric grad of a scalar loss w.r.t. arrs[idx]."""
    base = arrs[idx]
    num = np.zeros_like(base, dtype=np.float64)
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        mi = it.multi_index
        ap = [a.copy() for a in arrs]
        am = [a.copy() for a in arrs]
        ap[idx][mi] += eps
        am[idx][mi] -= eps
        lp = loss_fn(*make_tensors(*ap)).data.astype(np.float64)
        lm = loss_fn(*make_tensors(*am)).data.astype(np.float64)
        num[mi] = (lp - lm) / (2.0 * eps)
        it.iternext()
    return num


def _analytic_grads(make_tensors, loss_fn, arrs):
    """Run backward and return analytic grads for every array in arrs.
    Entries are None for inputs that do not require grad (e.g. targets)."""
    tensors = list(make_tensors(*[a.copy() for a in arrs]))
    for t in tensors:
        if getattr(t, "requires_grad", False):
            t.requires_grad_(True)
    L = loss_fn(*tensors)
    L.backward()
    out = []
    for t, a in zip(tensors, arrs):
        if t.grad is None:
            out.append(None)
        else:
            out.append(t.grad.data.astype(np.float64).reshape(a.shape))
    return out


def _check(name, make_tensors, loss_fn, arrs, atol=1e-3):
    anal = _analytic_grads(make_tensors, loss_fn, arrs)
    for i, a in enumerate(arrs):
        if anal[i] is None:
            continue
        num = _fd_loss_grad(make_tensors, loss_fn, arrs, i)
        err = np.max(np.abs(anal[i] - num))
        assert err < atol, f"{name}: grad[{i}] FD mismatch max|err|={err:.3e}"
    print(f"  PASS {name}")


def test_focal_loss_fd():
    np.random.seed(31)
    logits = np.random.randn(3, 4).astype("float64")
    target = np.array([1, 0, 2], dtype=np.int64)

    def mt(*a):
        return [Tensor(a[0], requires_grad=True), Tensor(a[1])]

    def lf(*t):
        return ops.FocalLoss.apply(t[0], t[1], 2.0, 1.0)

    _check("FocalLoss", mt, lf, [logits, target])


def test_cosine_embedding_loss_fd():
    np.random.seed(32)
    x1 = np.random.randn(3, 4).astype("float64")
    x2 = np.random.randn(3, 4).astype("float64")
    y = np.array([1, -1, 1], dtype=np.int64)

    def mt(*a):
        return [Tensor(a[0], requires_grad=True), Tensor(a[1], requires_grad=True), Tensor(a[2])]

    def lf(*t):
        return ops.CosineEmbeddingLoss.apply(t[0], t[1], t[2], 0.0)

    _check("CosineEmbeddingLoss", mt, lf, [x1, x2, y])


def test_triplet_margin_loss_fd():
    np.random.seed(33)
    a = np.random.randn(3, 4).astype("float64")
    p = np.random.randn(3, 4).astype("float64")
    n = np.random.randn(3, 4).astype("float64")

    def mt(*aa):
        return [Tensor(aa[0], requires_grad=True), Tensor(aa[1], requires_grad=True),
                Tensor(aa[2], requires_grad=True)]

    def lf(*t):
        return ops.TripletMarginLoss.apply(t[0], t[1], t[2], 1.0)

    _check("TripletMarginLoss", mt, lf, [a, p, n])


def test_contrastive_loss_fd():
    np.random.seed(34)
    x1 = np.random.randn(3, 4).astype("float64")
    x2 = np.random.randn(3, 4).astype("float64")
    y = np.array([0, 1, 0], dtype=np.int64)

    def mt(*aa):
        return [Tensor(aa[0], requires_grad=True), Tensor(aa[1], requires_grad=True), Tensor(aa[2])]

    def lf(*t):
        return ops.ContrastiveLoss.apply(t[0], t[1], t[2], 1.0)

    _check("ContrastiveLoss", mt, lf, [x1, x2, y])


def test_margin_ranking_loss_fd():
    np.random.seed(35)
    x1 = np.random.randn(3, 4).astype("float64")
    x2 = np.random.randn(3, 4).astype("float64")
    y = np.array([1, -1, 1], dtype=np.int64)

    def mt(*aa):
        return [Tensor(aa[0], requires_grad=True), Tensor(aa[1], requires_grad=True), Tensor(aa[2])]

    def lf(*t):
        return ops.MarginRankingLoss.apply(t[0], t[1], t[2], 0.0)

    _check("MarginRankingLoss", mt, lf, [x1, x2, y])


if __name__ == "__main__":
    test_focal_loss_fd()
    test_cosine_embedding_loss_fd()
    test_triplet_margin_loss_fd()
    test_contrastive_loss_fd()
    test_margin_ranking_loss_fd()
    print("ALL PASS")
