import sys
import os
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python")
)

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops
from tests.python.test_autograd_correctness import _rng, numerical_grads, _check


def test_conv1d():
    _check(lambda I: ops.Conv1d.apply(I[0], I[1]).sum(), [(1, 2, 8), (3, 2, 4)], [1, 2])


def test_conv1d_batch():
    _check(
        lambda I: ops.Conv1d.apply(I[0], I[1], 2, 1).sum(),
        [(3, 2, 9), (4, 2, 3)],
        [1, 2],
    )


def test_maxpool2d():
    def f(I):
        return ops.MaxPool2d.apply(I[0], 2, 2).sum()

    _check(f, [(1, 2, 8, 8)], [1])


def test_avgpool2d():
    def f(I):
        return ops.AvgPool2d.apply(I[0], 2, 2).sum()

    _check(f, [(1, 2, 8, 8)], [1])


def test_mseloss():
    tgt = Tensor(np.random.randn(4, 3).astype("float32"))

    def f(I):
        return ops.MSELoss.apply(I[0], tgt)

    _check(f, [(4, 3)], [1])


def test_cross_entropy():
    tgt = np.array([0, 1, 2, 3, 1], dtype=np.int64)

    def f(I):
        return ops.CrossEntropyLoss.apply(I[0], tgt)

    _check(f, [(5, 4)], [1])


def test_triplet_margin_loss_uses_p():
    a = Tensor(np.array([[0.0, 0.0]], dtype=np.float32), requires_grad=True)
    pos = Tensor(np.array([[1.0, 1.0]], dtype=np.float32), requires_grad=True)
    neg = Tensor(np.array([[0.5, 0.0]], dtype=np.float32), requires_grad=True)

    loss_p1 = ops.TripletMarginLoss.apply(a, pos, neg, 0.0, 1)
    loss_p2 = ops.TripletMarginLoss.apply(a, pos, neg, 0.0, 2)
    # ||a-pos||_1 = 2, ||a-neg||_1 = 0.5 -> hinge 1.5
    assert np.allclose(loss_p1.data, [1.5], atol=1e-5)
    # ||a-pos||_2 = sqrt(2), ||a-neg||_2 = 0.5 -> hinge sqrt(2) - 0.5
    assert np.allclose(loss_p2.data, [np.sqrt(2) - 0.5], atol=1e-5)
    assert not np.allclose(loss_p1.data, loss_p2.data)


def test_triplet_margin_loss_p_grads():
    # Positive is always much farther than negative, so the hinge stays active
    # and the p-norm gradient is exercised across all perturbations.
    def make_f(p):
        def f(I):
            pos = Tensor(np.array([[5.0, 0.0]], dtype=np.float32))
            neg = Tensor(np.array([[0.0, 0.1]], dtype=np.float32))
            return ops.TripletMarginLoss.apply(I[0], pos, neg, 0.0, p)

        return f

    _check(make_f(1), [(2, 2)], [3])
    _check(make_f(2), [(2, 2)], [3])


def test_nllloss():
    tgt = np.array([0, 1, 2, 3, 1], dtype=np.int64)

    def f(I):
        return ops.NLLLoss.apply(I[0], tgt)

    _check(f, [(5, 4)], [1])


def test_kldiv():
    p = np.abs(np.random.randn(5, 4).astype("float32")) + 0.1
    p = p / p.sum(axis=1, keepdims=True)

    def f(I):
        return ops.KLDivLoss.apply(I[0], p)

    _check(f, [(5, 4)], [1])


def test_bce():
    tgt = Tensor(np.clip(np.random.randn(4, 3).astype("float32") * 0.3 + 0.5, 0.1, 0.9))

    # map raw logits through sigmoid so the BCE input stays in (0,1)
    def f(I):
        return ops.BCELoss.apply(I[0].sigmoid(), tgt)

    _check(f, [(4, 3)], [1])


def test_smoothl1():
    tgt = Tensor(np.random.randn(4, 3).astype("float32"))

    def f(I):
        return ops.SmoothL1Loss.apply(I[0], tgt)

    _check(f, [(4, 3)], [1])


def test_huber():
    tgt = Tensor(np.random.randn(4, 3).astype("float32"))

    def f(I):
        return ops.HuberLoss.apply(I[0], tgt)

    _check(f, [(4, 3)], [1])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL OPS3 CORRECTNESS TESTS PASSED")
