import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops
from tests.python.test_autograd_correctness import _rng, numerical_grads, _check


def test_conv1d():
    _check(lambda I: ops.Conv1d.apply(I[0], I[1]).sum(), [(1, 2, 8), (3, 2, 4)], [1, 2])


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
    tgt = Tensor(
        np.clip(np.random.randn(4, 3).astype("float32") * 0.3 + 0.5, 0.1, 0.9)
    )
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
