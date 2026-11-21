import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops
from tests.python.test_autograd_correctness import _rng, numerical_grads, _check


def test_layernorm():
    def f(I):
        return ops.LayerNorm.apply(I[0], I[1], I[2]).sum()
    _check(f, [(4, 3), (3,), (3,)], [1, 2, 3])


def test_rmsnorm():
    def f(I):
        return ops.RMSNorm.apply(I[0], I[1]).sum()
    _check(f, [(4, 3), (3,)], [1, 2])


def test_transpose():
    def f(I):
        return ops.Transpose.apply(I[0], 0, 1).sum()
    _check(f, [(4, 3)], [1])


def test_reshape():
    def f(I):
        return ops.Reshape.apply(I[0], (2, 6)).sum()
    _check(f, [(4, 3)], [1])


def test_expand():
    def f(I):
        return ops.Expand.apply(I[0], (4, 3)).sum()
    _check(f, [(4, 1)], [1])


def test_getitem():
    def f(I):
        return ops.GetItem.apply(I[0], (slice(None), slice(1, 3))).sum()
    _check(f, [(4, 3)], [1])


def test_linear_with_bias():
    def f(I):
        return ops.LinearFn.apply(I[0], I[1], I[2]).sum()
    _check(f, [(5, 4), (6, 4), (6,)], [1, 2, 3])


def test_embedding():
    def f(I):
        return ops.EmbeddingFn.apply(I[0], np.array([0, 2, 1, 3])).sum()
    _check(f, [(5, 3)], [1])


def test_cat():
    def f(I):
        return ops.Cat.apply(I[0], I[1], 0).sum()
    _check(f, [(4, 3), (2, 3)], [1, 2])


def test_stack():
    def f(I):
        return ops.Stack.apply(I[0], I[1], 0).sum()
    _check(f, [(4, 3), (4, 3)], [1, 2])


def test_dropout():
    def f(I):
        return ops.DropoutFn.apply(I[0], 0.3, 123).sum()
    _check(f, [(4, 3)], [1])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL OPS2 CORRECTNESS TESTS PASSED")
