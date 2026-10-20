"""Tests for sneppx_serve.pytorch_api (sneppx-alg interface_bindings.pytorch_api)."""

import numpy as np

from SneppX_ALG.interface_bindings.pytorch_api import Tensor, Linear, Module


def test_tensor_shape():
    t = Tensor(np.array([1.0, 2, 3]))
    assert t.shape == (3,)


def test_tensor_dtype():
    t = Tensor(np.array([1.0, 2, 3]))
    assert t.dtype is not None


def test_linear_forward():
    m = Linear(3, 2)
    t = Tensor(np.array([1.0, 2, 3], dtype=np.float64))
    out = m(t)
    assert out.shape == (2,)


def test_module_zero_grad():
    m = Linear(3, 2)
    assert hasattr(m, "zero_grad")


def test_accuracy():
    from SneppX_ALG.interface_bindings.pytorch_api import accuracy

    a = accuracy(np.array([0, 1, 2]), np.array([0, 1, 3]))
    assert 0 <= a <= 1
