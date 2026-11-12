"""Regression tests for matrix-multiplication and multi-output gradients."""

import numpy as np

from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings.autograd import grad


def test_vector_dot_product_backward():
    left = Tensor([1.0, 2.0], requires_grad=True)
    right = Tensor([3.0, 4.0], requires_grad=True)

    (left @ right).backward()

    np.testing.assert_allclose(left.grad.data, [3.0, 4.0])
    np.testing.assert_allclose(right.grad.data, [1.0, 2.0])


def test_grad_accumulates_contributions_from_multiple_outputs():
    value = Tensor([2.0, 3.0], requires_grad=True)
    squared_sum = (value * value).sum()
    scaled_sum = (value * 3).sum()

    result = grad([squared_sum, scaled_sum], value)[0]

    np.testing.assert_allclose(result.data, [7.0, 9.0], rtol=1e-5, atol=1e-6)
