"""API-level regression tests for autograd.grad target handling."""

import numpy as np

from SneppX_ALG import Tensor
from SneppX_ALG.interface_bindings.autograd import grad


def test_grad_returns_gradient_for_non_leaf_intermediate():
    source = Tensor([2.0, 3.0], requires_grad=True)
    intermediate = source * source
    output = intermediate * 3.0

    result = grad(output, intermediate)[0]

    np.testing.assert_allclose(result.data, [3.0, 3.0], rtol=1e-6, atol=1e-7)
    assert source.grad is None
    assert intermediate.grad is None


def test_grad_still_propagates_through_intermediate_to_leaf():
    source = Tensor([2.0, 3.0], requires_grad=True)
    intermediate = source * source
    output = intermediate * 3.0

    result = grad(output, source)[0]

    np.testing.assert_allclose(result.data, [12.0, 18.0], rtol=1e-6, atol=1e-7)
    assert source.grad is None


def test_grad_of_tensor_with_respect_to_itself_is_identity():
    source = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    result = grad(source, source)[0]

    np.testing.assert_array_equal(result.data, np.ones_like(source.data))




def test_grad_accumulates_multiple_outputs_for_non_leaf_input():
    source = Tensor([1.0, 2.0], requires_grad=True)
    intermediate = source * 2.0
    first_output = (intermediate * 3.0).sum()
    second_output = (intermediate * 4.0).sum()

    result = grad([first_output, second_output], intermediate)[0]

    np.testing.assert_allclose(result.data, [7.0, 7.0], rtol=1e-6, atol=1e-7)
    assert source.grad is None


def test_grad_create_graph_supports_second_derivative():
    source = Tensor([1.5, -2.0, 0.25], requires_grad=True)
    loss = (source * source * source).sum()

    first = grad(loss, source, create_graph=True)[0]
    second = grad(first.sum(), source)[0]

    np.testing.assert_allclose(
        second.data,
        6.0 * source.data,
        rtol=1e-5,
        atol=1e-6,
    )


def test_grad_allow_unused_returns_none_for_unrelated_input():
    output_source = Tensor([1.0, 2.0], requires_grad=True)
    unrelated = Tensor([3.0, 4.0], requires_grad=True)

    result = grad(output_source, unrelated, allow_unused=True)

    assert result == [None]
