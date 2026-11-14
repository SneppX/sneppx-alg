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

def _finite_difference_matmul(a, b, epsilon=1e-3):
    def objective(left, right):
        return float(np.asarray(np.matmul(left, right)).sum())

    grad_a = np.zeros_like(a, dtype=np.float64)
    grad_b = np.zeros_like(b, dtype=np.float64)
    for index in np.ndindex(a.shape):
        plus = a.copy()
        minus = a.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        grad_a[index] = (
            objective(plus, b) - objective(minus, b)
        ) / (2.0 * epsilon)
    for index in np.ndindex(b.shape):
        plus = b.copy()
        minus = b.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        grad_b[index] = (
            objective(a, plus) - objective(a, minus)
        ) / (2.0 * epsilon)
    return grad_a, grad_b


def test_matmul_backward_matches_finite_differences_for_vector_and_batch_shapes():
    rng = np.random.default_rng(20261114)
    cases = [
        ((3,), (3,)),
        ((3,), (2, 3, 4)),
        ((2, 4, 3), (3,)),
        ((2, 1, 3, 4), (1, 5, 4, 2)),
    ]

    for a_shape, b_shape in cases:
        a_data = rng.normal(size=a_shape).astype(np.float32)
        b_data = rng.normal(size=b_shape).astype(np.float32)
        a = Tensor(a_data.copy(), requires_grad=True)
        b = Tensor(b_data.copy(), requires_grad=True)

        (a @ b).sum().backward()
        expected_a, expected_b = _finite_difference_matmul(a_data, b_data)

        np.testing.assert_allclose(
            a.grad.data, expected_a, rtol=2e-3, atol=2e-3,
            err_msg=f"left gradient mismatch for {a_shape} @ {b_shape}",
        )
        np.testing.assert_allclose(
            b.grad.data, expected_b, rtol=2e-3, atol=2e-3,
            err_msg=f"right gradient mismatch for {a_shape} @ {b_shape}",
        )


def test_batched_matmul_create_graph_preserves_second_order_dependencies():
    left = Tensor(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10.0,
        requires_grad=True,
    )
    right = Tensor(
        np.arange(20, dtype=np.float32).reshape(1, 4, 5) / 10.0,
        requires_grad=True,
    )

    first_left, first_right = grad(
        (left @ right).sum(), [left, right], create_graph=True
    )
    assert first_left.grad_fn is not None
    assert first_right.grad_fn is not None

    second_right = grad(first_left.sum(), right)[0]
    expected = np.full(right.shape, left.shape[0] * left.shape[1], dtype=np.float32)
    np.testing.assert_allclose(
        second_right.data, expected, rtol=1e-5, atol=1e-5
    )

def test_getitem_backward_accumulates_duplicate_advanced_indices():
    source = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    source[[0, 0, 2]].sum().backward()

    np.testing.assert_array_equal(source.grad.data, [2.0, 0.0, 1.0])


def test_getitem_backward_accumulates_duplicate_multidimensional_indices():
    source = Tensor(
        np.arange(1, 10, dtype=np.float32).reshape(3, 3),
        requires_grad=True,
    )

    source[[0, 0, 2], [1, 1, 0]].sum().backward()

    expected = np.zeros((3, 3), dtype=np.float32)
    expected[0, 1] = 2.0
    expected[2, 0] = 1.0
    np.testing.assert_array_equal(source.grad.data, expected)


def test_getitem_duplicate_indices_match_with_create_graph():
    source = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    result = grad(source[[0, 0, 2]].sum(), source, create_graph=True)[0]

    np.testing.assert_array_equal(result.data, [2.0, 0.0, 1.0])
    assert result.grad_fn is not None

def test_prod_backward_handles_single_and_multiple_zeros():
    single_zero = Tensor([0.0, 2.0, 3.0], requires_grad=True)
    single_zero.prod().backward()
    np.testing.assert_allclose(single_zero.grad.data, [6.0, 0.0, 0.0])

    multiple_zeros = Tensor([0.0, 2.0, 0.0, 3.0], requires_grad=True)
    multiple_zeros.prod().backward()
    np.testing.assert_array_equal(multiple_zeros.grad.data, [0.0, 0.0, 0.0, 0.0])


def test_prod_backward_handles_zero_groups_across_multiple_axes_and_keepdim():
    values = np.array(
        [
            [[0.0, 2.0], [3.0, 4.0]],
            [[5.0, 0.0], [0.0, 7.0]],
        ],
        dtype=np.float32,
    )
    source = Tensor(values.copy(), requires_grad=True)
    source.prod(dim=(1, 2), keepdim=True).sum().backward()

    expected = np.zeros_like(values)
    expected[0, 0, 0] = 2.0 * 3.0 * 4.0
    expected[1, 0, 1] = 5.0 * 0.0 * 7.0
    np.testing.assert_allclose(source.grad.data, expected)


def test_prod_create_graph_preserves_second_order_dependencies_at_zero():
    source = Tensor([0.0, 2.0, 3.0], requires_grad=True)
    first = grad(source.prod(), source, create_graph=True)[0]

    np.testing.assert_allclose(first.data, [6.0, 0.0, 0.0])
    assert first.grad_fn is not None

    second = grad(first[0], source)[0]
    np.testing.assert_allclose(second.data, [0.0, 3.0, 2.0])


def test_prod_backward_matches_finite_differences_with_zero_inputs():
    values = np.array([0.0, -2.0, 3.0], dtype=np.float32)
    source = Tensor(values.copy(), requires_grad=True)
    source.prod().backward()

    epsilon = 1e-3
    expected = np.empty_like(values)
    for index in range(values.size):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        expected[index] = (
            np.prod(plus) - np.prod(minus)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(source.grad.data, expected, rtol=2e-3, atol=2e-3)

