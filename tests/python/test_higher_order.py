"""Test to verify higher-order autograd support (grad-of-grad)."""

import numpy as np
import sys

sys.path.insert(0, "bindings/python")

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import functional as F


def test_grad_of_grad():
    # f(x) = x^3 => f'(x) = 3x^2 => f''(x) = 6x
    x = Tensor(np.array([2.0]), requires_grad=True, dtype="float64")

    def f(x):
        return x**3

    # First derivative
    grad_f_func = F.grad(f)
    grad_f = grad_f_func(x)[0]
    assert np.allclose(grad_f.data, 3.0 * (x.data**2))

    # Second derivative: grad of (grad_f)
    def grad_f_as_func(x):
        return F.grad(f)(x)[0]

    grad_grad_f = F.grad(grad_f_as_func)(x)[0]
    assert np.allclose(grad_grad_f.data, 6.0 * x.data)


if __name__ == "__main__":
    test_grad_of_grad()
    print("test_grad_of_grad: OK")
