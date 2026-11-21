"""Tests for c_tensor_expr.py — tensor expression kernels."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings import TensorExpr, TensorExprCompiler


def test_constant():
    data = np.array([1.0, 2.0, 3.0])
    expr = TensorExpr.constant(data)
    result = expr.evaluate()
    assert np.allclose(result, data)


def test_add():
    a = TensorExpr.constant(np.array([1.0, 2.0]))
    b = TensorExpr.constant(np.array([3.0, 4.0]))
    expr = TensorExpr.add(a, b)
    result = expr.evaluate()
    assert np.allclose(result, [4.0, 6.0])


def test_mul():
    a = TensorExpr.constant(np.array([2.0, 3.0]))
    b = TensorExpr.constant(np.array([4.0, 5.0]))
    expr = TensorExpr.mul(a, b)
    result = expr.evaluate()
    assert np.allclose(result, [8.0, 15.0])


def test_matmul():
    a = TensorExpr.constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = TensorExpr.constant(np.array([[5.0, 6.0], [7.0, 8.0]]))
    expr = TensorExpr.matmul(a, b)
    result = expr.evaluate()
    assert np.allclose(result, a.value @ b.value)


def test_relu():
    x = TensorExpr.constant(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    expr = TensorExpr.relu(x)
    result = expr.evaluate()
    assert np.allclose(result, [0.0, 0.0, 0.0, 1.0, 2.0])


def test_composite():
    a = TensorExpr.constant(np.array([1.0, 2.0]))
    b = TensorExpr.constant(np.array([3.0, 4.0]))
    s = TensorExpr.add(a, b)
    m = TensorExpr.mul(s, b)
    result = m.evaluate()
    expected = (a.value + b.value) * b.value
    assert np.allclose(result, expected)


def test_compiler_compile():
    comp = TensorExprCompiler()
    a = TensorExpr.constant(np.array([1.0]))
    b = TensorExpr.constant(np.array([2.0]))
    expr = TensorExpr.add(a, b)
    plan = comp.compile(expr)
    assert len(plan) >= 1
    assert isinstance(plan, list)


def test_repr():
    expr = TensorExpr("add",
                      TensorExpr.constant(np.array([1.0])),
                      TensorExpr.constant(np.array([2.0])))
    r = repr(expr)
    assert "TensorExpr" in r


if __name__ == "__main__":
    test_constant()
    test_add()
    test_mul()
    test_matmul()
    test_relu()
    test_composite()
    test_compiler_compile()
    test_repr()
    print("All c_tensor_expr tests passed.")
