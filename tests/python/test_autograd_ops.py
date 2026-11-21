"""Tests for Autograd Ops."""

import numpy as np
from SneppX_ALG.interface_bindings.autograd_ops import Add, Sub, Mul, Div, MatMul, Neg, Pow
from SneppX_ALG.interface_bindings.autograd import Context
from SneppX_ALG.interface_bindings.tensor import Tensor


def make_ctx():
    ctx = Context()
    ctx.save_for_backward = lambda **kwargs: None
    ctx.save_attr = lambda **kwargs: None
    ctx.get_saved_tensor = lambda name: Tensor.ones((2, 2))
    ctx.get_attr = lambda name: None
    return ctx


def test_add_forward():
    a = Tensor.ones((2, 3)) * 2.0
    b = Tensor.ones((2, 3)) * 3.0
    out = Add.forward(None, a, b)
    assert np.allclose(out.data, 5.0)


def test_add_scalar_forward():
    a = Tensor.ones((2, 3)) * 2.0
    out = Add.forward(None, a, 5.0)
    assert np.allclose(out.data, 7.0)


def test_add_backward():
    grad = Tensor.ones((2, 3))
    grads = Add.backward(None, grad)
    assert len(grads) == 2
    assert np.allclose(grads[0].data, 1.0)
    assert np.allclose(grads[1].data, 1.0)


def test_sub_forward():
    a = Tensor.ones((2, 3)) * 10.0
    b = Tensor.ones((2, 3)) * 3.0
    out = Sub.forward(None, a, b)
    assert np.allclose(out.data, 7.0)


def test_sub_backward():
    grad = Tensor.ones((2, 3))
    grads = Sub.backward(None, grad)
    assert np.allclose(grads[0].data, 1.0)
    assert np.allclose(grads[1].data, -1.0)


def test_mul_forward():
    a = Tensor.ones((2, 3)) * 3.0
    b = Tensor.ones((2, 3)) * 4.0
    out = Mul.forward(make_ctx(), a, b)
    assert np.allclose(out.data, 12.0)


def test_mul_scalar_forward():
    a = Tensor.ones((2, 3)) * 3.0
    out = Mul.forward(make_ctx(), a, 5.0)
    assert np.allclose(out.data, 15.0)


def test_div_forward():
    a = Tensor.ones((2, 3)) * 10.0
    b = Tensor.ones((2, 3)) * 2.0
    out = Div.forward(make_ctx(), a, b)
    assert np.allclose(out.data, 5.0)


def test_neg_forward():
    a = Tensor.ones((2, 3)) * 5.0
    out = Neg.forward(None, a)
    assert np.allclose(out.data, -5.0)


def test_neg_backward():
    grad = Tensor.ones((2, 3))
    grads = Neg.backward(None, grad)
    assert np.allclose(grads[0].data, -1.0)


def test_matmul_forward():
    a = Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]]))
    out = MatMul.forward(make_ctx(), a, b)
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(out.data, expected)


def test_pow_forward():
    a = Tensor.ones((2, 3)) * 3.0
    out = Pow.forward(make_ctx(), a, 2.0)
    assert np.allclose(out.data, 9.0)


if __name__ == "__main__":
    import sys
    locals_ = locals().copy()
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  PASS {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
