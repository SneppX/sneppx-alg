import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings.autograd import grad, jvp, vjp


def _fd_jvp(func, inputs, tangents, h=1e-4):
    # (f(x+h v) - f(x-h v)) / (2h)
    def eval_at(scale):
        pert = [Tensor(np.asarray(i.data) + scale * h * np.asarray(t.data)) for i, t in zip(inputs, tangents)]
        return func(*pert)

    fp = eval_at(1.0)
    fm = eval_at(-1.0)
    return np.asarray((fp.data - fm.data) / (2 * h)).astype(np.float64)


def test_jvp_linear():
    np.random.seed(61)
    x = Tensor(np.random.randn(2, 3).astype("float32")); x.requires_grad_(True)
    W = Tensor(np.random.randn(3, 4).astype("float32")); W.requires_grad_(True)
    vx = Tensor(np.random.randn(2, 3).astype("float32"))
    vW = Tensor(np.random.randn(3, 4).astype("float32"))

    def f(a, b):
        return a @ b

    _, jv = jvp(f, [x, W], [vx, vW])
    ref = vx.data @ W.data + x.data @ vW.data
    assert np.allclose(jv.data.astype(np.float64), ref.astype(np.float64), atol=1e-3), (
        f"JVP linear max|err|={np.max(np.abs(jv.data.astype(np.float64) - ref)):.3e}"
    )
    fd = _fd_jvp(f, [x, W], [vx, vW])
    assert np.allclose(jv.data.astype(np.float64), fd, atol=1e-3), (
        f"JVP linear FD max|err|={np.max(np.abs(jv.data.astype(np.float64) - fd)):.3e}"
    )


def test_vjp_linear():
    np.random.seed(63)
    x = Tensor(np.random.randn(2, 3).astype("float32")); x.requires_grad_(True)
    W = Tensor(np.random.randn(3, 4).astype("float32")); W.requires_grad_(True)
    u = Tensor(np.random.randn(2, 4).astype("float32"))  # cotangent

    def f(a, b):
        return a @ b

    out, vjps = vjp(f, [x, W], u)
    gx_ref = u.data @ W.data.T
    gW_ref = x.data.T @ u.data
    assert np.allclose(vjps[0].data.astype(np.float64), gx_ref.astype(np.float64), atol=1e-3), (
        f"VJP x max|err|={np.max(np.abs(vjps[0].data.astype(np.float64) - gx_ref)):.3e}"
    )
    assert np.allclose(vjps[1].data.astype(np.float64), gW_ref.astype(np.float64), atol=1e-3), (
        f"VJP W max|err|={np.max(np.abs(vjps[1].data.astype(np.float64) - gW_ref)):.3e}"
    )


def test_jvp_nonlinear_elementwise():
    np.random.seed(67)
    x = Tensor(np.random.randn(5).astype("float32")); x.requires_grad_(True)
    v = Tensor(np.random.randn(5).astype("float32"))

    def f(a):
        return a * a  # R^n -> R^n, J = diag(2a)

    _, jv = jvp(f, [x], [v])
    ref = 2.0 * x.data * v.data
    assert np.allclose(jv.data.astype(np.float64), ref.astype(np.float64), atol=1e-3), (
        f"JVP nonlinear max|err|={np.max(np.abs(jv.data.astype(np.float64) - ref)):.3e}"
    )
    fd = _fd_jvp(f, [x], [v])
    assert np.allclose(jv.data.astype(np.float64), fd, atol=1e-3), (
        f"JVP nonlinear FD max|err|={np.max(np.abs(jv.data.astype(np.float64) - fd)):.3e}"
    )


def test_jvp_chain_with_norm():
    # JVP through LayerNorm -> MatMul, validating the forward-mode path on a
    # real composite used in the higher-order suites.
    np.random.seed(69)
    x = Tensor(np.random.randn(3, 4).astype("float32")); x.requires_grad_(True)
    W = Tensor(np.random.randn(4, 2).astype("float32")); W.requires_grad_(True)
    g = Tensor(np.random.randn(3, 4).astype("float32"))
    vx = Tensor(np.random.randn(3, 4).astype("float32"))
    vW = Tensor(np.random.randn(4, 2).astype("float32"))

    def f(a, b):
        return ops.LayerNorm.apply(a, g, ops._as_const(0.0, g.dtype)) @ b

    _, jv = jvp(f, [x, W], [vx, vW])
    fd = _fd_jvp(f, [x, W], [vx, vW])
    assert np.allclose(jv.data.astype(np.float64), fd, atol=2e-3), (
        f"JVP chain FD max|err|={np.max(np.abs(jv.data.astype(np.float64) - fd)):.3e}"
    )


if __name__ == "__main__":
    test_jvp_linear()
    test_vjp_linear()
    test_jvp_nonlinear_elementwise()
    test_jvp_chain_with_norm()
    print("ALL JVP/VJP TESTS PASSED")
