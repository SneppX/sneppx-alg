"""Tests for JAX-style tracing transforms: jit/grad/value_and_grad/jacobian/hessian/vmap."""

import numpy as np
from SneppX_ALG.interface_bindings.jit import (
    jit,
    grad,
    value_and_grad,
    jacobian,
    hessian,
    vmap,
    trace_function,
)

RNG = np.random.default_rng(11)


def _mlp(x, w, b):
    h = (x @ w) + b
    return (h / (1 + np.exp(-h))).sum()


def _finite_diff(fn, arg, idx, h=1e-3):
    e = np.zeros_like(arg)
    e.flat[idx] = h
    return (fn(arg + e) - fn(arg - e)) / (2 * h)


def test_jit_matches_python():
    x = RNG.normal(size=6).astype(np.float32)
    w = RNG.normal(size=(6, 4)).astype(np.float32)
    b = RNG.normal(size=4).astype(np.float32)
    jf = jit(_mlp)
    assert abs(jf(x, w, b) - _mlp(x, w, b)) < 1e-5
    assert abs(jf(x, w, b) - _mlp(x, w, b)) < 1e-5
    print("  test_jit_matches_python PASS")


def test_trace_function():
    x = np.arange(4, dtype=np.float32)
    tr = trace_function(lambda a: (a * a).sum(), x)
    assert len(tr.inputs) == 1
    assert abs(tr.execute(x) - (x * x).sum()) < 1e-5
    print("  test_trace_function PASS")


def test_grad_single_argnum():
    x = RNG.normal(size=6).astype(np.float32)
    w = RNG.normal(size=(6, 4)).astype(np.float32)
    b = RNG.normal(size=4).astype(np.float32)
    gx = grad(_mlp, argnums=0)(x, w, b)
    num = np.array([_finite_diff(lambda t: _mlp(t, w, b), x, k)
                    for k in range(x.size)])
    assert gx.shape == x.shape
    assert np.abs(gx - num).max() < 1e-3
    gw = grad(_mlp, argnums=1)(x, w, b)
    numw = np.array([_finite_diff(lambda t: _mlp(x, t, b), w, k)
                     for k in range(w.size)]).reshape(w.shape)
    assert gw.shape == w.shape
    assert np.abs(gw - numw).max() < 1e-3
    gb = grad(_mlp, argnums=2)(x, w, b)
    numb = np.array([_finite_diff(lambda t: _mlp(x, w, t), b, k)
                     for k in range(b.size)]).reshape(b.shape)
    assert np.abs(gb - numb).max() < 1e-3
    print("  test_grad_single_argnum PASS")


def test_grad_multi_argnum_returns_tuple():
    x = RNG.normal(size=6).astype(np.float32)
    w = RNG.normal(size=(6, 4)).astype(np.float32)
    b = RNG.normal(size=4).astype(np.float32)
    g = grad(_mlp, argnums=[0, 1, 2])(x, w, b)
    assert isinstance(g, tuple) and len(g) == 3
    assert g[0].shape == x.shape and g[1].shape == w.shape and g[2].shape == b.shape
    num0 = np.array([_finite_diff(lambda t: _mlp(t, w, b), x, k)
                     for k in range(x.size)])
    num1 = np.array([_finite_diff(lambda t: _mlp(x, t, b), w, k)
                     for k in range(w.size)]).reshape(w.shape)
    num2 = np.array([_finite_diff(lambda t: _mlp(x, w, t), b, k)
                     for k in range(b.size)]).reshape(b.shape)
    assert np.abs(g[0] - num0).max() < 1e-3
    assert np.abs(g[1] - num1).max() < 1e-3
    assert np.abs(g[2] - num2).max() < 1e-3
    print("  test_grad_multi_argnum_returns_tuple PASS")


def test_value_and_grad():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    f = lambda a: (a * a).sum()
    v, g = value_and_grad(f)(x)
    assert abs(v - 14.0) < 1e-5
    assert np.allclose(g, [2.0, 4.0, 6.0], atol=1e-5)
    print("  test_value_and_grad PASS")


def test_jacobian():
    x = RNG.normal(size=4).astype(np.float32)
    f = lambda a: (a * a) + (2.0 * a)
    J = jacobian(f)(x)
    assert J.shape == (4, 4)
    assert np.allclose(J, np.diag(2.0 * x + 2.0), atol=1e-5)
    print("  test_jacobian PASS")


def test_hessian_via_finite_diff():
    x = RNG.normal(size=4).astype(np.float32)
    f = lambda a: (0.5 * a * a + np.exp(a)).sum()
    H = hessian(f)(x)
    assert H.shape == (4, 4)
    fg = grad(f)
    num = np.zeros((4, 4))
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1e-3
        num[i] = (fg(x + e) - fg(x - e)) / (2 * 1e-3)
    assert np.abs(H - num).max() < 1e-4
    print("  test_hessian_via_finite_diff PASS")


def test_hessian_cross_partials():
    f = lambda x: x[0] * x[0] * x[1]
    H = hessian(f)(np.array([0.7, 1.3], dtype=np.float32))
    assert H.shape == (2, 2)
    assert abs(H[0, 1] - 1.4) < 1e-4
    assert abs(H[1, 0] - 1.4) < 1e-4
    assert abs(H[0, 0] - 2.6) < 1e-4
    assert abs(H[1, 1]) < 1e-6
    # exact [[0,1],[1,0]] case
    H2 = hessian(lambda x: x[0] * x[1])(np.array([3.0, 4.0]))
    assert np.allclose(H2, [[0.0, 1.0], [1.0, 0.0]], atol=1e-6)
    print("  test_hessian_cross_partials PASS")


def test_mean_and_reduce_grads():
    gmean = grad(lambda a: a.mean())(np.arange(5, dtype=np.float32))
    assert np.allclose(gmean, 0.2, atol=1e-6)
    gsum = grad(lambda a: a.sum())(np.arange(5, dtype=np.float32))
    assert np.allclose(gsum, 1.0, atol=1e-6)
    gsum_axis = grad(lambda a: a.sum(axis=0))(np.arange(6).reshape(2, 3).astype(np.float32))
    assert gsum_axis.shape == (2, 3)
    assert np.allclose(gsum_axis, 1.0, atol=1e-6)
    print("  test_mean_and_reduce_grads PASS")


def test_getitem_and_reshape_grads():
    ggi = grad(lambda a: (a[0] * a[1]).sum())(np.array([2.0, 3.0, 4.0], dtype=np.float32))
    assert np.allclose(ggi, [3.0, 2.0, 0.0], atol=1e-5)
    ggr = grad(lambda a: a.reshape(2, 2).sum())(np.arange(4, dtype=np.float32))
    assert np.allclose(ggr, [1.0, 1.0, 1.0, 1.0], atol=1e-6)
    print("  test_getitem_and_reshape_grads PASS")


def test_matmul_gradients():
    w = RNG.normal(size=(3, 2)).astype(np.float32)
    body = lambda a: (a @ w).sum()
    x = RNG.normal(size=3).astype(np.float32)
    g = grad(body)(x)
    assert np.allclose(g, np.ones(2) @ w.T, atol=1e-5)
    num = np.array([_finite_diff(body, x, k) for k in range(3)])
    assert np.abs(g - num).max() < 1e-3
    # matrix operand gradient
    gm = grad(lambda M: (M @ x).sum())(RNG.normal(size=(2, 3)).astype(np.float32))
    assert gm.shape == (2, 3)
    print("  test_matmul_gradients PASS")


def test_vmap():
    w = RNG.normal(size=(3, 2)).astype(np.float32)
    X = RNG.normal(size=(5, 3)).astype(np.float32)
    body = lambda a: (a @ w).sum()
    out = vmap(body)(X)
    assert out.shape == (5,)
    ref = np.array([body(X[i]) for i in range(5)])
    assert np.allclose(out, ref, atol=1e-5)
    meanv = vmap(lambda a: (a * a).mean())(np.arange(12).reshape(4, 3).astype(np.float32))
    assert meanv.shape == (4,)
    assert np.allclose(meanv, np.array([(r * r).mean() for r in np.arange(12).reshape(4, 3)]), atol=1e-5)
    print("  test_vmap PASS")


def test_vmap_over_multiple_args():
    a = RNG.normal(size=(4, 3)).astype(np.float32)
    b = RNG.normal(size=(4, 3)).astype(np.float32)
    out = vmap(lambda p, q: (p * q).sum())(a, b)
    ref = np.array([(a[i] * b[i]).sum() for i in range(4)])
    assert np.allclose(out, ref, atol=1e-5)
    print("  test_vmap_over_multiple_args PASS")


if __name__ == "__main__":
    test_jit_matches_python()
    test_trace_function()
    test_grad_single_argnum()
    test_grad_multi_argnum_returns_tuple()
    test_value_and_grad()
    test_jacobian()
    test_hessian_via_finite_diff()
    test_hessian_cross_partials()
    test_mean_and_reduce_grads()
    test_getitem_and_reshape_grads()
    test_matmul_gradients()
    test_vmap()
    test_vmap_over_multiple_args()
    print("ALL jit TESTS PASS")
