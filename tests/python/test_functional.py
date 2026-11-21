import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import functional as F


def _T(arr):
    return Tensor(np.asarray(arr, dtype=np.float64), dtype="float64")


def test_jvp_scalar():
    # f(a,b) = a^2 * b + a^3, analytic jvp = (2ab+3a^2)*ta + a^2*tb
    def f(a, b):
        return (a * a) * b + a * a * a
    a = _T([1.3, 0.7])
    b = _T([2.0])
    ta = np.array([0.4, -0.2])
    tb = np.array([0.9])
    out, jv = F.jvp(f, [a, b], [_T(ta), _T(tb)])
    av, bv = a.data, b.data
    expected = (2 * av * bv + 3 * av**2) * ta + (av**2) * tb
    # out is vector-valued here; compare per-coordinate
    assert np.allclose(jv.data, expected, atol=1e-9), f"jvp {jv.data} vs {expected}"
    print("jvp scalar: OK")


def test_vjp_identity():
    # identity: v^T (J t) == (J^T v) . t  -> with v = ones, sum(jvp) == vjp_cts . t
    def f(a, b):
        return (a * a) * b + a * a * a
    a = _T([1.3, 0.7])
    b = _T([2.0])
    ta = np.array([0.4, -0.2])
    tb = np.array([0.9])
    out, jv = F.jvp(f, [a, b], [_T(ta), _T(tb)])
    _, vjp_cts = F.vjp(f, [a, b], Tensor(np.ones_like(out.data), dtype="float64"))
    lhs = float(np.sum(jv.data))
    rhs = float(np.sum(vjp_cts[0].data * ta) + np.sum(vjp_cts[1].data * tb))
    assert np.allclose(lhs, rhs, atol=1e-9), f"identity {lhs} vs {rhs}"
    print("vjp/jvp identity: OK")


def test_jacrev_matches_fd():
    # f(a) = [a0^2, a0*a1, a1^3]; use differentiable ops so the graph is built
    from SneppX_ALG.interface_bindings.autograd_ops import Stack
    def f(a):
        a0 = a[0]
        a1 = a[1]
        return Stack.apply(a0 * a0, a0 * a1, a1 * a1 * a1, 0)
    a = _T([1.3, 0.7])
    (J,) = F.jacrev(f, [a])
    expected = np.array([
        [2 * a.data[0], 0.0],
        [a.data[1], a.data[0]],
        [0.0, 3 * a.data[1]**2],
    ])
    assert np.allclose(J, expected, atol=1e-9), f"jacrev {J} vs {expected}"
    print("jacrev: OK")


def test_jacfwd_matches_jacrev():
    from SneppX_ALG.interface_bindings.autograd_ops import Stack
    def f(a):
        a0 = a[0]
        a1 = a[1]
        return Stack.apply(a0 * a0, a0 * a1, a1 * a1 * a1, 0)
    a = _T([1.3, 0.7])
    (Jr,) = F.jacrev(f, [a])
    (Jf,) = F.jacfwd(f, [a])
    assert np.allclose(Jr, Jf, atol=1e-9), f"jacfwd {Jf} vs jacrev {Jr}"
    print("jacfwd == jacrev: OK")


def test_jvp_mlp_loss():
    rng = np.random.RandomState(0)
    X = _T(rng.randn(2, 3))
    W1 = _T(rng.randn(3, 4))
    b1 = _T(rng.randn(4))
    W2 = _T(rng.randn(4, 2))
    b2 = _T(rng.randn(2))

    def f(X, w1, b1, w2, b2):
        h = (X @ w1 + b1).relu()
        y = h @ w2 + b2
        return (y * y).mean()

    tx = rng.randn(2, 3)
    tw1 = rng.randn(3, 4)
    tb1 = rng.randn(4)
    tw2 = rng.randn(4, 2)
    tb2 = rng.randn(2)
    eps = 1e-6
    base = [X.data.copy(), W1.data.copy(), b1.data.copy(), W2.data.copy(), b2.data.copy()]
    tan = [tx, tw1, tb1, tw2, tb2]

    def eval_at(scale):
        xs = _T(base[0] + scale * tan[0])
        w1s = _T(base[1] + scale * tan[1])
        b1s = _T(base[2] + scale * tan[2])
        w2s = _T(base[3] + scale * tan[3])
        b2s = _T(base[4] + scale * tan[4])
        return float(f(xs, w1s, b1s, w2s, b2s).data)

    num = (eval_at(eps) - eval_at(-eps)) / (2 * eps)
    _, jv = F.jvp(f, [X, W1, b1, W2, b2],
                  [_T(tx), _T(tw1), _T(tb1), _T(tw2), _T(tb2)])
    assert np.allclose(float(jv.data), num, atol=1e-4), f"mlp jvp {float(jv.data)} vs {num}"
    print("jvp mlp loss: OK")


if __name__ == "__main__":
    test_jvp_scalar()
    test_vjp_identity()
    test_jacrev_matches_fd()
    test_jacfwd_matches_jacrev()
    test_jvp_mlp_loss()
    print("ALL FUNCTIONAL OK")
