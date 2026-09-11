import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor

EPS = 1e-6


def fd(inp, loss_fn):
    base = inp.data.copy()
    g = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        o = base[idx]
        base[idx] = o + EPS
        lp = loss_fn(Tensor(base.copy(), dtype=inp.dtype)).data.flat[0]
        base[idx] = o - EPS
        lm = loss_fn(Tensor(base.copy(), dtype=inp.dtype)).data.flat[0]
        base[idx] = o
        g[idx] = (lp - lm) / (2 * EPS)
        it.iternext()
    return g


def check(name, inp, loss_fn, atol=1e-5):
    inp2 = Tensor(inp.data.copy())
    inp2.requires_grad_(True)
    loss_fn(inp2).backward()
    numeric = fd(inp, loss_fn)
    assert np.allclose(inp2.grad.data, numeric, atol=atol), \
        f"{name}: max|d|={np.abs(inp2.grad.data - numeric).max()}"
    print(f"{name}: OK")


def test_elementwise_math_backward_fd():
    x = Tensor(np.random.randn(4, 3).astype(np.float64) + 2.0, dtype="float64")

    def sq(i):
        return (i.square() * i.square()).mean()

    check("square", x, sq)

    def rec(i):
        return (i.reciprocal() * i.reciprocal()).mean()

    check("reciprocal", x, rec)

    def clamp_l(i):
        return (i.clamp(1.0, 2.5) ** 2).mean()

    check("clamp", x, clamp_l)

    def clamp_min(i):
        return (i.clamp_min(1.5) ** 2).mean()

    check("clamp_min", x, clamp_min)

    def clamp_max(i):
        return (i.clamp_max(2.0) ** 2).mean()

    check("clamp_max", x, clamp_max)

    def rem(i):
        return (i.remainder(2.0) ** 2).mean()

    check("remainder", x, rem)

    def mod(i):
        return (i.mod(3.0) ** 2).mean()

    check("mod", x, mod)


def test_sign_zero_grad():
    x = Tensor(np.random.randn(4, 3).astype(np.float64), dtype="float64")
    x.requires_grad_(True)
    (x.sign() * x.sign()).mean().backward()
    assert np.all(np.asarray(x.grad.data) == 0.0)
    assert np.allclose(np.asarray(x.sign().data),
                       np.sign(np.asarray(x.data)))


def test_reduction_backward_fd():
    x = Tensor(np.random.randn(3, 4).astype(np.float64) + 1.5, dtype="float64")

    def prod_dim(i):
        return i.prod(dim=1).sum()

    check("prod(dim=1)", x, prod_dim)

    def prod_all(i):
        return i.prod()

    check("prod()", x, prod_all)

    def lse(i):
        o = i.logsumexp(dim=0)
        return (o * o).sum()

    check("logsumexp(dim=0)", x, lse)

    def amin_d(i):
        return i.amin(dim=1).sum()

    check("amin(dim=1)", x, amin_d)

    def amax_d(i):
        return i.amax(dim=1).sum()

    check("amax(dim=1)", x, amax_d)

    def amin_all(i):
        return i.amin()

    check("amin()", x, amin_all)

    def amax_all(i):
        return i.amax()

    check("amax()", x, amax_all)


def test_trace_diagonal_backward_fd():
    x = Tensor(np.random.randn(4, 4).astype(np.float64), dtype="float64")

    def tr(i):
        return i.trace() * 2.0

    check("trace", x, tr)

    def diag(i):
        return (i.diagonal() ** 2).sum()

    check("diagonal", x, diag)

    def diag_off(i):
        return (i.diagonal(offset=1) ** 2).sum()

    check("diagonal(offset=1)", x, diag_off)

    def diag_neg(i):
        return (i.diagonal(offset=-1) ** 2).sum()

    check("diagonal(offset=-1)", x, diag_neg)


def test_argmin_argmax_values_and_shapes():
    arr = np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]])
    t = Tensor(arr.astype(np.float64))
    ai = np.asarray(t.argmax(dim=1).data)
    assert ai.tolist() == [1, 2]
    assert ai.dtype == np.int64
    ni = np.asarray(t.argmin(dim=1).data)
    assert ni.tolist() == [0, 1]
    keep = np.asarray(t.argmax(dim=1, keepdim=True).data)
    assert keep.shape == (2, 1) and keep[0, 0] == 1
    flat = np.asarray(t.argmax().data)
    assert flat.dtype == np.int64
    assert int(flat) == 5


def test_keepdim_and_higher_order_elementwise():
    arr = np.random.randn(3, 4).astype(np.float64) + 2.0
    t = Tensor(arr.copy())
    o = t.prod(dim=1, keepdim=True)
    assert o.shape == (3, 1)
    lse = t.logsumexp(dim=1, keepdim=True)
    assert lse.shape == (3, 1)
    a = Tensor(arr.copy())
    a.requires_grad_(True)
    (a.square() ** 2).sum().backward()
    assert np.allclose(np.asarray(a.grad.data), 4.0 * arr ** 3)

    # prod(dim=tuple)
    p = t.prod(dim=(0, 1))
    assert np.allclose(np.asarray(p.data), arr.prod())


if __name__ == "__main__":
    test_elementwise_math_backward_fd()
    test_sign_zero_grad()
    test_reduction_backward_fd()
    test_trace_diagonal_backward_fd()
    test_argmin_argmax_values_and_shapes()
    test_keepdim_and_higher_order_elementwise()
    print("all ok")