import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops


def _rng(*shape, seed=0):
    np.random.seed(seed)
    return Tensor(np.random.randn(*shape).astype("float32"))


def numerical_grads(f, inputs, eps=1e-3):
    """Central-difference numerical gradients of scalar f(inputs).

    Rebuilds input tensors (Tensor.data is a copy, not a view) so that
    perturbations do not alias the originals.
    """
    grads = []
    for idx in range(len(inputs)):
        base = inputs[idx].data
        ng = np.zeros(base.shape, dtype="float32")
        it = np.nditer(base, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            orig = base[i]
            arr_p = base.copy()
            arr_p[i] = orig + eps
            plus = list(inputs)
            plus[idx] = Tensor(arr_p)
            arr_m = base.copy()
            arr_m[i] = orig - eps
            minus = list(inputs)
            minus[idx] = Tensor(arr_m)
            fp = f(plus).data.sum()
            fm = f(minus).data.sum()
            ng[i] = (fp - fm) / (2.0 * eps)
        grads.append(ng)
    return grads


def _check(f, shapes, seeds, atol=1e-2, rtol=1e-2):
    inputs = [_rng(*s, seed=k) for s, k in zip(shapes, seeds)]
    for inp in inputs:
        inp.requires_grad_(True)
        inp.grad = None
    out = f(inputs)
    out.backward()
    analytic = [inp.grad.data.copy() for inp in inputs]
    numeric = numerical_grads(f, inputs)
    for a, n in zip(analytic, numeric):
        assert a.shape == n.shape, f"shape mismatch {a.shape} vs {n.shape}"
        assert np.allclose(a, n, atol=atol, rtol=rtol), (
            f"grad mismatch max|diff|={np.max(np.abs(a - n)):.3e}"
        )


def test_add():
    _check(lambda I: (I[0] + I[1]).sum(), [(4, 3), (4, 3)], [1, 2])


def test_sub():
    _check(lambda I: (I[0] - I[1]).sum(), [(4, 3), (4, 3)], [1, 2])


def test_mul():
    _check(lambda I: (I[0] * I[1]).sum(), [(4, 3), (4, 3)], [1, 2])


def test_div():
    _check(lambda I: (I[0] / (I[1] + 2.0)).sum(), [(4, 3), (4, 3)], [1, 2])


def test_matmul():
    _check(lambda I: (I[0] @ I[1]).sum(), [(5, 4), (4, 6)], [1, 2])


def test_sum():
    _check(lambda I: I[0].sum(), [(4, 3, 2)], [1])


def test_mean():
    _check(lambda I: I[0].mean(), [(4, 3, 2)], [1])


def test_relu():
    _check(lambda I: I[0].relu().sum(), [(4, 3)], [1])


def test_sigmoid():
    _check(lambda I: I[0].sigmoid().sum(), [(4, 3)], [1])


def test_tanh():
    _check(lambda I: I[0].tanh().sum(), [(4, 3)], [1])


def test_gelu():
    _check(lambda I: I[0].gelu().sum(), [(4, 3)], [1])


def test_silu():
    _check(lambda I: I[0].silu().sum(), [(4, 3)], [1])


def test_sqrt():
    _check(lambda I: (I[0] + 3.0).sqrt().sum(), [(4, 3)], [1])


def test_exp():
    _check(lambda I: (I[0] + 2.0).exp().sum(), [(4, 3)], [1])


def test_log():
    _check(lambda I: ((I[0] * I[0] + 2.0)).log().sum(), [(4, 3)], [1])


def test_pow():
    _check(lambda I: ops.Pow.apply(I[0] + 2.0, 3.0).sum(), [(4, 3)], [1])


def test_softmax():
    _check(lambda I: I[0].softmax(dim=-1).sum(), [(4, 3)], [1])


def test_logsoftmax():
    _check(lambda I: I[0].log_softmax(dim=-1).sum(), [(4, 3)], [1])


def test_composite_graph():
    def f(I):
        a, b, c = I
        h = ((a @ b) * c + 1.0).relu()
        h = h.tanh()
        return (h * h).sum()
    _check(f, [(5, 4), (4, 6), (5, 6)], [1, 2, 3])


def test_chain_with_activations():
    def f(I):
        x = I[0]
        x = x.sigmoid()
        x = x.gelu()
        x = x.softmax(dim=-1)
        return x.sum()
    _check(f, [(4, 3)], [1])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL AUTOGRAD CORRECTNESS TESTS PASSED")
