import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops


def _scalar(out):
    return float(np.asarray(out.data).astype(np.float64).reshape(-1)[0])


def _check_op(name, make_tensors, out_fn, arrays, atol=1e-3, eps=1e-3):
    """FD backward check: analytic grad of sum(out) vs central differences."""
    tensors = list(make_tensors(*[a.copy() for a in arrays]))
    for t in tensors:
        if getattr(t, "requires_grad", False):
            t.requires_grad_(True)
    L = out_fn(*tensors)
    L.backward()
    analytic = []
    for t, a in zip(tensors, arrays):
        if t.grad is None:
            analytic.append(None)
        else:
            analytic.append(t.grad.data.astype(np.float64).reshape(a.shape))

    for i, a in enumerate(arrays):
        if analytic[i] is None:
            continue
        num = np.zeros_like(a, dtype=np.float64)
        it = np.nditer(a, flags=["multi_index"])
        while not it.finished:
            mi = it.multi_index
            ap = [x.copy() for x in arrays]
            am = [x.copy() for x in arrays]
            ap[i][mi] += eps
            am[i][mi] -= eps
            Lp = _scalar(out_fn(*make_tensors(*ap)))
            Lm = _scalar(out_fn(*make_tensors(*am)))
            num[mi] = (Lp - Lm) / (2.0 * eps)
            it.iternext()
        err = np.max(np.abs(analytic[i] - num))
        assert err < atol, f"{name}: grad[{i}] FD mismatch max|err|={err:.3e}"
    print(f"  PASS {name}")


def test_unary_elementwise():
    np.random.seed(101)
    a = np.random.randn(3, 4).astype("float64")
    pos = np.exp(np.random.randn(3, 4)).astype("float64")  # >0 for Sqrt/Log

    def mt(*x):
        return [Tensor(x[0], requires_grad=True)]

    specs = [
        ("Gelu", lambda *t: ops.Gelu.apply(t[0]).sum(), 1e-3, 1e-3),
        ("Silu", lambda *t: ops.Silu.apply(t[0]).sum(), 1e-3, 1e-3),
        ("Sigmoid", lambda *t: ops.Sigmoid.apply(t[0]).sum(), 1e-3, 1e-3),
        ("Tanh", lambda *t: ops.Tanh.apply(t[0]).sum(), 1e-3, 1e-3),
        ("Relu", lambda *t: ops.Relu.apply(t[0]).sum(), 1e-3, 1e-3),
        ("Neg", lambda *t: ops.Neg.apply(t[0]).sum(), 1e-3, 1e-3),
        ("Abs", lambda *t: ops.Abs.apply(t[0]).sum(), 1e-3, 1e-3),
        ("Exp", lambda *t: ops.Exp.apply(t[0]).sum(), 1e-2, 2e-3),
        ("Sqrt", lambda *t: ops.Sqrt.apply(t[0]).sum(), 1e-3, 1e-3),
        ("Log", lambda *t: ops.Log.apply(t[0]).sum(), 1e-3, 1e-3),
    ]
    for nm, fn, ep, at in specs:
        _check_op(nm, mt, fn, [pos if nm in ("Sqrt", "Log") else a], atol=at, eps=ep)


def test_binary_ops():
    np.random.seed(102)
    a = np.random.randn(3, 4).astype("float64")
    b = (np.random.randn(3, 4) + 2.0).astype("float64")

    def mt(*x):
        return [Tensor(x[0], requires_grad=True), Tensor(x[1], requires_grad=True)]

    def mt_a(*x):
        return [Tensor(x[0], requires_grad=True)]

    _check_op("Div", mt, lambda *t: ops.Div.apply(t[0], t[1]).sum(), [a, b])
    _check_op("Pow", mt_a, lambda *t: ops.Pow.apply(t[0], ops._as_const(2.0, t[0].dtype)).sum(), [a])
    _check_op("Add", mt, lambda *t: ops.Add.apply(t[0], t[1]).sum(), [a, b])


def test_reductions():
    np.random.seed(103)
    a = np.random.randn(3, 4).astype("float64")
    sq = np.random.randn(2, 1, 3).astype("float64")
    un = np.random.randn(2, 3).astype("float64")

    def mt(*x):
        return [Tensor(x[0], requires_grad=True)]

    _check_op("Sum", mt, lambda *t: ops.Sum.apply(t[0]).sum(), [a])
    _check_op("Mean", mt, lambda *t: ops.Mean.apply(t[0]).sum(), [a])
    _check_op("Squeeze", mt, lambda *t: ops.Squeeze.apply(t[0], 1).sum(), [sq])
    _check_op("Unsqueeze", mt, lambda *t: ops.Unsqueeze.apply(t[0], 1).sum(), [un])
    _check_op("Softmax", mt, lambda *t: ops.Softmax.apply(t[0], -1).sum(), [a])
    _check_op("LogSoftmax", mt, lambda *t: ops.LogSoftmax.apply(t[0], -1).sum(), [a])


def test_pooling():
    np.random.seed(104)
    a = np.random.randn(1, 1, 5, 5).astype("float64")

    def mt(*x):
        return [Tensor(x[0], requires_grad=True)]

    _check_op("MaxPool2d", mt, lambda *t: ops.MaxPool2d.apply(t[0], 2, 2).sum(), [a])
    _check_op("AvgPool2d", mt, lambda *t: ops.AvgPool2d.apply(t[0], 2, 2).sum(), [a])


def test_losses_fd():
    np.random.seed(105)
    a = np.random.randn(3, 4).astype("float64")
    b = np.random.randn(3, 4).astype("float64")
    logits = np.random.randn(3, 5).astype("float64")
    logp = logits - logits.max(axis=1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))
    target = np.array([2, 0, 4], dtype=np.int64)
    prob = np.abs(np.random.randn(3, 4)).astype("float64") + 0.1
    prob = prob / prob.sum(axis=1, keepdims=True)
    prob_kl = np.abs(np.random.randn(3, 5)).astype("float64") + 0.1
    prob_kl = prob_kl / prob_kl.sum(axis=1, keepdims=True)
    logit_b = np.random.randn(3, 4).astype("float64")
    t01 = (np.random.randn(3, 4) > 0).astype("float64")

    def mt_in(*x):
        return [Tensor(x[0], requires_grad=True)]

    def mt_ab(*x):
        return [Tensor(x[0], requires_grad=True), Tensor(x[1], requires_grad=True)]

    def tgt():
        return Tensor(target)

    def t01t():
        return Tensor(t01)

    def tprobt():
        return Tensor(prob_kl)

    _check_op("MSELoss", mt_ab, lambda *t: ops.MSELoss.apply(t[0], t[1]).sum(), [a, b])
    _check_op("CrossEntropyLoss", mt_in, lambda *t: ops.CrossEntropyLoss.apply(t[0], tgt()).sum(), [logits])
    _check_op("NLLLoss", mt_in, lambda *t: ops.NLLLoss.apply(t[0], tgt()).sum(), [logp])
    _check_op("KLDivLoss", mt_in, lambda *t: ops.KLDivLoss.apply(t[0], tprobt()).sum(), [logp])
    _check_op("BCELoss", mt_in, lambda *t: ops.BCELoss.apply(t[0], t01t()).sum(), [prob])
    _check_op("BCEWithLogitsLoss", mt_in, lambda *t: ops.BCEWithLogitsLoss.apply(t[0], t01t()).sum(), [logit_b])
    _check_op("SmoothL1Loss", mt_ab, lambda *t: ops.SmoothL1Loss.apply(t[0], t[1], 1.0).sum(), [a, b])
    _check_op("HuberLoss", mt_ab, lambda *t: ops.HuberLoss.apply(t[0], t[1], 1.0).sum(), [a, b])


if __name__ == "__main__":
    test_unary_elementwise()
    test_binary_ops()
    test_reductions()
    test_pooling()
    test_losses_fd()
    print("ALL OPS5 TESTS PASSED")
