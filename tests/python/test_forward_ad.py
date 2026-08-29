import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

from SneppX_ALG.interface_bindings.tensor import Tensor
from SneppX_ALG.interface_bindings import autograd_ops as ops
from SneppX_ALG.interface_bindings.autograd import jvp as exact_jvp
from SneppX_ALG.interface_bindings.forward_ad import forward_jvp


def test_forward_jvp_linear():
    np.random.seed(71)
    x = Tensor(np.random.randn(2, 3).astype("float64")); x.requires_grad_(True)
    W = Tensor(np.random.randn(3, 4).astype("float64")); W.requires_grad_(True)
    vx = Tensor(np.random.randn(2, 3).astype("float64"))
    vW = Tensor(np.random.randn(3, 4).astype("float64"))

    def f(a, b):
        return ops.MatMul.apply(a, b)

    out_f, jv_f = forward_jvp(f, [x, W], [vx, vW])
    out_e, jv_e = exact_jvp(f, [x, W], [vx, vW])
    ref = vx.data.astype(np.float64) @ W.data.astype(np.float64) + x.data.astype(np.float64) @ vW.data.astype(np.float64)
    assert np.allclose(jv_f.data.astype(np.float64), jv_e.data.astype(np.float64), atol=1e-6), "fwd vs exact"
    assert np.allclose(jv_f.data.astype(np.float64), ref, atol=1e-3), "fwd vs analytic"
    print("  PASS forward_jvp_linear")


def test_forward_jvp_elementwise():
    np.random.seed(72)
    x = Tensor(np.random.randn(5).astype("float64")); x.requires_grad_(True)
    v = Tensor(np.random.randn(5).astype("float64"))

    def f(a):
        return ops.Mul.apply(a, a)

    out_f, jv_f = forward_jvp(f, [x], [v])
    out_e, jv_e = exact_jvp(f, [x], [v])
    ref = 2.0 * x.data.astype(np.float64) * v.data.astype(np.float64)
    assert np.allclose(jv_f.data.astype(np.float64), jv_e.data.astype(np.float64), atol=1e-6)
    assert np.allclose(jv_f.data.astype(np.float64), ref, atol=1e-3)
    print("  PASS forward_jvp_elementwise")


def test_forward_jvp_composite():
    np.random.seed(73)
    x = Tensor(np.random.randn(3, 4).astype("float64")); x.requires_grad_(True)
    W = Tensor(np.random.randn(4, 2).astype("float64")); W.requires_grad_(True)
    vx = Tensor(np.random.randn(3, 4).astype("float64"))
    vW = Tensor(np.random.randn(4, 2).astype("float64"))

    def f(a, b):
        return ops.Sigmoid.apply(ops.MatMul.apply(a, b))

    out_f, jv_f = forward_jvp(f, [x, W], [vx, vW])
    out_e, jv_e = exact_jvp(f, [x, W], [vx, vW])
    y = x.data.astype(np.float64) @ W.data.astype(np.float64)
    s = 1.0 / (1.0 + np.exp(-y))
    jv_ref = s * (1.0 - s) * (vx.data.astype(np.float64) @ W.data.astype(np.float64) + y * 0 + x.data.astype(np.float64) @ vW.data.astype(np.float64))
    assert np.allclose(jv_f.data.astype(np.float64), jv_e.data.astype(np.float64), atol=1e-6), "fwd vs exact"
    assert np.allclose(jv_f.data.astype(np.float64), jv_ref, atol=1e-3), "fwd vs analytic"
    print("  PASS forward_jvp_composite")


def test_forward_jvp_fallback():
    # CrossEntropyLoss has no registered tangent rule -> must fall back to the
    # exact reverse-of-reverse jvp and still be correct.
    np.random.seed(74)
    x = Tensor(np.random.randn(2, 3).astype("float64")); x.requires_grad_(True)
    t = Tensor(np.array([1, 0], dtype=np.int64))
    vx = Tensor(np.random.randn(2, 3).astype("float64"))

    def f(a):
        return ops.CrossEntropyLoss.apply(a, t)

    out_f, jv_f = forward_jvp(f, [x], [vx])
    out_e, jv_e = exact_jvp(f, [x], [vx])
    assert np.allclose(jv_f.data.astype(np.float64), jv_e.data.astype(np.float64), atol=1e-6), "fallback vs exact"
    print("  PASS forward_jvp_fallback (CrossEntropyLoss)")


def test_forward_jvp_norm_chain():
    # Exercises the newly-added LayerNorm + LinearFn tangent rules end-to-end
    # through a single forward pass; validated against the exact jvp.
    np.random.seed(76)
    x = Tensor(np.random.randn(3, 4).astype("float64")); x.requires_grad_(True)
    g = Tensor(np.ones(4).astype("float64")); g.requires_grad_(True)
    b = Tensor(np.zeros(4).astype("float64")); b.requires_grad_(True)
    W = Tensor(np.random.randn(4, 2).astype("float64")); W.requires_grad_(True)
    vx = Tensor(np.random.randn(3, 4).astype("float64"))
    vg = Tensor(np.random.randn(4).astype("float64"))
    vb = Tensor(np.random.randn(4).astype("float64"))
    vW = Tensor(np.random.randn(4, 2).astype("float64"))

    def f(a, ga, be, w):
        return ops.MatMul.apply(ops.LayerNorm.apply(a, ga, be), w)

    out_f, jv_f = forward_jvp(f, [x, g, b, W], [vx, vg, vb, vW])
    out_e, jv_e = exact_jvp(f, [x, g, b, W], [vx, vg, vb, vW])
    assert np.allclose(jv_f.data.astype(np.float64), jv_e.data.astype(np.float64), atol=1e-6), "norm chain fwd vs exact"
    print("  PASS forward_jvp_norm_chain")


def test_forward_jvp_softmax_chain():
    np.random.seed(75)
    x = Tensor(np.random.randn(2, 3).astype("float64")); x.requires_grad_(True)
    vx = Tensor(np.random.randn(2, 3).astype("float64"))

    def f(a):
        return ops.Softmax.apply(ops.Log.apply(ops.Exp.apply(a)), dim=-1)

    out_f, jv_f = forward_jvp(f, [x], [vx])
    out_e, jv_e = exact_jvp(f, [x], [vx])
    s = np.exp(x.data.astype(np.float64))
    s = s / s.sum(axis=-1, keepdims=True)
    sdt = (s * vx.data.astype(np.float64)).sum(axis=-1, keepdims=True)
    ref = s * (vx.data.astype(np.float64) - sdt)
    assert np.allclose(jv_f.data.astype(np.float64), jv_e.data.astype(np.float64), atol=1e-6)
    assert np.allclose(jv_f.data.astype(np.float64), ref, atol=1e-3)
    print("  PASS forward_jvp_softmax_chain")


if __name__ == "__main__":
    test_forward_jvp_linear()
    test_forward_jvp_elementwise()
    test_forward_jvp_composite()
    test_forward_jvp_fallback()
    test_forward_jvp_norm_chain()
    test_forward_jvp_softmax_chain()
    print("ALL FORWARD_AD TESTS PASSED")
