# -*- coding: utf-8 -*-
"""C `_Tensor` engine <-> numpy bridge tests (issue #14).

Validates that the true C tensor engine (``SneppX_ALG._SNEPPX_c._Tensor``)
round-trips numpy data losslessly and computes operations (add, matmul,
conv2d, activations, reductions, shape ops) in agreement with numpy.

Also guards two binding-correctness properties discovered while building the
bridge:
- ``t.data()`` must expose C *element* strides as numpy *byte* strides
  (``tensor_bindings.cpp`` conversion),
- ops that receive impossible inputs must raise ``RuntimeError`` (the C
  layer returns NULL) instead of segfaulting.
"""

import shutil
import tempfile

import numpy as np
import pytest

import SneppX_ALG.interface_bindings.c_tensor_bridge as cb


pytest.importorskip(
    "SneppX_ALG.interface_bindings.c_tensor_bridge",
    reason="bridge import failed",
)


def _ext_available():
    try:
        return cb.available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ext_available(), reason="C _SNEPPX_c extension not built"
)


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------


def test_round_trip_2d_float32():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(a, cb.round_trip(a))


def test_round_trip_3d_float64():
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    assert np.array_equal(a, cb.round_trip(a))


def test_round_trip_int_and_bool_dtypes():
    i32 = np.arange(8, dtype=np.int32).reshape(4, 2) * 3 - 7
    assert np.array_equal(i32, cb.round_trip(i32))
    b = np.array([[True, False], [False, True]])
    assert np.array_equal(b, cb.round_trip(b))


def test_python_tensor_independent_of_c_memory():
    # mutating the C tensor afterwards must not corrupt the numpy copy
    a = np.arange(4, dtype=np.float32)
    t = cb.from_numpy(a)
    out = cb.to_numpy(t)
    t.fill_f32(100.0)
    assert np.array_equal(out, a)


def test_shape_and_dtype_metadata():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = cb.from_numpy(a)
    assert t.shape == (3, 4)
    assert t.ndim == 2
    assert t.numel == 12
    assert t.dtype_name == "float32"


# ---------------------------------------------------------------------------
# operations parity
# ---------------------------------------------------------------------------


def _parity(a, b, c_ref, op):
    ta, tb = cb.from_numpy(a), cb.from_numpy(b)
    return np.max(np.abs(cb.to_numpy(op(ta, tb)) - c_ref))


def test_add_parity():
    a = np.random.RandomState(0).rand(4, 5).astype(np.float32)
    b = np.random.RandomState(1).rand(4, 5).astype(np.float32)
    err = _parity(a, b, a + b, lambda x, y: x.add(y))
    assert err <= 1e-6


def test_sub_mul_div_parity():
    rs = np.random.RandomState(2)
    a = (rs.rand(3, 3) + 1.0).astype(np.float32)
    b = (rs.rand(3, 3) + 1.0).astype(np.float32)
    assert _parity(a, b, a - b, lambda x, y: x.sub(y)) <= 1e-6
    assert _parity(a, b, a * b, lambda x, y: x.mul(y)) <= 1e-6
    assert _parity(a, b, a / b, lambda x, y: x.div(y)) <= 1e-5


def test_matmul_parity_square():
    a = np.random.RandomState(3).rand(4, 4).astype(np.float32)
    b = np.random.RandomState(4).rand(4, 4).astype(np.float32)
    err = _parity(a, b, a @ b, lambda x, y: x.matmul(y))
    assert err <= 1e-4


def test_matmul_parity_rectangular():
    a = np.random.RandomState(5).rand(2, 6).astype(np.float32)
    b = np.random.RandomState(6).rand(6, 3).astype(np.float32)
    err = _parity(a, b, a @ b, lambda x, y: x.matmul(y))
    assert err <= 1e-5


def test_matmul_parity_float64_exact_dtype():
    """f64 matmul must stay f64 (was hardcoded float32 downcast)."""
    a = np.random.RandomState(14).rand(3, 5)
    b = np.random.RandomState(15).rand(5, 4)
    ta, tb = cb.from_numpy(a), cb.from_numpy(b)
    out = ta.matmul(tb)
    assert cb.c_dtype_to_numpy(out.dtype) == np.float64
    got = cb.to_numpy(out)
    assert np.max(np.abs(got - (a @ b))) <= 1e-12


def test_matmul_parity_batched():
    a = np.random.RandomState(7).rand(3, 4, 5).astype(np.float32)
    b = np.random.RandomState(8).rand(3, 5, 2).astype(np.float32)
    ref = np.stack([a[i] @ b[i] for i in range(3)])
    ta, tb = cb.from_numpy(a), cb.from_numpy(b)
    got = cb.to_numpy(ta.matmul(tb))
    assert np.max(np.abs(got - ref)) <= 1e-4


def test_dot_parity_float32_float64():
    """dot() must read f64 storage correctly (was misread as f32 -> garbage)."""
    a = np.random.RandomState(21).rand(8).astype(np.float32)
    b = np.random.RandomState(22).rand(8).astype(np.float32)
    assert np.isclose(cb.from_numpy(a).dot(cb.from_numpy(b)), float(a @ b), rtol=1e-5)
    a64 = np.random.RandomState(23).rand(8)
    b64 = np.random.RandomState(24).rand(8)
    # float return type caps precision; garbage misread would be far off
    assert np.isclose(cb.from_numpy(a64).dot(cb.from_numpy(b64)), float(a64 @ b64), rtol=1e-6)


def test_reductions_sum_mean_float64_dtype_following():
    """f64 sum/mean must stay f64 and match numpy exactly."""
    a = np.random.RandomState(25).rand(3, 5)
    t = cb.from_numpy(a)
    s = t.sum(1)
    assert cb.c_dtype_to_numpy(s.dtype) == np.float64
    assert np.allclose(cb.to_numpy(s), a.sum(axis=1, keepdims=True), rtol=0, atol=1e-13)
    m = t.mean(0)
    assert cb.c_dtype_to_numpy(m.dtype) == np.float64
    assert np.allclose(cb.to_numpy(m), a.mean(axis=0, keepdims=True), rtol=0, atol=1e-13)
    # f32 inputs keep f32
    a32 = a.astype(np.float32)
    t32 = cb.from_numpy(a32)
    assert cb.c_dtype_to_numpy(t32.sum(1).dtype) == np.float32
    assert np.allclose(cb.to_numpy(t32.mean(0)), a32.mean(axis=0, keepdims=True), atol=1e-6)


def _conv2d_numpy(x, k, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    x = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    n, c, h, w = x.shape
    kh, kw = k.shape[-2:]
    out_h = (h - kh) // stride_h + 1
    out_w = (w - kw) // stride_w + 1
    c_out = k.shape[0]
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            win = x[
                :, :, i * stride_h : i * stride_h + kh, j * stride_w : j * stride_w + kw
            ]
            for co in range(c_out):
                out[:, co, i, j] = (win * k[co]).sum(axis=(1, 2, 3))
    return out


def test_conv2d_parity_stride1_pad0():
    rs = np.random.RandomState(9)
    x = rs.rand(1, 3, 8, 8).astype(np.float32)
    k = rs.rand(1, 3, 3, 3).astype(np.float32)
    # C conv2d follows torch semantics: w (C_out, C_in, KH, KW), x (N, C, H, W)
    # -> out (N, C_out, H_out, W_out)
    ref = _conv2d_numpy(x, k)
    tx, tk = cb.from_numpy(x), cb.from_numpy(k)
    got = cb.to_numpy(tx.conv2d(tk, 1, 1, 0, 0))
    assert got.shape == ref.shape, (got.shape, ref.shape)
    err = np.max(np.abs(got - ref))
    assert err <= 1e-4, f"conv2d mismatch {err}"


def test_relu_sigmoid_exp_parity():
    rs = np.random.RandomState(10)
    a = rs.rand(3, 4).astype(np.float32) * 4.0 - 2.0
    t = cb.from_numpy(a)
    assert np.allclose(cb.to_numpy(t.relu()), np.maximum(a, 0), atol=1e-6)
    assert np.allclose(cb.to_numpy(t.sigmoid()), 1 / (1 + np.exp(-a)), atol=1e-6)
    assert np.allclose(cb.to_numpy(t.exp()), np.exp(a), atol=1e-5)


def test_reductions_sum_mean_softmax():
    a = np.random.RandomState(11).rand(4, 5).astype(np.float32)
    t = cb.from_numpy(a)
    assert np.allclose(cb.to_numpy(t.sum(0)).ravel(), a.sum(axis=0), atol=1e-5)
    # C reductions keep the reduced dim: mean(dim=1) -> (N, 1)
    assert np.allclose(cb.to_numpy(t.mean(1)).ravel(), a.mean(axis=1), atol=1e-5)
    sm = cb.to_numpy(t.softmax(1))
    e = np.exp(a - a.max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    assert np.allclose(sm, ref, atol=1e-5)


def test_shape_ops_reshape_transpose_reshape_back():
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = cb.from_numpy(a)
    r = cb.to_numpy(t.reshape(np.array([6, 4]))).reshape(6, 4)
    assert np.array_equal(r, a.reshape(6, 4))
    # C transpose(dim1, dim2) swaps the two named axes (torch semantics)
    tr = cb.to_numpy(t.transpose(1, 2)).reshape(2, 4, 3)
    assert np.array_equal(tr, a.transpose(0, 2, 1))


# ---------------------------------------------------------------------------
# robustness guards
# ---------------------------------------------------------------------------


def test_invalid_op_raises_runtime_error_not_crash():
    a = cb.from_numpy(np.ones((2, 3), dtype=np.float32))
    # inner dims 3 vs 2 -> C returns NULL -> must raise, not segfault
    with pytest.raises(RuntimeError):
        a.matmul(cb.from_numpy(np.eye(2, dtype=np.float32)))


def test_metadata_round_trip_after_c_ops():
    a = np.random.RandomState(12).rand(2, 4).astype(np.float32)
    b = np.random.RandomState(13).rand(4, 3).astype(np.float32)
    out = cb.from_numpy(a).matmul(cb.from_numpy(b))
    assert out.shape == (2, 3)
    # result can be converted back and remains finite
    assert np.isfinite(cb.to_numpy(out)).all()
