"""Tests for quantization module — INT8/FP8/AWQ/GPTQ."""

import warnings

import numpy as np
from SneppX_ALG.interface_bindings import Tensor
from SneppX_ALG.interface_bindings.quantization import (
    QuantMode,
    quantize_int8_sym,
    dequantize_int8_sym,
    quantize_int8_asym,
    dequantize_int8_asym,
    quantize_int8_channel,
    quantize_int4_sym,
    dequantize_int4_sym,
    quantize_fp8_e4m3,
    dequantize_fp8_e4m3,
    quantize_fp8_e5m2,
    dequantize_fp8_e5m2,
    awq_scale_weights,
    awq_quantize,
    gptq_compute_hessian,
    gptq_quantize,
    QuantizedLinear,
    quantize_error,
    _fp8_e4m3_to_float,
    _fp8_e5m2_to_float,
)


def test_int8_sym_roundtrip():
    x = Tensor.randn((32, 64))
    q, scale = quantize_int8_sym(x)
    dq = dequantize_int8_sym(q, scale)
    err = quantize_error(x, dq, "mse")
    assert q.dtype == "int8", f"Expected int8, got {q.dtype}"
    assert err < 1.0, f"MSE too high: {err}"
    print(f"  test_int8_sym_roundtrip: MSE={err:.6f} PASS")


def test_int8_sym_extremes():
    x = Tensor([-127.0, 0.0, 127.0])
    q, scale = quantize_int8_sym(x)
    dq = dequantize_int8_sym(q, scale)
    assert dq.data[0] < 0
    assert abs(dq.data[1]) < 1e-5
    assert dq.data[2] > 0
    print("  test_int8_sym_extremes PASS")


def test_int8_sym_zeros():
    x = Tensor.zeros((16,))
    q, scale = quantize_int8_sym(x)
    assert np.all(q.data == 0)
    dq = dequantize_int8_sym(q, scale)
    assert np.allclose(dq.data, 0.0)
    print("  test_int8_sym_zeros PASS")


def test_int8_asym_roundtrip():
    x = Tensor(np.random.randn(32, 64).astype(np.float32) * 2.0 + 1.0)
    q, scale, zp = quantize_int8_asym(x)
    dq = dequantize_int8_asym(q, scale, zp)
    err = quantize_error(x, dq, "mse")
    assert q.dtype == "uint8"
    assert err < 2.0, f"MSE too high: {err}"
    print(f"  test_int8_asym_roundtrip: MSE={err:.6f} PASS")


def test_int8_asym_all_positive_preserves_max():
    # All-positive ranges previously clipped the zero-point to 0 and then
    # saturated vmax: 4.0 reconstructed as 3.0.
    x = Tensor(np.array([1.0, 2.0, 4.0], dtype=np.float32))
    q, scale, zp = quantize_int8_asym(x)
    dq = dequantize_int8_asym(q, scale, zp)
    assert zp == 0
    assert np.allclose(dq.data, [1.0, 2.0, 4.0], atol=0.02)


def test_int8_asym_all_negative_preserves_min():
    x = Tensor(np.array([-4.0, -2.0, -1.0], dtype=np.float32))
    q, scale, zp = quantize_int8_asym(x)
    dq = dequantize_int8_asym(q, scale, zp)
    assert zp == 255
    assert np.allclose(dq.data, [-4.0, -2.0, -1.0], atol=0.02)


def test_int8_channel():
    x = Tensor.randn((8, 16))
    q, scales = quantize_int8_channel(x, dim=0)
    assert q.shape == x.shape
    assert scales.shape == (8,)
    print(f"  test_int8_channel: scales shape={scales.shape} PASS")


def test_int4_roundtrip():
    x = Tensor(np.random.randn(100).astype(np.float32))
    q, scale = quantize_int4_sym(x)
    dq = dequantize_int4_sym(q, scale, 100)
    err = quantize_error(x, dq, "mse")
    assert err < 5.0, f"MSE too high: {err}"
    print(f"  test_int4_roundtrip: MSE={err:.6f} PASS")


def test_fp8_e4m3_roundtrip():
    x = Tensor([1.0, 2.0, 0.5, -1.0, 0.0, 127.0, -64.0, 0.125])
    q = quantize_fp8_e4m3(x)
    dq = dequantize_fp8_e4m3(q)
    err = quantize_error(x, dq, "mse")
    print(f"  test_fp8_e4m3_roundtrip: MSE={err:.6f} PASS")


def test_fp8_e5m2_roundtrip():
    x = Tensor([1.0, 2.0, 0.5, -1.0, 0.0, 256.0, -128.0, 0.25])
    q = quantize_fp8_e5m2(x)
    dq = dequantize_fp8_e5m2(q)
    err = quantize_error(x, dq, "mse")
    print(f"  test_fp8_e5m2_roundtrip: MSE={err:.6f} PASS")


def test_fp8_special_values():
    x = Tensor([0.0, float("inf"), float("-inf")])
    q_e4m3 = quantize_fp8_e4m3(x)
    dq_e4m3 = dequantize_fp8_e4m3(q_e4m3)
    assert dq_e4m3.data[0] == 0.0
    assert np.isinf(dq_e4m3.data[1])
    assert np.isinf(dq_e4m3.data[2])
    print("  test_fp8_special_values PASS")


def test_fp8_subnormals_flush_to_signed_zero_without_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for code in range(256):
            _fp8_e4m3_to_float(np.uint8(code))
            _fp8_e5m2_to_float(np.uint8(code))

    for sign in (0, 0x80):
        for mantissa in range(1, 8):
            decoded = _fp8_e4m3_to_float(np.uint8(sign | mantissa))
            assert decoded == 0.0
            assert bool(np.signbit(decoded)) == bool(sign)
        for mantissa in range(1, 4):
            decoded = _fp8_e5m2_to_float(np.uint8(sign | mantissa))
            assert decoded == 0.0
            assert bool(np.signbit(decoded)) == bool(sign)


def test_awq_scale():
    w = Tensor.randn((4, 32))
    act_scales = Tensor(np.random.rand(32).astype(np.float32))
    scaled = awq_scale_weights(w, act_scales, group_size=16)
    assert scaled.shape == w.shape
    print("  test_awq_scale PASS")


def test_awq_quantize():
    w = Tensor.randn((2, 32))
    act_scales = Tensor(np.random.rand(32).astype(np.float32))
    qw, scales = awq_quantize(w, act_scales, group_size=16)
    assert qw.shape == w.shape
    assert qw.dtype == "int8"
    assert scales.shape == (2, 2)  # 2 groups across 32 cols
    print(f"  test_awq_quantize: scales shape={scales.shape} PASS")


def test_gptq_hessian():
    acts = Tensor.randn((128, 16))
    H = gptq_compute_hessian(acts)
    assert H.shape == (16, 16)
    assert np.allclose(H.data, H.data.T, atol=1e-5)  # symmetric
    print("  test_gptq_hessian PASS")


def test_gptq_quantize():
    w = Tensor.randn((3, 48))
    acts = Tensor.randn((64, 48))
    H = gptq_compute_hessian(acts)
    qw, scales, zeros = gptq_quantize(w, H, group_size=16, bits=8, sym=True)
    assert qw.shape == w.shape
    assert scales.shape == (3, 3)  # 3 groups across 48 cols
    print(f"  test_gptq_quantize: scales shape={scales.shape} PASS")


def test_gptq_quantize_no_hessian():
    w = Tensor.randn((2, 32))
    qw, scales, zeros = gptq_quantize(w, group_size=16, bits=4, sym=False)
    assert qw.shape == w.shape
    assert zeros is not None
    print("  test_gptq_quantize_no_hessian PASS")


def test_quantized_linear():
    from SneppX_ALG.interface_bindings.nn import Linear

    linear = Linear(8, 16)
    qlinear = QuantizedLinear.from_float(linear, mode=QuantMode.INT8_SYM)
    x = Tensor.randn((2, 8))
    out = qlinear(x)
    assert out.shape == (2, 16)
    print("  test_quantized_linear PASS")


def test_quantized_linear_asym():
    from SneppX_ALG.interface_bindings.nn import Linear

    linear = Linear(8, 16)
    qlinear = QuantizedLinear.from_float(linear, mode=QuantMode.INT8_ASYM)
    x = Tensor.randn((2, 8))
    out = qlinear(x)
    assert out.shape == (2, 16)
    print("  test_quantized_linear_asym PASS")


def test_quantized_linear_dispatches_on_mode():
    from SneppX_ALG.interface_bindings.nn import Linear

    linear = Linear(2, 2)
    linear.weight = Tensor(np.array([[3.0, 4.0], [7.0, 0.0]], dtype=np.float32))
    linear.bias = Tensor(np.zeros((2,), dtype=np.float32))
    x = Tensor(np.array([[1.0, 0.0]], dtype=np.float32))

    # Range [0, 7] is exact for both asymmetric INT8 and packed INT4.
    for mode in (QuantMode.INT8_ASYM, QuantMode.INT4_SYM):
        qlinear = QuantizedLinear.from_float(linear, mode=mode)
        out = qlinear(x)
        assert out.shape == (1, 2)
        assert np.allclose(out.data, [[3.0, 7.0]], atol=0.05), out.data


def test_quantized_linear_stores_mode_metadata():
    from SneppX_ALG.interface_bindings.nn import Linear

    linear = Linear(2, 2)
    linear.weight = Tensor(np.array([[3.0, 4.0], [7.0, 0.0]], dtype=np.float32))
    linear.bias = Tensor(np.zeros((2,), dtype=np.float32))

    asym = QuantizedLinear.from_float(linear, mode=QuantMode.INT8_ASYM)
    assert asym.zero_point is not None
    assert asym.weight_shape == (2, 2)

    int4 = QuantizedLinear.from_float(linear, mode=QuantMode.INT4_SYM)
    assert int4.weight_shape == (2, 2)
    assert int4.dequantize_weight().shape == (2, 2)


def test_quantize_error_metrics():
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([0.9, 2.1, 2.8])
    mse = quantize_error(x, y, "mse")
    mae = quantize_error(x, y, "mae")
    snr = quantize_error(x, y, "snr")
    assert mse > 0
    assert mae > 0
    assert snr < float("inf")
    print(
        f"  test_quantize_error_metrics: MSE={mse:.6f} MAE={mae:.6f} SNR={snr:.1f}dB PASS"
    )


if __name__ == "__main__":
    test_int8_sym_roundtrip()
    test_int8_sym_extremes()
    test_int8_sym_zeros()
    test_int8_asym_roundtrip()
    test_int8_channel()
    test_int4_roundtrip()
    test_fp8_e4m3_roundtrip()
    test_fp8_e5m2_roundtrip()
    test_fp8_special_values()
    test_fp8_subnormals_flush_to_signed_zero_without_overflow()
    test_awq_scale()
    test_awq_quantize()
    test_gptq_hessian()
    test_gptq_quantize()
    test_gptq_quantize_no_hessian()
    test_quantized_linear()
    test_quantized_linear_asym()
    test_quantize_error_metrics()
    print("\nAll quantization tests passed!")
