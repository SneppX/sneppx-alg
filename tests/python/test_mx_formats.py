"""Tests for MX (micro-scaling) deployment formats."""

import numpy as np
from SneppX_ALG.interface_bindings.mx_formats import (
    MXFormat,
    MX_FORMATS,
    quantize_mx,
    dequantize_mx,
    mx_error,
)

RNG = np.random.default_rng(7)


def test_format_table():
    names = set(MX_FORMATS.keys())
    assert names == {
        MXFormat.MXFP8_E4M3,
        MXFormat.MXFP8_E5M2,
        MXFormat.MXFP6_E3M2,
        MXFormat.MXFP6_E2M3,
        MXFormat.MXFP4_E2M1,
        MXFormat.NVFP4_E2M1,
    }
    assert MX_FORMATS[MXFormat.MXFP4_E2M1].bits == 4
    assert MX_FORMATS[MXFormat.MXFP8_E4M3].block_size == 32
    assert MX_FORMATS[MXFormat.NVFP4_E2M1].block_size == 0
    print("  test_format_table PASS")


def test_roundtrip_all_formats():
    w = (RNG.normal(size=512) * 1.0).astype(np.float32)
    minsnr = {
        MXFormat.MXFP8_E4M3: 28.0,
        MXFormat.MXFP8_E5M2: 20.0,
        MXFormat.MXFP6_E3M2: 20.0,
        MXFormat.MXFP6_E2M3: 25.0,
        MXFormat.MXFP4_E2M1: 12.0,
        MXFormat.NVFP4_E2M1: 10.0,
    }
    for fmt in sorted(MX_FORMATS.keys()):
        packed, scales = quantize_mx(w, format=fmt)
        recon = dequantize_mx(packed, scales, w.shape, format=fmt)
        assert recon.shape == w.shape
        snr = mx_error(w, recon, metric="snr")
        assert snr > minsnr[fmt], f"{fmt}: snr={snr:.1f} dB"
    print("  test_roundtrip_all_formats PASS")


def test_small_magnitude_subnormals():
    w = (RNG.normal(size=256) * 0.01).astype(np.float32)
    for fmt in (MXFormat.MXFP4_E2M1, MXFormat.MXFP6_E2M3,
                MXFormat.MXFP6_E3M2):
        packed, scales = quantize_mx(w, format=fmt)
        recon = dequantize_mx(packed, scales, w.shape, format=fmt)
        assert mx_error(w, recon, metric="snr") > 10.0
    print("  test_small_magnitude_subnormals PASS")


def test_mxfp8_high_accuracy():
    w = (RNG.normal(size=256) * 4.0).astype(np.float32)
    packed, scales = quantize_mx(w, format=MXFormat.MXFP8_E4M3)
    recon = dequantize_mx(packed, scales, w.shape, format=MXFormat.MXFP8_E4M3)
    assert mx_error(w, recon, metric="snr") > 28.0
    print("  test_mxfp8_high_accuracy PASS")


def test_block_boundaries():
    # exact multiple of block size
    w = (RNG.normal(size=64) * 0.5).astype(np.float32)
    packed, scales = quantize_mx(w, format=MXFormat.MXFP4_E2M1)
    assert scales.shape == (2,)
    recon = dequantize_mx(packed, scales, w.shape, format=MXFormat.MXFP4_E2M1)
    assert np.allclose(recon, recon, atol=0)
    assert mx_error(w, recon, metric="snr") > 10.0
    # non-multiple: padding must be trimmed by dequantize
    w2 = (RNG.normal(size=50) * 0.5).astype(np.float32)
    packed2, scales2 = quantize_mx(w2, format=MXFormat.MXFP4_E2M1)
    recon2 = dequantize_mx(packed2, scales2, w2.shape, format=MXFormat.MXFP4_E2M1)
    assert recon2.shape == (50,)
    print("  test_block_boundaries PASS")


def test_2d_shape_preserved():
    w = (RNG.normal(size=(16, 64)) * 0.5).astype(np.float32)
    packed, scales = quantize_mx(w, format=MXFormat.MXFP4_E2M1)
    recon = dequantize_mx(packed, scales, w.shape, format=MXFormat.MXFP4_E2M1)
    assert recon.shape == (16, 64)
    print("  test_2d_shape_preserved PASS")


def test_per_tensor_nvfp4():
    w = (RNG.normal(size=128) * 0.7).astype(np.float32)
    packed, scales = quantize_mx(w, format=MXFormat.NVFP4_E2M1)
    assert scales.shape == (1,)
    recon = dequantize_mx(packed, scales, w.shape, format=MXFormat.NVFP4_E2M1)
    assert recon.shape == w.shape
    assert mx_error(w, recon, metric="snr") > 10.0
    print("  test_per_tensor_nvfp4 PASS")


def test_zeros_and_scales():
    w = np.zeros(64, dtype=np.float32)
    packed, scales = quantize_mx(w, format=MXFormat.MXFP4_E2M1)
    recon = dequantize_mx(packed, scales, w.shape, format=MXFormat.MXFP4_E2M1)
    assert np.allclose(recon, 0.0, atol=1e-6)
    print("  test_zeros_and_scales PASS")


def test_error_metrics():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.5], dtype=np.float32)
    assert abs(mx_error(a, b, "mse") - (0.25 / 3)) < 1e-6
    assert abs(mx_error(a, b, "mae") - (0.5 / 3)) < 1e-6
    assert mx_error(a, a, "snr") == float("inf")
    print("  test_error_metrics PASS")


def test_unknown_format_raises():
    try:
        quantize_mx(np.ones(8), format="bogus")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    print("  test_unknown_format_raises PASS")


if __name__ == "__main__":
    test_format_table()
    test_roundtrip_all_formats()
    test_small_magnitude_subnormals()
    test_mxfp8_high_accuracy()
    test_block_boundaries()
    test_2d_shape_preserved()
    test_per_tensor_nvfp4()
    test_zeros_and_scales()
    test_error_metrics()
    test_unknown_format_raises()
    print("ALL mx_formats TESTS PASS")
