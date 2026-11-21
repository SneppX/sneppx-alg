"""Tests for c_simd_gemm.py — SIMD GEMM kernels."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bindings", "python"))

import numpy as np
from SneppX_ALG.interface_bindings import SimdGemm, GemmConfig


def test_gemm_basic():
    gemm = SimdGemm(tile_m=8, tile_n=8, tile_k=8)
    a = np.random.randn(16, 12).astype(np.float32)
    b = np.random.randn(12, 20).astype(np.float32)
    c = gemm.gemm(a, b)
    expected = a @ b
    assert np.allclose(c, expected, atol=1e-4)


def test_gemm_alpha():
    gemm = SimdGemm()
    a = np.random.randn(8, 6).astype(np.float32)
    b = np.random.randn(6, 4).astype(np.float32)
    c = gemm.gemm(a, b, alpha=2.0)
    expected = 2.0 * (a @ b)
    assert np.allclose(c, expected, atol=1e-4)


def test_gemm_beta():
    gemm = SimdGemm()
    a = np.random.randn(6, 4).astype(np.float32)
    b = np.random.randn(4, 6).astype(np.float32)
    c_init = np.ones((6, 6), dtype=np.float32)
    c = gemm.gemm(a, b, beta=0.5, c=c_init)
    expected = 0.5 * c_init + a @ b
    assert np.allclose(c, expected, atol=1e-4)


def test_strassen():
    gemm = SimdGemm()
    a = np.random.randn(32, 32).astype(np.float32)
    b = np.random.randn(32, 32).astype(np.float32)
    c = gemm.strassen(a, b, threshold=8)
    assert np.allclose(c, a @ b, atol=1e-3)


def test_strassen_small():
    gemm = SimdGemm()
    a = np.random.randn(4, 4).astype(np.float32)
    b = np.random.randn(4, 4).astype(np.float32)
    c = gemm.strassen(a, b, threshold=8)
    assert np.allclose(c, a @ b, atol=1e-5)


def test_gemm_non_square():
    gemm = SimdGemm(tile_m=4, tile_n=8, tile_k=4)
    a = np.random.randn(5, 7).astype(np.float32)
    b = np.random.randn(7, 9).astype(np.float32)
    c = gemm.gemm(a, b)
    assert np.allclose(c, a @ b, atol=1e-4)


def test_gemm_config():
    config = GemmConfig(tile_m=32, tile_n=32, tile_k=32)
    gemm = config.create_gemm()
    assert gemm.tile_m == 32
    assert isinstance(gemm, SimdGemm)


def test_gemm_identity():
    gemm = SimdGemm()
    a = np.random.randn(8, 8).astype(np.float32)
    I = np.eye(8, dtype=np.float32)
    c = gemm.gemm(a, I)
    assert np.allclose(c, a, atol=1e-4)


if __name__ == "__main__":
    test_gemm_basic()
    test_gemm_alpha()
    test_gemm_beta()
    test_strassen()
    test_strassen_small()
    test_gemm_non_square()
    test_gemm_config()
    test_gemm_identity()
    print("All c_simd_gemm tests passed.")
