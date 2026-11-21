"""SIMD GEMM kernel bindings — optimized matrix multiplication.

Wraps C implementation in ``kernel/tensor/simd_gemm.c`` with pure-Python fallback.
Implements tiled GEMM with SIMD-like loop structure.
"""

from typing import Optional, Tuple

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_core_kernel")


class SimdGemm:
    """SIMD-optimized general matrix multiply (GEMM).

    Pure-Python fallback uses tiled NumPy matmul to simulate SIMD blocking.
    """

    def __init__(self, tile_m: int = 64, tile_n: int = 64, tile_k: int = 64):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self._has_c = _HAS_C

    def gemm(self, a: np.ndarray, b: np.ndarray,
             alpha: float = 1.0, beta: float = 0.0,
             c: Optional[np.ndarray] = None) -> np.ndarray:
        m, k_a = a.shape
        k_b, n = b.shape
        if k_a != k_b:
            raise ValueError(f"Inner dims must match: {k_a} vs {k_b}")
        if c is None:
            c = np.zeros((m, n), dtype=a.dtype)
        out = np.zeros((m, n), dtype=a.dtype)
        for i in range(0, m, self.tile_m):
            for j in range(0, n, self.tile_n):
                for kk in range(0, k_a, self.tile_k):
                    i_end = min(i + self.tile_m, m)
                    j_end = min(j + self.tile_n, n)
                    kk_end = min(kk + self.tile_k, k_a)
                    a_tile = a[i:i_end, kk:kk_end]
                    b_tile = b[kk:kk_end, j:j_end]
                    out[i:i_end, j:j_end] += alpha * (a_tile @ b_tile)
        if beta != 0:
            out += beta * c
        return out

    def strassen(self, a: np.ndarray, b: np.ndarray, threshold: int = 64) -> np.ndarray:
        m, k = a.shape
        _, n = b.shape
        if m <= threshold or k <= threshold or n <= threshold:
            return a @ b
        if m % 2 != 0 or k % 2 != 0 or n % 2 != 0:
            return a @ b
        m2, k2, n2 = m // 2, k // 2, n // 2
        a11, a12 = a[:m2, :k2], a[:m2, k2:]
        a21, a22 = a[m2:, :k2], a[m2:, k2:]
        b11, b12 = b[:k2, :n2], b[:k2, n2:]
        b21, b22 = b[k2:, :n2], b[k2:, n2:]
        p1 = self.strassen(a11 + a22, b11 + b22, threshold)
        p2 = self.strassen(a21 + a22, b11, threshold)
        p3 = self.strassen(a11, b12 - b22, threshold)
        p4 = self.strassen(a22, b21 - b11, threshold)
        p5 = self.strassen(a11 + a12, b22, threshold)
        p6 = self.strassen(a21 - a11, b11 + b12, threshold)
        p7 = self.strassen(a12 - a22, b21 + b22, threshold)
        c11 = p1 + p4 - p5 + p7
        c12 = p3 + p5
        c21 = p2 + p4
        c22 = p1 - p2 + p3 + p6
        out = np.zeros((m, n), dtype=a.dtype)
        out[:m2, :n2] = c11
        out[:m2, n2:] = c12
        out[m2:, :n2] = c21
        out[m2:, n2:] = c22
        return out


class GemmConfig:
    """GEMM configuration — tile sizes, algorithm selection."""

    def __init__(self, tile_m: int = 64, tile_n: int = 64, tile_k: int = 64,
                 use_strassen: bool = False, strassen_threshold: int = 128):
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.use_strassen = use_strassen
        self.strassen_threshold = strassen_threshold

    def create_gemm(self) -> SimdGemm:
        return SimdGemm(self.tile_m, self.tile_n, self.tile_k)
