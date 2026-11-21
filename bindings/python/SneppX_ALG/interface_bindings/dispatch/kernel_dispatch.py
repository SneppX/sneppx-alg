"""Kernel-domain dispatch — routes kernel-level operations (attention, memory, thread, etc.)

Priority:
1. Compiled ``neural_core_kernel`` shared library (ctypes)
2. Compiled ``_SNEPPX_c`` / ``_arix_c`` pybind11 module
3. Pure-Python / NumPy fallback
"""

from typing import Optional, Tuple

import numpy as np

from ..c_loader import load_library

_C_LIB, _HAS_C = load_library("neural_core_kernel")
if not _HAS_C:
    _C_LIB, _HAS_C = load_library("_SNEPPX_c")


class KernelBackend:
    """Unified kernel backend dispatching across C / pybind11 / Python."""

    def __init__(self):
        self.name = "kernel"
        self.has_c = _HAS_C
        self._lib = _C_LIB

    # ---- Capability ------------------------------------------------

    def available(self) -> bool:
        return self.has_c

    # ---- Attention -------------------------------------------------

    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        scale = k.shape[-1] ** 0.5
        scores = (q @ k.transpose(0, 1, -1, -2)) / scale
        if mask is not None:
            scores = scores + mask
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        return attn @ v

    @staticmethod
    def flash_attention(q, k, v, mask=None):
        return KernelBackend.scaled_dot_product(q, k, v, mask)

    @staticmethod
    def differential_attention(q, k, v, lambd=0.9):
        q1, q2 = np.split(q, 2, axis=-1)
        k1, k2 = np.split(k, 2, axis=-1)
        scale = (k1.shape[-1]) ** 0.5
        scores = (q1 @ k1.transpose(0, 1, -1, -2)) - lambd * (q2 @ k2.transpose(0, 1, -1, -2))
        scores = scores / scale
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        return attn @ v

    # ---- Memory ----------------------------------------------------

    @staticmethod
    def memory_alloc(size_bytes: int) -> int:
        return size_bytes

    @staticmethod
    def memory_free(ptr: int) -> None:
        pass

    # ---- Thread pool -----------------------------------------------

    @staticmethod
    def parallel_for(n: int, func, n_threads: int = 4):
        for i in range(n):
            func(i)

    # ---- Logger ----------------------------------------------------

    @staticmethod
    def log(level: str, message: str) -> None:
        print(f"[{level}] {message}")

    # Placeholder: additional kernel methods will be added as
    # bindings are created in Phase 5.
