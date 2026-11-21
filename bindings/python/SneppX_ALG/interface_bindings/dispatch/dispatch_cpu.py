"""CPU backend — NumPy operations with optional ctypes C acceleration."""

from typing import Optional, Tuple

import numpy as np

from ..c_loader import load_library

_C_LIB, _HAS_C = load_library("neural_core_kernel")
if not _HAS_C:
    _C_LIB, _HAS_C = load_library("_SNEPPX_c")


class CpuBackend:
    """Backend that runs all operations on the CPU.

    Uses NumPy for tensor arithmetic and delegates to the compiled C
    shared library (via ctypes) when available for accelerated kernels.
    """

    def __init__(self):
        self.name = "cpu"
        self.has_c = _HAS_C
        self._lib = _C_LIB

    # ---- Tensor arithmetic (always NumPy) --------------------------------

    @staticmethod
    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    @staticmethod
    def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    @staticmethod
    def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    @staticmethod
    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    @staticmethod
    def div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a / b

    @staticmethod
    def sum(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.sum(x, axis=axis)

    @staticmethod
    def mean(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.mean(x, axis=axis)

    @staticmethod
    def std(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.std(x, axis=axis)

    @staticmethod
    def var(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.var(x, axis=axis)

    # ---- Activation functions -------------------------------------------

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        return x * 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    # ---- Normalization --------------------------------------------------

    @staticmethod
    def layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                  eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        out = (x - mean) / np.sqrt(var + eps)
        return out * weight + bias

    @staticmethod
    def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        ss = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(ss + eps) * weight

    # ---- Random number generation ---------------------------------------

    @staticmethod
    def randn(*shape: int) -> np.ndarray:
        return np.random.randn(*shape).astype(np.float32)

    @staticmethod
    def rand(*shape: int) -> np.ndarray:
        return np.random.rand(*shape).astype(np.float32)

    @staticmethod
    def randint(low: int, high: int, size: Tuple[int, ...]) -> np.ndarray:
        return np.random.randint(low, high, size=size).astype(np.int64)

    # ---- Utility --------------------------------------------------------

    @staticmethod
    def to_numpy(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def from_numpy(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def zeros(*shape: int, dtype=np.float32) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(*shape: int, dtype=np.float32) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def empty(*shape: int, dtype=np.float32) -> np.ndarray:
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def copy(x: np.ndarray) -> np.ndarray:
        return x.copy()

    @staticmethod
    def transpose(x: np.ndarray, axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        return np.transpose(x, axes)

    @staticmethod
    def reshape(x: np.ndarray, *shape: int) -> np.ndarray:
        return x.reshape(*shape)

    @staticmethod
    def cat(xs: Tuple[np.ndarray, ...], axis: int = 0) -> np.ndarray:
        return np.concatenate(xs, axis=axis)

    @staticmethod
    def stack(xs: Tuple[np.ndarray, ...], axis: int = 0) -> np.ndarray:
        return np.stack(xs, axis=axis)
