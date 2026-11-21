"""Architecture kernel bindings — general arch utils, Mamba-2.

Wraps C implementations in ``kernel/arch/`` with pure-Python fallback.
"""

from typing import Optional, Tuple

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_core_kernel")


class ArchOps:
    """General architecture utility operations."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def activation(x: np.ndarray, kind: str = "silu") -> np.ndarray:
        if kind == "silu":
            return x / (1.0 + np.exp(-x))
        if kind == "gelu":
            import math
            return x * 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
        if kind == "relu":
            return np.maximum(x, 0)
        if kind == "tanh":
            return np.tanh(x)
        return x

    @staticmethod
    def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                   eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight + bias


class Mamba2:
    """Mamba-2 selective state-space model block."""

    def __init__(self, d_model: int = 64, d_state: int = 16):
        self.d_model = d_model
        self.d_state = d_state
        self._has_c = _HAS_C

    def forward(self, x: np.ndarray, dt: np.ndarray, A: np.ndarray,
                B: np.ndarray, C: np.ndarray) -> np.ndarray:
        batch, seq, d = x.shape
        h = np.zeros((batch, self.d_state), dtype=np.float32)
        out = np.zeros_like(x)
        for t in range(seq):
            delta = dt[:, t:t+1, :, None]  # (batch, 1, 1, 1)
            A_bar = np.exp((delta * A[None, None, :, :]).squeeze(1))
            u = x[:, t, :self.d_state]
            h = (A_bar @ h[..., None]).squeeze(-1) + u * B.squeeze(-1)
            ssm_out = np.sum(h * C[:, t, :], axis=-1)
            out[:, t, :self.d_state] = ssm_out[:, None] * np.ones(self.d_state, dtype=np.float32)
        return out

    @staticmethod
    def hippo_init(N: int) -> np.ndarray:
        A = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i > j:
                    A[i, j] = -np.sqrt(2 * i + 1) * np.sqrt(2 * j + 1)
                elif i == j:
                    A[i, j] = -(i + 1)
                else:
                    A[i, j] = 0
        return A
