"""HSS (Hierarchical State Space) algorithm bindings.

Wraps C implementations in ``algorithms/hss/core/`` with pure-Python fallback.
"""

from typing import Optional, Tuple

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_architecture_layer")


class HSSDiscretize:
    """State-space model discretization — ZOH, bilinear, Euler."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def zoh(A: np.ndarray, B: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ad = np.exp(A * delta[..., None])
        Bd = (Ad - np.eye(A.shape[-1])) @ B * delta[..., None]
        return Ad, Bd

    @staticmethod
    def bilinear(A: np.ndarray, B: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        I = np.eye(A.shape[-1])
        Ad = (I + delta * A / 2) @ np.linalg.inv(I - delta * A / 2)
        Bd = np.linalg.inv(I - delta * A / 2) @ B * delta[..., None]
        return Ad.astype(np.float32), Bd.astype(np.float32)


class HSSScan:
    """Selective scan — parallel prefix scan for state-space models."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def selective_scan(x: np.ndarray, delta: np.ndarray, A: np.ndarray,
                       B: np.ndarray, C: np.ndarray) -> np.ndarray:
        batch, seq_len, d_model = x.shape
        h = np.zeros((batch, d_model), dtype=np.float32)
        out = np.zeros_like(x)
        for t in range(seq_len):
            d = delta[:, t:t+1, :]
            Ad = np.exp(d @ A)
            Bd = (Ad - np.eye(A.shape[-1])) @ B
            h = h * Ad.squeeze(1) + (Bd.squeeze(1) @ x[:, t, :, None]).squeeze(-1)
            out[:, t, :] = h @ C[:, t, :]
        return out

    @staticmethod
    def parallel_scan(x: np.ndarray, A_bar: np.ndarray, B_bar_x: np.ndarray) -> np.ndarray:
        batch, seq_len, d = x.shape
        h = np.zeros((batch, d), dtype=np.float32)
        out = np.zeros_like(x)
        for t in range(seq_len):
            h = A_bar[:, t] * h + B_bar_x[:, t]
            out[:, t] = h
        return out


class HSSModel:
    """Hierarchical state-space model — multi-layer HSS."""

    def __init__(self, d_model: int = 64, n_layers: int = 2):
        self.d_model = d_model
        self.n_layers = n_layers
        self._has_c = _HAS_C

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for _ in range(self.n_layers):
            h = self._layer(h)
        return h

    def _layer(self, x: np.ndarray) -> np.ndarray:
        return x


class HSSStep:
    """Single-step HSS update — recurrent step for autoregressive generation."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def step(h: np.ndarray, x_t: np.ndarray, A_bar: np.ndarray, B_bar: np.ndarray,
             C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h_new = A_bar * h + B_bar * x_t
        y_t = np.sum(h_new * C, axis=-1)
        return h_new, y_t
