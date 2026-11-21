"""Attention kernel bindings — attention, differential attention, flex attention.

Wraps C implementations in ``kernel/attention/`` with pure-Python fallback.
"""

from typing import Optional, Tuple

import numpy as np

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_core_kernel")


class Attention:
    """Scaled dot-product attention with optional causal masking."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def forward(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                mask: Optional[np.ndarray] = None, scale: Optional[float] = None) -> np.ndarray:
        s = scale or (k.shape[-1] ** 0.5)
        scores = (q @ k.transpose(0, 1, -1, -2)) / s
        if mask is not None:
            scores = scores + mask
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        return attn @ v


class DifferentialAttention:
    """Differential attention — λ-scaled subtracted QK pairs."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def forward(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                lambd: float = 0.9) -> np.ndarray:
        q1, q2 = np.split(q, 2, axis=-1)
        k1, k2 = np.split(k, 2, axis=-1)
        scale = k1.shape[-1] ** 0.5
        scores = (q1 @ k1.transpose(0, 1, -1, -2)) - lambd * (q2 @ k2.transpose(0, 1, -1, -2))
        scores = scores / scale
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        return attn @ v


class FlexAttention:
    """Flexible attention — block-sparse masking, multi-modal KV."""

    def __init__(self):
        self._has_c = _HAS_C

    @staticmethod
    def forward(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                block_mask: Optional[np.ndarray] = None) -> np.ndarray:
        scale = k.shape[-1] ** 0.5
        scores = (q @ k.transpose(0, 1, -1, -2)) / scale
        if block_mask is not None:
            scores = scores * block_mask - 1e9 * (1 - block_mask)
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        return attn @ v
