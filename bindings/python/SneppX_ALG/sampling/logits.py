"""Logit processors for autoregressive decoding (pure NumPy).

All functions accept either a 1-D logit vector ``(V,)`` or a batched
``(..., V)`` array and operate on the final axis.
"""

from typing import Optional

import numpy as np

NEG_INF = -np.inf


def _as_float(x) -> np.ndarray:
    if hasattr(x, "data"):
        x = x.data
    return np.asarray(x, dtype=np.float64)


def temperature(logits, t: float):
    """Divide logits by temperature ``t`` (``t == 1`` is identity)."""
    if t <= 0:
        raise ValueError("temperature must be > 0")
    return _as_float(logits) / t


def top_k_filter(logits, k: int):
    """Keep the ``k`` highest logits, masking the rest to ``-inf``."""
    x = _as_float(logits)
    if k is None or k <= 0 or k >= x.shape[-1]:
        return x
    threshold = np.sort(x, axis=-1)[..., -k][..., None]
    return np.where(x >= threshold, x, NEG_INF)


def top_p_filter(logits, p: float):
    """Nucleus sampling: keep the smallest set with cumulative prob >= ``p``."""
    x = _as_float(logits)
    if p is None or p >= 1.0:
        return x
    x_max = x.max(axis=-1, keepdims=True)
    exp = np.exp(x - x_max)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    order = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, order, axis=-1)
    cumulative = np.cumsum(sorted_probs, axis=-1)
    sorted_mask = cumulative - sorted_probs >= p
    mask = np.zeros_like(probs, dtype=bool)
    np.put_along_axis(mask, order, sorted_mask, axis=-1)
    return np.where(mask, NEG_INF, x)


def min_p_filter(logits, min_p: float):
    """Drop tokens whose probability is below ``min_p * max_prob``."""
    x = _as_float(logits)
    if min_p is None or min_p <= 0.0:
        return x
    x_max = x.max(axis=-1, keepdims=True)
    exp = np.exp(x - x_max)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    cutoff = min_p * probs.max(axis=-1, keepdims=True)
    return np.where(probs < cutoff, NEG_INF, x)


def repetition_penalty(logits, prev_ids, penalty: float):
    """Divide positive / multiply negative logits of previously seen ids."""
    if penalty is None or penalty == 1.0 or prev_ids is None:
        return _as_float(logits)
    x = _as_float(logits).copy()
    ids = np.unique(np.asarray(prev_ids).ravel())
    ids = ids[(ids >= 0) & (ids < x.shape[-1])]
    if ids.size == 0:
        return x
    vals = x[..., ids]
    x[..., ids] = np.where(vals > 0, vals / penalty, vals * penalty)
    return x


def filter_logits(
    logits,
    temp: Optional[float] = None,
    k: Optional[int] = None,
    p: Optional[float] = None,
    min_p: Optional[float] = None,
):
    """Apply temperature -> top-k -> top-p -> min-p in sequence."""
    x = _as_float(logits)
    if temp is not None:
        x = temperature(x, temp)
    if k is not None:
        x = top_k_filter(x, k)
    if p is not None:
        x = top_p_filter(x, p)
    if min_p is not None:
        x = min_p_filter(x, min_p)
    return x


def softmax(logits) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = _as_float(logits)
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)
