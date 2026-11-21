"""Gumbel-based sampling utilities (pure NumPy)."""

from typing import Optional

import numpy as np


def gumbel_noise(shape, seed: Optional[int] = None) -> np.ndarray:
    """Draw standard Gumbel noise of ``shape`` via ``-log(-log(U))``."""
    rng = np.random.default_rng(seed)
    u = rng.random(shape)
    u = np.clip(u, 1e-20, 1.0 - 1e-20)
    return -np.log(-np.log(u))


def _softmax(x, axis=-1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)


def gumbel_softmax(
    logits,
    tau: float = 1.0,
    hard: bool = False,
    axis: int = -1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample from a categorical via the Gumbel-softmax (concrete) relaxation.

    With ``hard=True`` an exact one-hot sample is returned; otherwise the
    relaxed (soft) sample is returned.
    """
    logits = np.asarray(logits, dtype=np.float64)
    g = gumbel_noise(logits.shape, seed)
    y = _softmax((logits + g) / max(tau, 1e-10), axis=axis)
    if not hard:
        return y
    idx = np.argmax(y, axis=axis)
    eye = np.eye(logits.shape[axis])
    return eye[idx]


def gumbel_max_sample(logits, seed: Optional[int] = None) -> int:
    """Exact categorical sample via the argmax of ``logits + Gumbel``."""
    logits = np.asarray(logits, dtype=np.float64)
    g = gumbel_noise(logits.shape, seed)
    return int(np.argmax(logits + g))


def straight_through(logits, tau: float = 1.0, seed: Optional[int] = None):
    """Return a hard one-hot plus the soft gradient surrogate ``y_hard - y``."""
    soft = gumbel_softmax(logits, tau=tau, hard=False, seed=seed)
    hard = gumbel_softmax(logits, tau=tau, hard=True, seed=seed)
    return hard, soft
