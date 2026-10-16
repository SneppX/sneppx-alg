"""Mixup / CutMix data augmentation (pure NumPy)."""

from typing import Optional, Tuple

import numpy as np


def mixup(
    x, y, alpha: float = 0.2, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Mixup: convex combination of random example pairs.

    Returns ``(x_mixed, y_mixed, lam)`` where ``lam ~ Beta(alpha, alpha)``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if alpha > 0:
        lam = float(rng.beta(alpha, alpha))
    else:
        lam = 1.0
    n = x.shape[0]
    index = rng.permutation(n)
    x_mix = lam * x + (1.0 - lam) * x[index]
    y_mix = lam * y + (1.0 - lam) * y[index]
    return x_mix, y_mix, lam


def cutmix(
    x, y, alpha: float = 1.0, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """CutMix for ``(N, H, W[, C])`` inputs: paste a random box from another image."""
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y).copy()
    rng = np.random.default_rng(seed)
    if alpha > 0:
        lam = float(rng.beta(alpha, alpha))
    else:
        lam = 1.0
    n, h, w = x.shape[0], x.shape[1], x.shape[2]
    index = rng.permutation(n)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = rng.integers(h)
    cx = rng.integers(w)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, h)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    x[:, y1:y2, x1:x2, ...] = x[index, y1:y2, x1:x2, ...]
    box_area = (y2 - y1) * (x2 - x1)
    lam = 1.0 - box_area / float(h * w)
    y = lam * y + (1.0 - lam) * y[index]
    return x, y, float(lam)


def mixup_loss(loss_fn, pred, y_a, y_b, lam: float):
    """Combine two loss values: ``lam * L(pred, y_a) + (1-lam) * L(pred, y_b)``."""
    return lam * loss_fn(pred, y_a) + (1.0 - lam) * loss_fn(pred, y_b)
