"""Spectral normalization via power iteration (pure NumPy)."""

from typing import Optional, Tuple

import numpy as np


def power_iteration(
    A, num_iters: int = 20, seed: Optional[int] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Approximate the largest singular value of ``A`` by power iteration.

    Returns ``(sigma, u, v)`` with ``A v ~= sigma u``.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v) + 1e-12
    sigma = 0.0
    u = np.zeros(m)
    for _ in range(num_iters):
        u = A @ v
        sigma = np.linalg.norm(u)
        if sigma < 1e-12:
            return 0.0, np.zeros(m), v
        u /= sigma
        v = A.T @ u
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            break
        v /= v_norm
    return float(sigma), u, v


def spectral_norm(W, num_iters: int = 20, seed: Optional[int] = None) -> float:
    """Spectral norm (largest singular value) of a weight matrix."""
    sigma, _, _ = power_iteration(W, num_iters=num_iters, seed=seed)
    return sigma


def spectral_normalize(
    W, num_iters: int = 20, seed: Optional[int] = None
) -> np.ndarray:
    """Return ``W / sigma(W)`` so the result has spectral norm ~1."""
    W = np.asarray(W, dtype=np.float64)
    sigma = spectral_norm(W, num_iters=num_iters, seed=seed)
    return W / (sigma + 1e-12)


def spectral_penalty(W, num_iters: int = 20, seed: Optional[int] = None) -> float:
    """Penalize spectral norm in excess of one: ``max(0, sigma - 1)^2``."""
    sigma = spectral_norm(W, num_iters=num_iters, seed=seed)
    return float(max(0.0, sigma - 1.0) ** 2)
