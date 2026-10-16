"""Regularization penalties (pure NumPy)."""

import numpy as np


def _a(w) -> np.ndarray:
    if hasattr(w, "data"):
        w = w.data
    return np.asarray(w, dtype=np.float64)


def l1_penalty(w) -> float:
    """L1 / lasso: ``sum |w|``."""
    return float(np.sum(np.abs(_a(w))))


def l2_penalty(w) -> float:
    """L2 / ridge: ``sum w^2``."""
    return float(np.sum(_a(w) ** 2))


def elastic_net(w, l1: float = 0.5, l2: float = 0.5) -> float:
    """Weighted sum of L1 and L2 penalties."""
    return float(l1 * l1_penalty(w) + l2 * l2_penalty(w))


def group_lasso(groups) -> float:
    """Group lasso: sum of the L2 norm of each group of parameters."""
    total = 0.0
    for g in groups:
        total += float(np.sqrt(np.sum(_a(g) ** 2)))
    return total


def total_variation(x, beta: float = 1.0) -> float:
    """Anisotropic total-variation penalty over the last axis."""
    x = _a(x)
    return float(beta * np.sum(np.abs(np.diff(x, axis=-1))))


def orthogonality_penalty(W) -> float:
    """Penalize deviation of ``W`` from having orthonormal rows.

    ``|| W W^T - I ||_F^2`` (shape-agnostic: uses the smaller Gram matrix).
    """
    W = _a(W)
    if W.ndim != 2:
        raise ValueError("orthogonality_penalty expects a 2-D matrix")
    gram = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
    eye = np.eye(gram.shape[0])
    return float(np.sum((gram - eye) ** 2))


def proximity_penalty(w, w_ref) -> float:
    """Penalize drift from a reference (e.g. a frozen copy): ``||w - w_ref||^2``."""
    return float(np.sum((_a(w) - _a(w_ref)) ** 2))
