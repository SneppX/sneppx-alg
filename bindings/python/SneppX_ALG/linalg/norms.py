"""Vector / matrix norms and related quantities (pure NumPy)."""

from typing import Optional

import numpy as np


def vector_norm(x: np.ndarray, ord: float = 2.0) -> float:
    """Vector norm (default L2)."""
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).ravel(), ord=ord))


def matrix_norm(A: np.ndarray, ord: str = "fro") -> float:
    """Matrix norm (default Frobenius)."""
    return float(np.linalg.norm(np.asarray(A, dtype=np.float64), ord=ord))


def frobenius_norm(A: np.ndarray) -> float:
    return matrix_norm(A, "fro")


def nuclear_norm(A: np.ndarray) -> float:
    """Sum of singular values (trace norm)."""
    s = np.linalg.svd(np.asarray(A, dtype=np.float64), compute_uv=False)
    return float(np.sum(s))


def spectral_norm(A: np.ndarray) -> float:
    """Largest singular value."""
    s = np.linalg.svd(np.asarray(A, dtype=np.float64), compute_uv=False)
    return float(s[0]) if s.size else 0.0


def trace_norm(A: np.ndarray) -> float:
    return nuclear_norm(A)


def condition_number(A: np.ndarray, p: Optional[float] = None) -> float:
    return float(np.linalg.cond(np.asarray(A, dtype=np.float64), p=p))
