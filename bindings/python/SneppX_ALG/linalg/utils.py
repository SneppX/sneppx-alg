"""Matrix utilities — diag, triangular helpers, rank, determinant (pure NumPy)."""

from typing import Tuple

import numpy as np


def tril(A: np.ndarray, k: int = 0) -> np.ndarray:
    """Lower triangle (elements on/below the ``k``-th diagonal)."""
    return np.tril(np.asarray(A, dtype=np.float64), k)


def triu(A: np.ndarray, k: int = 0) -> np.ndarray:
    """Upper triangle (elements on/above the ``k``-th diagonal)."""
    return np.triu(np.asarray(A, dtype=np.float64), k)


def diag(x: np.ndarray, k: int = 0) -> np.ndarray:
    """Extract or construct a diagonal (NumPy semantics)."""
    return np.diag(np.asarray(x, dtype=np.float64), k)


def diagflat(x: np.ndarray, k: int = 0) -> np.ndarray:
    """Flatten input and build a diagonal matrix."""
    return np.diagflat(np.asarray(x, dtype=np.float64), k)


def eye(n: int, m: int = None, k: int = 0) -> np.ndarray:
    """Identity / shifted-diagonal matrix."""
    return np.eye(n, m, k, dtype=np.float64)


def matrix_rank(A: np.ndarray, tol: float = None) -> int:
    """Rank of a matrix."""
    return int(np.linalg.matrix_rank(np.asarray(A, dtype=np.float64), tol=tol))


def det(A: np.ndarray) -> float:
    """Determinant."""
    return float(np.linalg.det(np.asarray(A, dtype=np.float64)))


def slogdet(A: np.ndarray) -> Tuple[float, float]:
    """``(sign, log|det|)`` for numerical stability."""
    sign, logabsdet = np.linalg.slogdet(np.asarray(A, dtype=np.float64))
    return float(sign), float(logabsdet)


def is_symmetric(A: np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a matrix is symmetric within tolerance."""
    A = np.asarray(A, dtype=np.float64)
    return A.shape[0] == A.shape[1] and bool(np.allclose(A, A.T, atol=tol))


def is_positive_definite(A: np.ndarray) -> bool:
    """Check symmetric positive-definiteness via Cholesky."""
    A = np.asarray(A, dtype=np.float64)
    if not is_symmetric(A):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
