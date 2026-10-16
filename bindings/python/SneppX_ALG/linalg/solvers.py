"""Linear solvers and inverses (pure NumPy)."""

from typing import Optional, Tuple

import numpy as np


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the linear system ``A @ x = b``."""
    return np.linalg.solve(
        np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )


def lstsq(
    A: np.ndarray, b: np.ndarray, rcond: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Least-squares solution: ``(x, residuals, rank, singular_values)``."""
    return np.linalg.lstsq(
        np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64), rcond=rcond
    )


def inv(A: np.ndarray) -> np.ndarray:
    """Matrix inverse."""
    return np.linalg.inv(np.asarray(A, dtype=np.float64))


def pinv(A: np.ndarray, rcond: float = 1e-15) -> np.ndarray:
    """Moore-Penrose pseudo-inverse."""
    return np.linalg.pinv(np.asarray(A, dtype=np.float64), rcond=rcond)


def solve_triangular(A: np.ndarray, b: np.ndarray, lower: bool = True) -> np.ndarray:
    """Solve ``A @ x = b`` for a triangular ``A`` by back/forward substitution."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = A.shape[0]
    x = np.zeros_like(b, dtype=np.float64)
    order = range(n) if lower else range(n - 1, -1, -1)
    for i in order:
        if lower:
            acc = b[i] - A[i, :i] @ x[:i]
        else:
            acc = b[i] - A[i, i + 1 :] @ x[i + 1 :]
        x[i] = acc / A[i, i]
    return x


def cond(A: np.ndarray, p: Optional[float] = None) -> float:
    """Condition number of a matrix."""
    return float(np.linalg.cond(np.asarray(A, dtype=np.float64), p=p))
