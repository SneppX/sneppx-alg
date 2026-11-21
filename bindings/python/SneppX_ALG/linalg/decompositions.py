"""Matrix decompositions (pure NumPy)."""

from typing import Tuple

import numpy as np


def lu(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LU decomposition with partial pivoting: ``A = P @ L @ U``.

    ``P`` is a permutation matrix, ``L`` unit lower-triangular, ``U`` upper-triangular.
    """
    A = np.array(A, dtype=np.float64)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("lu requires a square matrix")
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(U[k:, k])))
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
        if abs(U[k, k]) < 1e-14:
            continue
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    return P, L, U


def qr(A: np.ndarray, mode: str = "reduced") -> Tuple[np.ndarray, np.ndarray]:
    """QR decomposition ``A = Q @ R``."""
    return np.linalg.qr(np.asarray(A, dtype=np.float64), mode=mode)


def svd(
    A: np.ndarray, full_matrices: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Singular value decomposition ``A = U @ diag(s) @ Vt``."""
    return np.linalg.svd(np.asarray(A, dtype=np.float64), full_matrices=full_matrices)


def cholesky(A: np.ndarray) -> np.ndarray:
    """Cholesky factor ``L`` with ``A = L @ L.T`` (A symmetric positive-definite)."""
    return np.linalg.cholesky(np.asarray(A, dtype=np.float64))


def eig(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """General eigenvalues/eigenvectors (non-symmetric)."""
    return np.linalg.eig(np.asarray(A, dtype=np.float64))


def eigh(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues/eigenvectors of a symmetric matrix (ascending eigenvalues)."""
    return np.linalg.eigh(np.asarray(A, dtype=np.float64))
