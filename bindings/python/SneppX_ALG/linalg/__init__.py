"""SNEPPX linear algebra — decompositions, solvers, norms, utilities.

from SneppX_ALG import linalg

Q, R = linalg.qr(A)
x = linalg.solve(A, b)
s = linalg.spectral_norm(A)
"""

from . import decompositions, norms, solvers, utils
from .decompositions import cholesky, eig, eigh, lu, qr, svd
from .norms import (
    condition_number,
    frobenius_norm,
    matrix_norm,
    nuclear_norm,
    spectral_norm,
    trace_norm,
    vector_norm,
)
from .solvers import cond, inv, lstsq, pinv, solve, solve_triangular
from .utils import (
    det,
    diag,
    diagflat,
    eye,
    is_positive_definite,
    is_symmetric,
    matrix_rank,
    slogdet,
    tril,
    triu,
)

__all__ = [
    "decompositions",
    "solvers",
    "norms",
    "utils",
    "lu",
    "qr",
    "svd",
    "cholesky",
    "eig",
    "eigh",
    "solve",
    "lstsq",
    "inv",
    "pinv",
    "solve_triangular",
    "cond",
    "vector_norm",
    "matrix_norm",
    "frobenius_norm",
    "nuclear_norm",
    "spectral_norm",
    "trace_norm",
    "condition_number",
    "tril",
    "triu",
    "diag",
    "diagflat",
    "eye",
    "matrix_rank",
    "det",
    "slogdet",
    "is_symmetric",
    "is_positive_definite",
]
