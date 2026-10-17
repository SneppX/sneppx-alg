"""Descriptive statistics (pure NumPy)."""

from typing import Optional

import numpy as np


def _a(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def mean(x) -> float:
    return float(np.mean(_a(x)))


def median(x) -> float:
    return float(np.median(_a(x)))


def mode(x):
    """Most common value (ties resolved by smallest value)."""
    x = np.asarray(x)
    vals, counts = np.unique(x, return_counts=True)
    return vals[np.argmax(counts)]


def variance(x, ddof: int = 0) -> float:
    return float(np.var(_a(x), ddof=ddof))


def std(x, ddof: int = 0) -> float:
    return float(np.std(_a(x), ddof=ddof))


def quantile(x, q: float) -> float:
    return float(np.quantile(_a(x), q))


def percentile(x, q: float) -> float:
    return float(np.percentile(_a(x), q))


def iqr(x) -> float:
    x = _a(x)
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))


def skewness(x, bias: bool = True) -> float:
    x = _a(x)
    n = x.shape[0]
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    g = np.mean(((x - m) / s) ** 3)
    if not bias and n > 2:
        g *= np.sqrt(n * (n - 1)) / (n - 2)
    return float(g)


def kurtosis(x, fisher: bool = True, bias: bool = True) -> float:
    x = _a(x)
    n = x.shape[0]
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    g = np.mean(((x - m) / s) ** 4)
    if not bias and n > 3:
        g = ((n + 1) * g - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3)) + 3
    return float(g - 3.0) if bias else float(g)


def zscore(x, ddof: int = 0) -> np.ndarray:
    x = _a(x)
    m = x.mean()
    s = x.std(ddof=ddof)
    if s == 0:
        return np.zeros_like(x)
    return (x - m) / s


def covariance(x, y, ddof: int = 1) -> float:
    return float(np.cov(_a(x), _a(y), ddof=ddof)[0, 1])


def correlation(x, y) -> float:
    x = _a(x)
    y = _a(y)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def zscore_all(x, axis: Optional[int] = None) -> np.ndarray:
    """Standardize array (per-axis) to zero mean / unit variance."""
    x = _a(x)
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd
