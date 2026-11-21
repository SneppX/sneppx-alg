"""Probability distributions and densities (pure NumPy)."""

import math
from typing import Optional

import numpy as np

_SQRT2 = math.sqrt(2.0)
_SQRTPI = math.sqrt(math.pi)


def _erf(x):
    return np.vectorize(math.erf)(x) if np.ndim(x) else math.erf(x)


def normal_pdf(x, mu: float = 0.0, sigma: float = 1.0):
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * _SQRT2 * _SQRTPI)


def normal_cdf(x, mu: float = 0.0, sigma: float = 1.0):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + _erf((x - mu) / (sigma * _SQRT2)))


def uniform_pdf(x, low: float = 0.0, high: float = 1.0):
    x = np.asarray(x, dtype=np.float64)
    return np.where((x >= low) & (x <= high), 1.0 / (high - low), 0.0)


def exponential_pdf(x, rate: float = 1.0):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, rate * np.exp(-rate * x), 0.0)


def bernoulli_pmf(k, p: float = 0.5):
    k = np.asarray(k, dtype=np.float64)
    return np.where(k == 1, p, np.where(k == 0, 1.0 - p, 0.0))


def binomial_pmf(k, n: int, p: float = 0.5):
    k = np.asarray(k, dtype=np.float64)
    coeff = np.vectorize(lambda j: math.comb(int(n), int(j)) if 0 <= j <= n else 0.0)
    return coeff(k) * (p**k) * ((1.0 - p) ** (n - k))


def poisson_pmf(k, lam: float = 1.0):
    k = np.asarray(k, dtype=np.float64)
    fac = np.vectorize(lambda j: math.factorial(int(j)) if j >= 0 else math.inf)
    return np.where(k >= 0, np.exp(-lam) * (lam**k) / fac(k), 0.0)


def categorical_pmf(k, probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    k = np.asarray(k, dtype=np.int64)
    out = np.zeros(k.shape, dtype=np.float64)
    valid = (k >= 0) & (k < probs.shape[0])
    out[valid] = probs[k[valid]]
    return out


def entropy(p, base: Optional[float] = None) -> float:
    """Shannon entropy of a probability vector."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    h = -np.sum(p * np.log(p))
    if base is not None:
        h /= np.log(base)
    return float(h)


def kl_divergence(p, q) -> float:
    """KL divergence ``D(p || q)`` for two probability vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def cross_entropy(p, q) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(q[mask])))
