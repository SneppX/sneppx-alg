"""Hypothesis tests, confidence intervals and resampling (pure Python/NumPy).

Student-t and chi-square p-values are computed from first principles
(regularized incomplete beta / gamma) so there is no SciPy dependency.
"""

import math
from typing import Callable, Optional, Tuple

import numpy as np

_FPMIN = 1e-300
_EPS = 1e-15


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must be in (0, 1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e1,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.223964580411365e-01,
        2.400758277161838e00,
        2.549732539343734e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    return num / den


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """CDF of Student's t distribution with ``df`` degrees of freedom."""
    x = df / (df + t * t)
    ib = _betainc(df / 2.0, 0.5, x)
    if t > 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def student_t_sf(t: float, df: float) -> float:
    """Survival function ``P(T > t)``."""
    return 1.0 - student_t_cdf(t, df)


def _gser(a: float, x: float) -> float:
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(500):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float) -> float:
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma ``Q(a, x)``."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("require a > 0 and x >= 0")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function ``P(X > x)``."""
    return gammaincc(df / 2.0, x / 2.0)


def _as1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def mean_confidence_interval(x, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Normal-approximation CI for the mean: ``(low, high, mean)``."""
    x = _as1d(x)
    n = x.shape[0]
    m = float(x.mean())
    if n < 2:
        return m, m, m
    se = float(x.std(ddof=1)) / math.sqrt(n)
    z = norm_ppf(0.5 + confidence / 2.0)
    return m - z * se, m + z * se, m


def one_sample_ttest(x, popmean: float = 0.0) -> dict:
    """One-sample two-sided t-test."""
    x = _as1d(x)
    n = x.shape[0]
    df = n - 1
    se = x.std(ddof=1) / math.sqrt(n)
    t = 0.0 if se == 0 else (x.mean() - popmean) / se
    p = 2.0 * student_t_sf(abs(t), df)
    return {"statistic": float(t), "pvalue": float(p), "df": df}


def independent_ttest(a, b, equal_var: bool = True) -> dict:
    """Two-sample two-sided t-test (Student or Welch)."""
    a = _as1d(a)
    b = _as1d(b)
    na, nb = a.shape[0], b.shape[0]
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if equal_var:
        df = na + nb - 2
        sp2 = ((na - 1) * va + (nb - 1) * vb) / df
        se = math.sqrt(sp2 * (1.0 / na + 1.0 / nb))
    else:
        se = math.sqrt(va / na + vb / nb)
        num = (va / na + vb / nb) ** 2
        den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
        df = num / den if den > 0 else 1.0
    t = 0.0 if se == 0 else (a.mean() - b.mean()) / se
    p = 2.0 * student_t_sf(abs(t), df)
    return {"statistic": float(t), "pvalue": float(p), "df": float(df)}


def paired_ttest(a, b) -> dict:
    """Paired two-sided t-test."""
    return one_sample_ttest(_as1d(a) - _as1d(b), popmean=0.0)


def chi_square_test(observed, expected) -> dict:
    """Pearson chi-square goodness-of-fit / independence test."""
    o = np.asarray(observed, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if o.shape != e.shape:
        raise ValueError("observed and expected must have the same shape")
    df = o.size - 1
    stat = float(np.sum((o - e) ** 2 / e))
    p = float(chi2_sf(stat, df))
    return {"statistic": stat, "pvalue": p, "df": df}


def bootstrap_ci(
    data,
    stat_fn: Optional[Callable] = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for a statistic."""
    data = _as1d(data)
    if stat_fn is None:
        stat_fn = lambda s: float(np.mean(s))
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    stats = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = data[rng.integers(0, n, size=n)]
        stats[i] = stat_fn(sample)
    lo = float(np.quantile(stats, (1.0 - confidence) / 2.0))
    hi = float(np.quantile(stats, 1.0 - (1.0 - confidence) / 2.0))
    return lo, hi


def permutation_test(
    a,
    b,
    stat_fn: Optional[Callable] = None,
    n_resamples: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """Two-sided permutation test for a difference statistic."""
    a = _as1d(a)
    b = _as1d(b)
    if stat_fn is None:
        stat_fn = lambda x, y: float(np.mean(x) - np.mean(y))
    observed = stat_fn(a, b)
    pooled = np.concatenate([a, b])
    na = a.shape[0]
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_resamples):
        perm = rng.permutation(pooled)
        stat = stat_fn(perm[:na], perm[na:])
        if abs(stat) >= abs(observed):
            count += 1
    p = (count + 1) / (n_resamples + 1)
    return {"statistic": float(observed), "pvalue": float(p)}
