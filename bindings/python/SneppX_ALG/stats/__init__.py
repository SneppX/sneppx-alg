"""SNEPPX statistics — descriptive stats, distributions, inference.

from SneppX_ALG import stats

mu, sd = stats.mean(x), stats.std(x)
res = graph_level = stats.independent_ttest(a, b)
"""

from . import descriptive, distributions, inference
from .descriptive import (
    correlation,
    covariance,
    iqr,
    kurtosis,
    mean,
    median,
    mode,
    percentile,
    quantile,
    skewness,
    std,
    variance,
    zscore,
    zscore_all,
)
from .distributions import (
    bernoulli_pmf,
    binomial_pmf,
    categorical_pmf,
    cross_entropy,
    entropy,
    exponential_pdf,
    kl_divergence,
    normal_cdf,
    normal_pdf,
    poisson_pmf,
    uniform_pdf,
)
from .inference import (
    bootstrap_ci,
    chi2_sf,
    chi_square_test,
    independent_ttest,
    mean_confidence_interval,
    norm_ppf,
    one_sample_ttest,
    paired_ttest,
    permutation_test,
    student_t_cdf,
)

__all__ = [
    "descriptive",
    "distributions",
    "inference",
    "mean",
    "median",
    "mode",
    "variance",
    "std",
    "quantile",
    "percentile",
    "iqr",
    "skewness",
    "kurtosis",
    "zscore",
    "zscore_all",
    "covariance",
    "correlation",
    "normal_pdf",
    "normal_cdf",
    "uniform_pdf",
    "exponential_pdf",
    "bernoulli_pmf",
    "binomial_pmf",
    "poisson_pmf",
    "categorical_pmf",
    "entropy",
    "kl_divergence",
    "cross_entropy",
    "norm_ppf",
    "student_t_cdf",
    "mean_confidence_interval",
    "one_sample_ttest",
    "independent_ttest",
    "paired_ttest",
    "chi2_sf",
    "chi_square_test",
    "bootstrap_ci",
    "permutation_test",
]
