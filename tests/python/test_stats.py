import math

import numpy as np

from SneppX_ALG import stats as st


def test_descriptive():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert st.mean(x) == 3.0
    assert st.median(x) == 3.0
    assert abs(st.variance(x) - 2.0) < 1e-9
    assert abs(st.std(x, ddof=1) - np.std(x, ddof=1)) < 1e-9
    assert st.quantile(x, 0.5) == 3.0
    assert st.iqr(x) == 2.0
    assert st.mode([1, 2, 2, 3]) == 2


def test_correlation_covariance():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    assert abs(st.correlation(x, y) - 1.0) < 1e-9
    assert abs(st.covariance(x, y) - 2.0) < 1e-9
    assert np.allclose(st.zscore(x), [-1.22474487, 0.0, 1.22474487])


def test_skew_kurtosis():
    sym = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(st.skewness(sym)) < 1e-9
    assert abs(st.kurtosis(sym)) < 2.0  # roughly platykurtic, no crash


def test_normal_dist():
    assert abs(st.normal_pdf(0.0) - 1.0 / math.sqrt(2 * math.pi)) < 1e-12
    assert abs(st.normal_cdf(0.0) - 0.5) < 1e-9
    assert abs(st.normal_cdf(1.96) - 0.975) < 1e-3


def test_norm_ppf():
    assert abs(st.norm_ppf(0.5) - 0.0) < 1e-9
    assert abs(st.norm_ppf(0.975) - 1.959964) < 1e-4
    assert abs(st.norm_ppf(0.025) + 1.959964) < 1e-4


def test_entropy_kl():
    p = np.array([0.5, 0.5])
    assert abs(st.entropy(p) - math.log(2)) < 1e-12
    q = np.array([0.5, 0.5])
    assert abs(st.kl_divergence(p, q)) < 1e-12
    assert abs(st.cross_entropy(p, q) - math.log(2)) < 1e-12


def test_student_t_cdf():
    assert abs(st.student_t_cdf(0.0, 5) - 0.5) < 1e-9
    # t=2.015, df=5 -> one-sided ~0.95
    assert abs(st.student_t_cdf(2.015, 5) - 0.95) < 5e-3


def test_ttests():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    r = st.independent_ttest(a, b)
    assert r["statistic"] < 0
    assert 0.0 <= r["pvalue"] <= 1.0
    rw = st.independent_ttest(a, b, equal_var=False)
    assert 0.0 <= rw["pvalue"] <= 1.0
    one = st.one_sample_ttest(a, popmean=3.0)
    assert abs(one["statistic"]) < 1e-9
    paired = st.paired_ttest(a, np.array([2.0, 2.0, 3.0, 5.0, 6.0]))
    assert paired["statistic"] < 0


def test_chi_square():
    obs = np.array([10.0, 20.0, 30.0])
    exp = np.array([20.0, 20.0, 20.0])
    r = st.chi_square_test(obs, exp)
    assert abs(r["statistic"] - 10.0) < 1e-9
    assert 0.0 <= r["pvalue"] <= 1.0


def test_confidence_interval_and_resampling():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, size=100)
    lo, hi, m = st.mean_confidence_interval(x, 0.95)
    assert lo < m < hi
    blo, bhi = st.bootstrap_ci(x, n_resamples=200, seed=0)
    assert blo < bhi
    a = rng.normal(0.0, 1.0, size=30)
    b = rng.normal(0.0, 1.0, size=30)
    p = st.permutation_test(a, b, n_resamples=200, seed=0)
    assert 0.0 <= p["pvalue"] <= 1.0
