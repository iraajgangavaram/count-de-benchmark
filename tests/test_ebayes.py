import numpy as np
import pytest
from scipy.special import polygamma

from countde.ebayes import fit_f_dist, moderated_t_test, squeeze_var, trigamma_inverse


@pytest.mark.parametrize("x", [0.01, 0.2, 1.0, 5.0, 100.0])
def test_trigamma_inverse_round_trip(x):
    y = trigamma_inverse(x)
    assert float(polygamma(1, y)) == pytest.approx(x, rel=1e-6)


def _scaled_inv_chi2_variances(rng, n, df0, s0sq, df1):
    """Draw sample variances from the model the estimator assumes."""
    sigma2 = s0sq * df0 / rng.chisquare(df0, size=n)  # true variances (prior)
    return sigma2 * rng.chisquare(df1, size=n) / df1  # observed on df1 d.o.f.


@pytest.mark.parametrize("df0,s0sq,df1", [(8, 0.5, 4), (4, 1.0, 4), (20, 0.2, 10)])
def test_fit_f_dist_recovers_prior(df0, s0sq, df1):
    rng = np.random.default_rng(0)
    s2 = _scaled_inv_chi2_variances(rng, 30000, df0, s0sq, df1)
    s20, df2 = fit_f_dist(s2, df1)
    assert df2 == pytest.approx(df0, rel=0.10)
    assert float(np.atleast_1d(s20)[0]) == pytest.approx(s0sq, rel=0.05)


def test_fit_f_dist_recovers_a_mean_variance_trend():
    rng = np.random.default_rng(1)
    n, df0, df1 = 20000, 10, 4
    cov = rng.uniform(0, 10, n)
    s0sq = np.exp(1.5 - 0.3 * cov)  # variance falls with expression
    sigma2 = s0sq * df0 / rng.chisquare(df0, size=n)
    s2 = sigma2 * rng.chisquare(df1, size=n) / df1
    s20, df2 = fit_f_dist(s2, df1, covariate=cov)
    assert df2 == pytest.approx(df0, rel=0.15)
    assert np.median(np.abs(s20 / s0sq - 1)) < 0.05


def test_too_few_genes_means_no_moderation():
    s20, df2 = fit_f_dist(np.array([0.5, 1.0]), 4)
    assert df2 == 0.0 and np.isnan(s20).all()
    post, df, _ = squeeze_var(np.array([0.5, 1.0]), 4)
    assert np.allclose(post, [0.5, 1.0]) and df == 4


def test_squeezed_variances_lie_between_raw_and_prior_and_shrink_spread():
    rng = np.random.default_rng(2)
    s2 = _scaled_inv_chi2_variances(rng, 5000, 6, 0.8, 4)
    post, df_total, s20 = squeeze_var(s2, 4)
    lo, hi = np.minimum(s2, s20), np.maximum(s2, s20)
    assert ((post >= lo - 1e-12) & (post <= hi + 1e-12)).all()
    assert post.std() < s2.std()
    assert df_total > 4  # borrowed degrees of freedom


def test_moderated_t_has_extra_degrees_of_freedom_and_correct_sign():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, size=(3000, 6))
    x[:100, 3:] += 3.0  # first 100 genes are up in group B
    in_a, in_b = np.array([1, 1, 1, 0, 0, 0], bool), np.array([0, 0, 0, 1, 1, 1], bool)
    diff, p, info = moderated_t_test(x, in_a, in_b)
    assert info["df"] > 4 and info["prior_df"] > 0
    assert (diff[:100] > 0).mean() > 0.95
    assert (p[:100] < 0.01).mean() > 0.8
    assert (p[100:] < 0.05).mean() < 0.08
