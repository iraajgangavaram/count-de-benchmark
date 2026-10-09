"""Empirical-Bayes variance moderation (Smyth 2004), implemented from scratch.

Gene-wise variances from few replicates are very noisy. The empirical-Bayes
model treats true gene variances as draws from a scaled inverse chi-square
prior, sigma_g^2 ~ s0^2 * d0 / chi2_d0, estimates (s0^2, d0) from the observed
variances, and replaces each variance by a posterior value that borrows
strength across genes. The moderated t-statistic then has d0 + d extra
degrees of freedom.

The mean-variance trend is handled by letting s0^2 depend on average
expression through a cubic polynomial (a simple stand-in for the spline used
by limma-trend). This module has been validated against simulated ground
truth (see tests/test_ebayes.py), not against the R limma package.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import digamma, polygamma


def trigamma(x):
    return polygamma(1, x)


def trigamma_inverse(x: float) -> float:
    """Solve trigamma(y) = x for y by Newton's method (as in limma::trigammaInverse)."""
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = float(trigamma(y))
        dif = tri * (1.0 - tri / x) / float(polygamma(2, y))
        y += dif
        if -dif / y < 1e-8:
            break
    return y


def fit_f_dist(
    s2: np.ndarray,
    df1: float,
    covariate: Optional[np.ndarray] = None,
    trend_degree: int = 3,
) -> Tuple[np.ndarray, float]:
    """Estimate the prior (s0^2 per gene, d0) from sample variances.

    Parameters
    ----------
    s2 : per-gene sample variances, each estimated on `df1` degrees of freedom.
    covariate : optional per-gene average expression; if given (and enough genes
        are available) the prior variance follows a polynomial trend in it.

    Returns (s20, df2) where s20 has the same length as s2 and df2 may be inf.
    Returns (nan array, 0.0) when there are too few usable genes to estimate a
    prior; callers should then skip moderation.
    """
    s2 = np.asarray(s2, dtype=float)
    ok = np.isfinite(s2) & (s2 > -1e-15)
    n_ok = int(ok.sum())
    if n_ok < 3 or df1 <= 0:
        return np.full_like(s2, np.nan), 0.0

    x = np.clip(s2[ok], 0.0, None)
    m = np.median(x)
    if m == 0:
        m = 1.0
    x = np.maximum(x, 1e-5 * m)  # zero-variance genes would give log(0)
    e = np.log(x) - digamma(df1 / 2.0) + np.log(df1 / 2.0)

    n_params = 1
    if covariate is not None and n_ok >= 10 * (trend_degree + 1):
        cov = np.asarray(covariate, dtype=float)
        mu, sd = cov[ok].mean(), cov[ok].std() or 1.0
        coef = np.polyfit((cov[ok] - mu) / sd, e, trend_degree)
        fit_all = np.polyval(coef, (cov - mu) / sd)
        fit_ok = fit_all[ok]
        n_params = trend_degree + 1
    else:
        fit_all = np.full_like(s2, e.mean())
        fit_ok = fit_all[ok]

    evar = np.sum((e - fit_ok) ** 2) / (n_ok - n_params) - float(trigamma(df1 / 2.0))
    if evar > 0:
        df2 = 2.0 * trigamma_inverse(evar)
        if df2 > 1e6:
            df2 = np.inf
    else:
        df2 = np.inf
    if np.isinf(df2):
        s20 = np.exp(fit_all)
    else:
        s20 = np.exp(fit_all + digamma(df2 / 2.0) - np.log(df2 / 2.0))
    return s20, float(df2)


def squeeze_var(
    s2: np.ndarray, df1: float, covariate: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Posterior variances. Returns (s2_post, df_total, s20)."""
    s2 = np.asarray(s2, dtype=float)
    s20, df2 = fit_f_dist(s2, df1, covariate)
    if df2 == 0.0:
        return s2.copy(), df1, s20
    if np.isinf(df2):
        return s20.copy(), np.inf, s20
    post = (df2 * s20 + df1 * s2) / (df2 + df1)
    return post, df1 + df2, s20


def moderated_t_test(
    log_expr: np.ndarray, in_a: np.ndarray, in_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Moderated two-group t-test on a genes x samples log-expression matrix.

    Returns (logFC b - a, p-values, info) where info reports the estimated
    prior degrees of freedom `prior_df` and residual `df`.
    """
    xa, xb = log_expr[:, in_a], log_expr[:, in_b]
    na, nb = xa.shape[1], xb.shape[1]
    df1 = na + nb - 2
    diff = xb.mean(axis=1) - xa.mean(axis=1)
    pooled = ((na - 1) * xa.var(axis=1, ddof=1) + (nb - 1) * xb.var(axis=1, ddof=1)) / df1
    amean = log_expr.mean(axis=1)

    s2_post, df_total, _ = squeeze_var(pooled, df1, covariate=amean)
    se = np.sqrt(s2_post * (1.0 / na + 1.0 / nb))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = diff / se
    if np.isinf(df_total):
        p = 2 * stats.norm.sf(np.abs(t))
    else:
        p = 2 * stats.t.sf(np.abs(t), df_total)
    p = np.where(np.isnan(p), 1.0, p)  # zero variance and zero difference
    return diff, p, {"prior_df": float(df_total - df1), "df": float(df_total)}
