"""Differential expression between two groups with three selectable tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .ebayes import moderated_t_test
from .normalize import normalize, validate_counts

METHODS = ("moderated", "welch", "mannwhitney")


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, returned in the input order."""
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p
    if np.isnan(p).any():
        raise ValueError("p-values must not contain NaN")
    order = np.argsort(p)
    scaled = p[order] * p.size / np.arange(1, p.size + 1)
    # enforce monotonicity from the largest p-value downwards
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def estimate_null_fraction(pvalues, lam: float = 0.5) -> float:
    """Storey's pi0: share of genes whose p-values look like pure noise.

    Under the null, p-values are uniform, so the proportion above `lam` is
    about (1 - lam) * pi0. Returns a value in (0, 1]; values near 1 mean the
    p-value histogram is flat, i.e. there is little evidence of any signal.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0 or not 0.0 < lam < 1.0:
        raise ValueError("need at least one p-value and 0 < lam < 1")
    return float(min(1.0, max((p > lam).mean() / (1.0 - lam), 1.0 / p.size)))


def log_cpm(norm: pd.DataFrame, prior_count: float = 0.5) -> np.ndarray:
    """log2 counts-per-million of size-factor-normalised counts.

    All samples share the same scaling constant (the mean normalised library
    size), so this is a pure shift of log2(norm + prior): it stabilises low
    counts without reintroducing composition effects through per-sample CPM.
    """
    lib = norm.sum(axis=0).mean()
    return np.log2((norm.to_numpy() + prior_count) / (lib + 1.0) * 1e6)


def differential_expression(
    counts: pd.DataFrame,
    groups: pd.Series,
    reference: str | None = None,
    min_total_count: int = 10,
    method: str = "moderated",
) -> pd.DataFrame:
    """Test every gene for a difference between exactly two groups.

    Parameters
    ----------
    counts : genes x samples raw integer counts.
    groups : group label per sample, indexed by sample name.
    reference : baseline group (default: first label alphabetically).
        log2_fold_change is (other group) / reference.
    min_total_count : genes with fewer total reads are dropped before testing.
    method : one of
        "moderated"   empirical-Bayes moderated t-test on log-CPM (default);
        "welch"       Welch's t-test on log-CPM;
        "mannwhitney" rank-sum test on log-CPM.

    All methods use median-of-ratios normalisation and Benjamini-Hochberg FDR.
    None of them models count dispersion directly the way DESeq2 or edgeR do.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    validate_counts(counts)
    groups = groups.reindex(counts.columns)
    if groups.isna().any():
        raise ValueError("every sample in the count matrix needs a group label")
    levels = sorted(groups.unique())
    if len(levels) != 2:
        raise ValueError(f"exactly two groups are required, found {len(levels)}")
    ref = reference if reference is not None else levels[0]
    if ref not in levels:
        raise ValueError(f"reference group {ref!r} not found in groups")
    other = next(g for g in levels if g != ref)
    in_ref, in_other = (groups == ref).to_numpy(), (groups == other).to_numpy()
    if in_ref.sum() < 2 or in_other.sum() < 2:
        raise ValueError("each group needs at least 2 samples")

    norm = normalize(counts)  # size factors use all genes, before filtering
    norm = norm[counts.sum(axis=1) >= min_total_count]
    if norm.empty:
        raise ValueError("no genes pass the min_total_count filter")
    logx = log_cpm(norm)

    if method == "moderated":
        diff, p, _ = moderated_t_test(logx, in_ref, in_other)
    else:
        with warnings.catch_warnings():
            # Identical values trigger scipy precision warnings; NaN is handled below.
            warnings.simplefilter("ignore", RuntimeWarning)
            if method == "welch":
                _, p = stats.ttest_ind(logx[:, in_other], logx[:, in_ref], axis=1, equal_var=False)
            else:
                _, p = stats.mannwhitneyu(
                    logx[:, in_other], logx[:, in_ref], axis=1, alternative="two-sided"
                )
        p = np.where(np.isnan(p), 1.0, p)  # no variation at all -> no evidence
        diff = logx[:, in_other].mean(axis=1) - logx[:, in_ref].mean(axis=1)

    result = pd.DataFrame(
        {
            "base_mean": norm.mean(axis=1).to_numpy(),
            "log2_fold_change": diff,
            "pvalue": p,
            "padj": benjamini_hochberg(p),
        },
        index=norm.index,
    )
    return result.sort_values(["padj", "pvalue"])
