"""Differential expression: normalise, Welch t-test on log2 counts, BH correction."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .normalize import normalize, validate_counts


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


def differential_expression(
    counts: pd.DataFrame,
    groups: pd.Series,
    reference: str | None = None,
    min_total_count: int = 10,
) -> pd.DataFrame:
    """Test every gene for a difference between exactly two groups.

    Parameters
    ----------
    counts : genes x samples raw integer counts.
    groups : group label per sample, indexed by sample name.
    reference : group treated as the baseline (default: first label alphabetically).
        log2_fold_change is (other group) / reference.
    min_total_count : genes with fewer total reads are dropped before testing,
        which reduces the multiple-testing burden.

    Method: median-of-ratios normalisation, log2(x + 1) transform, Welch's
    t-test per gene, Benjamini-Hochberg FDR. This is a lightweight approach;
    it does not model count dispersion like DESeq2 or edgeR and will have
    less power on small experiments.
    """
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
    keep = counts.sum(axis=1) >= min_total_count
    norm = norm[keep]
    if norm.empty:
        raise ValueError("no genes pass the min_total_count filter")

    logn = np.log2(norm.to_numpy() + 1.0)
    with warnings.catch_warnings():
        # Genes with identical values everywhere trigger scipy precision warnings;
        # they yield NaN, which is handled explicitly below.
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p = stats.ttest_ind(logn[:, in_other], logn[:, in_ref], axis=1, equal_var=False)
    p = np.where(np.isnan(p), 1.0, p)  # zero variance in both groups -> no evidence

    mean_other = norm.to_numpy()[:, in_other].mean(axis=1)
    mean_ref = norm.to_numpy()[:, in_ref].mean(axis=1)
    result = pd.DataFrame(
        {
            "base_mean": norm.to_numpy().mean(axis=1),
            "log2_fold_change": np.log2((mean_other + 1.0) / (mean_ref + 1.0)),
            "pvalue": p,
            "padj": benjamini_hochberg(p),
        },
        index=norm.index,
    )
    return result.sort_values(["padj", "pvalue"])
