"""Library-size normalisation for RNA-seq count matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_counts(counts: pd.DataFrame) -> None:
    """Raise ValueError unless counts is a genes x samples matrix of non-negative integers."""
    if counts.empty:
        raise ValueError("count matrix is empty")
    arr = counts.to_numpy(dtype=float)
    if np.isnan(arr).any():
        raise ValueError("count matrix contains missing values")
    if (arr < 0).any():
        raise ValueError("count matrix contains negative values")
    if not np.allclose(arr, np.round(arr)):
        raise ValueError("count matrix must contain integer counts (raw, not normalised)")


def size_factors(counts: pd.DataFrame) -> pd.Series:
    """Median-of-ratios size factors (the method used by DESeq2).

    Each sample is compared with a pseudo-reference sample (the per-gene
    geometric mean). Only genes with non-zero counts in every sample can
    contribute, because the geometric mean is zero otherwise. The size factor
    is the median ratio of a sample to the reference over those genes.
    Factors are relative: only ratios between samples are meaningful.
    """
    validate_counts(counts)
    arr = counts.to_numpy(dtype=float)
    usable = (arr > 0).all(axis=1)
    if not usable.any():
        raise ValueError(
            "no gene has non-zero counts in every sample; "
            "median-of-ratios normalisation is not possible"
        )
    logs = np.log(arr[usable])
    log_ref = logs.mean(axis=1, keepdims=True)
    return pd.Series(np.exp(np.median(logs - log_ref, axis=0)), index=counts.columns)


def normalize(counts: pd.DataFrame) -> pd.DataFrame:
    """Divide each sample's counts by its size factor."""
    return counts / size_factors(counts)
