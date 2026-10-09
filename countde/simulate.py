"""Simulate negative-binomial RNA-seq counts with known differentially expressed genes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Simulation:
    counts: pd.DataFrame  # genes x samples
    groups: pd.Series  # sample -> "A" / "B"
    is_de: pd.Series  # gene -> bool (ground truth)
    true_size_factors: pd.Series


def simulate_counts(
    n_genes: int = 5000,
    n_per_group: int = 3,
    frac_de: float = 0.1,
    fold_change: float = 3.0,
    dispersion: float = 0.2,
    size_factor_sd: float = 0.25,
    seed: int = 0,
) -> Simulation:
    """Gamma-Poisson (negative binomial) counts.

    Gene means are log-normal (median 100). A fraction `frac_de` of genes are
    changed by `fold_change` in group B, half up and half down. Variance is
    mu + dispersion * mu^2. Each sample has its own random library size.
    """
    if not 0 <= frac_de <= 1:
        raise ValueError("frac_de must be between 0 and 1")
    if fold_change < 1 or dispersion <= 0 or n_per_group < 1:
        raise ValueError("need fold_change >= 1, dispersion > 0 and n_per_group >= 1")
    rng = np.random.default_rng(seed)
    n_samples = 2 * n_per_group

    base = rng.lognormal(mean=np.log(100), sigma=1.5, size=n_genes)
    n_de = int(round(frac_de * n_genes))
    de_idx = rng.permutation(n_genes)[:n_de]
    lfc = np.zeros(n_genes)
    lfc[de_idx] = np.log2(fold_change) * np.where(np.arange(n_de) % 2 == 0, 1, -1)

    in_b = np.r_[np.zeros(n_per_group), np.ones(n_per_group)]
    sf = rng.lognormal(0.0, size_factor_sd, size=n_samples)
    mu = base[:, None] * 2.0 ** (lfc[:, None] * in_b[None, :]) * sf[None, :]
    lam = rng.gamma(shape=1.0 / dispersion, scale=mu * dispersion)
    counts = rng.poisson(lam)

    samples = [f"A{i + 1}" for i in range(n_per_group)] + [f"B{i + 1}" for i in range(n_per_group)]
    genes = [f"gene{i + 1:05d}" for i in range(n_genes)]
    is_de = np.zeros(n_genes, dtype=bool)
    is_de[de_idx] = True
    return Simulation(
        counts=pd.DataFrame(counts, index=genes, columns=samples),
        groups=pd.Series(["A"] * n_per_group + ["B"] * n_per_group, index=samples),
        is_de=pd.Series(is_de, index=genes),
        true_size_factors=pd.Series(sf, index=samples),
    )
