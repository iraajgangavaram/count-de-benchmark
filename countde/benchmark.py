"""Measure power and false discovery rate on simulated data with known truth."""

from __future__ import annotations

import pandas as pd

from .de import differential_expression
from .simulate import simulate_counts


def evaluate(result: pd.DataFrame, is_de: pd.Series, alpha: float = 0.05) -> dict:
    """Compare calls at padj < alpha with ground truth.

    Genes removed by the low-count filter count as not called, so power is
    measured over *all* truly DE genes, not just the ones that were tested.
    """
    called = set(result.index[result["padj"] < alpha])
    truth = set(is_de.index[is_de])
    true_pos = len(called & truth)
    false_pos = len(called - truth)
    return {
        "n_called": len(called),
        "power": true_pos / len(truth) if truth else float("nan"),
        "observed_fdr": false_pos / len(called) if called else 0.0,
    }


def run_benchmark(
    n_per_group=(3, 6, 10),
    fold_changes=(2.0, 4.0),
    reps: int = 5,
    n_genes: int = 5000,
    alpha: float = 0.05,
    dispersion: float = 0.2,
) -> pd.DataFrame:
    """Average power and observed FDR over `reps` simulations per setting."""
    rows = []
    for n in n_per_group:
        for fc in fold_changes:
            metrics = []
            for rep in range(reps):
                sim = simulate_counts(
                    n_genes=n_genes, n_per_group=n, fold_change=fc,
                    dispersion=dispersion, seed=1000 * n + rep,
                )
                res = differential_expression(sim.counts, sim.groups)
                metrics.append(evaluate(res, sim.is_de, alpha))
            m = pd.DataFrame(metrics).mean()
            rows.append(
                {"n_per_group": n, "fold_change": fc, "power": m["power"],
                 "observed_fdr": m["observed_fdr"], "target_fdr": alpha}
            )
    return pd.DataFrame(rows)
