"""Measure power, FDR control, ranking quality and calibration on simulated data with known truth."""

from __future__ import annotations

import pandas as pd
from scipy.stats import rankdata

from .de import METHODS, differential_expression
from .simulate import simulate_counts


def auc_from_pvalues(pvalues: pd.Series, is_de: pd.Series) -> float:
    """Area under the ROC curve for ranking truly DE genes above null genes by p-value.

    Computed via the rank-sum identity; 0.5 is chance, 1.0 is perfect separation.
    Genes missing from `pvalues` (removed by the low-count filter) are ignored.
    """
    truth = is_de.reindex(pvalues.index).to_numpy(dtype=bool)
    n_pos, n_neg = int(truth.sum()), int((~truth).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(-pvalues.to_numpy())  # small p-value -> high rank
    return float((ranks[truth].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(result: pd.DataFrame, is_de: pd.Series, alpha: float = 0.05) -> dict:
    """Compare calls at padj < alpha with ground truth.

    Genes removed by the low-count filter count as not called, so power is
    measured over *all* truly DE genes, not just the ones that were tested.
    Also reports AUC and the fraction of truly null genes with raw p < alpha
    (should be close to alpha if p-values are calibrated).
    """
    called = set(result.index[result["padj"] < alpha])
    truth = set(is_de.index[is_de])
    true_pos = len(called & truth)
    false_pos = len(called - truth)
    null_p = result.loc[result.index.intersection(is_de.index[~is_de]), "pvalue"]
    return {
        "n_called": len(called),
        "power": true_pos / len(truth) if truth else float("nan"),
        "observed_fdr": false_pos / len(called) if called else 0.0,
        "auc": auc_from_pvalues(result["pvalue"], is_de),
        "null_p_below_alpha": float((null_p < alpha).mean()) if len(null_p) else float("nan"),
    }


def run_benchmark(
    n_per_group=(3, 6, 10),
    fold_changes=(2.0, 4.0),
    methods=METHODS,
    reps: int = 5,
    n_genes: int = 5000,
    alpha: float = 0.05,
    dispersion: float = 0.2,
) -> pd.DataFrame:
    """Average metrics over `reps` simulations for every method x design.

    Each (design, replicate) uses the same simulated dataset for every method,
    so methods are compared on identical data.
    """
    rows = []
    for n in n_per_group:
        for fc in fold_changes:
            per_method = {m: [] for m in methods}
            for rep in range(reps):
                sim = simulate_counts(
                    n_genes=n_genes, n_per_group=n, fold_change=fc,
                    dispersion=dispersion, seed=1000 * n + rep,
                )
                for m in methods:
                    res = differential_expression(sim.counts, sim.groups, method=m)
                    per_method[m].append(evaluate(res, sim.is_de, alpha))
            for m in methods:
                mean = pd.DataFrame(per_method[m]).mean()
                rows.append(
                    {
                        "method": m, "n_per_group": n, "fold_change": fc,
                        "dispersion": dispersion, "power": mean["power"],
                        "observed_fdr": mean["observed_fdr"], "auc": mean["auc"],
                        "null_p_below_alpha": mean["null_p_below_alpha"],
                        "n_called": mean["n_called"],
                    }
                )
    return pd.DataFrame(rows)
