"""Apply every countde method to a real two-group count matrix and summarise honestly.

    python scripts/reanalyse_real_counts.py \\
        --counts expression_matrix.csv --samples sample_groups.csv \\
        --reference Control --name gse163877 \\
        --original differential_expression_results.csv

counts:   gene IDs in column 1, raw integer counts per sample
samples:  CSV with columns Sample,Group   (or sample,group)
original: optional earlier results (Gene_ID, log2FoldChange, pvalue) to compare with

Writes docs/results/<name>_summary.json, docs/results/<name>_top_genes.csv and
docs/figures/<name>_pvalue_histogram.png. The summary reports how many genes
are significant AND how many would be expected by chance, so a null result is
visible as a null result.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from countde import METHODS, differential_expression  # noqa: E402
from countde.de import estimate_null_fraction, log_cpm  # noqa: E402
from countde.ebayes import moderated_t_test  # noqa: E402
from countde.normalize import normalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--original")
    ap.add_argument("--min-total-count", type=int, default=10)
    args = ap.parse_args()

    counts = pd.read_csv(args.counts, index_col=0)
    samples = pd.read_csv(args.samples)
    samples.columns = [c.lower() for c in samples.columns]
    groups = samples.set_index("sample")["group"]
    other = next(g for g in sorted(groups.unique()) if g != args.reference)

    out = {
        "dataset": args.name,
        "n_genes_in_matrix": int(counts.shape[0]),
        "group_sizes": groups.value_counts().to_dict(),
        "reference": args.reference,
        "comparison": f"{other} vs {args.reference}",
        "min_total_count": args.min_total_count,
        "methods": {},
    }

    results = {}
    for m in METHODS:
        res = differential_expression(counts, groups, reference=args.reference,
                                      min_total_count=args.min_total_count, method=m)
        results[m] = res
        n = len(res)
        out["methods"][m] = {
            "genes_tested": n,
            "raw_p_below_0.05": int((res.pvalue < 0.05).sum()),
            "expected_by_chance_at_0.05": round(0.05 * n, 1),
            "padj_below_0.10": int((res.padj < 0.10).sum()),
            "padj_below_0.05": int((res.padj < 0.05).sum()),
            "min_padj": round(float(res.padj.min()), 4),
            "estimated_null_fraction_pi0": round(estimate_null_fraction(res.pvalue.to_numpy()), 3),
        }

    norm = normalize(counts)
    norm = norm[counts.sum(axis=1) >= args.min_total_count]
    in_ref = (groups.reindex(norm.columns) == args.reference).to_numpy()
    _, _, info = moderated_t_test(log_cpm(norm), in_ref, ~in_ref)
    out["moderated_prior_degrees_of_freedom"] = round(info["prior_df"], 1)

    if args.original:
        orig = pd.read_csv(args.original, index_col=0).dropna(subset=["pvalue"])
        mod = results["moderated"]
        common = mod.index.intersection(orig.index)
        top = lambda s, k: set(s.sort_values().index[:k])
        out["comparison_with_original"] = {
            "genes_compared": int(len(common)),
            "original_genes_with_p": int(len(orig)),
            "spearman_log2fc": round(float(spearmanr(
                mod.loc[common, "log2_fold_change"], orig.loc[common, "log2FoldChange"])[0]), 3),
            "spearman_neg_log10_p": round(float(spearmanr(
                -np.log10(mod.loc[common, "pvalue"]), -np.log10(orig.loc[common, "pvalue"]))[0]), 3),
            "top100_overlap_by_p": len(top(mod.loc[common, "pvalue"], 100)
                                       & top(orig.loc[common, "pvalue"], 100)),
            "original_padj_non_missing": int(
                pd.read_csv(args.original, index_col=0).get("padj", pd.Series(dtype=float)).notna().sum()),
        }

    res_dir = ROOT / "docs" / "results"
    fig_dir = ROOT / "docs" / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{args.name}_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    results["moderated"].head(100).to_csv(res_dir / f"{args.name}_top_genes.csv",
                                          float_format="%.6g")

    _style.apply()
    p = results["moderated"]["pvalue"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.hist(p, bins=20, range=(0, 1), color="#2a78d6", edgecolor=_style.SURFACE, linewidth=1.2)
    ax.axhline(len(p) / 20, color=_style.INK_2, linestyle="--", linewidth=1)
    ax.annotate("flat = no signal", (0.98, len(p) / 20), xytext=(0, 5),
                textcoords="offset points", ha="right", color=_style.INK_2, fontsize=8.5)
    sig = out["methods"]["moderated"]["padj_below_0.10"]
    ax.set_title(f"{args.name}: {len(p):,} genes, {sig} at FDR < 10%")
    ax.set_xlabel("Raw p-value (moderated t-test)")
    ax.set_ylabel("Genes")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.name}_pvalue_histogram.png", dpi=200)
    plt.close(fig)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
