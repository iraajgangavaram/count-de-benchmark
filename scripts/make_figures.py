"""Regenerate every benchmark table and figure in docs/ from scratch.

    python scripts/make_figures.py            # simulation figures + tables
    python scripts/make_figures.py --reps 20  # more replicates, smoother curves

Everything is seeded, so the same command reproduces the same numbers.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import _style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from countde import run_benchmark, simulate_counts  # noqa: E402
from countde.de import log_cpm  # noqa: E402
from countde.ebayes import squeeze_var  # noqa: E402
from countde.normalize import normalize  # noqa: E402

FIG = ROOT / "docs" / "figures"
RES = ROOT / "docs" / "results"
ALPHA = 0.05
N_GRID = (3, 4, 5, 6, 8, 10)


def _label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                color=color, fontsize=8.5, va="center", fontweight="bold")


def fig_power(table: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, fc in zip(axes, (2.0, 4.0)):
        sub = table[table.fold_change == fc]
        for m, st in _style.METHOD_STYLE.items():
            d = sub[sub.method == m].sort_values("n_per_group")
            ax.plot(d.n_per_group, d.power, color=st["color"], marker=st["marker"],
                    label=st["label"], markeredgecolor=_style.SURFACE, markeredgewidth=1.2)
        ax.set_title(f"True fold change {fc:g}x")
        ax.set_xlabel("Replicates per group")
        ax.set_xticks(N_GRID)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(2.6, 11.6)
    axes[0].set_ylabel(f"Power at FDR < {ALPHA:.0%}")
    axes[1].legend(loc="lower right")
    fig.suptitle("Variance moderation recovers power when replicates are few",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "power_vs_replicates.png", dpi=200)
    plt.close(fig)


def fig_error_control(table: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sub = table[table.fold_change == 4.0]
    ax = axes[0]
    for m, st in _style.METHOD_STYLE.items():
        d = sub[sub.method == m].sort_values("n_per_group")
        d = d[d.n_called >= 10]  # FDR is meaningless when almost nothing is called
        ax.plot(d.n_per_group, d.observed_fdr, color=st["color"], marker=st["marker"],
                label=st["label"], markeredgecolor=_style.SURFACE, markeredgewidth=1.2)
    ax.axhline(ALPHA, color=_style.INK_2, linestyle="--", linewidth=1)
    ax.annotate("nominal 5%", (3, ALPHA), xytext=(0, 5), textcoords="offset points",
                color=_style.INK_2, fontsize=8.5)
    ax.annotate("points omitted where < 10 genes were called", (0.02, 0.02),
                xycoords="axes fraction", color=_style.INK_2, fontsize=8)
    ax.set_title("Observed FDR (4x changes)")
    ax.set_xlabel("Replicates per group")
    ax.set_ylabel("Share of calls that are false")
    ax.set_xlim(2.6, 10.4)
    ax.set_xticks(N_GRID)
    ax.set_ylim(0, 0.12)
    ax.legend(loc="upper right")

    ax = axes[1]
    for m, st in _style.METHOD_STYLE.items():
        d = table[(table.method == m) & (table.fold_change == 4.0)].sort_values("n_per_group")
        ax.plot(d.n_per_group, d.null_p_below_alpha, color=st["color"], marker=st["marker"],
                markeredgecolor=_style.SURFACE, markeredgewidth=1.2)
    ax.axhline(ALPHA, color=_style.INK_2, linestyle="--", linewidth=1)
    ax.annotate("ideal: 5%", (3, ALPHA), xytext=(0, 5), textcoords="offset points",
                color=_style.INK_2, fontsize=8.5)
    ax.set_title("Calibration on truly unchanged genes")
    ax.set_xlabel("Replicates per group")
    ax.set_ylabel("Null genes with raw p < 0.05")
    ax.set_xlim(2.6, 10.4)
    ax.set_xticks(N_GRID)
    ax.set_ylim(0, 0.12)
    fig.tight_layout()
    fig.savefig(FIG / "error_control.png", dpi=200)
    plt.close(fig)


def fig_shrinkage(seed: int = 3):
    sim = simulate_counts(n_genes=6000, n_per_group=3, fold_change=2.0, seed=seed)
    norm = normalize(sim.counts)
    norm = norm[sim.counts.sum(axis=1) >= 10]
    x = log_cpm(norm)
    a = np.array([g == "A" for g in sim.groups.reindex(norm.columns)])
    b = ~a
    df1 = 4
    pooled = (x[:, a].var(axis=1, ddof=1) * 2 + x[:, b].var(axis=1, ddof=1) * 2) / df1
    post, df_total, _ = squeeze_var(pooled, df1, covariate=x.mean(axis=1))
    amean = x.mean(axis=1)
    lim = (min(pooled.min(), post.min()) * 0.8, max(pooled.max(), post.max()) * 1.2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, v, title in ((axes[0], pooled, "Raw per-gene variance (4 d.f.)"),
                         (axes[1], post, f"Moderated variance (~{df_total:.0f} d.f.)")):
        ax.scatter(amean, v, s=4, alpha=0.25, color="#2a78d6", linewidths=0, rasterized=True)
        ax.set_yscale("log")
        ax.set_ylim(*lim)
        ax.set_title(title)
        ax.set_xlabel("Average log2 expression")
    axes[0].set_ylabel("Within-group variance")
    fig.suptitle("Empirical Bayes pulls noisy 3-vs-3 variances toward the trend",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "variance_shrinkage.png", dpi=200)
    plt.close(fig)
    return float(df_total - df1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-genes", type=int, default=5000)
    args = ap.parse_args()
    _style.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    main_tbl = run_benchmark(n_per_group=N_GRID, fold_changes=(2.0, 4.0),
                             reps=args.reps, n_genes=args.n_genes, alpha=ALPHA)
    main_tbl.to_csv(RES / "benchmark_main.csv", index=False, float_format="%.4f")
    fig_power(main_tbl)
    fig_error_control(main_tbl)
    prior_df = fig_shrinkage()
    print(f"prior d.f. estimated in the shrinkage example: {prior_df:.1f}")

    # Sensitivity to biological variability at the hardest design (3 vs 3, 4x)
    rows = []
    for disp in (0.05, 0.1, 0.2, 0.4, 0.8):
        t = run_benchmark(n_per_group=(3,), fold_changes=(4.0,), dispersion=disp,
                          reps=args.reps, n_genes=args.n_genes, alpha=ALPHA)
        rows.append(t)
    pd.concat(rows).to_csv(RES / "benchmark_dispersion.csv", index=False, float_format="%.4f")
    print("wrote", sorted(p.name for p in FIG.glob("*.png")), sorted(p.name for p in RES.glob("*.csv")))


if __name__ == "__main__":
    main()
