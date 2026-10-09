"""Command-line interface: python -m countde {run,benchmark} ..."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .benchmark import run_benchmark
from .de import differential_expression


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="countde", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="differential expression between two groups")
    r.add_argument("counts", help="CSV: gene IDs in first column, one column per sample")
    r.add_argument("samples", help="CSV with columns: sample,group")
    r.add_argument("--reference", help="baseline group (default: alphabetically first)")
    r.add_argument("--min-total-count", type=int, default=10)
    r.add_argument("-o", "--output", help="write results CSV here (default: stdout)")

    b = sub.add_parser("benchmark", help="power / FDR on simulated data")
    b.add_argument("--reps", type=int, default=5)
    b.add_argument("--n-genes", type=int, default=5000)

    args = p.parse_args(argv)

    if args.cmd == "run":
        counts = pd.read_csv(args.counts, index_col=0)
        samples = pd.read_csv(args.samples)
        if not {"sample", "group"} <= set(samples.columns):
            p.error("samples CSV needs columns: sample,group")
        groups = samples.set_index("sample")["group"]
        res = differential_expression(counts, groups, args.reference, args.min_total_count)
        res.to_csv(args.output or sys.stdout, index_label="gene")
    else:
        table = run_benchmark(reps=args.reps, n_genes=args.n_genes)
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
