# count-de-benchmark

A lightweight differential-expression pipeline for RNA-seq count data, together
with a simulation benchmark that measures how well it actually works: how many
truly changed genes it finds (power) and how often its discoveries are wrong
(observed false discovery rate).

The aim is both a usable tool and a worked example of **validating a
statistical method against ground truth** instead of assuming it works.

## Method

1. **Normalisation:** median-of-ratios size factors (as in DESeq2).
2. **Filtering:** drop genes with fewer than `min_total_count` reads (default 10).
3. **Test:** `log2(normalised count + 1)`, then Welch's t-test per gene.
4. **Multiple testing:** Benjamini-Hochberg false discovery rate.

This is deliberately simple. It does **not** model count dispersion or share
information across genes the way DESeq2 or edgeR do, and it is not a
replacement for them.

## Benchmark results

Simulated negative-binomial counts: 5,000 genes, 10% truly differential
(half up, half down), dispersion 0.2, gene means log-normal (median 100),
random library sizes. Averaged over 5 simulations per row; call threshold
`padj < 0.05`. Reproduce with `python -m countde benchmark`.

| Replicates per group | True fold change | Power | Observed FDR | Target FDR |
|---:|---:|---:|---:|---:|
| 3  | 2x | 0.000 | 0.000 | 0.05 |
| 3  | 4x | 0.000 | 0.000 | 0.05 |
| 6  | 2x | 0.000 | 0.300 | 0.05 |
| 6  | 4x | 0.747 | 0.040 | 0.05 |
| 10 | 2x | 0.315 | 0.038 | 0.05 |
| 10 | 4x | 0.986 | 0.037 | 0.05 |

What this shows:

- **With enough replicates the method is well behaved.** At 10 per group the
  observed FDR stays at or below the 5% target, and power is high for large
  effects.
- **With 3 replicates per group it finds essentially nothing.** A t-test with
  so few samples cannot reach the very small p-values that 5,000 simultaneous
  tests demand. Small designs are hard for every method; methods that share
  dispersion information across genes exist precisely to help here, but they
  are not evaluated in this project.
- **Where almost nothing is called, observed FDR is unreliable.** The 0.300
  at 6 replicates / 2x comes from a handful of calls (power is 0), so it is
  noise rather than evidence of poor FDR control.
- Also checked (see `tests/`): size factors are recovered to within 5% of the
  simulated truth, and p-values for genuinely unchanged genes are calibrated
  (about 5% fall below 0.05).

These numbers depend on the simulation settings above; treat them as a
demonstration of the approach, not universal performance figures.

## Usage

```bash
pip install -e .

# Differential expression
python -m countde run counts.csv samples.csv --reference control -o results.csv

# Reproduce the benchmark
python -m countde benchmark --reps 5
```

- `counts.csv`: gene IDs in the first column, one column of **raw integer
  counts** per sample.
- `samples.csv`: columns `sample,group`, with exactly two groups and at least
  two samples in each.
- `log2_fold_change` is (non-reference group) / reference.

Python API:

```python
from countde import differential_expression, simulate_counts, evaluate

sim = simulate_counts(n_per_group=10, fold_change=4.0, seed=1)
res = differential_expression(sim.counts, sim.groups)
print(evaluate(res, sim.is_de, alpha=0.05))
```

## Limitations

- Two-group comparisons only; no covariates or batch terms.
- Genes need non-zero counts in every sample to contribute to size-factor
  estimation, so very sparse data (for example single-cell) is not supported.
- The simulation assumes a negative-binomial model with a single dispersion;
  real data has gene-specific dispersion and other structure.

## Tests

```bash
pip install pytest
pytest
```
