import numpy as np
import pandas as pd
import pytest

from countde import (
    benjamini_hochberg,
    differential_expression,
    evaluate,
    size_factors,
    simulate_counts,
)


# ---------- Benjamini-Hochberg ----------

def test_bh_matches_hand_calculation():
    # sorted p: .005 .01 .03 .04 -> p*n/rank: .02 .02 .04 .04 (already monotone)
    adj = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.02, 0.04, 0.04, 0.02])


def test_bh_enforces_monotonicity_and_cap():
    # sorted p: .01 .04 .5 -> p*n/rank = .03 .06 .5 (original order: .5, .03, .06)
    adj = benjamini_hochberg([0.5, 0.01, 0.04])
    assert np.allclose(adj, [0.5, 0.03, 0.06])
    # sorted p: .3 .45 -> p*n/rank = .6 .45; the smaller p-value may not get a
    # larger adjusted value than the next one, so .6 is lowered to .45
    adj = benjamini_hochberg([0.45, 0.3])
    assert np.allclose(adj, [0.45, 0.45])
    assert (benjamini_hochberg([0.9, 0.95, 1.0]) <= 1.0).all()


def test_bh_edge_cases():
    assert benjamini_hochberg([]).size == 0
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1, float("nan")])


# ---------- size factors ----------

def test_size_factors_recover_known_library_sizes():
    sim = simulate_counts(n_genes=3000, n_per_group=4, seed=1)
    est = np.log(size_factors(sim.counts))
    true = np.log(sim.true_size_factors)
    est, true = est - est.mean(), true - true.mean()  # factors are relative
    assert np.abs(est - true).max() < 0.05


def test_size_factors_exactly_scale_invariant():
    base = pd.DataFrame({"s1": [10, 20, 30, 40], "s2": [10, 20, 30, 40]})
    doubled = base.assign(s2=base["s2"] * 2)
    sf = size_factors(doubled)
    assert sf["s2"] / sf["s1"] == pytest.approx(2.0)


def test_size_factors_need_a_gene_expressed_everywhere():
    counts = pd.DataFrame({"s1": [0, 5], "s2": [5, 0]}, index=["g1", "g2"])
    with pytest.raises(ValueError):
        size_factors(counts)


@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame({"s1": [-1, 5], "s2": [3, 4]}),
        pd.DataFrame({"s1": [1.5, 5], "s2": [3, 4]}),
        pd.DataFrame({"s1": [np.nan, 5], "s2": [3, 4]}),
    ],
)
def test_invalid_counts_rejected(bad):
    with pytest.raises(ValueError):
        size_factors(bad)


# ---------- differential expression ----------

def _toy():
    counts = pd.DataFrame(
        {
            "A1": [100, 50, 60, 70, 80],
            "A2": [110, 50, 60, 70, 80],
            "A3": [95, 50, 60, 70, 80],
            "B1": [400, 50, 60, 70, 80],
            "B2": [420, 50, 60, 70, 80],
            "B3": [390, 50, 60, 70, 80],
        },
        index=["up", "c1", "c2", "c3", "c4"],
    )
    groups = pd.Series(["A"] * 3 + ["B"] * 3, index=counts.columns)
    return counts, groups


def test_fold_change_direction_and_reference_flip():
    counts, groups = _toy()
    res = differential_expression(counts, groups, min_total_count=0)
    assert res.loc["up", "log2_fold_change"] == pytest.approx(np.log2(403.33 / 102.0), abs=0.05)
    assert res.loc["up", "pvalue"] < 0.01
    flipped = differential_expression(counts, groups, reference="B", min_total_count=0)
    assert flipped.loc["up", "log2_fold_change"] == pytest.approx(
        -res.loc["up", "log2_fold_change"], abs=1e-9
    )


def test_constant_genes_get_p_of_one_not_nan():
    counts, groups = _toy()
    res = differential_expression(counts, groups, min_total_count=0)
    assert res.loc["c1", "pvalue"] == 1.0
    assert not res.isna().any().any()


def test_low_count_filter_drops_genes():
    counts, groups = _toy()
    counts.loc["tiny"] = [1, 0, 1, 0, 1, 0]
    res = differential_expression(counts, groups, min_total_count=10)
    assert "tiny" not in res.index and "up" in res.index


def test_group_validation():
    counts, groups = _toy()
    with pytest.raises(ValueError):  # three groups
        differential_expression(counts, pd.Series(["A", "B", "C", "A", "B", "C"], index=counts.columns))
    with pytest.raises(ValueError):  # only one replicate in a group
        differential_expression(counts, pd.Series(["A", "B", "B", "B", "B", "B"], index=counts.columns))
    with pytest.raises(ValueError):  # unlabeled sample
        differential_expression(counts, groups.drop("B3"))
    with pytest.raises(ValueError):  # unknown reference
        differential_expression(counts, groups, reference="Z")


# ---------- statistical behaviour on simulated truth ----------

def test_pvalues_are_calibrated_for_non_de_genes():
    sim = simulate_counts(n_genes=8000, n_per_group=8, seed=3)
    res = differential_expression(sim.counts, sim.groups, min_total_count=0)
    null = res.loc[~sim.is_de.reindex(res.index), "pvalue"]
    assert 0.03 < (null < 0.05).mean() < 0.07
    assert 0.003 < (null < 0.01).mean() < 0.02


def test_fdr_controlled_and_power_high_with_adequate_replication():
    sim = simulate_counts(n_genes=5000, n_per_group=10, fold_change=4.0, seed=11)
    res = differential_expression(sim.counts, sim.groups)
    m = evaluate(res, sim.is_de, alpha=0.05)
    assert m["power"] > 0.9
    assert m["observed_fdr"] < 0.10  # nominal 0.05, allow sampling noise


def test_no_true_de_genes_gives_few_calls():
    sim = simulate_counts(n_genes=3000, n_per_group=6, frac_de=0.0, seed=5)
    res = differential_expression(sim.counts, sim.groups)
    assert (res["padj"] < 0.05).sum() <= 5


def test_simulation_is_reproducible_and_validates_inputs():
    a, b = simulate_counts(seed=2), simulate_counts(seed=2)
    assert a.counts.equals(b.counts)
    assert a.is_de.sum() == 500
    with pytest.raises(ValueError):
        simulate_counts(frac_de=1.5)
