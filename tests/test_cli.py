import pandas as pd

from countde.cli import main


def test_run_command_writes_results(tmp_path):
    counts = pd.DataFrame(
        {"A1": [100, 50, 60], "A2": [110, 50, 60], "B1": [400, 50, 60], "B2": [420, 50, 60]},
        index=pd.Index(["up", "c1", "c2"], name="gene"),
    )
    counts.to_csv(tmp_path / "counts.csv")
    pd.DataFrame(
        {"sample": ["A1", "A2", "B1", "B2"], "group": ["A", "A", "B", "B"]}
    ).to_csv(tmp_path / "samples.csv", index=False)
    out = tmp_path / "out.csv"

    code = main(["run", str(tmp_path / "counts.csv"), str(tmp_path / "samples.csv"),
                 "--min-total-count", "0", "-o", str(out)])

    assert code == 0
    res = pd.read_csv(out, index_col="gene")
    assert list(res.columns) == ["base_mean", "log2_fold_change", "pvalue", "padj"]
    assert res.index[0] == "up" and res.loc["up", "log2_fold_change"] > 1.5


def test_method_option_changes_the_test_and_rejects_unknown_methods(tmp_path):
    import pytest

    rng_counts = pd.DataFrame(
        {f"{g}{i}": [20 + 3 * i + j for j in range(40)] for g in "AB" for i in range(3)},
        index=[f"g{j}" for j in range(40)],
    )
    rng_counts.loc["g0", ["B0", "B1", "B2"]] = [200, 210, 190]
    rng_counts.index.name = "gene"
    rng_counts.to_csv(tmp_path / "c.csv")
    pd.DataFrame(
        {"sample": list(rng_counts.columns), "group": [c[0] for c in rng_counts.columns]}
    ).to_csv(tmp_path / "s.csv", index=False)

    outputs = {}
    for method in ("moderated", "welch"):
        out = tmp_path / f"{method}.csv"
        assert main(["run", str(tmp_path / "c.csv"), str(tmp_path / "s.csv"),
                     "--method", method, "--min-total-count", "0", "-o", str(out)]) == 0
        outputs[method] = pd.read_csv(out, index_col="gene")
    assert outputs["moderated"].index[0] == outputs["welch"].index[0] == "g0"
    assert not outputs["moderated"]["pvalue"].equals(outputs["welch"]["pvalue"])

    with pytest.raises(SystemExit):  # argparse rejects unknown choices
        main(["run", str(tmp_path / "c.csv"), str(tmp_path / "s.csv"), "--method", "deseq2"])
