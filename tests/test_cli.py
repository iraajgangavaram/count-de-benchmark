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
