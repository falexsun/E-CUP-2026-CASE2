import csv
from pathlib import Path


def test_experiment_log_is_rectangular_unique_and_contains_public_lineage() -> None:
    path = Path(__file__).resolve().parents[1] / "experiments.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows
    assert all(None not in row for row in rows)
    identifiers = [row["id"] for row in rows]
    assert len(identifiers) == len(set(identifiers))

    public = {
        row["id"]: float(row["score"])
        for row in rows
        if row["metric"] == "public_macro_f1"
    }
    assert public["qwen3vl_reference_full_oof_v9_public"] == 0.745090
    assert public["qwen3vl_reference_v10_public"] == 0.793388
    assert public["qwen3vl_knn_v19_public"] == 0.808892
    assert public["qwen3vl_v27_public"] == 0.800964
    assert public["qwen3vl_standardized_regex_v45_public"] == 0.814222
