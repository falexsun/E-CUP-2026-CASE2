import json

import pandas as pd

from ozon_quality.audit import write_data_audit, write_leakage_report
from ozon_quality.oof import multilabel_group_folds, run_group_oof
from ozon_quality.splitting import group_holdout


def _dataset(path, rows=20) -> str:
    frame = pd.DataFrame(
        {
            "id": range(rows),
            "name": [f"product {index}" for index in range(rows)],
            "target": ["ok" if index % 2 else "bad" for index in range(rows)],
            "seller": [f"seller-{index // 2}" for index in range(rows)],
        }
    )
    frame.to_csv(path, index=False)
    return str(path)


def _schema(path) -> str:
    payload = {
        "schema_version": 1,
        "columns": {"id": "id", "title": "name", "label": "target", "group": "seller"},
        "text_fields": ["title"],
        "task_type": "auto",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_audit_and_group_split(tmp_path) -> None:
    source = _dataset(tmp_path / "all.csv")
    schema = _schema(tmp_path / "schema.json")
    report = tmp_path / "DATA_AUDIT.md"
    write_data_audit(source, str(report), schema=schema, image_sample=0)
    assert "Inferred task: **single-label**" in report.read_text(encoding="utf-8")
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    summary = group_holdout(
        source,
        str(train_path),
        str(valid_path),
        schema=schema,
        valid_size=0.2,
        seed=42,
        candidates=8,
    )
    assert summary["train_rows"] + summary["valid_rows"] == 20
    train_sellers = set(pd.read_csv(train_path)["seller"])
    valid_sellers = set(pd.read_csv(valid_path)["seller"])
    assert not train_sellers & valid_sellers
    validation_report = tmp_path / "VALIDATION.md"
    assert write_leakage_report(
        str(train_path), str(valid_path), str(validation_report), schema=schema
    )


def test_leakage_report_blocks_duplicate_ids(tmp_path) -> None:
    source = _dataset(tmp_path / "train.csv", rows=6)
    schema = _schema(tmp_path / "schema.json")
    report = tmp_path / "VALIDATION.md"
    assert not write_leakage_report(source, source, str(report), schema=schema)
    assert "Status: **BLOCKED**" in report.read_text(encoding="utf-8")


def test_group_oof_writes_reusable_blend_config(tmp_path) -> None:
    source = _dataset(tmp_path / "all.csv", rows=20)
    schema = _schema(tmp_path / "schema.json")
    output = tmp_path / "oof"
    metrics = run_group_oof(
        source,
        str(output),
        schema=schema,
        text_model="debug-hash",
        vision_model="debug-pixels",
        text_revision=None,
        vision_revision=None,
        device="cpu",
        batch_size=4,
        seed=42,
        folds=2,
    )
    assert metrics["folds"] == 2
    assert (output / "oof_predictions.csv").exists()
    blend = json.loads((output / "blend_config.json").read_text(encoding="utf-8"))
    assert blend["selected_on"] == "group_oof"


def test_multilabel_group_folds_keep_groups_together() -> None:
    targets = pd.DataFrame([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]]).to_numpy()
    groups = pd.Series(["a", "a", "b", "b", "c", "c"])
    folds = multilabel_group_folds(targets, groups, n_splits=2, seed=42)
    assert len(set(folds)) == 2
    assert folds[0] == folds[1]
    assert folds[2] == folds[3]
