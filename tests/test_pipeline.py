import json

import pandas as pd
from PIL import Image

from ozon_quality.pipeline import predict, train


def test_debug_pipeline_end_to_end(tmp_path) -> None:
    Image.new("RGB", (8, 8), "white").save(tmp_path / "white.png")
    Image.new("RGB", (8, 8), "black").save(tmp_path / "black.png")
    schema = {
        "schema_version": 1,
        "columns": {"id": "id", "title": "name", "images": "photo", "label": "target"},
        "text_fields": ["title"],
        "image_separator": "|",
        "label_values": {},
    }
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "id": range(8),
            "name": ["разрешен товар", "разрешен нормальный", "хороший товар", "обычный товар", "запрещен товар", "опасный товар", "плохой товар", "нарушение правил"],
            "photo": ["white.png"] * 4 + ["black.png"] * 4,
            "target": ["ok"] * 4 + ["bad"] * 4,
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {"id": [10, 11, 12, 13], "name": ["хороший", "нормальный", "опасный", "нарушение"], "photo": ["white.png", "white.png", "black.png", "black.png"], "target": ["ok", "ok", "bad", "bad"]}
    ).to_csv(valid_path, index=False)
    pd.read_csv(valid_path).drop(columns="target").to_csv(test_path, index=False)
    artifact_dir = tmp_path / "artifact"
    blend_path = tmp_path / "blend.json"
    blend_path.write_text(
        json.dumps(
            {
                "task_type": "single_label",
                "classes": ["bad", "ok"],
                "dense_weight": 0.5,
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    metrics = train(
        str(train_path),
        str(valid_path),
        str(artifact_dir),
        schema=str(schema_path),
        text_model="debug-hash",
        vision_model="debug-pixels",
        device="cpu",
        batch_size=2,
        seed=42,
        blend_config=str(blend_path),
        refit_all=True,
    )
    assert metrics["macro_f1"] >= 0.5
    assert metrics["refit_all"] is True
    assert (artifact_dir / "uncertain_100.csv").exists()
    baseline_metrics = json.loads(
        (artifact_dir / "baseline_metrics.json").read_text(encoding="utf-8")
    )
    assert {"lexical", "fusion", "ensemble"} <= set(baseline_metrics)
    result = predict(
        str(test_path),
        str(artifact_dir / "model.joblib"),
        str(tmp_path / "submission.csv"),
        schema=None,
        device="cpu",
        batch_size=2,
        cache_dir=None,
        prediction_column="target",
    )
    assert result.columns.tolist() == ["id", "target", "probability_bad", "probability_ok"]
    assert len(result) == 4


def test_multilabel_pipeline(tmp_path) -> None:
    schema = {
        "schema_version": 1,
        "columns": {"id": "id", "title": "name", "label": "target"},
        "text_fields": ["title"],
        "task_type": "multi_label",
        "label_separator": "|",
    }
    schema_path = tmp_path / "schema_multi.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    train_path = tmp_path / "train_multi.csv"
    valid_path = tmp_path / "valid_multi.csv"
    test_path = tmp_path / "test_multi.csv"
    pd.DataFrame(
        {
            "id": range(8),
            "name": ["a", "a b", "b", "b c", "c", "a c", "a b c", "b again"],
            "target": ["A", "A|B", "B", "B|C", "C", "A|C", "A|B|C", "B"],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {"id": [10, 11, 12], "name": ["a", "b", "a c"], "target": ["A", "B", "A|C"]}
    ).to_csv(valid_path, index=False)
    pd.read_csv(valid_path).drop(columns="target").to_csv(test_path, index=False)
    artifact_dir = tmp_path / "multi_artifact"
    metrics = train(
        str(train_path),
        str(valid_path),
        str(artifact_dir),
        schema=str(schema_path),
        text_model="debug-hash",
        vision_model="debug-pixels",
        device="cpu",
        batch_size=2,
        seed=42,
    )
    assert metrics["task_type"] == "multi_label"
    assert metrics["classes"] == ["A", "B", "C"]
    result = predict(
        str(test_path),
        str(artifact_dir / "model.joblib"),
        str(tmp_path / "multi_submission.csv"),
        schema=None,
        device="cpu",
        batch_size=2,
        cache_dir=None,
        prediction_column="target",
    )
    assert result.columns.tolist() == [
        "id",
        "target",
        "probability_A",
        "probability_B",
        "probability_C",
    ]
    assert all(value.startswith("[") for value in result["target"])
