import numpy as np
import pandas as pd
import pytest

from ozon_quality.data import (
    adapt_frame,
    infer_columns,
    parse_images,
    parse_labels,
    schema_suggestion,
)


def test_schema_inference_and_suggestion() -> None:
    mapping, ambiguous = infer_columns(["item_id", "name", "description", "image_urls", "target"])
    assert not ambiguous
    assert mapping == {
        "id": "item_id",
        "title": "name",
        "description": "description",
        "images": "image_urls",
        "label": "target",
    }
    assert schema_suggestion(mapping.values())["metadata"]["unresolved"] == []


def test_parse_images_accepts_json_and_separator() -> None:
    assert parse_images('["a.jpg", "b.jpg"]') == ["a.jpg", "b.jpg"]
    assert parse_images("a.jpg|b.jpg") == ["a.jpg", "b.jpg"]
    assert parse_images(None) == []
    assert parse_images(np.asarray(["a.jpg", "b.jpg"])) == ["a.jpg", "b.jpg"]
    assert parse_labels('["legal", "image_issue"]') == ["legal", "image_issue"]
    assert parse_labels("legal|image_issue") == ["legal", "image_issue"]


def test_adapter_composes_text_and_resolves_relative_images(tmp_path) -> None:
    raw = pd.DataFrame(
        {"sku": [1], "name": ["Чай"], "desc": ["100 г"], "pics": ["a.jpg|https://x/y.jpg"], "answer": ["ok"]}
    )
    config = {
        "columns": {"id": "sku", "title": "name", "description": "desc", "images": "pics", "label": "answer"},
        "text_fields": ["title", "description"],
        "image_separator": "|",
    }
    frame = adapt_frame(raw, config, source_dir=tmp_path, require_label=True)
    assert frame.loc[0, "text"] == "[TITLE] Чай [DESCRIPTION] 100 г"
    assert frame.loc[0, "images"] == [str((tmp_path / "a.jpg").resolve()), "https://x/y.jpg"]
    assert frame.loc[0, "labels"] == ["ok"]


def test_adapter_rejects_label_in_text() -> None:
    raw = pd.DataFrame({"name": ["x"], "target": [1]})
    config = {"columns": {"title": "name", "label": "target"}, "text_fields": ["title", "label"]}
    with pytest.raises(ValueError, match="leakage"):
        adapt_frame(raw, config, source_dir=".", require_label=True)


def test_adapter_discovers_official_id_image_directories(tmp_path) -> None:
    image_dir = tmp_path / "images" / "42"
    image_dir.mkdir(parents=True)
    (image_dir / "1.jpg").write_bytes(b"image")
    (image_dir / "ignore.txt").write_text("x", encoding="utf-8")
    raw = pd.DataFrame({"id": [42], "name": ["Товар"], "target": [1]})
    config = {
        "columns": {"id": "id", "title": "name", "label": "target"},
        "text_fields": ["title"],
        "images_directory": "images",
    }
    frame = adapt_frame(raw, config, source_dir=tmp_path, require_label=True)
    assert frame.loc[0, "images"] == [str((image_dir / "1.jpg").resolve())]
