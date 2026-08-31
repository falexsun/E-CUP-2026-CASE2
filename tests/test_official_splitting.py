import json

import pandas as pd

from ozon_quality.data import adapt_frame
from ozon_quality.official_splitting import (
    derive_entity_groups,
    official_holdout,
    write_grouped_official_data,
)


def test_derived_groups_connect_duplicate_titles(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Одинаковый товар", "Одинаковый товар!", "Другой товар"],
            "description": ["a", "b", "c"],
            "category": ["БАД"] * 3,
            "label": [1, 0, 1],
        }
    )
    config = {
        "columns": {"id": "id", "title": "name", "description": "description", "category": "category", "label": "label"},
        "text_fields": ["title", "description", "category"],
    }
    frame = adapt_frame(raw, config, source_dir=tmp_path, require_label=True)
    groups = derive_entity_groups(frame, use_first_image=False)
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[0] != groups.iloc[2]


def test_official_split_preserves_duplicate_entities(tmp_path) -> None:
    rows = []
    for index in range(40):
        category = "БАД" if index < 20 else "Легковоспламеняющиеся"
        rows.append({"id": index, "name": f"товар {index // 2}", "description": f"описание {index // 2}", "category": category, "label": index % 2})
    source = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "columns": {"id": "id", "title": "name", "description": "description", "category": "category", "label": "label", "group": "entity_group"},
                "text_fields": ["title", "description", "category"],
            }
        ),
        encoding="utf-8",
    )
    train_path, valid_path = tmp_path / "train.csv", tmp_path / "valid.csv"
    official_holdout(str(source), str(train_path), str(valid_path), schema=str(schema), valid_size=0.25, seed=42, candidates=16)
    train, valid = pd.read_csv(train_path), pd.read_csv(valid_path)
    assert not set(train.entity_group) & set(valid.entity_group)

    grouped_path = tmp_path / "full_grouped.csv"
    summary = write_grouped_official_data(
        str(source), str(grouped_path), schema=str(schema)
    )
    grouped = pd.read_csv(grouped_path)
    assert summary["rows"] == len(rows)
    assert grouped["entity_group"].notna().all()
