import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ozon_quality.official_multimodal import (
    _cosine_reference_features,
    attach_reference_knn,
    combine_embedding_caches,
    load_embedding_cache,
)


def test_cosine_reference_features_return_nearest_label() -> None:
    pytest.importorskip("torch")
    references = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype="float32")
    labels = np.asarray([1, 0, 0], dtype="int8")
    queries = np.asarray([[1, 0], [-1, 0]], dtype="float32")

    difference, similarity, nearest_label = _cosine_reference_features(
        queries, references, labels, batch_size=1
    )

    np.testing.assert_allclose(difference, [1, -2])
    np.testing.assert_allclose(similarity, [1, 1])
    np.testing.assert_array_equal(nearest_label, [1, 0])


def test_combine_embedding_caches_preserves_alignment(tmp_path) -> None:
    ids = ["a", "b"]
    sources = []
    for index, values in enumerate((np.asarray([[1, 2], [3, 4]]), np.asarray([[5], [6]]))):
        source = tmp_path / f"source_{index}"
        source.mkdir()
        pd.DataFrame({"id": ids}).to_csv(source / "ids.csv", index=False)
        np.save(source / "embeddings.npy", values.astype("float16"))
        (source / "state.json").write_text(
            json.dumps({"complete": True, "next_index": 2, "rows": 2})
        )
        sources.append(str(source))
    output = tmp_path / "combined"
    state = combine_embedding_caches(sources, str(output))
    result = load_embedding_cache(str(output), ids)
    assert state["component_dimensions"] == [2, 1]
    np.testing.assert_array_equal(result, [[1, 2, 5], [3, 4, 6]])


def test_attach_reference_knn_validates_and_persists_bank(tmp_path) -> None:
    source = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "name": ["a", "b", "c"],
            "description": ["a", "b", "c"],
            "category": ["Легковоспламеняющиеся"] * 3,
            "label": [0, 1, 0],
            "entity_group": ["g1", "g2", "g3"],
        }
    ).to_csv(source, index=False)
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "columns": {
                    "id": "id",
                    "title": "name",
                    "description": "description",
                    "category": "category",
                    "label": "label",
                    "group": "entity_group",
                },
                "text_fields": ["title", "description", "category"],
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({"id": ["a", "b", "c"]}).to_csv(cache / "ids.csv", index=False)
    np.save(cache / "embeddings.npy", np.asarray([[3, 0], [0, 2], [-4, 0]], dtype="float16"))
    (cache / "state.json").write_text(
        json.dumps({"complete": True, "next_index": 3, "rows": 3}), encoding="utf-8"
    )
    base_artifact = tmp_path / "base.joblib"
    joblib.dump({"artifact_version": 1}, base_artifact)
    config = tmp_path / "knn.json"
    config.write_text(
        json.dumps(
            {
                "category": "Легковоспламеняющиеся",
                "knn_threshold": 0.0,
                "base_threshold": 0.5,
                "margin_scale": 0.01,
                "knn_alpha": 0.3,
                "blend_threshold": 0.25,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "with_knn.joblib"
    summary = attach_reference_knn(
        str(source),
        str(output),
        schema=str(schema),
        embedding_cache=str(cache),
        base_artifact=str(base_artifact),
        config_path=str(config),
    )
    artifact = joblib.load(output)
    override = artifact["knn_overrides"]["Легковоспламеняющиеся"]
    assert summary["rows"] == 3
    assert override["reference_labels"].tolist() == [0, 1, 0]
    np.testing.assert_allclose(
        np.linalg.norm(override["reference_embeddings"].astype("float32"), axis=1), 1
    )
