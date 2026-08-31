from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATEGORY = "Легковоспламеняющиеся"


def test_v45_builders_apply_frozen_config_and_remove_risky_overrides(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "category": [CATEGORY] * 4,
            "label": [0, 1, 0, 1],
        }
    )
    data_path = tmp_path / "train.csv"
    frame.to_csv(data_path, index=False)

    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame({"id": frame["id"]}).to_csv(cache / "ids.csv", index=False)
    np.save(
        cache / "embeddings.npy",
        np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype="float32"),
    )

    base_path = tmp_path / "v19.joblib"
    joblib.dump(
        {
            "knn_overrides": {
                CATEGORY: {
                    "nearest_override_threshold": 0.99,
                    "reference_embeddings": np.eye(2, dtype="float32"),
                    "reference_labels": np.asarray([0, 1], dtype="int8"),
                }
            },
            "exact_title_overrides": {CATEGORY: {"known title": 1}},
        },
        base_path,
    )

    intermediate = tmp_path / "linear.joblib"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_standardized_head_artifact.py"),
            "--base-artifact",
            str(base_path),
            "--data",
            str(data_path),
            "--embedding-cache",
            str(cache),
            "--config",
            str(ROOT / "configs/v45.json"),
            "--output",
            str(intermediate),
        ],
        check=True,
    )

    output = tmp_path / "v45.joblib"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_regex_only_artifact.py"),
            "--base-artifact",
            str(intermediate),
            "--config",
            str(ROOT / "configs/v45.json"),
            "--output",
            str(output),
        ],
        check=True,
    )

    artifact = joblib.load(output)
    head = artifact["linear_head_overrides"][CATEGORY]
    assert head["C"] == 0.03
    assert head["alpha_head"] == 0.25
    assert head["blend_threshold"] == 0.26
    assert "exact_title_overrides" not in artifact
    assert "nearest_override_threshold" not in artifact["knn_overrides"][CATEGORY]
    assert [rule["name"] for rule in artifact["regex_title_overrides"][CATEGORY]] == [
        "standalone_coal_or_fuel_briquette",
        "disposable_grill_with_coal",
        "gas_for_portable_stove_or_burner",
        "smoke_bomb",
        "barbecue_fuel_briquette",
    ]
