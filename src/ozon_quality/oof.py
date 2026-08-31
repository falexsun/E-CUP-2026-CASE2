"""Leakage-aware out-of-fold evaluation for ensemble selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ozon_quality.data import load_data
from ozon_quality.features import build_features
from ozon_quality.lexical import build_lexical_classifier
from ozon_quality.pipeline import (
    _classifier,
    _encode_targets,
    _evaluate_probabilities,
    _feature_slices,
    _search_blend,
)


def multilabel_group_folds(
    targets: np.ndarray,
    groups: pd.Series,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Greedily balance multilabel counts while keeping complete groups together."""
    unique_groups, inverse = np.unique(groups.astype(str), return_inverse=True)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} unique groups, got {len(unique_groups)}")
    group_counts = np.zeros((len(unique_groups), targets.shape[1]), dtype="float64")
    group_sizes = np.bincount(inverse).astype("float64")
    for row, group_index in enumerate(inverse):
        group_counts[group_index] += targets[row]
    total_counts = group_counts.sum(axis=0)
    rarity = (group_counts / np.maximum(total_counts, 1)).sum(axis=1)
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(unique_groups)) * 1e-9
    order = np.lexsort((jitter, -group_sizes, -rarity))
    fold_counts = np.zeros((n_splits, targets.shape[1]), dtype="float64")
    fold_sizes = np.zeros(n_splits, dtype="float64")
    assignment = np.full(len(unique_groups), -1, dtype="int64")
    target_counts = total_counts / n_splits
    target_size = len(targets) / n_splits
    for position, group_index in enumerate(order):
        if position < n_splits:
            fold = position
        else:
            scores = []
            for candidate in range(n_splits):
                label_error = np.mean(
                    np.square(
                        (fold_counts[candidate] + group_counts[group_index] - target_counts)
                        / np.maximum(target_counts, 1)
                    )
                )
                size_error = (
                    (fold_sizes[candidate] + group_sizes[group_index] - target_size)
                    / max(target_size, 1)
                ) ** 2
                scores.append(label_error + 0.25 * size_error)
            fold = int(np.argmin(scores))
        assignment[group_index] = fold
        fold_counts[fold] += group_counts[group_index]
        fold_sizes[fold] += group_sizes[group_index]
    return assignment[inverse]


def run_group_oof(
    input_path: str,
    output: str,
    *,
    schema: str | None,
    text_model: str,
    vision_model: str,
    text_revision: str | None,
    vision_revision: str | None,
    device: str,
    batch_size: int,
    seed: int,
    folds: int,
) -> dict[str, Any]:
    frame, config = load_data(input_path, schema, require_label=True)
    if "group" not in frame or frame["group"].eq("").any():
        raise ValueError("OOF requires a complete configured group/entity column")
    y, _, classes, task_type = _encode_targets(
        frame["labels"], frame["labels"], str(config.get("task_type", "auto"))
    )
    if frame["group"].nunique() < folds:
        raise ValueError(f"Need at least {folds} unique groups")
    if task_type == "single_label":
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_ids = np.full(len(frame), -1, dtype="int64")
        scalar = np.asarray(y)
        for fold, (_, valid_index) in enumerate(
            splitter.split(frame, scalar, groups=frame["group"])
        ):
            fold_ids[valid_index] = fold
    else:
        fold_ids = multilabel_group_folds(y, frame["group"], folds, seed)
    if (fold_ids < 0).any() or len(np.unique(fold_ids)) != folds:
        raise RuntimeError("Failed to assign every row to an OOF fold")
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    features, feature_meta = build_features(
        frame,
        text_model=text_model,
        vision_model=vision_model,
        text_revision=text_revision,
        vision_revision=vision_revision,
        device=device,
        batch_size=batch_size,
        cache_dir=output_dir / "cache",
    )
    fusion_slice = _feature_slices(feature_meta)["fusion"]
    dense_probability = np.zeros((len(frame), len(classes)), dtype="float32")
    lexical_probability = np.zeros_like(dense_probability)
    for fold in range(folds):
        valid_mask = fold_ids == fold
        train_mask = ~valid_mask
        if set(frame.loc[train_mask, "group"]) & set(frame.loc[valid_mask, "group"]):
            raise AssertionError("Group leakage detected inside OOF")
        dense = _classifier(seed + fold, task_type)
        dense.fit(features[train_mask, fusion_slice], y[train_mask])
        dense_probability[valid_mask] = dense.predict_proba(
            features[valid_mask, fusion_slice]
        )
        lexical = build_lexical_classifier(seed + fold, task_type)
        lexical.fit(frame.loc[train_mask, "text"], y[train_mask])
        lexical_probability[valid_mask] = lexical.predict_proba(
            frame.loc[valid_mask, "text"]
        )
    dense_metrics, _, _ = _evaluate_probabilities(dense_probability, y, task_type)
    lexical_metrics, _, _ = _evaluate_probabilities(lexical_probability, y, task_type)
    weight, ensemble_metrics, prediction, probability, threshold = _search_blend(
        dense_probability, lexical_probability, y, task_type
    )
    metrics = {
        "task_type": task_type,
        "classes": classes,
        "folds": folds,
        "rows": len(frame),
        "groups": int(frame["group"].nunique()),
        "dense": dense_metrics,
        "lexical": lexical_metrics,
        "ensemble": ensemble_metrics,
    }
    blend_config = {
        "task_type": task_type,
        "classes": classes,
        "dense_weight": weight,
        "lexical_weight": 1 - weight,
        "threshold": threshold,
        "selected_on": "group_oof",
        "folds": folds,
        "seed": seed,
        "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
    }
    predictions = pd.DataFrame({"id": frame["id"], "group": frame["group"], "fold": fold_ids})
    if task_type == "single_label":
        predictions["label"] = frame["labels"].map(lambda value: value[0])
        predictions["prediction"] = np.asarray(classes)[prediction]
    else:
        predictions["labels"] = frame["labels"].map(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
        predictions["prediction"] = [
            json.dumps(
                [label for label, active in zip(classes, row, strict=True) if active],
                ensure_ascii=False,
            )
            for row in prediction
        ]
    for index, label in enumerate(classes):
        predictions[f"dense_{label}"] = dense_probability[:, index]
        predictions[f"lexical_{label}"] = lexical_probability[:, index]
        predictions[f"ensemble_{label}"] = probability[:, index]
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "blend_config.json").write_text(
        json.dumps(blend_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
