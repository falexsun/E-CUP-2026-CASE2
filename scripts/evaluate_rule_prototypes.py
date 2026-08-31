#!/usr/bin/env python3
"""Evaluate fixed semantic rule prototypes as a calibrated addition to v19."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

CATEGORY = "Легковоспламеняющиеся"


def logit(values: np.ndarray | float) -> np.ndarray:
    values = np.clip(values, 1e-5, 1 - 1e-5)
    return np.log(values / (1 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def best_threshold(y_true: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, values)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(f1.argmax())
    return float(f1[index]), float(thresholds[min(index, len(thresholds) - 1)])


def v19_probability(base: np.ndarray, difference: np.ndarray, override: dict) -> np.ndarray:
    alpha = float(override["knn_alpha"])
    margin = (1 - alpha) * (logit(base) - logit(float(override["base_threshold"]))) + alpha * (
        difference - float(override["knn_threshold"])
    ) / float(override["margin_scale"])
    return sigmoid(margin).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--prototype-cache", required=True)
    parser.add_argument("--v8-dir", required=True)
    parser.add_argument("--v19-features", required=True)
    parser.add_argument("--v19-artifact", required=True)
    parser.add_argument("--outer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data, dtype={"id": str})
    cache = Path(args.embedding_cache)
    ids = pd.read_csv(cache / "ids.csv", dtype={"id": str})["id"].tolist()
    if ids != data["id"].tolist():
        raise ValueError("Embedding cache is not aligned")
    full_x = np.load(cache / "embeddings.npy").astype("float32")
    full_x /= np.maximum(np.linalg.norm(full_x, axis=1, keepdims=True), 1e-8)
    prototype_cache = Path(args.prototype_cache)
    prototype_x = np.load(prototype_cache / "embeddings.npy").astype("float32")
    prototype_x /= np.maximum(np.linalg.norm(prototype_x, axis=1, keepdims=True), 1e-8)
    prototype_y = np.load(prototype_cache / "labels.npy")
    similarity = full_x @ prototype_x.T
    prototype_score = similarity[:, prototype_y == 1].max(axis=1) - similarity[
        :, prototype_y == 0
    ].max(axis=1)
    id_to_index = {value: index for index, value in enumerate(data["id"])}

    v8 = Path(args.v8_dir)
    train = pd.read_csv(v8 / "oof_predictions.csv", dtype={"id": str})
    valid = pd.read_csv(v8 / "validation_predictions.csv", dtype={"id": str})
    train = train[train["category"].eq(CATEGORY)].reset_index(drop=True)
    valid = valid[valid["category"].eq(CATEGORY)].reset_index(drop=True)
    train_indices = np.asarray([id_to_index[value] for value in train["id"]])
    valid_indices = np.asarray([id_to_index[value] for value in valid["id"]])
    train_y = train["label"].to_numpy(dtype="int8")
    valid_y = valid["label"].to_numpy(dtype="int8")

    old_features = np.load(args.v19_features)
    artifact = joblib.load(args.v19_artifact)
    override = artifact["knn_overrides"][CATEGORY]
    v19_train = v19_probability(
        train["blend_probability"].to_numpy(dtype="float32"),
        old_features["main_diff1_oof"],
        override,
    )
    v19_valid = v19_probability(
        valid["blend_probability"].to_numpy(dtype="float32"),
        old_features["main_diff1_valid"],
        override,
    )
    train_score = prototype_score[train_indices]
    valid_score = prototype_score[valid_indices]

    candidates = []
    for center in np.quantile(train_score, np.arange(0.5, 0.991, 0.01)):
        for scale in (0.005, 0.01, 0.02, 0.04, 0.08):
            for alpha in np.arange(0.05, 0.41, 0.05):
                blend = sigmoid(
                    (1 - alpha) * logit(v19_train) + alpha * (train_score - center) / scale
                )
                score, threshold = best_threshold(train_y, blend)
                candidates.append(
                    {
                        "center": float(center),
                        "scale": scale,
                        "alpha": float(alpha),
                        "oof_f1": score,
                        "threshold": threshold,
                    }
                )
    selected = max(candidates, key=lambda row: (row["oof_f1"], -row["alpha"]))

    def combine(base: np.ndarray, score: np.ndarray) -> np.ndarray:
        return sigmoid(
            (1 - selected["alpha"]) * logit(base)
            + selected["alpha"] * (score - selected["center"]) / selected["scale"]
        )

    valid_blend = combine(v19_valid, valid_score)
    category_mask = data["category"].eq(CATEGORY).to_numpy()
    outer_y = data.loc[category_mask, "label"].to_numpy(dtype="int8")
    outer_score = prototype_score[category_mask]
    outer = np.load(args.outer)
    outer_blend = combine(outer["p"], outer_score)
    inherited_threshold = float(override["blend_threshold"])
    selected.update(
        {
            "prototype_train_ap": float(average_precision_score(train_y, train_score)),
            "prototype_valid_ap": float(average_precision_score(valid_y, valid_score)),
            "prototype_outer_ap": float(average_precision_score(outer_y, outer_score)),
            "v19_valid_f1": float(f1_score(valid_y, v19_valid >= inherited_threshold)),
            "valid_f1": float(f1_score(valid_y, valid_blend >= selected["threshold"])),
            "v19_valid_ap": float(average_precision_score(valid_y, v19_valid)),
            "valid_ap": float(average_precision_score(valid_y, valid_blend)),
            "v19_outer_f1": float(f1_score(outer_y, outer["p"] >= inherited_threshold)),
            "outer_f1": float(f1_score(outer_y, outer_blend >= selected["threshold"])),
            "v19_outer_ap": float(average_precision_score(outer_y, outer["p"])),
            "outer_ap": float(average_precision_score(outer_y, outer_blend)),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prototype_score=prototype_score,
        valid_blend=valid_blend,
        outer_blend=outer_blend,
    )
    Path(str(output) + ".json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
