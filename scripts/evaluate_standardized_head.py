#!/usr/bin/env python3
"""Select a standardized linear head on repeated train OOF and verify on two holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

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


def fit_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    score_x: np.ndarray,
    c_value: float,
) -> tuple[np.ndarray, StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000).fit(
        scaler.transform(train_x), train_y
    )
    return (
        model.predict_proba(scaler.transform(score_x))[:, 1].astype("float32"),
        scaler,
        model,
    )


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
    id_to_index = {value: index for index, value in enumerate(data["id"])}

    v8 = Path(args.v8_dir)
    train = pd.read_csv(v8 / "oof_predictions.csv", dtype={"id": str})
    valid = pd.read_csv(v8 / "validation_predictions.csv", dtype={"id": str})
    train = train[train["category"].eq(CATEGORY)].reset_index(drop=True)
    valid = valid[valid["category"].eq(CATEGORY)].reset_index(drop=True)
    train_x = full_x[[id_to_index[value] for value in train["id"]]]
    valid_x = full_x[[id_to_index[value] for value in valid["id"]]]
    train_y = train["label"].to_numpy(dtype="int8")
    valid_y = valid["label"].to_numpy(dtype="int8")

    features = np.load(args.v19_features)
    np.testing.assert_array_equal(features["y"], train_y)
    np.testing.assert_array_equal(features["yv"], valid_y)
    artifact = joblib.load(args.v19_artifact)
    override = artifact["knn_overrides"][CATEGORY]
    v19_oof = v19_probability(
        train["blend_probability"].to_numpy(dtype="float32"),
        features["main_diff1_oof"],
        override,
    )
    v19_valid = v19_probability(
        valid["blend_probability"].to_numpy(dtype="float32"),
        features["main_diff1_valid"],
        override,
    )

    seeds = (41, 43, 47, 51, 59)
    c_values = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03)
    alpha_values = tuple(np.arange(0.05, 0.41, 0.05))
    candidates = []
    heads = {}
    for c_value in c_values:
        repeated_oof = np.zeros(len(train), dtype="float32")
        for seed in seeds:
            seed_oof = np.zeros(len(train), dtype="float32")
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for fit_indices, score_indices in splitter.split(
                train_x, train_y, groups=train["group"].astype(str)
            ):
                probability, _, _ = fit_head(
                    train_x[fit_indices],
                    train_y[fit_indices],
                    train_x[score_indices],
                    c_value,
                )
                seed_oof[score_indices] = probability
            repeated_oof += seed_oof / len(seeds)
        valid_probability, scaler, model = fit_head(train_x, train_y, valid_x, c_value)
        heads[c_value] = (repeated_oof, valid_probability, scaler, model)
        for alpha in alpha_values:
            blend = sigmoid((1 - alpha) * logit(v19_oof) + alpha * logit(repeated_oof))
            score, threshold = best_threshold(train_y, blend)
            candidates.append(
                {
                    "C": c_value,
                    "alpha_head": float(alpha),
                    "oof_f1": score,
                    "threshold": threshold,
                }
            )

    selected = max(candidates, key=lambda row: (row["oof_f1"], -row["alpha_head"]))
    head_oof, head_valid, scaler, model = heads[selected["C"]]
    valid_blend = sigmoid(
        (1 - selected["alpha_head"]) * logit(v19_valid) + selected["alpha_head"] * logit(head_valid)
    )
    selected["head_oof_ap"] = float(average_precision_score(train_y, head_oof))
    selected["head_valid_ap"] = float(average_precision_score(valid_y, head_valid))
    selected["v19_valid_f1"] = float(
        f1_score(valid_y, v19_valid >= float(override["blend_threshold"]))
    )
    selected["valid_f1"] = float(f1_score(valid_y, valid_blend >= selected["threshold"]))
    selected["v19_valid_ap"] = float(average_precision_score(valid_y, v19_valid))
    selected["valid_ap"] = float(average_precision_score(valid_y, valid_blend))

    # Independent outer seed, with all hyperparameters frozen above.
    category_mask = data["category"].eq(CATEGORY).to_numpy()
    outer_x = full_x[category_mask]
    outer_y = data.loc[category_mask, "label"].to_numpy(dtype="int8")
    outer = np.load(args.outer)
    outer_fold = outer["fold"]
    outer_v19 = outer["p"]
    outer_head = np.zeros(len(outer_y), dtype="float32")
    for fold_index in range(5):
        fit_mask = outer_fold != fold_index
        score_mask = ~fit_mask
        probability, _, _ = fit_head(
            outer_x[fit_mask],
            outer_y[fit_mask],
            outer_x[score_mask],
            selected["C"],
        )
        outer_head[score_mask] = probability
    outer_blend = sigmoid(
        (1 - selected["alpha_head"]) * logit(outer_v19) + selected["alpha_head"] * logit(outer_head)
    )
    selected["v19_outer_f1"] = float(f1_score(outer_y, outer_v19 >= 0.26))
    selected["outer_f1"] = float(f1_score(outer_y, outer_blend >= selected["threshold"]))
    selected["v19_outer_ap"] = float(average_precision_score(outer_y, outer_v19))
    selected["outer_ap"] = float(average_precision_score(outer_y, outer_blend))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "selection": selected,
            "scaler": scaler,
            "model": model,
            "candidates": candidates,
        },
        output,
        compress=3,
    )
    Path(str(output) + ".json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
