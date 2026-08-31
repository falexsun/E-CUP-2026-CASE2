#!/usr/bin/env python3
"""Evaluate an RBF head without using either holdout for model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

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


def fit_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    score_x: np.ndarray,
    *,
    c_value: float,
    gamma: float,
) -> tuple[np.ndarray, SVC]:
    model = SVC(
        C=c_value,
        gamma=gamma,
        class_weight="balanced",
        cache_size=12000,
        probability=False,
    ).fit(train_x, train_y)
    return model.decision_function(score_x).astype("float32"), model


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
    # Reference embeddings are already L2 normalized. Float32 avoids numerical
    # differences between research and runtime inference.
    full_x = np.load(cache / "embeddings.npy").astype("float32")
    full_x /= np.maximum(np.linalg.norm(full_x, axis=1, keepdims=True), 1e-8)
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

    c_values = (0.3, 1.0, 3.0, 10.0, 30.0)
    gamma_values = (0.3, 1.0, 3.0, 10.0)
    alpha_values = tuple(np.arange(0.05, 0.41, 0.05))
    splitter = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=41).split(
            train_x, train_y, groups=train["group"].astype(str)
        )
    )
    candidates: list[dict] = []
    scores: dict[tuple[float, float], np.ndarray] = {}
    for c_value in c_values:
        for gamma in gamma_values:
            oof = np.zeros(len(train), dtype="float32")
            for fit_indices, score_indices in splitter:
                oof[score_indices], _ = fit_score(
                    train_x[fit_indices],
                    train_y[fit_indices],
                    train_x[score_indices],
                    c_value=c_value,
                    gamma=gamma,
                )
            # Map an uncalibrated SVC margin to a stable probability-like value.
            head = sigmoid(oof)
            scores[(c_value, gamma)] = head
            for alpha in alpha_values:
                blend = sigmoid((1 - alpha) * logit(v19_oof) + alpha * logit(head))
                score, threshold = best_threshold(train_y, blend)
                candidates.append(
                    {
                        "C": c_value,
                        "gamma": gamma,
                        "alpha_head": float(alpha),
                        "oof_f1": score,
                        "threshold": threshold,
                        "head_oof_ap": float(average_precision_score(train_y, head)),
                    }
                )

    selected = max(candidates, key=lambda row: (row["oof_f1"], -row["alpha_head"]))
    valid_margin, model = fit_score(
        train_x,
        train_y,
        valid_x,
        c_value=selected["C"],
        gamma=selected["gamma"],
    )
    valid_head = sigmoid(valid_margin)
    valid_blend = sigmoid(
        (1 - selected["alpha_head"]) * logit(v19_valid)
        + selected["alpha_head"] * logit(valid_head)
    )
    inherited_threshold = float(override["blend_threshold"])
    selected.update(
        {
            "v19_valid_f1": float(f1_score(valid_y, v19_valid >= inherited_threshold)),
            "valid_f1_selected_threshold": float(
                f1_score(valid_y, valid_blend >= selected["threshold"])
            ),
            "valid_f1_inherited_threshold": float(
                f1_score(valid_y, valid_blend >= inherited_threshold)
            ),
            "v19_valid_ap": float(average_precision_score(valid_y, v19_valid)),
            "valid_ap": float(average_precision_score(valid_y, valid_blend)),
        }
    )

    category_mask = data["category"].eq(CATEGORY).to_numpy()
    outer_x = full_x[category_mask]
    outer_y = data.loc[category_mask, "label"].to_numpy(dtype="int8")
    outer = np.load(args.outer)
    np.testing.assert_array_equal(outer["y"], outer_y)
    outer_head = np.zeros(len(outer_y), dtype="float32")
    for fold_index in range(5):
        fit_mask = outer["fold"] != fold_index
        score_mask = ~fit_mask
        margin, _ = fit_score(
            outer_x[fit_mask],
            outer_y[fit_mask],
            outer_x[score_mask],
            c_value=selected["C"],
            gamma=selected["gamma"],
        )
        outer_head[score_mask] = sigmoid(margin)
    outer_blend = sigmoid(
        (1 - selected["alpha_head"]) * logit(outer["p"])
        + selected["alpha_head"] * logit(outer_head)
    )
    selected.update(
        {
            "v19_outer_f1": float(f1_score(outer_y, outer["p"] >= inherited_threshold)),
            "outer_f1_selected_threshold": float(
                f1_score(outer_y, outer_blend >= selected["threshold"])
            ),
            "outer_f1_inherited_threshold": float(
                f1_score(outer_y, outer_blend >= inherited_threshold)
            ),
            "v19_outer_ap": float(average_precision_score(outer_y, outer["p"])),
            "outer_ap": float(average_precision_score(outer_y, outer_blend)),
        }
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"selection": selected, "model": model, "candidates": candidates},
        output,
        compress=3,
    )
    Path(str(output) + ".json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
