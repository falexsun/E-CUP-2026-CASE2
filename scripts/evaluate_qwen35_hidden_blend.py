#!/usr/bin/env python3
"""Select Qwen3.5 frozen-hidden blend on train OOF and evaluate once on valid."""

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
from sklearn.preprocessing import normalize

CATEGORY = "Легковоспламеняющиеся"


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-5, 1 - 1e-5)
    return np.log(values / (1 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def best_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    score = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(score.argmax())
    return float(score[index]), float(thresholds[min(index, len(thresholds) - 1)])


def v19_probability(base: np.ndarray, difference: np.ndarray, override: dict) -> np.ndarray:
    alpha = float(override["knn_alpha"])
    margin = (1 - alpha) * (
        logit(base) - logit(np.asarray(float(override["base_threshold"])))
    ) + alpha * (difference - float(override["knn_threshold"])) / float(override["margin_scale"])
    return sigmoid(margin).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--hidden-cache", required=True)
    parser.add_argument("--v8-dir", required=True)
    parser.add_argument("--v19-features", required=True)
    parser.add_argument("--v19-artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data, dtype={"id": str})
    hidden_ids = pd.read_csv(Path(args.hidden_cache) / "ids.csv", dtype={"id": str})
    if hidden_ids["id"].tolist() != data["id"].tolist():
        raise ValueError("Hidden cache and data IDs are not aligned")
    hidden = normalize(
        np.load(Path(args.hidden_cache) / "embeddings.npy", mmap_mode="r").astype("float32")
    )
    id_to_hidden = {value: index for index, value in enumerate(data["id"])}

    v8_dir = Path(args.v8_dir)
    train = pd.read_csv(v8_dir / "oof_predictions.csv", dtype={"id": str})
    valid = pd.read_csv(v8_dir / "validation_predictions.csv", dtype={"id": str})
    train = train[train["category"].eq(CATEGORY)].reset_index(drop=True)
    valid = valid[valid["category"].eq(CATEGORY)].reset_index(drop=True)
    train_x = hidden[[id_to_hidden[value] for value in train["id"]]]
    valid_x = hidden[[id_to_hidden[value] for value in valid["id"]]]
    y_train = train["label"].to_numpy(dtype="int8")
    y_valid = valid["label"].to_numpy(dtype="int8")

    features = np.load(args.v19_features)
    np.testing.assert_array_equal(features["y"], y_train)
    np.testing.assert_array_equal(features["yv"], y_valid)
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

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    c_values = (0.1, 1.0, 10.0, 100.0)
    alpha_values = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
    candidates = []
    probabilities: dict[float, tuple[np.ndarray, np.ndarray, LogisticRegression]] = {}
    for c_value in c_values:
        oof = np.zeros(len(train), dtype="float32")
        for fit_indices, score_indices in splitter.split(
            train_x, y_train, groups=train["group"].astype(str)
        ):
            model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000).fit(
                train_x[fit_indices], y_train[fit_indices]
            )
            oof[score_indices] = model.predict_proba(train_x[score_indices])[:, 1]
        final_model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000).fit(
            train_x, y_train
        )
        qwen_valid = final_model.predict_proba(valid_x)[:, 1].astype("float32")
        probabilities[c_value] = (oof, qwen_valid, final_model)
        for alpha in alpha_values:
            blended = sigmoid((1 - alpha) * logit(v19_oof) + alpha * logit(oof))
            score, threshold = best_threshold(y_train, blended)
            candidates.append(
                {
                    "C": c_value,
                    "alpha_qwen35": alpha,
                    "oof_f1": score,
                    "threshold": threshold,
                }
            )

    selected = max(candidates, key=lambda row: (row["oof_f1"], -row["alpha_qwen35"]))
    qwen_oof, qwen_valid, final_model = probabilities[selected["C"]]
    valid_blend = sigmoid(
        (1 - selected["alpha_qwen35"]) * logit(v19_valid)
        + selected["alpha_qwen35"] * logit(qwen_valid)
    )
    selected["valid_f1"] = float(f1_score(y_valid, valid_blend >= selected["threshold"]))
    selected["valid_ap"] = float(average_precision_score(y_valid, valid_blend))
    selected["qwen35_oof_ap"] = float(average_precision_score(y_train, qwen_oof))
    selected["qwen35_valid_ap"] = float(average_precision_score(y_valid, qwen_valid))
    selected["v19_valid_f1_frozen"] = float(
        f1_score(y_valid, v19_valid >= float(override["blend_threshold"]))
    )
    selected["v19_valid_ap"] = float(average_precision_score(y_valid, v19_valid))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "selection": selected,
            "model": final_model,
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
