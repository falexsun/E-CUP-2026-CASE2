#!/usr/bin/env python3
"""Evaluate a prompt-specific embedding as a leakage-safe nearest-neighbour signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

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


def nearest_difference(
    reference_x: np.ndarray,
    reference_y: np.ndarray,
    score_x: np.ndarray,
) -> np.ndarray:
    reference = torch.from_numpy(reference_x).to("cuda", dtype=torch.float32)
    score = torch.from_numpy(score_x).to("cuda", dtype=torch.float32)
    positive = reference[torch.from_numpy(reference_y == 1).to("cuda")]
    negative = reference[torch.from_numpy(reference_y == 0).to("cuda")]
    output = torch.empty(len(score_x), device="cpu", dtype=torch.float32)
    for start in range(0, len(score_x), 1024):
        end = min(start + 1024, len(score_x))
        batch = score[start:end]
        positive_max = (batch @ positive.T).amax(dim=1)
        negative_max = (batch @ negative.T).amax(dim=1)
        output[start:end] = (positive_max - negative_max).cpu()
    return output.numpy()


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

    old_features = np.load(args.v19_features)
    artifact = joblib.load(args.v19_artifact)
    override = artifact["knn_overrides"][CATEGORY]
    v19_oof = v19_probability(
        train["blend_probability"].to_numpy(dtype="float32"),
        old_features["main_diff1_oof"],
        override,
    )
    v19_valid = v19_probability(
        valid["blend_probability"].to_numpy(dtype="float32"),
        old_features["main_diff1_valid"],
        override,
    )

    seeds = (41, 43, 47, 51, 59)
    prompt_oof = np.zeros(len(train), dtype="float32")
    for seed in seeds:
        seed_oof = np.zeros(len(train), dtype="float32")
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fit_indices, score_indices in splitter.split(
            train_x, train_y, groups=train["group"].astype(str)
        ):
            seed_oof[score_indices] = nearest_difference(
                train_x[fit_indices], train_y[fit_indices], train_x[score_indices]
            )
        prompt_oof += seed_oof / len(seeds)
    prompt_valid = nearest_difference(train_x, train_y, valid_x)

    candidates: list[dict] = []
    for center in np.quantile(prompt_oof, np.arange(0.75, 0.991, 0.01)):
        for scale in (0.005, 0.01, 0.02, 0.04):
            for alpha in np.arange(0.05, 0.51, 0.05):
                margin = (1 - alpha) * logit(v19_oof) + alpha * (prompt_oof - center) / scale
                blend = sigmoid(margin)
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

    def combine(base: np.ndarray, prompt: np.ndarray) -> np.ndarray:
        return sigmoid(
            (1 - selected["alpha"]) * logit(base)
            + selected["alpha"] * (prompt - selected["center"]) / selected["scale"]
        )

    valid_blend = combine(v19_valid, prompt_valid)
    inherited_threshold = float(override["blend_threshold"])
    selected.update(
        {
            "prompt_oof_ap": float(average_precision_score(train_y, prompt_oof)),
            "prompt_valid_ap": float(average_precision_score(valid_y, prompt_valid)),
            "v19_valid_f1": float(f1_score(valid_y, v19_valid >= inherited_threshold)),
            "valid_f1": float(f1_score(valid_y, valid_blend >= selected["threshold"])),
            "v19_valid_ap": float(average_precision_score(valid_y, v19_valid)),
            "valid_ap": float(average_precision_score(valid_y, valid_blend)),
        }
    )

    category_mask = data["category"].eq(CATEGORY).to_numpy()
    outer_x = full_x[category_mask]
    outer_y = data.loc[category_mask, "label"].to_numpy(dtype="int8")
    outer = np.load(args.outer)
    np.testing.assert_array_equal(outer["y"], outer_y)
    prompt_outer = np.zeros(len(outer_y), dtype="float32")
    for fold_index in range(5):
        fit_mask = outer["fold"] != fold_index
        score_mask = ~fit_mask
        prompt_outer[score_mask] = nearest_difference(
            outer_x[fit_mask], outer_y[fit_mask], outer_x[score_mask]
        )
    outer_blend = combine(outer["p"], prompt_outer)
    selected.update(
        {
            "prompt_outer_ap": float(average_precision_score(outer_y, prompt_outer)),
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
        prompt_oof=prompt_oof,
        prompt_valid=prompt_valid,
        prompt_outer=prompt_outer,
        valid_blend=valid_blend,
        outer_blend=outer_blend,
    )
    Path(str(output) + ".json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
