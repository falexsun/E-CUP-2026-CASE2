#!/usr/bin/env python3
"""Evaluate centered/PCA-cleaned prototype metrics with repeated group OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize


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


def transform_embeddings(
    train: np.ndarray, valid: np.ndarray, remove_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    centered_train = train - mean
    centered_valid = valid - mean
    components = np.empty((0, train.shape[1]), dtype="float32")
    if remove_components:
        components = (
            PCA(n_components=remove_components, svd_solver="randomized", random_state=42)
            .fit(centered_train)
            .components_.astype("float32")
        )
        centered_train -= (centered_train @ components.T) @ components
        centered_valid -= (centered_valid @ components.T) @ components
    return (
        normalize(centered_train).astype("float32"),
        normalize(centered_valid).astype("float32"),
        mean.astype("float32"),
        components,
    )


def prototype_score(
    query: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    positive = torch.from_numpy(reference[labels == 1]).to("cuda")
    negative = torch.from_numpy(reference[labels == 0]).to("cuda")
    result = []
    for start in range(0, len(query), batch_size):
        batch = torch.from_numpy(query[start : start + batch_size]).to("cuda")
        result.append(
            ((batch @ positive.T).max(1).values - (batch @ negative.T).max(1).values).cpu().numpy()
        )
    return np.concatenate(result).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--v8-dir", required=True)
    parser.add_argument("--outer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data, dtype={"id": str})
    embedding_ids = pd.read_csv(Path(args.embedding_cache) / "ids.csv", dtype={"id": str})[
        "id"
    ].tolist()
    if embedding_ids != data["id"].tolist():
        raise ValueError("Embedding cache and data are not aligned")
    full_x = np.load(Path(args.embedding_cache) / "embeddings.npy").astype("float32")
    id_to_index = {value: index for index, value in enumerate(data["id"])}
    v8_dir = Path(args.v8_dir)
    train = pd.read_csv(v8_dir / "oof_predictions.csv", dtype={"id": str})
    valid = pd.read_csv(v8_dir / "validation_predictions.csv", dtype={"id": str})
    category = "Легковоспламеняющиеся"
    train = train[train["category"].eq(category)].reset_index(drop=True)
    valid = valid[valid["category"].eq(category)].reset_index(drop=True)
    train_x_raw = full_x[[id_to_index[value] for value in train["id"]]]
    valid_x_raw = full_x[[id_to_index[value] for value in valid["id"]]]
    y_train = train["label"].to_numpy(dtype="int8")
    y_valid = valid["label"].to_numpy(dtype="int8")
    base_oof = train["blend_probability"].to_numpy(dtype="float32")
    base_valid = valid["blend_probability"].to_numpy(dtype="float32")
    base_threshold = 0.3310261532664299
    seeds = (41, 43, 47, 51, 59)

    outer = np.load(args.outer)
    outer_y = outer["y"]
    outer_base = np.load("artifacts/v19_outer_probabilities_2026.npz")["p"]

    reports = []
    saved = {}
    for remove in (0, 2, 4, 8):
        train_x, valid_x, mean, components = transform_embeddings(train_x_raw, valid_x_raw, remove)
        repeated = np.zeros(len(train), dtype="float32")
        for seed in seeds:
            fold_score = np.zeros(len(train), dtype="float32")
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for fit_indices, score_indices in splitter.split(
                train_x, y_train, groups=train["group"].astype(str)
            ):
                fold_score[score_indices] = prototype_score(
                    train_x[score_indices], train_x[fit_indices], y_train[fit_indices]
                )
            repeated += fold_score / len(seeds)
        valid_score = prototype_score(valid_x, train_x, y_train)
        standalone_f1, knn_threshold = best_threshold(y_train, repeated)

        best = None
        for scale in (0.005, 0.01, 0.02, 0.04):
            normalized_score = (repeated - knn_threshold) / scale
            for alpha in np.arange(0.05, 0.51, 0.05):
                margin = (1 - alpha) * (logit(base_oof) - logit(base_threshold))
                margin += alpha * normalized_score
                blend = sigmoid(margin)
                blend_f1, threshold = best_threshold(y_train, blend)
                candidate = (blend_f1, -alpha, -scale, threshold, alpha, scale)
                if best is None or candidate > best:
                    best = candidate
        assert best is not None
        _, _, _, blend_threshold, alpha, scale = best
        valid_margin = (1 - alpha) * (logit(base_valid) - logit(base_threshold))
        valid_margin += alpha * (valid_score - knn_threshold) / scale
        valid_blend = sigmoid(valid_margin)
        valid_f1 = f1_score(y_valid, valid_blend >= blend_threshold)

        outer_score = outer[f"remove{remove}"] if remove else outer["center"]
        outer_margin = (1 - alpha) * (logit(outer_base) - logit(base_threshold))
        outer_margin += alpha * (outer_score - knn_threshold) / scale
        outer_blend = sigmoid(outer_margin)
        outer_f1 = f1_score(outer_y, outer_blend >= blend_threshold)
        report = {
            "remove_components": remove,
            "standalone_oof_f1": standalone_f1,
            "standalone_oof_ap": float(average_precision_score(y_train, repeated)),
            "knn_threshold": knn_threshold,
            "alpha": float(alpha),
            "margin_scale": float(scale),
            "blend_oof_f1": float(best[0]),
            "blend_threshold": float(blend_threshold),
            "valid_f1": float(valid_f1),
            "outer_f1": float(outer_f1),
        }
        reports.append(report)
        saved[remove] = {
            "mean": mean,
            "components": components,
            "oof_score": repeated,
            "valid_score": valid_score,
        }
        print(json.dumps(report, ensure_ascii=False), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        reports=np.asarray(reports, dtype=object),
        **{
            f"remove{remove}_{key}": value
            for remove, values in saved.items()
            for key, value in values.items()
        },
    )
    Path(str(output) + ".json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
