#!/usr/bin/env python3
"""Evaluate title-aware lexical views as an independent addition to v19."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

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


def render(frame: pd.DataFrame, view: str) -> pd.Series:
    title = frame["name"].fillna("").astype(str)
    description = frame["description"].fillna("").astype(str)
    if view == "title":
        return title
    if view == "title3_desc":
        return title + "\n" + title + "\n" + title + "\n" + description
    if view == "title_desc_head800":
        return title + "\n" + description.str[:800]
    raise ValueError(view)


def make_model(c_value: float) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b\w+\b",
                    max_features=80_000,
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 6),
                    max_features=120_000,
                    min_df=2,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("tfidf", features),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=3000,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--v8-dir", required=True)
    parser.add_argument("--v19-features", required=True)
    parser.add_argument("--v19-artifact", required=True)
    parser.add_argument("--outer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data, dtype={"id": str})
    id_to_index = {value: index for index, value in enumerate(data["id"])}
    v8 = Path(args.v8_dir)
    train = pd.read_csv(v8 / "oof_predictions.csv", dtype={"id": str})
    valid = pd.read_csv(v8 / "validation_predictions.csv", dtype={"id": str})
    train = train[train["category"].eq(CATEGORY)].reset_index(drop=True)
    valid = valid[valid["category"].eq(CATEGORY)].reset_index(drop=True)
    train_frame = data.iloc[[id_to_index[value] for value in train["id"]]].reset_index(drop=True)
    valid_frame = data.iloc[[id_to_index[value] for value in valid["id"]]].reset_index(drop=True)
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

    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=41).split(
            train, train_y, groups=train["group"].astype(str)
        )
    )
    candidates: list[dict] = []
    oof_values: dict[tuple[str, float], np.ndarray] = {}
    for view in ("title", "title3_desc", "title_desc_head800"):
        text = render(train_frame, view)
        for c_value in (0.3, 1.0, 3.0, 10.0):
            oof = np.zeros(len(train), dtype="float32")
            for fit_indices, score_indices in splits:
                model = make_model(c_value).fit(text.iloc[fit_indices], train_y[fit_indices])
                oof[score_indices] = model.predict_proba(text.iloc[score_indices])[:, 1]
            oof_values[(view, c_value)] = oof
            for alpha in np.arange(0.05, 0.51, 0.05):
                blend = sigmoid((1 - alpha) * logit(v19_oof) + alpha * logit(oof))
                score, threshold = best_threshold(train_y, blend)
                candidates.append(
                    {
                        "view": view,
                        "C": c_value,
                        "alpha": float(alpha),
                        "oof_f1": score,
                        "threshold": threshold,
                        "head_oof_ap": float(average_precision_score(train_y, oof)),
                    }
                )
    selected = max(candidates, key=lambda row: (row["oof_f1"], -row["alpha"]))
    train_text = render(train_frame, selected["view"])
    valid_text = render(valid_frame, selected["view"])
    model = make_model(selected["C"]).fit(train_text, train_y)
    valid_head = model.predict_proba(valid_text)[:, 1]
    valid_blend = sigmoid(
        (1 - selected["alpha"]) * logit(v19_valid) + selected["alpha"] * logit(valid_head)
    )
    inherited_threshold = float(override["blend_threshold"])
    selected.update(
        {
            "v19_valid_f1": float(f1_score(valid_y, v19_valid >= inherited_threshold)),
            "valid_f1": float(f1_score(valid_y, valid_blend >= selected["threshold"])),
            "v19_valid_ap": float(average_precision_score(valid_y, v19_valid)),
            "valid_ap": float(average_precision_score(valid_y, valid_blend)),
        }
    )

    category_frame = data[data["category"].eq(CATEGORY)].reset_index(drop=True)
    outer_text = render(category_frame, selected["view"])
    outer_y = category_frame["label"].to_numpy(dtype="int8")
    outer = np.load(args.outer)
    outer_head = np.zeros(len(outer_y), dtype="float32")
    for fold_index in range(5):
        fit_mask = outer["fold"] != fold_index
        score_mask = ~fit_mask
        fold_model = make_model(selected["C"]).fit(outer_text[fit_mask], outer_y[fit_mask])
        outer_head[score_mask] = fold_model.predict_proba(outer_text[score_mask])[:, 1]
    outer_blend = sigmoid(
        (1 - selected["alpha"]) * logit(outer["p"])
        + selected["alpha"] * logit(outer_head)
    )
    selected.update(
        {
            "v19_outer_f1": float(f1_score(outer_y, outer["p"] >= inherited_threshold)),
            "outer_f1": float(f1_score(outer_y, outer_blend >= selected["threshold"])),
            "v19_outer_ap": float(average_precision_score(outer_y, outer["p"])),
            "outer_ap": float(average_precision_score(outer_y, outer_blend)),
        }
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"selection": selected, "model": model, "candidates": candidates}, output, compress=3
    )
    Path(str(output) + ".json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
