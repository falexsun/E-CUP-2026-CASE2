"""Fast category-specific lexical baseline for the released competition task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from ozon_quality.data import load_data
from ozon_quality.lexical import build_lexical_classifier
from ozon_quality.official import OFFICIAL_CATEGORIES, official_category_f1

RULE_PATTERNS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "БАД": {
        "bad_direct": (
            r"\bбад\b",
            r"биологически\s+активн",
            r"dietary\s+supplement",
            r"не\s+является\s+лекарств",
            r"свидетельств.{0,20}государственн.{0,20}регистрац",
        ),
        "sports_food": (
            r"спортивн.{0,12}питан",
            r"\bbcaa\b",
            r"\bпротеин",
            r"аминокислот",
            r"l[\s-]?карнитин",
            r"предтрен",
            r"гейнер",
        ),
        "explicit_not_bad": (r"не\s+(?:является|относится).{0,20}\bбад\b",),
    },
    "Легковоспламеняющиеся": {
        "ignition_source": (
            r"зажигал",
            r"\bспич",
            r"огнив",
            r"источник.{0,15}(?:огня|пламени|воспламен)",
        ),
        "fuel_content": (
            r"жидкост.{0,15}(?:розжиг|зажигал)",
            r"газ.{0,15}(?:баллон|зажигал|горюч)",
            r"\bбутан\b",
            r"\bпропан\b",
            r"\bкеросин\b",
            r"\bбензин\b",
            r"легковоспламен",
        ),
        "without_content": (
            r"без\s+(?:газа|топлива|жидкости|спичек|угля)",
            r"не\s+входит\s+в\s+комплект",
            r"поставляется\s+без",
        ),
        "construction_only": (
            r"\bмангал",
            r"\bгрил",
            r"газов.{0,10}плит",
            r"горелк.{0,20}(?:встроен|пьезо)",
        ),
    },
}


def add_rule_tokens(text: pd.Series, category: str) -> pd.Series:
    output = text.fillna("").astype(str).str.replace(r"<[^>]+>", " ", regex=True)
    lowered = output.str.casefold()
    for family, patterns in RULE_PATTERNS[category].items():
        expression = "(?:" + ")|(?:".join(patterns) + ")"
        active = lowered.str.contains(expression, regex=True, na=False)
        output = output + active.map(
            {True: f" __RULE_{family.upper()}__", False: ""}
        )
    return output


def choose_positive_threshold(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype="int8")
    probability = np.asarray(probability, dtype="float64")
    if labels.ndim != 1 or probability.ndim != 1 or len(labels) != len(probability):
        raise ValueError("labels and probability must be equal-length one-dimensional arrays")
    if len(labels) == 0 or not np.isfinite(probability).all():
        raise ValueError("threshold selection requires finite non-empty probabilities")
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 197), probability])
    order = np.argsort(probability, kind="stable")
    sorted_probability = probability[order]
    cumulative_positive = np.r_[0, np.cumsum(labels[order], dtype="int64")]
    first_predicted = np.searchsorted(sorted_probability, candidates, side="left")
    predicted_positive = len(labels) - first_predicted
    true_positive = cumulative_positive[-1] - cumulative_positive[first_predicted]
    denominator = cumulative_positive[-1] + predicted_positive
    scores = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(candidates, dtype="float64"),
        where=denominator != 0,
    )
    best = np.flatnonzero(scores == scores.max())
    index = best[np.argmin(np.abs(candidates[best] - 0.5))]
    return float(candidates[index]), float(scores[index])


def train_official_text_baseline(
    train_path: str,
    valid_path: str,
    output: str,
    *,
    schema: str,
    seed: int,
    oof_folds: int = 5,
    tune_hyperparameters: bool = False,
) -> dict[str, Any]:
    train, config = load_data(train_path, schema, require_label=True)
    valid, _ = load_data(valid_path, schema, require_label=True)
    train_labels = train["labels"].map(lambda values: int(values[0])).to_numpy()
    valid_labels = valid["labels"].map(lambda values: int(values[0])).to_numpy()
    models: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    probability = np.zeros(len(valid), dtype="float32")
    prediction = np.zeros(len(valid), dtype="int8")
    category_metrics: dict[str, Any] = {}
    oof_probability = np.full(len(train), np.nan, dtype="float32")
    for offset, category in enumerate(OFFICIAL_CATEGORIES):
        train_mask = train["category"].eq(category).to_numpy()
        valid_mask = valid["category"].eq(category).to_numpy()
        train_text = add_rule_tokens(train.loc[train_mask, "text"], category)
        valid_text = add_rule_tokens(valid.loc[valid_mask, "text"], category)
        category_labels = train_labels[train_mask]
        category_indices = np.flatnonzero(train_mask)
        if "group" not in train:
            raise ValueError("Official OOF baseline requires entity_group in training data")
        category_groups = train.loc[train_mask, "group"].astype(str).to_numpy()
        splitter = StratifiedGroupKFold(
            n_splits=oof_folds, shuffle=True, random_state=seed + offset
        )
        splits = list(splitter.split(train_text, category_labels, groups=category_groups))
        candidates = (
            [(c, weight) for c in (0.1, 0.3, 1.0, 3.0, 10.0) for weight in (None, "balanced")]
            if tune_hyperparameters
            else [(4.0, "balanced")]
        )
        candidate_oof = {
            (c, weight): np.zeros(len(category_indices), dtype="float32")
            for c, weight in candidates
        }
        for fold, (fold_train, fold_valid) in enumerate(splits):
            template = build_lexical_classifier(
                seed + offset * 100 + fold, "single_label"
            )
            vectorizer = template.named_steps["tfidf"]
            fit_x = vectorizer.fit_transform(train_text.iloc[fold_train])
            holdout_x = vectorizer.transform(train_text.iloc[fold_valid])
            for c, weight in candidates:
                estimator = LogisticRegression(
                    C=c,
                    class_weight=weight,
                    max_iter=2000,
                    random_state=seed + offset * 100 + fold,
                    solver="liblinear",
                )
                estimator.fit(fit_x, category_labels[fold_train])
                candidate_oof[(c, weight)][fold_valid] = estimator.predict_proba(
                    holdout_x
                )[:, 1]
        ranked: list[tuple[float, float, float, str | None, float, np.ndarray]] = []
        for (c, weight), values in candidate_oof.items():
            candidate_threshold, candidate_score = choose_positive_threshold(
                category_labels, values
            )
            ranked.append(
                (
                    candidate_score,
                    -abs(np.log10(c) - np.log10(3.0)),
                    c,
                    weight,
                    candidate_threshold,
                    values,
                )
            )
        _, _, best_c, best_weight, threshold, category_oof = max(
            ranked, key=lambda value: value[:2]
        )
        oof_score = max(value[0] for value in ranked)
        oof_probability[category_indices] = category_oof
        model = build_lexical_classifier(seed + offset, "single_label")
        model.set_params(
            classifier__C=best_c,
            classifier__class_weight=best_weight,
        )
        model.fit(train_text, category_labels)
        category_probability = model.predict_proba(valid_text)[:, 1]
        probability[valid_mask] = category_probability
        prediction[valid_mask] = category_probability >= threshold
        models[category] = model
        thresholds[category] = threshold
        category_metrics[category] = {
            "threshold": threshold,
            "oof_f1": oof_score,
            "valid_f1": float(
                f1_score(
                    valid_labels[valid_mask], prediction[valid_mask], zero_division=0
                )
            ),
            "train_rows": int(train_mask.sum()),
            "valid_rows": int(valid_mask.sum()),
            "valid_positives": int(valid_labels[valid_mask].sum()),
            "oof_folds": oof_folds,
            "C": best_c,
            "class_weight": best_weight,
            "tuned": tune_hyperparameters,
        }
    metrics = official_category_f1(valid_labels, prediction, valid["category"])
    metrics["categories"] = category_metrics
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_version": 1,
        "models": models,
        "thresholds": thresholds,
        "schema": config,
        "rules": RULE_PATTERNS,
        "hyperparameters": {
            category: {
                "C": category_metrics[category]["C"],
                "class_weight": category_metrics[category]["class_weight"],
            }
            for category in OFFICIAL_CATEGORIES
        },
    }
    joblib.dump(artifact, output_dir / "official_text_baseline.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    errors = valid[["id", "title", "description", "category"]].copy()
    errors["label"] = valid_labels
    errors["probability"] = probability
    errors["prediction"] = prediction
    errors["error"] = np.where(
        (errors.label == 1) & (errors.prediction == 0),
        "FN",
        np.where((errors.label == 0) & (errors.prediction == 1), "FP", ""),
    )
    errors.to_csv(output_dir / "validation_predictions.csv", index=False)
    errors[errors.error.ne("")].sort_values("probability", ascending=False).to_csv(
        output_dir / "errors.csv", index=False
    )
    oof = train[["id", "title", "category", "group"]].copy()
    oof["label"] = train_labels
    oof["probability"] = oof_probability
    oof["prediction"] = [
        int(probability_value >= thresholds[category])
        for probability_value, category in zip(
            oof_probability, train["category"], strict=True
        )
    ]
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    return metrics
