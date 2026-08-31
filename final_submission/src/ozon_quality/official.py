"""Official E-CUP 2026 quality-control metric and output contract."""

from __future__ import annotations

import re
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OFFICIAL_CATEGORIES: Final[tuple[str, str]] = ("БАД", "Легковоспламеняющиеся")
RESULT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^<комментарий>(?P<comment>[\s\S]{50,300})<вердикт>(?P<verdict>бан|не бан)$"
)


def official_category_f1(
    labels: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    categories: np.ndarray | pd.Series,
) -> dict[str, float]:
    truth = np.asarray(labels, dtype="int64")
    predicted = np.asarray(predictions, dtype="int64")
    category_values = np.asarray(categories, dtype=str)
    if not (len(truth) == len(predicted) == len(category_values)):
        raise ValueError("labels, predictions and categories must have equal length")
    unknown = sorted(set(category_values) - set(OFFICIAL_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown official categories: {unknown}")
    scores: dict[str, float] = {}
    for category in OFFICIAL_CATEGORIES:
        mask = category_values == category
        if not mask.any():
            raise ValueError(f"Category {category!r} is absent")
        scores[f"f1_{category}"] = float(
            f1_score(truth[mask], predicted[mask], zero_division=0)
        )
    scores["official_score"] = float(np.mean(list(scores.values())))
    return scores


def format_result(comment: str, label: int) -> str:
    normalized = " ".join(str(comment).split()).strip()
    if len(normalized) < 50:
        normalized = (
            normalized + " Решение принято по названию, описанию и изображениям товара."
        ).strip()
    if len(normalized) > 300:
        normalized = normalized[:300].rsplit(" ", 1)[0].rstrip(" ,;:-")
    verdict = "не бан" if int(label) == 1 else "бан"
    result = f"<комментарий>{normalized}<вердикт>{verdict}"
    if not RESULT_PATTERN.fullmatch(result):
        raise ValueError("Formatted result violates the official output contract")
    return result


def validate_submission(frame: pd.DataFrame, expected_ids: pd.Series | None = None) -> None:
    if frame.columns.tolist() != ["id", "result"]:
        raise ValueError("Submission must contain exactly the columns: id, result")
    if frame["id"].duplicated().any() or frame["id"].isna().any():
        raise ValueError("Submission IDs must be complete and unique")
    invalid = [index for index, value in enumerate(frame["result"]) if not RESULT_PATTERN.fullmatch(str(value))]
    if invalid:
        raise ValueError(f"Invalid result format at rows: {invalid[:10]}")
    if expected_ids is not None and frame["id"].astype(str).tolist() != expected_ids.astype(str).tolist():
        raise ValueError("Submission IDs/order do not match the input")
