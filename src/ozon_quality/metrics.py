"""Validation metrics and binary threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def choose_binary_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.r_[0.01, np.linspace(0.05, 0.95, 181), 0.99, probability])
    scores = np.asarray([f1_score(y_true, probability >= value, average="macro") for value in candidates])
    best = np.flatnonzero(scores == scores.max())
    index = best[np.argmin(np.abs(candidates[best] - 0.5))]
    return float(candidates[index]), float(scores[index])


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def choose_multilabel_thresholds(y_true: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.full(y_true.shape[1], 0.5, dtype="float32")
    scores = np.zeros(y_true.shape[1], dtype="float32")
    for index in range(y_true.shape[1]):
        if np.unique(y_true[:, index]).size < 2:
            scores[index] = f1_score(
                y_true[:, index], probability[:, index] >= 0.5, zero_division=0
            )
            continue
        thresholds[index], scores[index] = choose_binary_threshold(
            y_true[:, index], probability[:, index]
        )
    return thresholds, scores
