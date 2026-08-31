"""Group-aware holdout creation with label-distribution search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

from ozon_quality.data import load_data, read_table


def _write_table(frame: pd.DataFrame, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.casefold()
    if suffix == ".csv":
        frame.to_csv(destination, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(destination, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        frame.to_json(destination, orient="records", lines=True, force_ascii=False)
    elif suffix == ".json":
        frame.to_json(destination, orient="records", force_ascii=False)
    else:
        raise ValueError("Output must be CSV, Parquet, JSON or JSONL")


def group_holdout(
    input_path: str,
    train_output: str,
    valid_output: str,
    *,
    schema: str | None,
    valid_size: float,
    seed: int,
    candidates: int = 64,
) -> dict[str, float | int]:
    if not 0 < valid_size < 1:
        raise ValueError("valid_size must be between 0 and 1")
    raw = read_table(input_path)
    frame, _ = load_data(input_path, schema, require_label=True)
    if "group" not in frame or frame["group"].eq("").any():
        raise ValueError("A complete group/seller/entity mapping is required; random split is forbidden")
    targets = MultiLabelBinarizer().fit_transform(frame["labels"])
    global_rate = targets.mean(axis=0)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for offset in range(candidates):
        splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed + offset)
        train_index, valid_index = next(splitter.split(frame, groups=frame["group"]))
        valid_rate = targets[valid_index].mean(axis=0)
        score = float(np.abs(valid_rate - global_rate).mean()) + abs(
            len(valid_index) / len(frame) - valid_size
        )
        if best is None or score < best[0]:
            best = score, train_index, valid_index
    assert best is not None
    score, train_index, valid_index = best
    prevalence_mae = float(np.abs(targets[valid_index].mean(axis=0) - global_rate).mean())
    _write_table(raw.iloc[train_index].reset_index(drop=True), train_output)
    _write_table(raw.iloc[valid_index].reset_index(drop=True), valid_output)
    return {
        "train_rows": len(train_index),
        "valid_rows": len(valid_index),
        "train_groups": frame.iloc[train_index]["group"].nunique(),
        "valid_groups": frame.iloc[valid_index]["group"].nunique(),
        "selection_objective": score,
        "label_prevalence_mae": prevalence_mae,
    }
