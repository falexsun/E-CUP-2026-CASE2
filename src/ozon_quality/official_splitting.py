"""Leakage-aware split tailored to the released E-CUP quality dataset."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from ozon_quality.data import load_data, read_table
from ozon_quality.splitting import _write_table


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _normalize_text(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value).casefold())
    return re.sub(r"\s+", " ", re.sub(r"[^a-zа-яё0-9]+", " ", text)).strip()


def _first_image_signature(paths: list[str]) -> str | None:
    if not paths:
        return None
    try:
        with Image.open(paths[0]) as image:
            rgb = image.convert("RGB")
            gray = np.asarray(rgb.convert("L").resize((9, 8)))
            difference = (gray[:, 1:] > gray[:, :-1]).tobytes()
            color = np.asarray(rgb.resize((8, 8)), dtype="float32").mean((0, 1))
            color_bucket = np.round(color / 16).astype("uint8").tobytes()
            return hashlib.sha256(difference + color_bucket).hexdigest()
    except (OSError, ValueError):
        return None


def derive_entity_groups(frame: pd.DataFrame, *, use_first_image: bool = True) -> pd.Series:
    """Connect exact text/title and conservative first-image duplicates."""
    size = len(frame)
    dsu = _DisjointSet(size)
    keys: list[tuple[str, pd.Series, int]] = []
    title = frame["title"].map(_normalize_text) if "title" in frame else frame["text"].map(_normalize_text)
    full_text = frame["text"].map(_normalize_text)
    keys.extend(
        [
            ("title", title, 1_000),
            ("full_text", full_text, 1_000),
            ("text_head_800", full_text.str[:800].where(full_text.str.len().ge(800), ""), 1_000),
            ("text_tail_400", full_text.str[-400:].where(full_text.str.len().ge(800), ""), 1_000),
        ]
    )
    if use_first_image:
        signatures = frame["images"].map(_first_image_signature)
        keys.append(("first_image", signatures, 20))
    for _, values, max_frequency in keys:
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(values):
            if value and len(str(value)) >= 8:
                buckets[str(value)].append(index)
        for members in buckets.values():
            if 1 < len(members) <= max_frequency:
                anchor = members[0]
                for member in members[1:]:
                    dsu.union(anchor, member)
    roots = [dsu.find(index) for index in range(size)]
    root_to_group = {root: f"entity-{number}" for number, root in enumerate(sorted(set(roots)))}
    return pd.Series([root_to_group[root] for root in roots], index=frame.index, name="entity_group")


def write_grouped_official_data(
    input_path: str,
    output_path: str,
    *,
    schema: str,
) -> dict[str, Any]:
    """Persist the full released table with the same entity groups used by validation."""
    raw = read_table(input_path)
    frame, _ = load_data(input_path, schema, require_label=True)
    groups = derive_entity_groups(frame)
    enriched = raw.copy()
    enriched["entity_group"] = groups
    _write_table(enriched, output_path)
    return {
        "rows": len(enriched),
        "entity_groups": int(groups.nunique()),
        "output": str(output_path),
    }


def official_holdout(
    input_path: str,
    train_output: str,
    valid_output: str,
    *,
    schema: str,
    valid_size: float,
    seed: int,
    candidates: int = 512,
) -> dict[str, Any]:
    raw = read_table(input_path)
    frame, _ = load_data(input_path, schema, require_label=True)
    groups = derive_entity_groups(frame)
    labels = frame["labels"].map(lambda values: int(values[0]))
    strata = frame["category"].astype(str) + "::" + labels.astype(str)
    stratum_names = sorted(strata.unique())
    target_matrix = np.column_stack(
        [(strata == stratum).to_numpy(dtype="float32") for stratum in stratum_names]
    )
    global_rate = target_matrix.mean(axis=0)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for offset in range(candidates):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=valid_size, random_state=seed + offset
        )
        train_index, valid_index = next(splitter.split(frame, groups=groups))
        valid_rate = target_matrix[valid_index].mean(axis=0)
        relative_error = np.abs(valid_rate - global_rate) / np.maximum(global_rate, 1e-4)
        objective = float(relative_error.mean()) + abs(
            len(valid_index) / len(frame) - valid_size
        )
        if best is None or objective < best[0]:
            best = objective, train_index, valid_index
    assert best is not None
    objective, train_index, valid_index = best
    enriched = raw.copy()
    enriched["entity_group"] = groups
    _write_table(enriched.iloc[train_index].reset_index(drop=True), train_output)
    _write_table(enriched.iloc[valid_index].reset_index(drop=True), valid_output)
    train_groups = set(groups.iloc[train_index])
    valid_groups = set(groups.iloc[valid_index])
    if train_groups & valid_groups:
        raise AssertionError("Entity groups overlap after official split")
    summary: dict[str, Any] = {
        "train_rows": len(train_index),
        "valid_rows": len(valid_index),
        "train_groups": len(train_groups),
        "valid_groups": len(valid_groups),
        "selection_objective": objective,
        "strata": {},
    }
    for stratum in stratum_names:
        summary["strata"][stratum] = {
            "all": int((strata == stratum).sum()),
            "train": int((strata.iloc[train_index] == stratum).sum()),
            "valid": int((strata.iloc[valid_index] == stratum).sum()),
        }
    return summary
