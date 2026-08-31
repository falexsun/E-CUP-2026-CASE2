#!/usr/bin/env python3
"""Add conservative, reproducible title-family overrides to a trained artifact."""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import joblib
import pandas as pd

FLAMMABLE_TITLE_RULES = [
    {
        "name": "standalone_coal_or_fuel_briquette",
        "pattern": (
            r"\b(?:уголь\s+(?:древесн\w*|каменн\w*)|"
            r"брикетированн\w*\s+топлив\w*)\b"
        ),
        "label": 1,
    },
    {
        "name": "disposable_grill_with_coal",
        "pattern": (
            r"(?:мангал\w*.*одноразов\w*.*угл|"
            r"одноразов\w*.*мангал\w*.*угл)"
        ),
        "label": 1,
    },
    {
        "name": "gas_for_portable_stove_or_burner",
        "pattern": r"\bгаз\w*\s+(?:для|к)\s+портативн\w*\s+(?:плит\w*|горел\w*)\b",
        "label": 1,
    },
    {
        "name": "smoke_bomb",
        "pattern": r"\b(?:дым\w*\s+шашк\w*|шашк\w*\s+дым\w*)\b",
        "label": 1,
    },
    {
        "name": "barbecue_fuel_briquette",
        "pattern": r"\bбрикет\w*\s+для\s+грил\w*\b",
        "label": 1,
    },
    {
        "name": "dry_fuel_with_ignition_source",
        "pattern": r"\bсух\w*\s+горюч\w*.*(?:\bс\s+поджиг\w*|\bспичк\w*)",
        "label": 1,
    },
]


def normalize_title(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", value).strip()


def build_exact_lookup(frame: pd.DataFrame, category: str, *, min_count: int) -> dict[str, int]:
    subset = frame.loc[frame["category"].eq(category), ["name", "label"]].copy()
    subset["normalized_title"] = subset["name"].map(normalize_title)
    statistics = subset.groupby("normalized_title")["label"].agg(["size", "nunique", "first"])
    statistics = statistics[
        (statistics["size"] >= min_count)
        & statistics["nunique"].eq(1)
        & statistics.index.to_series().astype(bool)
    ]
    return {title: int(row["first"]) for title, row in statistics.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-title-count", type=int, default=2)
    parser.add_argument("--nearest-cosine-threshold", type=float, default=0.99)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    required = {"name", "category", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing training columns: {missing}")

    artifact = joblib.load(args.base_artifact)
    categories = sorted(frame["category"].astype(str).unique())
    artifact["exact_title_overrides"] = {
        category: build_exact_lookup(frame, category, min_count=args.min_title_count)
        for category in categories
    }
    artifact["regex_title_overrides"] = {"Легковоспламеняющиеся": FLAMMABLE_TITLE_RULES}
    flammable_knn = artifact.get("knn_overrides", {}).get("Легковоспламеняющиеся")
    if flammable_knn is None:
        raise ValueError("Base artifact has no flammable KNN reference override")
    flammable_knn["nearest_override_threshold"] = args.nearest_cosine_threshold
    artifact["title_override_metadata"] = {
        "source": str(Path(args.data).name),
        "min_title_count": args.min_title_count,
        "exact_mapping_sizes": {
            category: len(mapping)
            for category, mapping in artifact["exact_title_overrides"].items()
        },
        "rule_names": [rule["name"] for rule in FLAMMABLE_TITLE_RULES],
        "nearest_cosine_threshold": args.nearest_cosine_threshold,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    print(artifact["title_override_metadata"])


if __name__ == "__main__":
    sys.exit(main())
