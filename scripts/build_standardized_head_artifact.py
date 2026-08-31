#!/usr/bin/env python3
"""Fit the frozen v45 standardized linear head on all released labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CATEGORY = "Легковоспламеняющиеся"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--C", type=float, default=0.03)
    parser.add_argument("--alpha-head", type=float, default=0.25)
    parser.add_argument("--blend-threshold", type=float, default=0.26)
    args = parser.parse_args()

    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))["linear_head"]
        args.C = float(config["C"])
        args.alpha_head = float(config["alpha_head"])
        args.blend_threshold = float(config["blend_threshold"])

    frame = pd.read_csv(args.data, dtype={"id": str})
    cache = Path(args.embedding_cache)
    cache_ids = pd.read_csv(cache / "ids.csv", dtype={"id": str})["id"].tolist()
    if cache_ids != frame["id"].tolist():
        raise ValueError("Embedding cache and training data are not aligned")
    embeddings = np.load(cache / "embeddings.npy").astype("float32")
    mask = frame["category"].eq(CATEGORY).to_numpy()
    train_x = embeddings[mask]
    train_y = frame.loc[mask, "label"].to_numpy(dtype="int8")
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000).fit(
        scaler.transform(train_x), train_y
    )

    artifact = joblib.load(args.base_artifact)
    artifact["linear_head_overrides"] = {
        CATEGORY: {
            "contract": "raw_embedding_standard_scaler_logistic_regression",
            "C": args.C,
            "alpha_head": args.alpha_head,
            "blend_threshold": args.blend_threshold,
            "scaler": scaler,
            "model": model,
        }
    }
    # Keep the serialized artifact independent of the machine-specific config path.
    # The versioned JSON remains the source of build provenance.
    artifact.pop("linear_head_provenance", None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    print(
        {
            "category": CATEGORY,
            "rows": int(mask.sum()),
            "positives": int(train_y.sum()),
            "C": args.C,
            "alpha_head": args.alpha_head,
            "blend_threshold": args.blend_threshold,
        }
    )


if __name__ == "__main__":
    main()
