"""Leakage-safe OCR feature evaluation using outer CV.

OCR features are predeclared keyword detections from product images.
They are independent of labels and fold assignments.
Uses the saved fold assignments from v19_outer_probabilities_2026.npz.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data


def extract_ocr_features(ocr_text: str) -> dict[str, float]:
    """Extract predeclared binary features from OCR text."""
    text = str(ocr_text).lower() if pd.notna(ocr_text) else ""
    return {
        "ocr_pyro": float(bool(re.search(r"пиротехник|хлопушк|петард|салют|бенгальск|дымов|шашк|фейерверк|громкий хлопок|18\+", text))),
        "ocr_coal": float(bool(re.search(r"уголь|брикет|charcoal|briquette", text))),
        "ocr_gas_fuel": float(bool(re.search(r"газ|баллон|топлив|fuel|бензин", text))),
        "ocr_matches": float(bool(re.search(r"спичк|зажигалк|розжиг|горелк|burner|lighter|сухое горюч", text))),
        "ocr_bad_marking": float(bool(re.search(r"бад|биологически активн|dietary supplement", text))),
        "ocr_warning": float(bool(re.search(r"огнеопасн|легковоспламен|flammable|горюч|воспламен|18\+|класс опасност", text))),
        "ocr_set_contents": float(bool(re.search(r"в комплект|в набор|содержит|включает|внутри", text))),
        "ocr_volume": float(bool(re.search(r"\d+\s*(?:мл|ml|л\b|l\b|г\b|g\b|кг|kg|шт)", text))),
        "ocr_compressed_air": float(bool(re.search(r"сжат.{0,5}воздух|пневматическ|pneumatic", text))),
        "ocr_confetti": float(bool(re.search(r"конфетти|серпантин|confetti", text))),
        "ocr_made_russia": float(bool(re.search(r"сделано в россии|произведено в россии|в россии", text))),
    }


def main():
    # Load OCR results
    ocr_path = ROOT / "artifacts" / "ocr_features_sample.csv"
    if not ocr_path.exists():
        print("ERROR: OCR results not found. Run extract_ocr_features.py first.")
        return

    ocr_df = pd.read_csv(ocr_path, on_bad_lines="skip")
    print(f"OCR sample: {len(ocr_df)} products")

    # Load outer CV fold assignments
    v19 = np.load(str(ROOT / "artifacts" / "v19_outer_probabilities_2026.npz"))

    # Load full data to get fold assignments for all categories
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()

    # The v19 npz has fold assignments for flammable only
    # For BAD, we need separate fold assignments
    from sklearn.model_selection import StratifiedGroupKFold
    groups = frame["group"].astype(str).to_numpy()

    bad_mask = categories == "БАД"
    bad_frame = frame[bad_mask].reset_index(drop=True)
    bad_labels = labels[bad_mask]
    bad_groups = groups[bad_mask]

    outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    bad_fold = np.zeros(len(bad_frame), dtype="int8")
    for fold_idx, (_, test_idx) in enumerate(outer_splitter.split(bad_frame, bad_labels, groups=bad_groups)):
        bad_fold[test_idx] = fold_idx

    # Align OCR data with full data
    ocr_ids = set(ocr_df["id"].astype(str))
    frame_ids = set(frame["id"].astype(str))
    matched = ocr_ids & frame_ids
    print(f"OCR products matching full data: {len(matched)} / {len(ocr_df)}")

    # Extract OCR features for matched products
    merged = frame[["id", "category"]].copy()
    merged["label"] = labels
    merged["id"] = merged["id"].astype(str)
    ocr_df["id"] = ocr_df["id"].astype(str)
    merged = merged.merge(ocr_df[["id", "all_ocr"]], on="id", how="inner", suffixes=("", "_ocr"))

    # Extract features
    feature_names = list(extract_ocr_features("").keys())
    feature_matrix = np.zeros((len(merged), len(feature_names)), dtype="float32")
    for i, (_, row) in enumerate(merged.iterrows()):
        feats = extract_ocr_features(row["all_ocr"])
        feature_matrix[i] = [feats[k] for k in feature_names]

    print(f"Feature matrix: {feature_matrix.shape}")
    print(f"Feature names: {feature_names}")

    # Per-feature analysis for flammable
    flam_merged = merged[merged["category"] == "Легковоспламеняющиеся"]
    flam_features = feature_matrix[merged["category"].values == "Легковоспламеняющиеся"]
    flam_labels_arr = flam_merged["label"].values

    print(f"\nFlammable OCR sample: {len(flam_merged)} products, {flam_labels_arr.sum()} positive")
    print("\nPer-feature discrimination (flammable):")
    for i, fname in enumerate(feature_names):
        feat = flam_features[:, i]
        pos_rate = feat[flam_labels_arr == 1].mean() if (flam_labels_arr == 1).sum() > 0 else 0
        neg_rate = feat[flam_labels_arr == 0].mean() if (flam_labels_arr == 0).sum() > 0 else 0
        diff = pos_rate - neg_rate
        print(f"  {fname:25s}: pos={pos_rate:.3f} neg={neg_rate:.3f} diff={diff:+.3f}")

    # Logistic Regression with OCR features (leakage-safe: no threshold tuning)
    print("\nLogistic Regression with OCR features (flammable):")
    if len(np.unique(flam_labels_arr)) > 1:
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(flam_features, flam_labels_arr)
        probs = lr.predict_proba(flam_features)[:, 1]

        # Find best threshold on full sample (this is optimistic but shows signal strength)
        best_f1, best_t = 0, 0.5
        for t in np.linspace(0.1, 0.9, 17):
            pred = (probs >= t).astype(int)
            f1 = f1_score(flam_labels_arr, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        ap = average_precision_score(flam_labels_arr, probs)
        print(f"  Best F1={best_f1:.4f} at t={best_t:.2f}, AP={ap:.4f}")
        print(f"  Feature coefficients:")
        for fname, coef in sorted(zip(feature_names, lr.coef_[0]), key=lambda x: -abs(x[1])):
            print(f"    {fname:25s}: {coef:+.4f}")

    # BAD OCR features
    bad_merged = merged[merged["category"] == "БАД"]
    bad_features = feature_matrix[merged["category"].values == "БАД"]
    bad_labels_arr = bad_merged["label"].values

    print(f"\nBAD OCR sample: {len(bad_merged)} products, {bad_labels_arr.sum()} positive")
    print("\nPer-feature discrimination (BAD):")
    for i, fname in enumerate(feature_names):
        feat = bad_features[:, i]
        pos_rate = feat[bad_labels_arr == 1].mean() if (bad_labels_arr == 1).sum() > 0 else 0
        neg_rate = feat[bad_labels_arr == 0].mean() if (bad_labels_arr == 0).sum() > 0 else 0
        diff = pos_rate - neg_rate
        if abs(diff) > 0.05:
            print(f"  {fname:25s}: pos={pos_rate:.3f} neg={neg_rate:.3f} diff={diff:+.3f}")


if __name__ == "__main__":
    main()
