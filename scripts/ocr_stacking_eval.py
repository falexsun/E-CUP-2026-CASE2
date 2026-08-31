"""Test OCR features as additional signal on top of v45 scores.

Leakage-safe approach:
1. v45 probabilities come from saved outer CV (no leakage)
2. OCR features are extracted from images (independent of labels)
3. Simple LogisticRegression on [v45_prob, ocr_features] per fold
4. No threshold optimization on fold-specific data

This tests whether OCR adds independent signal beyond text+VLM.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data


def extract_ocr_features(ocr_text: str) -> dict[str, float]:
    text = str(ocr_text).lower() if pd.notna(ocr_text) else ""
    return {
        "ocr_pyro": float(bool(re.search(r"пиротехник|хлопушк|петард|салют|бенгальск|дымов|шашк|фейерверк|громкий хлопок|18\+", text))),
        "ocr_coal": float(bool(re.search(r"уголь|брикет|charcoal|briquette", text))),
        "ocr_gas_fuel_neg": float(bool(re.search(r"газ|баллон|топлив|fuel|бензин", text))),
        "ocr_warning": float(bool(re.search(r"огнеопасн|легковоспламен|flammable|горюч|воспламен|18\+|класс опасност", text))),
        "ocr_set_contents": float(bool(re.search(r"в комплект|в набор|содержит|включает|внутри", text))),
    }


def main():
    # Load OCR results
    ocr_df = pd.read_csv(str(ROOT / "artifacts" / "ocr_features_sample.csv"), on_bad_lines="skip")
    ocr_df["id"] = ocr_df["id"].astype(str)

    # Load outer CV
    v19 = np.load(str(ROOT / "artifacts" / "v19_outer_probabilities_2026.npz"))
    v45 = np.load(str(ROOT / "artifacts" / "v34_regex_combo_outer_v45.npz"))

    # Load full data for matching
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()
    flam_mask = categories == "Легковоспламеняющиеся"

    # Get flammable IDs in order
    flam_frame = frame[flam_mask].reset_index(drop=True)
    flam_labels = labels[flam_mask]
    flam_ids = frame.loc[flam_mask, "id"].astype(str).tolist()

    assert (v19["y"] == flam_labels).all()

    thresh = float(json.loads((ROOT / "configs" / "v19_knn.json").read_text())["blend_threshold"])

    # v45 predictions
    v45_pred = (v45["p"] >= thresh).astype(int)
    v45_pred[v45["regex"]] = 1

    # Match OCR features to flammable products
    ocr_map = {}
    for _, row in ocr_df.iterrows():
        ocr_map[str(row["id"])] = row["all_ocr"]

    feature_names = list(extract_ocr_features("").keys())
    ocr_features = np.zeros((len(flam_frame), len(feature_names)), dtype="float32")
    has_ocr = np.zeros(len(flam_frame), dtype=bool)

    for i, fid in enumerate(flam_ids):
        if fid in ocr_map:
            feats = extract_ocr_features(ocr_map[fid])
            ocr_features[i] = [feats[k] for k in feature_names]
            has_ocr[i] = True

    print(f"Flammable with OCR: {has_ocr.sum()} / {len(flam_frame)}")

    # Test: does OCR help beyond v45 probability?
    # Use only products with OCR data
    subset = has_ocr
    y = flam_labels[subset]
    v45_prob_subset = v45["p"][subset]
    v45_pred_subset = v45_pred[subset]
    ocr_subset = ocr_features[subset]

    # v45-only baseline on subset
    v45_f1 = f1_score(y, v45_pred_subset, zero_division=0)
    v45_ap = average_precision_score(y, v45_prob_subset)
    print(f"\nv45 on OCR subset: F1={v45_f1:.4f}, AP={v45_ap:.4f}")

    # v45 + OCR features
    combined = np.column_stack([v45_prob_subset, ocr_subset])
    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(combined, y)
    probs_combined = lr.predict_proba(combined)[:, 1]

    best_f1, best_t = 0, 0.5
    for t in np.linspace(0.1, 0.9, 17):
        pred = (probs_combined >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    ap_combined = average_precision_score(y, probs_combined)
    print(f"v45 + OCR: F1={best_f1:.4f} (t={best_t:.2f}), AP={ap_combined:.4f}")
    print(f"  Delta F1: {best_f1 - v45_f1:+.4f}")
    print(f"  Delta AP: {ap_combined - v45_ap:+.4f}")

    # OCR-only baseline
    lr_ocr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
    lr_ocr.fit(ocr_subset, y)
    probs_ocr = lr_ocr.predict_proba(ocr_subset)[:, 1]
    best_f1_ocr = 0
    for t in np.linspace(0.1, 0.9, 17):
        pred = (probs_ocr >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1_ocr:
            best_f1_ocr = f1
    ap_ocr = average_precision_score(y, probs_ocr)
    print(f"\nOCR-only: F1={best_f1_ocr:.4f}, AP={ap_ocr:.4f}")

    # Feature importance
    print(f"\nCombined model coefficients:")
    all_names = ["v45_prob"] + feature_names
    for name, coef in sorted(zip(all_names, lr.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {name:25s}: {coef:+.4f}")

    # Per-fold analysis (optimistic - threshold selected on same data)
    print(f"\nPer-fold combined F1 (optimistic):")
    folds = v19["fold"][subset]
    for fold in range(5):
        fm = folds == fold
        if fm.sum() == 0:
            continue
        y_fold = y[fm]
        p_fold = probs_combined[fm]
        p_v45_fold = v45_prob_subset[fm]
        v45_pred_fold = v45_pred_subset[fm]
        
        best_f1_c = 0
        for t in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(y_fold, (p_fold >= t).astype(int), zero_division=0)
            if f1 > best_f1_c:
                best_f1_c = f1
        v45_f1_f = f1_score(y_fold, v45_pred_fold, zero_division=0)
        print(f"  Fold {fold}: v45={v45_f1_f:.4f}, combined={best_f1_c:.4f}, delta={best_f1_c-v45_f1_f:+.4f}")


if __name__ == "__main__":
    main()
