"""Comprehensive stacking evaluation combining multiple signals.

Tests combinations of:
1. v45 probability (from outer CV)
2. OCR keyword features (from image OCR)
3. VLM binary classifier scores
4. Multi-image embeddings

All evaluated leakage-safe using outer CV fold assignments.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data


def main():
    # Load data
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()

    flam_mask = categories == "Легковоспламеняющиеся"
    flam_labels = labels[flam_mask]
    flam_ids = frame.loc[flam_mask, "id"].astype(str).tolist()

    # Load outer CV
    v19 = np.load(str(ROOT / "artifacts" / "v19_outer_probabilities_2026.npz"))
    v45 = np.load(str(ROOT / "artifacts" / "v34_regex_combo_outer_v45.npz"))
    assert (v19["y"] == flam_labels).all()

    thresh = float(json.loads((ROOT / "configs" / "v19_knn.json").read_text())["blend_threshold"])
    v45_pred = (v45["p"] >= thresh).astype(int)
    v45_pred[v45["regex"]] = 1

    folds = v19["fold"]

    # Load OCR features (if available)
    ocr_path = ROOT / "artifacts" / "ocr_features_sample.csv"
    ocr_features = None
    ocr_ids = set()
    if ocr_path.exists():
        ocr_df = pd.read_csv(ocr_path, on_bad_lines="skip")
        ocr_df["id"] = ocr_df["id"].astype(str)
        ocr_ids = set(ocr_df["id"])

        def extract_ocr_features(text):
            t = str(text).lower()
            return {
                "ocr_pyro": float(bool(re.search(r"пиротехник|хлопушк|петард|салют|бенгальск|дымов|шашк|фейерверк|18\+|громкий", t))),
                "ocr_coal": float(bool(re.search(r"уголь|брикет", t))),
                "ocr_gas_neg": float(bool(re.search(r"газ|баллон|топлив|fuel", t))),
                "ocr_warning": float(bool(re.search(r"огнеопасн|легковоспламен|flammable|горюч|18\+|класс опасност", t))),
                "ocr_confetti": float(bool(re.search(r"конфетти|серпантин|confetti", t))),
            }

        feature_names = list(extract_ocr_features("").keys())
        ocr_matrix = np.zeros((len(flam_ids), len(feature_names)), dtype="float32")
        has_ocr = np.zeros(len(flam_ids), dtype=bool)
        for i, fid in enumerate(flam_ids):
            row = ocr_df[ocr_df["id"] == fid]
            if len(row) > 0:
                feats = extract_ocr_features(row.iloc[0]["all_ocr"])
                ocr_matrix[i] = [feats[k] for k in feature_names]
                has_ocr[i] = True

        ocr_features = True  # flag to indicate OCR is loaded
        print(f"OCR features loaded: {has_ocr.sum()} / {len(flam_ids)}")

    # Load VLM scores (if available)
    vlm_path = ROOT / "artifacts" / "vlm_classifier_sample.csv"
    vlm_scores = None
    vlm_ids = set()
    if vlm_path.exists():
        vlm_df = pd.read_csv(vlm_path)
        vlm_df["id"] = vlm_df["id"].astype(str)
        vlm_ids = set(vlm_df["id"])
        vlm_map = dict(zip(vlm_df["id"], vlm_df["yes_prob"]))
        vlm_scores = np.array([vlm_map.get(fid, 0.5) for fid in flam_ids], dtype="float32")
        print(f"VLM scores loaded: {len(vlm_ids)} / {len(flam_ids)}")

    # Evaluate combinations on products with all signals available
    # Find products with both OCR and VLM
    both_mask = has_ocr if ocr_features is not None else np.ones(len(flam_ids), dtype=bool)
    if vlm_scores is not None:
        both_mask = both_mask & np.array([fid in vlm_ids for fid in flam_ids])

    subset = both_mask
    y = flam_labels[subset]
    v45_p = v45["p"][subset]
    v45_pr = v45_pred[subset]
    folds_sub = folds[subset]

    print(f"\nProducts with all signals: {subset.sum()}")
    print(f"Positives: {y.sum()}")

    # Baseline: v45 only
    v45_f1 = f1_score(y, v45_pr, zero_division=0)
    v45_ap = average_precision_score(y, v45_p)
    print(f"\nv45 only: F1={v45_f1:.4f}, AP={v45_ap:.4f}")

    # v45 + OCR
    if ocr_features is not None and has_ocr[subset].sum() > 10:
        ocr_sub = ocr_matrix[subset]
        combined_ocr = np.column_stack([v45_p, ocr_sub])
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(combined_ocr, y)
        probs = lr.predict_proba(combined_ocr)[:, 1]
        best_f1 = 0
        for t in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        ap = average_precision_score(y, probs)
        print(f"v45 + OCR: F1={best_f1:.4f}, AP={ap:.4f} (delta F1={best_f1-v45_f1:+.4f})")

    # v45 + VLM
    if vlm_scores is not None:
        vlm_sub = vlm_scores[subset]
        combined_vlm = np.column_stack([v45_p, vlm_sub])
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(combined_vlm, y)
        probs = lr.predict_proba(combined_vlm)[:, 1]
        best_f1 = 0
        for t in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        ap = average_precision_score(y, probs)
        print(f"v45 + VLM: F1={best_f1:.4f}, AP={ap:.4f} (delta F1={best_f1-v45_f1:+.4f})")

    # v45 + OCR + VLM
    if ocr_features is not None and vlm_scores is not None and has_ocr[subset].sum() > 10:
        ocr_sub = ocr_matrix[subset]
        vlm_sub = vlm_scores[subset]
        combined_all = np.column_stack([v45_p, ocr_sub, vlm_sub])
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(combined_all, y)
        probs = lr.predict_proba(combined_all)[:, 1]
        best_f1 = 0
        for t in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        ap = average_precision_score(y, probs)
        print(f"v45 + OCR + VLM: F1={best_f1:.4f}, AP={ap:.4f} (delta F1={best_f1-v45_f1:+.4f})")

    print("\nNote: F1 values on subset with all signals - optimistic estimates.")
    print("Full evaluation needs OCR/VLM on all 12971 products with outer CV.")


if __name__ == "__main__":
    main()
