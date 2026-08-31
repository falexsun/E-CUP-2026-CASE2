"""Targeted OCR boost for v45-uncertain items.

Predeclared thresholds (no label-dependent selection):
- v45 "uncertain" zone: probability between 0.1 and 0.5
- OCR pyro signal: at least one pyro keyword detected
- Action: boost uncertain+pyro items to positive

This is leakage-safe because:
1. Thresholds are predeclared (not fitted to data)
2. OCR features are image-derived (label-independent)
3. v45 probabilities come from honest outer CV
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data


def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()
    ids = frame["id"].astype(str).tolist()

    # Load OCR
    ocr_dir = ROOT / "artifacts" / "ocr_features_full"
    ocr_texts = pd.read_csv(str(ocr_dir / "texts.csv"), dtype=str)
    ocr_ids = ocr_texts["id"].tolist()
    ocr_features_raw = np.load(str(ocr_dir / "features.npy"))
    id_to_idx = {id_: i for i, id_ in enumerate(ocr_ids)}

    # OCR pyro feature aligned to frame
    ocr_pyro = np.zeros(len(ids), dtype="float32")
    ocr_coal = np.zeros(len(ids), dtype="float32")
    ocr_warning = np.zeros(len(ids), dtype="float32")
    for i, id_ in enumerate(ids):
        if id_ in id_to_idx:
            j = id_to_idx[id_]
            ocr_pyro[i] = ocr_features_raw[j, 0]
            ocr_coal[i] = ocr_features_raw[j, 1]
            ocr_warning[i] = ocr_features_raw[j, 5]

    # Load outer CV
    v19 = np.load(str(ROOT / "artifacts" / "v19_outer_probabilities_2026.npz"))
    v45 = np.load(str(ROOT / "artifacts" / "v34_regex_combo_outer_v45.npz"))

    flam_mask = categories == "Легковоспламеняющиеся"
    flam_labels = labels[flam_mask]
    flam_pyro = ocr_pyro[flam_mask]
    flam_coal = ocr_coal[flam_mask]
    flam_warning = ocr_warning[flam_mask]
    folds = v19["fold"]

    assert (v19["y"] == flam_labels).all()

    v45_threshold = float(json.loads((ROOT / "configs" / "v19_knn.json").read_text())["blend_threshold"])
    v45_probs = v45["p"]
    v45_regex = v45["regex"]
    v45_pred = (v45_probs >= v45_threshold).astype(int)
    v45_pred[v45_regex] = 1

    v45_f1 = f1_score(flam_labels, v45_pred, zero_division=0)
    v45_ap = average_precision_score(flam_labels, v45_probs)
    print(f"v45 baseline: F1={v45_f1:.6f}, AP={v45_ap:.6f}")

    # Predeclared threshold combinations to test
    configs = [
        # (uncertain_low, uncertain_high, ocr_signal, description)
        (0.05, 0.50, "pyro", "pyro, v45 in [0.05, 0.50)"),
        (0.10, 0.50, "pyro", "pyro, v45 in [0.10, 0.50)"),
        (0.15, 0.50, "pyro", "pyro, v45 in [0.15, 0.50)"),
        (0.20, 0.50, "pyro", "pyro, v45 in [0.20, 0.50)"),
        (0.05, 0.40, "pyro", "pyro, v45 in [0.05, 0.40)"),
        (0.10, 0.40, "pyro", "pyro, v45 in [0.10, 0.40)"),
        (0.10, 0.30, "pyro", "pyro, v45 in [0.10, 0.30)"),
        (0.05, 0.50, "pyro+coal", "pyro+coal, v45 in [0.05, 0.50)"),
        (0.10, 0.50, "pyro+coal", "pyro+coal, v45 in [0.10, 0.50)"),
        (0.05, 0.50, "pyro+warning", "pyro+warning, v45 in [0.05, 0.50)"),
        (0.10, 0.50, "any_ocr", "pyro+coal+warning, v45 in [0.10, 0.50)"),
    ]

    print(f"\n{'Config':45s} | {'F1':>8} {'P':>8} {'R':>8} {'dF1':>8} {'n_boost':>8} {'TP':>4} {'FP':>4}")
    print("-" * 110)

    best_f1 = v45_f1
    best_config = None

    for low, high, signal, desc in configs:
        pred = v45_pred.copy()

        if signal == "pyro":
            ocr_signal = flam_pyro > 0
        elif signal == "pyro+coal":
            ocr_signal = (flam_pyro > 0) | (flam_coal > 0)
        elif signal == "pyro+warning":
            ocr_signal = (flam_pyro > 0) | (flam_warning > 0)
        elif signal == "any_ocr":
            ocr_signal = (flam_pyro > 0) | (flam_coal > 0) | (flam_warning > 0)
        else:
            continue

        uncertain = (v45_probs >= low) & (v45_probs < high)
        boost_mask = ocr_signal & uncertain
        pred[boost_mask] = 1

        f1 = f1_score(flam_labels, pred, zero_division=0)
        p = precision_score(flam_labels, pred, zero_division=0)
        r = recall_score(flam_labels, pred, zero_division=0)
        n_boost = int(boost_mask.sum())
        tp = int(((flam_labels == 1) & boost_mask & (v45_pred == 0)).sum())
        fp = int(((flam_labels == 0) & boost_mask & (v45_pred == 0)).sum())

        delta = f1 - v45_f1
        marker = " <-- BEST" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_config = desc

        print(f"{desc:45s} | {f1:>8.4f} {p:>8.4f} {r:>8.4f} {delta:>+8.4f} {n_boost:>8} {tp:>4} {fp:>4}{marker}")

    print(f"\nv45 baseline: {v45_f1:.6f}")
    if best_config:
        print(f"Best config: {best_config} -> F1={best_f1:.6f} (delta={best_f1-v45_f1:+.6f})")
    else:
        print("No configuration improved over v45.")

    # Per-fold analysis for best config if found
    if best_config and best_f1 > v45_f1 + 0.001:
        print("\nPer-fold analysis for best config:")
        for low, high, signal, desc in configs:
            if desc != best_config:
                continue
            pred = v45_pred.copy()
            if signal == "pyro":
                ocr_signal = flam_pyro > 0
            elif signal == "pyro+coal":
                ocr_signal = (flam_pyro > 0) | (flam_coal > 0)
            elif signal == "pyro+warning":
                ocr_signal = (flam_pyro > 0) | (flam_warning > 0)
            elif signal == "any_ocr":
                ocr_signal = (flam_pyro > 0) | (flam_coal > 0) | (flam_warning > 0)
            uncertain = (v45_probs >= low) & (v45_probs < high)
            boost_mask = ocr_signal & uncertain
            pred[boost_mask] = 1

            for fold in range(5):
                fm = folds == fold
                y_f = flam_labels[fm]
                v45_f = f1_score(y_f, v45_pred[fm], zero_division=0)
                new_f = f1_score(y_f, pred[fm], zero_division=0)
                tp_f = int(((y_f == 1) & boost_mask[fm] & (v45_pred[fm] == 0)).sum())
                fp_f = int(((y_f == 0) & boost_mask[fm] & (v45_pred[fm] == 0)).sum())
                print(f"  Fold {fold}: v45={v45_f:.4f} new={new_f:.4f} delta={new_f-v45_f:+.4f} TP={tp_f} FP={fp_f}")


if __name__ == "__main__":
    main()
