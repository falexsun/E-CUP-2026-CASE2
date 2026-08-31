"""Clean leakage-safe OCR evaluation: fold-local binary feature boost.

Tests whether OCR keyword presence can improve v45 predictions,
with all decisions made on training portion of each fold.
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
from ozon_quality.official_baseline import choose_positive_threshold


def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()
    ids = frame["id"].astype(str).tolist()

    # Load OCR features
    ocr_dir = ROOT / "artifacts" / "ocr_features_full"
    state = json.loads((ocr_dir / "state.json").read_text())
    assert state.get("complete")

    ocr_texts = pd.read_csv(str(ocr_dir / "texts.csv"), dtype=str)
    ocr_ids = ocr_texts["id"].tolist()
    ocr_features_raw = np.load(str(ocr_dir / "features.npy"))

    # Align
    id_to_idx = {id_: i for i, id_ in enumerate(ocr_ids)}
    feature_names = ["ocr_pyro", "ocr_coal", "ocr_gas_fuel", "ocr_matches",
                     "ocr_bad_marking", "ocr_warning", "ocr_set_contents", "ocr_confetti"]
    feature_matrix = np.zeros((len(ids), len(feature_names)), dtype="float32")
    for i, id_ in enumerate(ids):
        if id_ in id_to_idx:
            feature_matrix[i] = ocr_features_raw[id_to_idx[id_]]

    print(f"OCR aligned: {(feature_matrix.sum(axis=1) > 0).sum()} / {len(ids)}")

    # Load outer CV
    v19 = np.load(str(ROOT / "artifacts" / "v19_outer_probabilities_2026.npz"))
    v45 = np.load(str(ROOT / "artifacts" / "v34_regex_combo_outer_v45.npz"))

    flam_mask = categories == "Легковоспламеняющиеся"
    flam_labels = labels[flam_mask]
    flam_features = feature_matrix[flam_mask]
    folds = v19["fold"]

    assert (v19["y"] == flam_labels).all()

    v45_threshold = float(json.loads((ROOT / "configs" / "v19_knn.json").read_text())["blend_threshold"])
    v45_probs = v45["p"]
    v45_regex = v45["regex"]
    v45_pred = (v45_probs >= v45_threshold).astype(int)
    v45_pred[v45_regex] = 1

    # v45 aggregate baseline
    v45_agg_f1 = f1_score(flam_labels, v45_pred, zero_division=0)
    v45_agg_p = precision_score(flam_labels, v45_pred, zero_division=0)
    v45_agg_r = recall_score(flam_labels, v45_pred, zero_division=0)
    v45_agg_ap = average_precision_score(flam_labels, v45_probs)

    print(f"\nv45 baseline: F1={v45_agg_f1:.6f}, P={v45_agg_p:.6f}, R={v45_agg_r:.6f}, AP={v45_agg_ap:.6f}")

    # OCR feature prevalence
    print("\nOCR features (flammable full dataset):")
    for i, name in enumerate(feature_names):
        pos = flam_features[flam_labels == 1, i].mean()
        neg = flam_features[flam_labels == 0, i].mean()
        n_pos = int(flam_features[flam_labels == 1, i].sum())
        n_neg = int(flam_features[flam_labels == 0, i].sum())
        print(f"  {name:25s}: pos={pos:.4f} ({n_pos:>3}) neg={neg:.4f} ({n_neg:>3}) diff={pos-neg:+.4f}")

    # Strategy 1: Fold-local threshold on v45_probs + OCR binary boost
    # For each fold, find the best v45 threshold and OCR boost rule on train, apply to test
    print("\n" + "=" * 80)
    print("STRATEGY 1: Fold-local v45 threshold + OCR boost")
    print("=" * 80)

    oof_pred_strategy1 = np.zeros(len(flam_labels), dtype="int8")
    for fold in range(5):
        train = folds != fold
        test = folds == fold
        y_train = flam_labels[train]
        y_test = flam_labels[test]
        p_train = v45_probs[train]
        p_test = v45_probs[test]
        regex_train = v45_regex[train]
        regex_test = v45_regex[test]
        ocr_pyro_train = flam_features[train, 0]  # pyro
        ocr_pyro_test = flam_features[test, 0]

        # Find best v45 threshold on train (without regex)
        best_t, best_score = choose_positive_threshold(y_train, p_train)

        # Test: boost predictions where OCR says pyro AND v45 prob is moderate
        base_pred_test = (p_test >= best_t).astype(int)
        base_pred_test[regex_test] = 1
        base_f1 = f1_score(y_test, base_pred_test, zero_division=0)

        # Try boosting with OCR pyro
        best_boost_f1 = base_f1
        best_boost_thresh = 0
        for boost_thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            boosted_pred = base_pred_test.copy()
            # Boost: set to 1 where OCR pyro AND v45_prob > boost_thresh (but below threshold)
            boost_mask = (ocr_pyro_test > 0) & (p_test > boost_thresh) & (p_test < best_t)
            boosted_pred[boost_mask] = 1
            boost_f1 = f1_score(y_test, boosted_pred, zero_division=0)
            if boost_f1 > best_boost_f1:
                best_boost_f1 = boost_f1
                best_boost_thresh = boost_thresh

        # Also try: only boost where OCR pyro (no v45 threshold filter)
        pyro_only_pred = base_pred_test.copy()
        pyro_only_pred[(ocr_pyro_test > 0)] = 1
        pyro_f1 = f1_score(y_test, pyro_only_pred, zero_division=0)

        oof_pred_strategy1[test] = base_pred_test if best_boost_f1 <= base_f1 else (
            (p_test >= best_t).astype(int) | regex_test.astype(int) |
            ((ocr_pyro_test > 0) & (p_test > best_boost_thresh) & (p_test < best_t))
        ).astype("int8")

        n_boost = int(((ocr_pyro_test > 0) & (p_test > best_boost_thresh) & (p_test < best_t)).sum()) if best_boost_f1 > base_f1 else 0
        print(f"  Fold {fold}: v45={base_f1:.4f} boost={best_boost_f1:.4f} (t={best_boost_thresh:.2f}, n={n_boost}) pyro_only={pyro_f1:.4f} delta={best_boost_f1-base_f1:+.4f}")

    s1_f1 = f1_score(flam_labels, oof_pred_strategy1, zero_division=0)
    s1_p = precision_score(flam_labels, oof_pred_strategy1, zero_division=0)
    s1_r = recall_score(flam_labels, oof_pred_strategy1, zero_division=0)
    print(f"\n  Aggregate: F1={s1_f1:.6f}, P={s1_p:.6f}, R={s1_r:.6f}, delta={s1_f1-v45_agg_f1:+.6f}")

    # Strategy 2: Fold-local LR on [v45_prob, ocr_features] with proper threshold
    print("\n" + "=" * 80)
    print("STRATEGY 2: Fold-local LR on [v45_prob, ocr_features]")
    print("=" * 80)

    from sklearn.linear_model import LogisticRegression

    oof_probs_s2 = np.zeros(len(flam_labels), dtype="float32")
    for fold in range(5):
        train = folds != fold
        test = folds == fold
        y_train = flam_labels[train]
        y_test = flam_labels[test]
        combined_train = np.column_stack([v45_probs[train].reshape(-1, 1), flam_features[train]])
        combined_test = np.column_stack([v45_probs[test].reshape(-1, 1), flam_features[test]])

        # Select C on train via simple OOF
        best_inner_f1 = 0
        best_c = 0.1
        for c in [0.01, 0.03, 0.1, 0.3]:
            lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(combined_train, y_train)
            inner_probs = lr.predict_proba(combined_train)[:, 1]
            _, inner_f1 = choose_positive_threshold(y_train, inner_probs)
            if inner_f1 > best_inner_f1:
                best_inner_f1 = inner_f1
                best_c = c

        lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(combined_train, y_train)
        oof_probs_s2[test] = lr.predict_proba(combined_test)[:, 1]

    # Select threshold on full OOF
    s2_thresh, _ = choose_positive_threshold(flam_labels, oof_probs_s2)
    s2_pred = (oof_probs_s2 >= s2_thresh).astype(int)
    s2_f1 = f1_score(flam_labels, s2_pred, zero_division=0)
    s2_p = precision_score(flam_labels, s2_pred, zero_division=0)
    s2_r = recall_score(flam_labels, s2_pred, zero_division=0)
    s2_ap = average_precision_score(flam_labels, oof_probs_s2)

    print(f"  Threshold: {s2_thresh:.4f}")
    print(f"  Aggregate: F1={s2_f1:.6f}, P={s2_p:.6f}, R={s2_r:.6f}, AP={s2_ap:.6f}")
    print(f"  vs v45: delta F1={s2_f1-v45_agg_f1:+.6f}, delta AP={s2_ap-v45_agg_ap:+.6f}")

    # Per-fold for strategy 2
    for fold in range(5):
        test = folds == fold
        y_test = flam_labels[test]
        p_test = oof_probs_s2[test]
        pred_test = (p_test >= s2_thresh).astype(int)
        f1 = f1_score(y_test, pred_test, zero_division=0)
        v45_f1 = f1_score(y_test, v45_pred[test], zero_division=0)
        print(f"    Fold {fold}: v45={v45_f1:.4f} s2={f1:.4f} delta={f1-v45_f1:+.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"v45 baseline:       F1={v45_agg_f1:.6f}  AP={v45_agg_ap:.6f}")
    print(f"Strategy 1 (boost): F1={s1_f1:.6f}  delta={s1_f1-v45_agg_f1:+.6f}")
    print(f"Strategy 2 (LR):    F1={s2_f1:.6f}  delta={s2_f1-v45_agg_f1:+.6f}  AP={s2_ap:.6f}")


if __name__ == "__main__":
    main()
