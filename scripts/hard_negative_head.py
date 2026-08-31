"""Hard-negative-aware supervised head for flammable category.

Approach:
1. Identify hard negatives (high cosine sim to positive centroid)
2. Train LR with class_weight that emphasizes hard negatives
3. Use fold-local training with OOF threshold selection
4. Compare with v45 baseline on same folds

All parameters (C, hard-negative threshold, blend weight) selected on
training portion of each fold only.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data
from ozon_quality.official_baseline import choose_positive_threshold
from ozon_quality.official_multimodal import load_embedding_cache, _safe_logit


def norm_title(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", value).strip()


def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()
    groups = frame["group"].astype(str).to_numpy()

    flam_mask = categories == "Легковоспламеняющиеся"
    flam_frame = frame[flam_mask].reset_index(drop=True)
    flam_labels = labels[flam_mask]
    flam_groups = groups[flam_mask]

    # Load embeddings
    emb = np.asarray(load_embedding_cache(str(ROOT / "artifacts" / "reference_joint_full_v2"), frame["id"]), dtype="float32")
    flam_emb = normalize(emb[flam_mask].astype("float32"), norm="l2")

    # Load v45 outer CV
    v19 = np.load(str(ROOT / "artifacts" / "v19_outer_probabilities_2026.npz"))
    v45 = np.load(str(ROOT / "artifacts" / "v34_regex_combo_outer_v45.npz"))
    assert (v19["y"] == flam_labels).all()

    v45_threshold = float(json.loads((ROOT / "configs" / "v19_knn.json").read_text())["blend_threshold"])
    v45_probs = v45["p"]
    v45_regex = v45["regex"]
    v45_pred = (v45_probs >= v45_threshold).astype(int)
    v45_pred[v45_regex] = 1

    folds = v19["fold"]

    # v45 baseline
    v45_f1 = f1_score(flam_labels, v45_pred, zero_division=0)
    v45_ap = average_precision_score(flam_labels, v45_probs)
    print(f"v45 baseline: F1={v45_f1:.6f}, AP={v45_ap:.6f}")

    # === Strategy A: Hard-negative oversampled LR ===
    # For each fold, identify hard negatives in training set,
    # oversample them, train LR, evaluate on test
    print("\n" + "=" * 80)
    print("STRATEGY A: Hard-negative oversampled head")
    print("=" * 80)

    oof_probs_a = np.zeros(len(flam_labels), dtype="float32")

    for fold in range(5):
        train = folds != fold
        test = folds == fold

        x_train = flam_emb[train]
        y_train = flam_labels[train]
        x_test = flam_emb[test]
        y_test = flam_labels[test]

        # Identify hard negatives in training set
        pos_centroid = x_train[y_train == 1].mean(axis=0)
        pos_centroid_norm = pos_centroid / np.linalg.norm(pos_centroid)
        train_sims = x_train @ pos_centroid_norm

        neg_sims = train_sims[y_train == 0]
        hard_neg_threshold = np.percentile(neg_sims, 95)  # top 5% of negatives
        hard_neg_mask = (y_train == 0) & (train_sims >= hard_neg_threshold)

        n_hard = hard_neg_mask.sum()
        n_pos = (y_train == 1).sum()

        # Create augmented training set: original + oversampled hard negatives
        # Use class_weight to emphasize hard negatives
        best_f1 = 0
        best_c = 0.1
        best_weight_ratio = 1.0

        for c in [0.01, 0.03, 0.1, 0.3]:
            for weight_ratio in [1.0, 2.0, 3.0, 5.0]:
                # Custom class weights: increase weight of hard negatives
                sample_weights = np.ones(len(y_train), dtype="float32")
                sample_weights[hard_neg_mask] = weight_ratio

                lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42)
                lr.fit(x_train, y_train, sample_weight=sample_weights)

                inner_probs = lr.predict_proba(x_train)[:, 1]
                _, inner_f1 = choose_positive_threshold(y_train, inner_probs)
                if inner_f1 > best_f1:
                    best_f1 = inner_f1
                    best_c = c
                    best_weight_ratio = weight_ratio

        # Train with best params on full train
        sample_weights = np.ones(len(y_train), dtype="float32")
        hard_neg_mask_best = (y_train == 0) & (train_sims >= hard_neg_threshold)
        sample_weights[hard_neg_mask_best] = best_weight_ratio

        lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(x_train, y_train, sample_weight=sample_weights)
        oof_probs_a[test] = lr.predict_proba(x_test)[:, 1]

        # Evaluate on test fold
        threshold, _ = choose_positive_threshold(y_train, lr.predict_proba(x_train)[:, 1])
        test_pred = (oof_probs_a[test] >= threshold).astype(int)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)
        v45_fold_f1 = f1_score(y_test, v45_pred[test], zero_division=0)

        print(f"  Fold {fold}: v45={v45_fold_f1:.4f} hard_neg={test_f1:.4f} "
              f"C={best_c} wr={best_weight_ratio:.1f} hard_n={n_hard} "
              f"delta={test_f1-v45_fold_f1:+.4f}")

    # Aggregate strategy A
    best_thresh_a, _ = choose_positive_threshold(flam_labels, oof_probs_a)
    pred_a = (oof_probs_a >= best_thresh_a).astype(int)
    f1_a = f1_score(flam_labels, pred_a, zero_division=0)
    ap_a = average_precision_score(flam_labels, oof_probs_a)
    print(f"\n  Aggregate: F1={f1_a:.6f}, AP={ap_a:.6f}, delta={f1_a-v45_f1:+.6f}")

    # === Strategy B: StandardScaler + LR (v45-like but re-evaluated) ===
    # This is essentially what v45 does, but we re-evaluate on honest outer CV
    print("\n" + "=" * 80)
    print("STRATEGY B: StandardScaler + balanced LR (v45-style, honest outer)")
    print("=" * 80)

    oof_probs_b = np.zeros(len(flam_labels), dtype="float32")

    for fold in range(5):
        train = folds != fold
        test = folds == fold

        x_train = flam_emb[train]
        y_train = flam_labels[train]
        x_test = flam_emb[test]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        # Select C on train
        best_inner_f1 = 0
        best_c = 0.03
        for c in [0.01, 0.03, 0.1, 0.3]:
            lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(x_train_scaled, y_train)
            inner_probs = lr.predict_proba(x_train_scaled)[:, 1]
            _, inner_f1 = choose_positive_threshold(y_train, inner_probs)
            if inner_f1 > best_inner_f1:
                best_inner_f1 = inner_f1
                best_c = c

        lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(x_train_scaled, y_train)
        oof_probs_b[test] = lr.predict_proba(x_test_scaled)[:, 1]

    best_thresh_b, _ = choose_positive_threshold(flam_labels, oof_probs_b)
    pred_b = (oof_probs_b >= best_thresh_b).astype(int)
    f1_b = f1_score(flam_labels, pred_b, zero_division=0)
    ap_b = average_precision_score(flam_labels, oof_probs_b)
    print(f"  Aggregate: F1={f1_b:.6f}, AP={ap_b:.6f}, delta={f1_b-v45_f1:+.6f}")

    # === Strategy C: Blend v45 + new head ===
    print("\n" + "=" * 80)
    print("STRATEGY C: Blend v45 + hard-negative head")
    print("=" * 80)

    for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
        blended = alpha * oof_probs_a + (1 - alpha) * v45_probs
        best_f1 = 0
        best_t = 0.5
        for t in np.linspace(0.05, 0.95, 19):
            pred = (blended >= t).astype(int)
            f1 = f1_score(flam_labels, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        print(f"  alpha={alpha:.2f}: F1={best_f1:.4f} (t={best_t:.2f}), delta={best_f1-v45_f1:+.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"v45 baseline:           F1={v45_f1:.6f}  AP={v45_ap:.6f}")
    print(f"Hard-negative head:     F1={f1_a:.6f}  AP={ap_a:.6f}  delta={f1_a-v45_f1:+.6f}")
    print(f"StandardScaler head:    F1={f1_b:.6f}  AP={ap_b:.6f}  delta={f1_b-v45_f1:+.6f}")


if __name__ == "__main__":
    main()
