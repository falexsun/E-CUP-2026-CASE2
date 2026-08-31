"""Test multi-image embeddings as additional signal on top of v45.

Uses the outer CV fold assignments from v19 npz.
Tests whether multi-image embeddings add independent signal
beyond the v45 probability scores.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data


def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()

    flam_mask = categories == "Легковоспламеняющиеся"
    flam_labels = labels[flam_mask]

    v19 = np.load(str(ROOT / "artifacts" / "v19_outer_probabilities_2026.npz"))
    v45 = np.load(str(ROOT / "artifacts" / "v34_regex_combo_outer_v45.npz"))
    assert (v19["y"] == flam_labels).all()

    thresh = float(json.loads((ROOT / "configs" / "v19_knn.json").read_text())["blend_threshold"])
    v45_pred = (v45["p"] >= thresh).astype(int)
    v45_pred[v45["regex"]] = 1

    folds = v19["fold"]

    # Load multi-image embeddings
    multi_dir = ROOT / "artifacts" / "multi_image_embeddings"
    mean_emb = np.load(str(multi_dir / "mean_embeddings.npy")).astype("float32")
    max_emb = np.load(str(multi_dir / "max_embeddings.npy")).astype("float32")

    # Load single-image embeddings for comparison
    from ozon_quality.official_multimodal import load_embedding_cache
    single_emb = np.asarray(
        load_embedding_cache(str(ROOT / "artifacts" / "reference_joint_full_v2"), frame["id"]),
        dtype="float32"
    )

    # Normalize
    single_norm = normalize(single_emb[flam_mask], norm="l2")
    mean_norm = normalize(mean_emb[flam_mask], norm="l2")
    max_norm = normalize(max_emb[flam_mask], norm="l2")

    y = flam_labels
    v45_p = v45["p"]

    # v45 baseline
    v45_f1 = f1_score(y, v45_pred, zero_division=0)
    v45_ap = average_precision_score(y, v45_p)
    print(f"v45 baseline: F1={v45_f1:.4f}, AP={v45_ap:.4f}")

    # Test: LR on [v45_prob, single_embedding] features
    # Use v45 probability + first N principal components of embeddings
    for name, emb in [("single", single_norm), ("mean_multi", mean_norm), ("max_multi", max_norm)]:
        # Use only top 32 PCA components to avoid overfitting
        from sklearn.decomposition import PCA
        pca = PCA(n_components=32, random_state=42)
        emb_pca = pca.fit_transform(emb)

        combined = np.column_stack([v45_p.reshape(-1, 1), emb_pca])

        # Evaluate with fold-local LR
        oof_probs = np.zeros(len(y), dtype="float32")
        for fold in range(5):
            train = folds != fold
            test = folds == fold
            lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(combined[train], y[train])
            oof_probs[test] = lr.predict_proba(combined[test])[:, 1]

        best_f1 = 0
        best_t = 0.5
        for t in np.linspace(0.1, 0.9, 17):
            pred = (oof_probs >= t).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        ap = average_precision_score(y, oof_probs)
        print(f"v45 + {name:12s} (PCA-32): F1={best_f1:.4f} (t={best_t:.2f}), AP={ap:.4f}, delta={best_f1-v45_f1:+.4f}")

    # Also test: simple addition of multi-image score
    for name, emb in [("mean_multi", mean_norm), ("max_multi", max_norm)]:
        # Fold-local LR on multi-image alone
        multi_probs = np.zeros(len(y), dtype="float32")
        for fold in range(5):
            train = folds != fold
            test = folds == fold
            lr = LogisticRegression(C=0.03, class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(emb[train], y[train])
            multi_probs[test] = lr.predict_proba(emb[test])[:, 1]

        # Blend v45 + multi
        for alpha in [0.05, 0.1, 0.15, 0.2]:
            blended = alpha * multi_probs + (1 - alpha) * v45_p
            best_f1 = 0
            for t in np.linspace(0.1, 0.9, 17):
                pred = (blended >= t).astype(int)
                f1 = f1_score(y, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
            print(f"v45 + alpha={alpha:.2f}*{name}: F1={best_f1:.4f}, delta={best_f1-v45_f1:+.4f}")


if __name__ == "__main__":
    main()
