"""Evaluate multi-image embeddings vs single-image on outer CV.

Compares:
1. Single-image (main image) embeddings - baseline v45
2. Mean-pooled multi-image embeddings
3. Max-pooled multi-image embeddings

Uses fold-local LogisticRegression with frozen v45 hyperparameters.
"""

import json
import sys
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
from ozon_quality.official import OFFICIAL_CATEGORIES
from ozon_quality.official_baseline import choose_positive_threshold
from ozon_quality.official_multimodal import load_embedding_cache, _safe_logit, _cosine_reference_features


def evaluate_embedding_set(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    category_mask: np.ndarray,
    category: str,
    n_folds: int = 5,
    seed: int = 2026,
) -> dict:
    """Evaluate embeddings using fold-local LR with v45 frozen params."""
    x = normalize(embeddings[category_mask].astype("float32"), norm="l2")
    y = labels[category_mask]
    g = groups[category_mask]

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(x), dtype="float32")
    fold_assignments = np.zeros(len(x), dtype="int8")

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=g)):
        fold_assignments[test_idx] = fold_idx

        # Use v45 frozen params: C=0.03, balanced
        lr = LogisticRegression(C=0.03, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
        lr.fit(x[train_idx], y[train_idx])
        oof_probs[test_idx] = lr.predict_proba(x[test_idx])[:, 1]

    # Select threshold on OOF
    threshold, _ = choose_positive_threshold(y, oof_probs)
    oof_preds = (oof_probs >= threshold).astype(int)

    f1 = f1_score(y, oof_preds, zero_division=0)
    p = precision_score(y, oof_preds, zero_division=0)
    r = recall_score(y, oof_preds, zero_division=0)
    ap = average_precision_score(y, oof_probs)

    # Per-fold
    fold_metrics = []
    for fold in range(n_folds):
        fm = fold_assignments == fold
        if fm.sum() == 0:
            continue
        fold_f1 = f1_score(y[fm], oof_preds[fm], zero_division=0)
        fold_p = precision_score(y[fm], oof_preds[fm], zero_division=0)
        fold_r = recall_score(y[fm], oof_preds[fm], zero_division=0)
        fold_metrics.append({"fold": fold, "f1": fold_f1, "p": fold_p, "r": fold_r, "pos": int(y[fm].sum())})

    return {
        "f1": f1, "p": p, "r": r, "ap": ap, "threshold": threshold,
        "fold_metrics": fold_metrics,
    }


def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()
    groups = frame["group"].astype(str).to_numpy()

    # Load single-image embeddings (baseline)
    single_emb = np.asarray(
        load_embedding_cache(str(ROOT / "artifacts" / "reference_joint_full_v2"), frame["id"]),
        dtype="float32"
    )
    print(f"Single-image embeddings: {single_emb.shape}")

    # Load multi-image embeddings
    multi_dir = ROOT / "artifacts" / "multi_image_embeddings"
    if not (multi_dir / "mean_embeddings.npy").exists():
        print("Multi-image embeddings not found. Run extract_multi_image_embeddings.py first.")
        return

    mean_emb = np.load(str(multi_dir / "mean_embeddings.npy")).astype("float32")
    max_emb = np.load(str(multi_dir / "max_embeddings.npy")).astype("float32")
    print(f"Mean multi-image embeddings: {mean_emb.shape}")
    print(f"Max multi-image embeddings: {max_emb.shape}")

    # Verify alignment
    multi_ids = pd.read_csv(str(multi_dir / "ids.csv"), dtype=str)["id"].tolist()
    frame_ids = frame["id"].astype(str).tolist()
    assert multi_ids == frame_ids, "ID mismatch!"

    # Evaluate for flammable category
    flam_mask = categories == "Легковоспламеняющиеся"

    print("\n" + "=" * 70)
    print("FLAMMABLE CATEGORY EVALUATION")
    print("=" * 70)

    for name, emb in [("single_image", single_emb), ("mean_multi", mean_emb), ("max_multi", max_emb)]:
        result = evaluate_embedding_set(emb, labels, groups, flam_mask, "Легковоспламеняющиеся")
        print(f"\n{name}:")
        print(f"  F1={result['f1']:.4f}, P={result['p']:.4f}, R={result['r']:.4f}, AP={result['ap']:.4f}")
        print(f"  Threshold: {result['threshold']:.4f}")
        for fm in result["fold_metrics"]:
            print(f"    Fold {fm['fold']}: F1={fm['f1']:.4f}, pos={fm['pos']}")

    # Also test concatenated single + multi
    concat_emb = np.concatenate([single_emb, mean_emb], axis=1)
    print(f"\nConcatenated single+mean: {concat_emb.shape}")
    result = evaluate_embedding_set(concat_emb, labels, groups, flam_mask, "Легковоспламеняющиеся")
    print(f"  F1={result['f1']:.4f}, P={result['p']:.4f}, R={result['r']:.4f}, AP={result['ap']:.4f}")


if __name__ == "__main__":
    main()
