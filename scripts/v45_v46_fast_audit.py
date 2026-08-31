"""Fast outer CV audit using v45 frozen parameters.

Instead of re-selecting hyperparameters in each fold, uses the v45 frozen
parameters directly. This is valid for understanding per-fold error patterns
since v45 parameters were already validated via outer CV.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data
from ozon_quality.lexical import build_lexical_classifier
from ozon_quality.official import OFFICIAL_CATEGORIES
from ozon_quality.official_baseline import add_rule_tokens, choose_positive_threshold
from ozon_quality.official_multimodal import (
    _cosine_reference_features,
    _safe_logit,
    load_embedding_cache,
)


def _norm_title(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", value).strip()


# Exact v45 regex rules
V45_RULES = [
    (re.compile(r"\b(?:уголь\s+(?:древесн\w*|каменн\w*)|брикетированн\w*\s+топлив\w*)\b", re.I), 1),
    (re.compile(r"(?:мангал\w*.*одноразов\w*.*угл|одноразов\w*.*мангал\w*.*угл)", re.I), 1),
    (re.compile(r"\bгаз\w*\s+(?:для|к)\s+портативн\w*\s+(?:плит\w*|горел\w*)\b", re.I), 1),
    (re.compile(r"\b(?:дым\w*\s+шашк\w*|шашк\w*\s+дым\w*)\b", re.I), 1),
    (re.compile(r"\bбрикет\w*\s+для\s+грил\w*\b", re.I), 1),
]

# New v46 rules
V46_NEW_RULES = [
    (re.compile(r"\bспичк\w*\s+(?:длительн\w*\s+)?горен\w*.*(?:турист|50\s*мм|5\s*см)\b", re.I), 1),
    (re.compile(r"\b(?:шашк\w*\s+(?:дымов|страйкбол|учебн|имитацион|бел)|гранат\w*\s+страйкбол)\b", re.I), 1),
    (re.compile(r"бенгальск\w*\s+свеч", re.I), 1),
    (re.compile(r"мокс\w*|моксотерап", re.I), 1),
]


def main():
    schema = str(ROOT / "configs" / "ozon_schema.json")
    full_data_path = str(ROOT / "data" / "full_grouped.csv")
    embedding_cache = str(ROOT / "artifacts" / "reference_joint_full_v2")
    # Try multiple possible paths
    candidates = [
        ROOT / "artifacts" / "official_multimodal_v45_standardized_regex_fullrefit.joblib",
        ROOT / "artifacts" / "reproduced_v45_repo_review.joblib",
        ROOT / "artifacts" / "official_multimodal.joblib",
    ]
    artifact_path = None
    for p in candidates:
        if p.exists():
            artifact_path = str(p)
            break
    if artifact_path is None:
        raise FileNotFoundError(f"No v45 artifact found. Tried: {candidates}")

    print("Loading data and artifact...")
    frame, _ = load_data(full_data_path, schema, require_label=True)
    labels = frame["labels"].map(lambda values: int(values[0])).to_numpy()
    categories = frame["category"].to_numpy()
    groups = frame["group"].astype(str).to_numpy()

    embeddings_raw = load_embedding_cache(embedding_cache, frame["id"])
    embeddings = np.asarray(embeddings_raw, dtype="float32")
    x_norm = normalize(embeddings, norm="l2")

    artifact = joblib.load(artifact_path)
    print(f"Loaded artifact with keys: {list(artifact.keys())}")

    n_folds = 5
    outer_splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=2026)
    outer_splits = list(outer_splitter.split(x_norm, labels, groups=groups))

    # Storage for OOF predictions
    oof_v45 = np.zeros(len(frame), dtype="float32")
    oof_v45_pred = np.zeros(len(frame), dtype="int8")
    oof_v46 = np.zeros(len(frame), dtype="float32")
    oof_v46_pred = np.zeros(len(frame), dtype="int8")
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")

        test_frame = frame.iloc[test_idx].reset_index(drop=True)
        test_y = labels[test_idx]
        test_x = x_norm[test_idx]
        test_emb = embeddings[test_idx]

        # Use v45 frozen models to predict on test fold
        # 1. Lexical predictions
        lexical_test = np.zeros(len(test_idx), dtype="float32")
        for category in OFFICIAL_CATEGORIES:
            mask = test_frame["category"].eq(category).to_numpy()
            if not mask.any():
                continue
            text = add_rule_tokens(test_frame.loc[mask, "text"], category)
            lexical_test[mask] = artifact["lexical"]["models"][category].predict_proba(text)[:, 1]

        # 2. VLM predictions
        vlm_test = np.zeros(len(test_idx), dtype="float32")
        for category in OFFICIAL_CATEGORIES:
            mask = test_frame["category"].eq(category).to_numpy()
            if not mask.any():
                continue
            vlm_test[mask] = artifact["models"][category].predict_proba(test_x[mask])[:, 1]

        # 3. Blend
        blend_test = np.zeros(len(test_idx), dtype="float32")
        for category in OFFICIAL_CATEGORIES:
            mask = test_frame["category"].eq(category).to_numpy()
            if not mask.any():
                continue
            decision = artifact["decisions"][category]
            alpha = float(decision["blend_alpha_vlm"])
            strategy = decision.get("blend_strategy", "raw_probability")
            if strategy == "raw_probability":
                blend_test[mask] = alpha * vlm_test[mask] + (1 - alpha) * lexical_test[mask]
            else:
                lex_thresh = float(artifact["lexical"]["thresholds"][category])
                vlm_thresh = float(decision["vlm_threshold"])
                margin = alpha * (_safe_logit(vlm_test[mask]) - _safe_logit(vlm_thresh)) + \
                    (1 - alpha) * (_safe_logit(lexical_test[mask]) - _safe_logit(lex_thresh))
                blend_test[mask] = 1 / (1 + np.exp(-margin))

        # 4. kNN for flammable (fold-local reference bank)
        flam_category = "Легковоспламеняющиеся"
        flam_train_mask = frame.iloc[train_idx]["category"].eq(flam_category).to_numpy()
        flam_test_mask = test_frame["category"].eq(flam_category).to_numpy()

        if flam_train_mask.any() and flam_test_mask.any():
            train_flam_x = x_norm[train_idx][flam_train_mask]
            train_flam_y = labels[train_idx][flam_train_mask]

            # Use v45 kNN parameters
            knn_config = artifact["knn_overrides"][flam_category]
            knn_score_test, _, _ = _cosine_reference_features(
                test_x[flam_test_mask],
                train_flam_x,
                train_flam_y,
                batch_size=512,
            )

            alpha = float(knn_config["knn_alpha"])
            margin_scale = float(knn_config["margin_scale"])
            knn_threshold = float(knn_config["knn_threshold"])
            base_threshold = float(knn_config["base_threshold"])
            blend_threshold = float(knn_config["blend_threshold"])

            margin = (1 - alpha) * (
                _safe_logit(blend_test[flam_test_mask]) - _safe_logit(base_threshold)
            ) + alpha * (knn_score_test - knn_threshold) / margin_scale
            blend_test[flam_test_mask] = 1 / (1 + np.exp(-margin))

            # 5. Standardized head
            head_config = artifact["linear_head_overrides"][flam_category]
            raw_test_flam = test_emb[flam_test_mask]
            head_prob = head_config["model"].predict_proba(
                head_config["scaler"].transform(raw_test_flam)
            )[:, 1]
            alpha_head = float(head_config["alpha_head"])
            blended_logit = (1 - alpha_head) * _safe_logit(blend_test[flam_test_mask]) + \
                alpha_head * _safe_logit(head_prob)
            blend_test[flam_test_mask] = 1 / (1 + np.exp(-blended_logit))

        # 6. Apply thresholds and regex rules
        pred_v45 = np.zeros(len(test_idx), dtype="int8")
        for category in OFFICIAL_CATEGORIES:
            mask = test_frame["category"].eq(category).to_numpy()
            if mask.any():
                threshold = float(artifact["decisions"][category]["blend_threshold"])
                pred_v45[mask] = (blend_test[mask] >= threshold).astype("int8")

        # Apply v45 regex
        for pattern, label in V45_RULES:
            for idx in range(len(test_frame)):
                if test_frame.iloc[idx]["category"] == flam_category:
                    title = _norm_title(test_frame.iloc[idx]["title"])
                    if pattern.search(title):
                        pred_v45[idx] = label
                        blend_test[idx] = float(label)

        # v46 = v45 + new rules
        pred_v46 = pred_v45.copy()
        blend_v46 = blend_test.copy()
        for pattern, label in V46_NEW_RULES:
            for idx in range(len(test_frame)):
                if test_frame.iloc[idx]["category"] == flam_category:
                    title = _norm_title(test_frame.iloc[idx]["title"])
                    if pattern.search(title):
                        pred_v46[idx] = label
                        blend_v46[idx] = float(label)

        # Store OOF
        oof_v45[test_idx] = blend_test
        oof_v45_pred[test_idx] = pred_v45
        oof_v46[test_idx] = blend_v46
        oof_v46_pred[test_idx] = pred_v46

        # Fold metrics
        fold_data = {"fold": fold_idx, "categories": {}}
        for category in OFFICIAL_CATEGORIES:
            mask = test_frame["category"].eq(category).to_numpy()
            if not mask.any():
                continue
            y_true = test_y[mask]
            for version, preds in [("v45", pred_v45), ("v46", pred_v46)]:
                y_pred = preds[mask]
                f1 = f1_score(y_true, y_pred, zero_division=0)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                key = f"{category}_{version}"
                fold_data["categories"][key] = {
                    "f1": f1, "precision": prec, "recall": rec,
                    "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
                    "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
                    "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
                    "positives": int(y_true.sum()),
                }

        # Macro F1
        for version, preds in [("v45", pred_v45), ("v46", pred_v46)]:
            bad_f1 = fold_data["categories"].get(f"БАД_{version}", {}).get("f1", 0)
            flam_f1 = fold_data["categories"].get(f"Легковоспламеняющиеся_{version}", {}).get("f1", 0)
            fold_data[f"macro_f1_{version}"] = (bad_f1 + flam_f1) / 2

        fold_metrics.append(fold_data)
        print(f"  v45 Macro: {fold_data['macro_f1_v45']:.4f}, v46 Macro: {fold_data['macro_f1_v46']:.4f}")
        for cat in OFFICIAL_CATEGORIES:
            v45_data = fold_data["categories"].get(f"{cat}_v45", {})
            v46_data = fold_data["categories"].get(f"{cat}_v46", {})
            print(f"    {cat}: v45 F1={v45_data.get('f1', 0):.4f} (TP={v45_data.get('tp', 0)}, FP={v45_data.get('fp', 0)}, FN={v45_data.get('fn', 0)})")
            print(f"           v46 F1={v46_data.get('f1', 0):.4f} (TP={v46_data.get('tp', 0)}, FP={v46_data.get('fp', 0)}, FN={v46_data.get('fn', 0)})")

    # Aggregate
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS:")
    for version, preds in [("v45", oof_v45_pred), ("v46", oof_v46_pred)]:
        print(f"\n  {version}:")
        for category in OFFICIAL_CATEGORIES:
            mask = categories == category
            f1 = f1_score(labels[mask], preds[mask], zero_division=0)
            prec = precision_score(labels[mask], preds[mask], zero_division=0)
            rec = recall_score(labels[mask], preds[mask], zero_division=0)
            print(f"    {category}: F1={f1:.6f}, P={prec:.6f}, R={rec:.6f}")
        macro = np.mean([
            f1_score(labels[categories == c], preds[categories == c], zero_division=0)
            for c in OFFICIAL_CATEGORIES
        ])
        print(f"    Macro F1: {macro:.6f}")

    # Save
    output_dir = ROOT / "artifacts" / "v45_v46_outer_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_df = frame[["id", "title", "description", "category"]].copy()
    oof_df["label"] = labels
    oof_df["group"] = groups
    oof_df["v45_probability"] = oof_v45
    oof_df["v45_prediction"] = oof_v45_pred
    oof_df["v46_probability"] = oof_v46
    oof_df["v46_prediction"] = oof_v46_pred
    oof_df["v45_error"] = np.where(
        (labels == 1) & (oof_v45_pred == 0), "FN",
        np.where((labels == 0) & (oof_v45_pred == 1), "FP", ""),
    )
    oof_df["v46_error"] = np.where(
        (labels == 1) & (oof_v46_pred == 0), "FN",
        np.where((labels == 0) & (oof_v46_pred == 1), "FP", ""),
    )
    oof_df.to_csv(output_dir / "outer_oof_predictions.csv", index=False)

    # Error analysis
    flam_mask = categories == "Легковоспламеняющиеся"
    flam_df = oof_df[flam_mask].copy()
    flam_df.to_csv(output_dir / "flammable_oof.csv", index=False)

    # v45 FN (missed positives)
    v45_fn = flam_df[(flam_df["label"] == 1) & (flam_df["v45_prediction"] == 0)]
    v45_fn.to_csv(output_dir / "v45_flammable_fn.csv", index=False)
    print(f"\nv45 flammable FN: {len(v45_fn)}")

    # v46 newly caught
    v46_new = flam_df[(flam_df["v45_prediction"] == 0) & (flam_df["v46_prediction"] == 1)]
    v46_new.to_csv(output_dir / "v46_newly_caught.csv", index=False)
    print(f"v46 newly caught: {len(v46_new)} (TP: {(v46_new['label']==1).sum()}, FP: {(v46_new['label']==0).sum()})")

    # Top scored negatives
    top_neg = flam_df[flam_df["label"] == 0].nlargest(200, "v46_probability")
    top_neg.to_csv(output_dir / "flammable_top_negatives.csv", index=False)

    with open(output_dir / "fold_metrics.json", "w") as f:
        json.dump(fold_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
