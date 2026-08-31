"""Outer 5-fold group CV audit for v45 architecture.

Reproduces the full v45 pipeline (lexical + VLM blend + kNN + standardized head + regex)
fold-locally and collects per-fold metrics, FP/FN details, and error families.
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
from ozon_quality.official import OFFICIAL_CATEGORIES, official_category_f1
from ozon_quality.official_baseline import add_rule_tokens, choose_positive_threshold
from ozon_quality.official_multimodal import (
    _cosine_reference_features,
    _safe_logit,
    load_embedding_cache,
)


def _normalize_lookup_title(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", value).strip()


# Regex rules from v45 config
V45_RULES = [
    (re.compile(r"(?:^|\s)(уголь|брикет)\S*\s+(?:для\s+)?(?:барбекю|гриля|мангала|шашлык)", re.I), 1),
    (re.compile(r"(?:^|\s)(?:топливн\S*\s+)?брикет\S*\s+(?:для\s+)?(?:розжиг|мангал|барбекю|гриль)", re.I), 1),
    (re.compile(r"(?:одноразов\S*\s+)?мангал\S*\s+.*(?:уголь|брикет|розжиг)", re.I), 1),
    (re.compile(r"(?:газ|баллон\S*)\s+.*(?:для\s+)?(?:портативн\S*\s+)?(?:плит|горелк)", re.I), 1),
    (re.compile(r"(?:дымов\S*\s+)?(?:шашк|бомб)\S*\s+(?:для\s+)?(?:сигнализаци|фейерверк|пейнтбол)", re.I), 1),
]


def main():
    schema = str(ROOT / "configs" / "ozon_schema.json")
    full_data_path = str(ROOT / "data" / "full_grouped.csv")
    embedding_cache = str(ROOT / "artifacts" / "reference_joint_full_v2")

    print("Loading data...")
    frame, _ = load_data(full_data_path, schema, require_label=True)
    labels = frame["labels"].map(lambda values: int(values[0])).to_numpy()
    categories = frame["category"].to_numpy()
    groups = frame["group"].astype(str).to_numpy()
    ids = frame["id"].astype(str).tolist()

    print(f"Loaded {len(frame)} rows, {len(set(groups))} groups")
    print(f"Embedding cache: {embedding_cache}")

    embeddings_raw = load_embedding_cache(embedding_cache, frame["id"])
    embeddings = np.asarray(embeddings_raw, dtype="float32")
    x_norm = normalize(embeddings, norm="l2")

    n_folds = 5
    outer_splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=2026)

    # Category-stratified split: use a combined stratification key
    strat_key = (frame["category"] + "_" + frame["labels"].map(lambda v: str(v[0]))).to_numpy()
    outer_splits = list(outer_splitter.split(x_norm, labels, groups=groups))

    all_oof = np.zeros(len(frame), dtype="float32")
    all_oof_preds = np.zeros(len(frame), dtype="int8")
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")

        train_frame = frame.iloc[train_idx].reset_index(drop=True)
        test_frame = frame.iloc[test_idx].reset_index(drop=True)
        train_y = labels[train_idx]
        test_y = labels[test_idx]
        train_groups = groups[train_idx]
        train_x = x_norm[train_idx]
        test_x = x_norm[test_idx]

        # --- 1. Lexical branch (fold-local) ---
        lexical_oof = np.zeros(len(train_idx), dtype="float32")
        lexical_test = np.zeros(len(test_idx), dtype="float32")
        lexical_thresholds = {}

        for cat_idx, category in enumerate(OFFICIAL_CATEGORIES):
            train_mask = train_frame["category"].eq(category).to_numpy()
            test_mask = test_frame["category"].eq(category).to_numpy()
            if not train_mask.any():
                continue

            text_train = add_rule_tokens(train_frame.loc[train_mask, "text"], category)
            text_test = add_rule_tokens(test_frame.loc[test_mask, "text"], category)
            y_cat = train_y[train_mask]
            groups_cat = train_groups[train_mask]

            # Inner OOF for lexical
            inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + cat_idx)
            inner_splits = list(inner_splitter.split(text_train, y_cat, groups=groups_cat))

            best_lex = None
            for c_val in (0.01, 0.03, 0.1, 0.3, 1.0):
                for weight in (None, "balanced"):
                    candidate_oof = np.zeros(len(text_train), dtype="float32")
                    for inner_train, inner_test in inner_splits:
                        lex = build_lexical_classifier(42 + cat_idx, "single_label")
                        lex.set_params(classifier__C=c_val, classifier__class_weight=weight)
                        lex.fit(text_train.iloc[inner_train], y_cat[inner_train])
                        candidate_oof[inner_test] = lex.predict_proba(text_train.iloc[inner_test])[:, 1]
                    threshold, score = choose_positive_threshold(y_cat, candidate_oof)
                    if best_lex is None or score > best_lex[0]:
                        best_lex = (score, c_val, weight, candidate_oof, threshold)

            _, best_c, best_w, cat_lex_oof, lex_threshold = best_lex
            lexical_thresholds[category] = lex_threshold

            # Refit on full train fold
            lex_final = build_lexical_classifier(42 + cat_idx, "single_label")
            lex_final.set_params(classifier__C=best_c, classifier__class_weight=best_w)
            lex_final.fit(text_train, y_cat)
            lexical_oof[train_mask] = cat_lex_oof
            lexical_test[test_mask] = lex_final.predict_proba(text_test)[:, 1]

        # --- 2. VLM branch (fold-local) ---
        vlm_oof = np.zeros(len(train_idx), dtype="float32")
        vlm_test = np.zeros(len(test_idx), dtype="float32")
        vlm_thresholds = {}
        vlm_models = {}

        for cat_idx, category in enumerate(OFFICIAL_CATEGORIES):
            train_mask = train_frame["category"].eq(category).to_numpy()
            test_mask = test_frame["category"].eq(category).to_numpy()
            if not train_mask.any():
                continue

            x_cat = train_x[train_mask]
            y_cat = train_y[train_mask]
            groups_cat = train_groups[train_mask]

            inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + cat_idx)
            inner_splits = list(inner_splitter.split(x_cat, y_cat, groups=groups_cat))

            best_vlm = None
            for c_val in (0.01, 0.03, 0.1, 0.3, 1.0):
                for weight in (None, "balanced"):
                    candidate_oof = np.zeros(len(x_cat), dtype="float32")
                    for inner_train, inner_test in inner_splits:
                        lr = LogisticRegression(
                            C=c_val, class_weight=weight, max_iter=1000, solver="lbfgs",
                            random_state=42 + cat_idx,
                        )
                        lr.fit(x_cat[inner_train], y_cat[inner_train])
                        candidate_oof[inner_test] = lr.predict_proba(x_cat[inner_test])[:, 1]
                    threshold, score = choose_positive_threshold(y_cat, candidate_oof)
                    if best_vlm is None or score > best_vlm[0]:
                        best_vlm = (score, c_val, weight, candidate_oof, threshold)

            _, best_c, best_w, cat_vlm_oof, vlm_threshold = best_vlm
            vlm_thresholds[category] = vlm_threshold

            # Refit
            lr_final = LogisticRegression(
                C=best_c, class_weight=best_w, max_iter=1000, solver="lbfgs",
                random_state=42 + cat_idx,
            ).fit(x_cat, y_cat)
            vlm_models[category] = lr_final
            vlm_oof[train_mask] = cat_vlm_oof
            vlm_test[test_mask] = lr_final.predict_proba(test_x[test_mask])[:, 1]

        # --- 3. Blend (fold-local, selected on train OOF) ---
        blend_oof = np.zeros(len(train_idx), dtype="float32")
        blend_test = np.zeros(len(test_idx), dtype="float32")
        blend_thresholds = {}
        blend_decisions = {}

        for category in OFFICIAL_CATEGORIES:
            train_mask = train_frame["category"].eq(category).to_numpy()
            test_mask = test_frame["category"].eq(category).to_numpy()
            if not train_mask.any():
                continue

            y_cat = train_y[train_mask]
            lex_thresh = lexical_thresholds[category]
            vlm_thresh = vlm_thresholds[category]

            lex_margin = _safe_logit(lexical_oof[train_mask]) - _safe_logit(lex_thresh)
            vlm_margin = _safe_logit(vlm_oof[train_mask]) - _safe_logit(vlm_thresh)

            best_blend = None
            for alpha in np.linspace(0, 1, 21):
                raw_prob = alpha * vlm_oof[train_mask] + (1 - alpha) * lexical_oof[train_mask]
                margin = alpha * vlm_margin + (1 - alpha) * lex_margin
                margin_prob = 1 / (1 + np.exp(-margin))
                for strategy, prob in [("raw_probability", raw_prob), ("threshold_normalized_logit", margin_prob)]:
                    threshold, score = choose_positive_threshold(y_cat, prob)
                    if best_blend is None or score > best_blend[0]:
                        best_blend = (score, strategy, alpha, threshold, prob)

            _, strategy, alpha, blend_thresh, cat_blend_oof = best_blend
            blend_thresholds[category] = blend_thresh
            blend_decisions[category] = {
                "blend_strategy": strategy,
                "blend_alpha_vlm": alpha,
                "blend_threshold": blend_thresh,
                "vlm_threshold": vlm_thresh,
                "lexical_threshold": lex_thresh,
            }

            blend_oof[train_mask] = cat_blend_oof
            if strategy == "raw_probability":
                blend_test[test_mask] = alpha * vlm_test[test_mask] + (1 - alpha) * lexical_test[test_mask]
            else:
                valid_margin = alpha * (_safe_logit(vlm_test[test_mask]) - _safe_logit(vlm_thresh)) + \
                    (1 - alpha) * (_safe_logit(lexical_test[test_mask]) - _safe_logit(lex_thresh))
                blend_test[test_mask] = 1 / (1 + np.exp(-valid_margin))

        # --- 4. kNN for flammable (fold-local reference bank) ---
        flammable_category = "Легковоспламеняющиеся"
        flammable_train_mask = train_frame["category"].eq(flammable_category).to_numpy()
        flammable_test_mask = test_frame["category"].eq(flammable_category).to_numpy()

        if flammable_train_mask.any() and flammable_test_mask.any():
            ref_embeddings = train_x[flammable_train_mask].astype("float32")
            ref_labels = train_y[flammable_train_mask]

            # Compute kNN score for test
            knn_score_test, _, _ = _cosine_reference_features(
                test_x[flammable_test_mask], ref_embeddings, ref_labels, batch_size=512,
            )
            knn_score_train, _, _ = _cosine_reference_features(
                train_x[flammable_train_mask], ref_embeddings, ref_labels, batch_size=512,
            )

            # Select knn parameters on train OOF
            best_knn = None
            for knn_alpha in (0.2, 0.25, 0.3, 0.35, 0.4):
                for margin_scale in (0.005, 0.01, 0.02):
                    for knn_thresh in np.percentile(knn_score_train, [5, 10, 25, 50]):
                        margin = (1 - knn_alpha) * (
                            _safe_logit(blend_oof[flammable_train_mask]) - _safe_logit(blend_thresholds[flammable_category])
                        ) + knn_alpha * (knn_score_train - knn_thresh) / margin_scale
                        prob = 1 / (1 + np.exp(-margin))
                        threshold, score = choose_positive_threshold(train_y[flammable_train_mask], prob)
                        if best_knn is None or score > best_knn[0]:
                            best_knn = (score, knn_alpha, knn_thresh, margin_scale, threshold, prob)

            _, knn_alpha, knn_thresh, margin_scale, knn_blend_thresh, knn_train_prob = best_knn

            # Apply to test
            margin_test = (1 - knn_alpha) * (
                _safe_logit(blend_test[flammable_test_mask]) - _safe_logit(blend_thresholds[flammable_category])
            ) + knn_alpha * (knn_score_test - knn_thresh) / margin_scale
            knn_test_prob = 1 / (1 + np.exp(-margin_test))
            blend_test[flammable_test_mask] = knn_test_prob
            blend_oof[flammable_train_mask] = knn_train_prob
            blend_thresholds[flammable_category] = knn_blend_thresh

            print(f"  kNN: alpha={knn_alpha:.2f}, scale={margin_scale:.3f}, thresh={knn_thresh:.6f}")

        # --- 5. Standardized head for flammable (fold-local) ---
        if flammable_train_mask.any() and flammable_test_mask.any():
            raw_train_flam = embeddings[train_idx][flammable_train_mask]
            raw_test_flam = embeddings[test_idx][flammable_test_mask]
            y_flam = train_y[flammable_train_mask]

            scaler = StandardScaler()
            x_scaled_train = scaler.fit_transform(raw_train_flam)
            x_scaled_test = scaler.transform(raw_test_flam)

            best_head = None
            for c_val in (0.01, 0.03, 0.1, 0.3):
                head_oof = np.zeros(len(x_scaled_train), dtype="float32")
                inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                flam_groups = train_groups[flammable_train_mask]
                for inner_train, inner_test in inner_splitter.split(
                    x_scaled_train, y_flam, groups=flam_groups
                ):
                    head = LogisticRegression(
                        C=c_val, class_weight="balanced", max_iter=1000, solver="lbfgs",
                        random_state=42,
                    )
                    head.fit(x_scaled_train[inner_train], y_flam[inner_train])
                    head_oof[inner_test] = head.predict_proba(x_scaled_train[inner_test])[:, 1]
                _, score = choose_positive_threshold(y_flam, head_oof)
                if best_head is None or score > best_head[0]:
                    best_head = (score, c_val, head_oof)

            _, best_c_head, head_oof_train = best_head
            head_final = LogisticRegression(
                C=best_c_head, class_weight="balanced", max_iter=1000, solver="lbfgs",
                random_state=42,
            ).fit(x_scaled_train, y_flam)
            head_prob_test = head_final.predict_proba(x_scaled_test)[:, 1]

            # Logit blend
            alpha_head = 0.25
            blend_thresh_flam = blend_thresholds[flammable_category]

            blended_logit_train = (1 - alpha_head) * _safe_logit(blend_oof[flammable_train_mask]) + \
                alpha_head * _safe_logit(head_oof_train)
            blend_oof[flammable_train_mask] = 1 / (1 + np.exp(-blended_logit_train))

            blended_logit_test = (1 - alpha_head) * _safe_logit(blend_test[flammable_test_mask]) + \
                alpha_head * _safe_logit(head_prob_test)
            blend_test[flammable_test_mask] = 1 / (1 + np.exp(-blended_logit_test))

            # Re-select threshold
            _, new_thresh = choose_positive_threshold(train_y[flammable_train_mask], blend_oof[flammable_train_mask])
            blend_thresholds[flammable_category] = new_thresh

        # --- 6. Apply thresholds ---
        test_preds = np.zeros(len(test_idx), dtype="int8")
        for category in OFFICIAL_CATEGORIES:
            mask = test_frame["category"].eq(category).to_numpy()
            if mask.any():
                test_preds[mask] = (blend_test[mask] >= blend_thresholds[category]).astype("int8")

        # --- 7. Regex rules ---
        for cat, rules in [("Легковоспламеняющиеся", V45_RULES)]:
            mask = test_frame["category"].eq(cat).to_numpy()
            indices = np.flatnonzero(mask)
            for idx in indices:
                title = _normalize_lookup_title(test_frame.iloc[idx]["title"])
                for pattern, label in rules:
                    if pattern.search(title):
                        test_preds[idx] = label
                        blend_test[idx] = float(label)
                        break

        # Store OOF
        all_oof[test_idx] = blend_test
        all_oof_preds[test_idx] = test_preds

        # --- Fold metrics ---
        fold_cat_metrics = {}
        for category in OFFICIAL_CATEGORIES:
            mask = test_frame["category"].eq(category).to_numpy()
            if not mask.any():
                continue
            y_true = test_y[mask]
            y_pred = test_preds[mask]
            y_prob = blend_test[mask]
            f1 = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
            fold_cat_metrics[category] = {
                "f1": f1, "precision": prec, "recall": rec, "ap": ap,
                "positives": int(y_true.sum()), "total": len(y_true),
                "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
                "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
                "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
                "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
            }
            print(f"  {category}: F1={f1:.4f}, P={prec:.4f}, R={rec:.4f}, AP={ap:.4f}, "
                  f"pos={int(y_true.sum())}, TP={fold_cat_metrics[category]['tp']}, "
                  f"FP={fold_cat_metrics[category]['fp']}, FN={fold_cat_metrics[category]['fn']}")

        bad_f1 = fold_cat_metrics.get("БАД", {}).get("f1", 0)
        flam_f1 = fold_cat_metrics.get("Легковоспламеняющиеся", {}).get("f1", 0)
        macro = (bad_f1 + flam_f1) / 2
        fold_metrics.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "macro_f1": macro,
            "categories": fold_cat_metrics,
        })
        print(f"  Macro F1: {macro:.4f}")

    # --- Aggregate metrics ---
    print(f"\n{'='*60}")
    print("AGGREGATE OUTER CV RESULTS:")

    for category in OFFICIAL_CATEGORIES:
        mask = frame["category"].eq(category).to_numpy()
        y_true = labels[mask]
        y_pred = all_oof_preds[mask]
        y_prob = all_oof[mask]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        print(f"  {category}: F1={f1:.6f}, P={prec:.6f}, R={rec:.6f}, AP={ap:.6f}")

    macro_f1 = np.mean([
        f1_score(labels[frame["category"].eq(c)], all_oof_preds[frame["category"].eq(c)], zero_division=0)
        for c in OFFICIAL_CATEGORIES
    ])
    print(f"  Macro F1: {macro_f1:.6f}")

    # --- Save detailed error analysis ---
    output_dir = ROOT / "artifacts" / "v45_outer_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save OOF predictions
    oof_df = frame[["id", "title", "description", "category"]].copy()
    oof_df["label"] = labels
    oof_df["group"] = groups
    oof_df["oof_probability"] = all_oof
    oof_df["oof_prediction"] = all_oof_preds
    oof_df["error"] = np.where(
        (labels == 1) & (all_oof_preds == 0), "FN",
        np.where((labels == 0) & (all_oof_preds == 1), "FP", ""),
    )
    oof_df.to_csv(output_dir / "outer_oof_predictions.csv", index=False)

    # Save fold metrics
    with open(output_dir / "fold_metrics.json", "w") as f:
        json.dump(fold_metrics, f, ensure_ascii=False, indent=2)

    # Save error details
    errors = oof_df[oof_df["error"].ne("")]
    errors.to_csv(output_dir / "outer_errors.csv", index=False)

    # Per-error-family analysis for flammable
    flammable_errors = oof_df[
        (oof_df["category"] == "Легковоспламеняющиеся") & (oof_df["error"].ne(""))
    ].copy()
    flammable_errors.to_csv(output_dir / "flammable_errors.csv", index=False)

    # Top-scored negatives
    flammable_neg = oof_df[
        (oof_df["category"] == "Легковоспламеняющиеся") & (oof_df["label"] == 0)
    ].nlargest(200, "oof_probability")
    flammable_neg.to_csv(output_dir / "flammable_top_negatives.csv", index=False)

    print(f"\nSaved audit to {output_dir}")
    print(f"Total errors: {len(errors)}")
    print(f"Flammable errors: {len(flammable_errors)}")

    # Save summary
    summary = {
        "aggregate_macro_f1": macro_f1,
        "per_fold_macro_f1": [f["macro_f1"] for f in fold_metrics],
        "per_category_f1": {
            c: f1_score(labels[frame["category"].eq(c)], all_oof_preds[frame["category"].eq(c)], zero_division=0)
            for c in OFFICIAL_CATEGORIES
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
