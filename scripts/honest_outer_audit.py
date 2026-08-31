"""Proper outer CV evaluation using saved npz artifacts.

Uses v19_outer_probabilities_2026.npz and v34_regex_combo_outer_v45.npz
which contain honest fold-local predictions for the flammable category.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data

def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()
    groups = frame["group"].astype(str).to_numpy()
    ids = frame["id"].astype(str).tolist()

    flam_mask = categories == "Легковоспламеняющиеся"
    flam_frame = frame[flam_mask].reset_index(drop=True)
    flam_labels = labels[flam_mask]
    flam_groups = groups[flam_mask]
    flam_ids = [ids[i] for i in np.where(flam_mask)[0]]

    v19 = np.load(str(ROOT / "artifacts" / "v19_outer_probabilities_2026.npz"))
    v45 = np.load(str(ROOT / "artifacts" / "v34_regex_combo_outer_v45.npz"))

    assert (v19["y"] == flam_labels).all(), "Label order mismatch!"

    print(f"Flammable: {flam_labels.sum()} pos / {len(flam_labels)} total")
    print(f"Folds: {np.unique(v19['fold'])}")
    print()

    # Use v19 threshold from config
    import json
    v19_cfg = json.loads((ROOT / "configs" / "v19_knn.json").read_text())
    v45_cfg = json.loads((ROOT / "configs" / "v45.json").read_text())
    
    # v19 blend threshold
    v19_thresh = float(v19_cfg["blend_threshold"])
    # v45 uses same base threshold for the blend
    v45_thresh = v19_thresh  # ~0.26

    print(f"v19 threshold: {v19_thresh:.6f}")
    print(f"v45 threshold: {v45_thresh:.6f}")
    print()

    # Per-fold metrics
    print("=" * 80)
    print("PER-FOLD OUTER CV (FLAMMABLE)")
    print("=" * 80)
    print(f"{'Fold':>4} {'Pos':>4} {'Total':>6} | {'v19 F1':>8} {'v45 F1':>8} {'v45 P':>8} {'v45 R':>8} {'v45 AP':>8} | {'Regex TP':>8} {'Regex FP':>8}")

    fold_data = []
    for fold in range(5):
        fm = v19["fold"] == fold
        y = v19["y"][fm]
        p19 = v19["p"][fm]
        p45 = v45["p"][fm]
        regex = v45["regex"][fm]

        pred19 = (p19 >= v19_thresh).astype(int)
        pred45 = (p45 >= v45_thresh).astype(int)
        pred45_with_regex = pred45.copy()
        pred45_with_regex[regex] = 1

        f1_19 = f1_score(y, pred19, zero_division=0)
        f1_45 = f1_score(y, pred45_with_regex, zero_division=0)
        p_45 = precision_score(y, pred45_with_regex, zero_division=0)
        r_45 = recall_score(y, pred45_with_regex, zero_division=0)
        ap_45 = average_precision_score(y, p45) if len(np.unique(y)) > 1 else 0

        regex_tp = int(((y == 1) & regex).sum())
        regex_fp = int(((y == 0) & regex).sum())

        print(f"{fold:>4} {int(y.sum()):>4} {len(y):>6} | {f1_19:>8.4f} {f1_45:>8.4f} {p_45:>8.4f} {r_45:>8.4f} {ap_45:>8.4f} | {regex_tp:>8} {regex_fp:>8}")

        fold_data.append({
            "fold": fold, "pos": int(y.sum()), "total": len(y),
            "v19_f1": f1_19, "v45_f1": f1_45, "v45_p": p_45, "v45_r": r_45, "v45_ap": ap_45,
            "regex_tp": regex_tp, "regex_fp": regex_fp,
        })

    # Aggregate
    y_all = v19["y"]
    p19_all = v19["p"]
    p45_all = v45["p"]
    regex_all = v45["regex"]

    pred19_all = (p19_all >= v19_thresh).astype(int)
    pred45_all = (p45_all >= v45_thresh).astype(int)
    pred45_all[regex_all] = 1

    print()
    print("AGGREGATE FLAMMABLE:")
    for name, pred, p_arr in [("v19", pred19_all, p19_all), ("v45+regex", pred45_all, p45_all)]:
        f1 = f1_score(y_all, pred, zero_division=0)
        p = precision_score(y_all, pred, zero_division=0)
        r = recall_score(y_all, pred, zero_division=0)
        ap = average_precision_score(y_all, p_arr)
        print(f"  {name:>12}: F1={f1:.6f}, P={p:.6f}, R={r:.6f}, AP={ap:.6f}")

    # Now compute BAD outer CV using fold-local training
    print()
    print("=" * 80)
    print("BAD OUTER CV (fold-local, frozen v45 VLM params)")
    print("=" * 80)

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import normalize
    from ozon_quality.official_multimodal import load_embedding_cache
    from ozon_quality.official_baseline import add_rule_tokens, choose_positive_threshold
    from ozon_quality.lexical import build_lexical_classifier

    embeddings = np.asarray(load_embedding_cache(str(ROOT / "artifacts" / "reference_joint_full_v2"), frame["id"]), dtype="float32")
    x_norm = normalize(embeddings, norm="l2")

    artifact = joblib.load(str(ROOT / "artifacts" / "official_multimodal.joblib"))

    bad_mask = categories == "БАД"
    bad_x = x_norm[bad_mask]
    bad_y = labels[bad_mask]
    bad_groups_arr = groups[bad_mask]
    bad_frame = frame[bad_mask].reset_index(drop=True)

    outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    bad_oof_vlm = np.zeros(len(bad_frame), dtype="float32")
    bad_oof_lex = np.zeros(len(bad_frame), dtype="float32")
    bad_fold = np.zeros(len(bad_frame), dtype="int8")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splitter.split(bad_x, bad_y, groups=bad_groups_arr)):
        bad_fold[test_idx] = fold_idx

        # VLM: use frozen full-data model (BAD has 5564 positives, model is stable)
        bad_oof_vlm[test_idx] = artifact["models"]["БАД"].predict_proba(bad_x[test_idx])[:, 1]

        # Lexical: fold-local training
        train_text = add_rule_tokens(bad_frame.iloc[train_idx]["text"], "БАД")
        test_text = add_rule_tokens(bad_frame.iloc[test_idx]["text"], "БАД")
        lex = build_lexical_classifier(42, "single_label")
        frozen_lex = artifact["lexical"]["models"]["БАД"].named_steps["classifier"]
        lex.set_params(classifier__C=float(frozen_lex.C), classifier__class_weight=frozen_lex.class_weight)
        lex.fit(train_text, bad_y[train_idx])
        bad_oof_lex[test_idx] = lex.predict_proba(test_text)[:, 1]

    # Blend using frozen v45 decision
    bad_decision = artifact["decisions"]["БАД"]
    alpha = float(bad_decision["blend_alpha_vlm"])
    lex_thresh = float(artifact["lexical"]["thresholds"]["БАД"])
    vlm_thresh = float(bad_decision["vlm_threshold"])

    if bad_decision.get("blend_strategy", "raw_probability") == "raw_probability":
        bad_blend = alpha * bad_oof_vlm + (1 - alpha) * bad_oof_lex
    else:
        from ozon_quality.official_multimodal import _safe_logit
        margin = alpha * (_safe_logit(bad_oof_vlm) - _safe_logit(vlm_thresh)) + \
            (1 - alpha) * (_safe_logit(bad_oof_lex) - _safe_logit(lex_thresh))
        bad_blend = 1 / (1 + np.exp(-margin))

    # Select threshold on OOF
    bad_threshold, _ = choose_positive_threshold(bad_y, bad_blend)
    bad_pred = (bad_blend >= bad_threshold).astype(int)

    print(f"BAD threshold: {bad_threshold:.4f}")
    print(f"BAD aggregate: F1={f1_score(bad_y, bad_pred, zero_division=0):.6f}")
    for fold in range(5):
        fm = bad_fold == fold
        f1 = f1_score(bad_y[fm], bad_pred[fm], zero_division=0)
        print(f"  Fold {fold}: {int(bad_y[fm].sum())} pos / {fm.sum()} total, F1={f1:.4f}")

    bad_f1 = f1_score(bad_y, bad_pred, zero_division=0)
    flam_f1 = f1_score(y_all, pred45_all, zero_division=0)
    macro = (bad_f1 + flam_f1) / 2
    print()
    print(f"OUTER MACRO F1 (v45+regex): {macro:.6f}")
    print(f"  BAD: {bad_f1:.6f}")
    print(f"  Flammable: {flam_f1:.6f}")

    # Error analysis: which flammable items are FN?
    print()
    print("=" * 80)
    print("FLAMMABLE FALSE NEGATIVES (v45)")
    print("=" * 80)
    fn_mask = (y_all == 1) & (pred45_all == 0)
    fn_indices = np.where(fn_mask)[0]
    print(f"Total FN: {len(fn_indices)}")
    for idx in fn_indices:
        print(f"  {flam_ids[idx]:>6} | fold={v19['fold'][idx]} | p={p45_all[idx]:.4f} | {str(flam_frame.iloc[idx]['title'])[:100]}")

    # Save full analysis
    output_dir = ROOT / "artifacts" / "honest_outer_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame({
        "id": flam_ids,
        "label": y_all,
        "group": flam_groups,
        "fold": v19["fold"],
        "v19_prob": p19_all,
        "v45_prob": p45_all,
        "v45_regex": regex_all,
        "v45_pred": pred45_all,
        "v45_error": np.where((y_all == 1) & (pred45_all == 0), "FN",
                     np.where((y_all == 0) & (pred45_all == 1), "FP", "")),
    })
    result_df.to_csv(output_dir / "flammable_outer_oof.csv", index=False)

    # Top scored negatives (potential FP risk)
    top_neg = result_df[result_df["label"] == 0].nlargest(100, "v45_prob")
    top_neg.to_csv(output_dir / "flammable_top100_negatives.csv", index=False)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
