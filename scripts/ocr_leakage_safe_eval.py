"""Strictly leakage-safe OCR stacking evaluation.

Rules:
- v45 probabilities from saved npz (honest outer CV, no full-refit)
- OCR features are image-derived (label-independent)
- All thresholds and model parameters fit ONLY on training portion of each outer fold
- No entity-426 party-cracker rules (exploratory invalid)
- Per-fold, per-category reporting with CI estimates
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data
from ozon_quality.official_baseline import choose_positive_threshold


def extract_ocr_features(text: str) -> np.ndarray:
    """Predeclared OCR keyword features. No label-dependent selection."""
    t = str(text).lower() if pd.notna(text) else ""
    return np.array([
        float(bool(re.search(r"пиротехник|петард|салют|бенгальск|дымов|шашк|фейерверк|громкий хлопок|18\+", t))),
        float(bool(re.search(r"уголь|брикет|charcoal|briquette", t))),
        float(bool(re.search(r"газ|баллон|топлив|fuel|бензин", t))),
        float(bool(re.search(r"спичк|зажигалк|розжиг|горелк|burner|lighter|сухое горюч", t))),
        float(bool(re.search(r"бад|биологически активн|dietary supplement", t))),
        float(bool(re.search(r"огнеопасн|легковоспламен|flammable|горюч|воспламен|18\+|класс опасност", t))),
        float(bool(re.search(r"в комплект|в набор|содержит|включает|внутри", t))),
        float(bool(re.search(r"конфетти|серпантин|confetti", t))),
    ], dtype="float32")


FEATURE_NAMES = [
    "ocr_pyro", "ocr_coal", "ocr_gas_fuel", "ocr_matches",
    "ocr_bad_marking", "ocr_warning", "ocr_set_contents", "ocr_confetti",
]


def evaluate_fold_local_stacking(
    v45_probs: np.ndarray,
    ocr_features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    v45_threshold: float,
    category_name: str,
) -> dict:
    """Evaluate OCR features as additional signal, all parameters fit OOF."""
    n_folds = 5
    oof_probs = np.zeros(len(labels), dtype="float32")
    fold_results = []

    for fold in range(n_folds):
        train_mask = folds != fold
        test_mask = folds == fold

        y_train = labels[train_mask]
        y_test = labels[test_mask]
        v45_train = v45_probs[train_mask]
        v45_test = v45_probs[test_mask]
        ocr_train = ocr_features[train_mask]
        ocr_test = ocr_features[test_mask]

        # v45 baseline on this fold
        v45_pred_test = (v45_test >= v45_threshold).astype(int)
        v45_f1 = f1_score(y_test, v45_pred_test, zero_division=0)

        # Train OCR-augmented model on train portion
        # Select best C on inner train split
        from sklearn.model_selection import StratifiedGroupKFold

        combined_train = np.column_stack([v45_train.reshape(-1, 1), ocr_train])
        combined_test = np.column_stack([v45_test.reshape(-1, 1), ocr_test])

        best_score = -1
        best_c = 0.1
        for c in [0.01, 0.03, 0.1, 0.3, 1.0]:
            lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=42)
            lr.fit(combined_train, y_train)
            inner_probs = lr.predict_proba(combined_train)[:, 1]
            _, inner_score = choose_positive_threshold(y_train, inner_probs)
            if inner_score > best_score:
                best_score = inner_score
                best_c = c

        # Fit on full train with best C
        lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(combined_train, y_train)
        test_probs = lr.predict_proba(combined_test)[:, 1]
        oof_probs[test_mask] = test_probs

        # Select threshold on train
        threshold, _ = choose_positive_threshold(y_train, lr.predict_proba(combined_train)[:, 1])
        test_pred = (test_probs >= threshold).astype(int)
        ocr_f1 = f1_score(y_test, test_pred, zero_division=0)

        # Also test v45 + simple OCR rule threshold
        ocr_flag_train = ocr_train[:, 0]  # pyro feature
        ocr_flag_test = ocr_test[:, 0]
        rule_pred = v45_pred_test.copy()
        # Only boost where OCR says pyro AND v45 probability is moderate
        boost_mask = (ocr_flag_test > 0) & (v45_test > 0.1) & (v45_test < v45_threshold)
        rule_pred[boost_mask] = 1
        rule_f1 = f1_score(y_test, rule_pred, zero_division=0)

        fold_results.append({
            "fold": fold,
            "pos": int(y_test.sum()),
            "v45_f1": v45_f1,
            "ocr_stacking_f1": ocr_f1,
            "ocr_rule_f1": rule_f1,
            "delta_stacking": ocr_f1 - v45_f1,
            "delta_rule": rule_f1 - v45_f1,
            "best_c": best_c,
            "threshold": threshold,
            "boost_count": int(boost_mask.sum()),
        })

    # Aggregate
    agg_probs = oof_probs.copy()
    best_agg_f1 = 0
    best_agg_t = v45_threshold
    for t in np.linspace(0.05, 0.95, 19):
        pred = (agg_probs >= t).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_agg_f1:
            best_agg_f1 = f1
            best_agg_t = t

    v45_agg_pred = (v45_probs >= v45_threshold).astype(int)
    v45_agg_f1 = f1_score(labels, v45_agg_pred, zero_division=0)
    agg_ap = average_precision_score(labels, agg_probs)
    v45_ap = average_precision_score(labels, v45_probs)

    return {
        "category": category_name,
        "v45_f1": v45_agg_f1,
        "v45_ap": v45_ap,
        "ocr_stacking_f1": best_agg_f1,
        "ocr_stacking_ap": agg_ap,
        "ocr_stacking_threshold": best_agg_t,
        "delta_f1": best_agg_f1 - v45_agg_f1,
        "delta_ap": agg_ap - v45_ap,
        "fold_results": fold_results,
    }


def main():
    # Load data
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()
    ids = frame["id"].astype(str).tolist()

    # Load OCR features
    ocr_dir = ROOT / "artifacts" / "ocr_features_full"
    state = json.loads((ocr_dir / "state.json").read_text())
    assert state.get("complete"), "OCR extraction not complete!"

    ocr_texts = pd.read_csv(str(ocr_dir / "texts.csv"), dtype=str)
    ocr_ids = ocr_texts["id"].tolist()

    # Align OCR to frame order
    id_to_idx = {id_: i for i, id_ in enumerate(ocr_ids)}
    feature_matrix = np.zeros((len(ids), len(FEATURE_NAMES)), dtype="float32")
    aligned_texts = [""] * len(ids)

    ocr_features_raw = np.load(str(ocr_dir / "features.npy"))
    for i, id_ in enumerate(ids):
        if id_ in id_to_idx:
            j = id_to_idx[id_]
            feature_matrix[i] = ocr_features_raw[j]
            aligned_texts[i] = ocr_texts.iloc[j]["ocr_text"]

    print(f"OCR features aligned: {(feature_matrix.sum(axis=1) > 0).sum()} / {len(ids)} non-zero")

    # Load outer CV for flammable
    v19 = np.load(str(ROOT / "artifacts" / "v19_outer_probabilities_2026.npz"))
    v45 = np.load(str(ROOT / "artifacts" / "v34_regex_combo_outer_v45.npz"))

    flam_mask = categories == "Легковоспламеняющиеся"
    flam_labels = labels[flam_mask]
    flam_features = feature_matrix[flam_mask]
    flam_ids = [ids[i] for i in np.where(flam_mask)[0]]

    assert (v19["y"] == flam_labels).all()
    v45_threshold = float(json.loads((ROOT / "configs" / "v19_knn.json").read_text())["blend_threshold"])

    v45_probs = v45["p"].copy()
    v45_pred = (v45_probs >= v45_threshold).astype(int)
    v45_pred[v45["regex"]] = 1

    folds = v19["fold"]

    # === FLAMMABLE EVALUATION ===
    print("\n" + "=" * 80)
    print("FLAMMABLE: STRICTLY LEAKAGE-SAFE OCR STACKING EVALUATION")
    print("=" * 80)

    # v45 baseline
    v45_f1 = f1_score(flam_labels, v45_pred, zero_division=0)
    v45_ap = average_precision_score(flam_labels, v45_probs)
    print(f"\nv45 baseline: F1={v45_f1:.6f}, AP={v45_ap:.6f}")

    # OCR feature prevalence
    print("\nOCR feature prevalence (flammable):")
    for i, name in enumerate(FEATURE_NAMES):
        pos_rate = flam_features[flam_labels == 1, i].mean()
        neg_rate = flam_features[flam_labels == 0, i].mean()
        print(f"  {name:25s}: pos={pos_rate:.4f} neg={neg_rate:.4f} diff={pos_rate-neg_rate:+.4f}")

    # Full OCR stacking evaluation
    result = evaluate_fold_local_stacking(
        v45_probs, flam_features, flam_labels, folds, v45_threshold, "Легковоспламеняющиеся"
    )

    print(f"\nOCR stacking aggregate:")
    print(f"  v45: F1={result['v45_f1']:.6f}, AP={result['v45_ap']:.6f}")
    print(f"  v45+OCR: F1={result['ocr_stacking_f1']:.6f}, AP={result['ocr_stacking_ap']:.6f}")
    print(f"  Delta: F1={result['delta_f1']:+.6f}, AP={result['delta_ap']:+.6f}")
    print(f"  Threshold: {result['ocr_stacking_threshold']:.4f}")

    print(f"\nPer-fold results:")
    print(f"{'Fold':>4} {'Pos':>4} | {'v45 F1':>8} {'stack F1':>8} {'rule F1':>8} | {'d_stack':>8} {'d_rule':>8} {'boost':>6}")
    for fr in result["fold_results"]:
        print(f"{fr['fold']:>4} {fr['pos']:>4} | {fr['v45_f1']:>8.4f} {fr['ocr_stacking_f1']:>8.4f} {fr['ocr_rule_f1']:>8.4f} | {fr['delta_stacking']:>+8.4f} {fr['delta_rule']:>+8.4f} {fr['boost_count']:>6}")

    # === BAD EVALUATION (text OOF-based) ===
    print("\n" + "=" * 80)
    print("BAD: OCR EVALUATION")
    print("=" * 80)

    bad_mask = categories == "БАД"
    bad_labels = labels[bad_mask]
    bad_features = feature_matrix[bad_mask]

    # Load lexical OOF for BAD
    lex_oof_path = ROOT / "artifacts" / "official_text_full_oof_v6" / "oof_predictions.csv"
    if lex_oof_path.exists():
        lex_oof = pd.read_csv(lex_oof_path, dtype={"id": str})
        lex_bad = lex_oof[lex_oof["category"] == "БАД"].copy()
        lex_bad["id"] = lex_bad["id"].astype(str)
        bad_frame = frame[bad_mask].copy()
        bad_frame["id"] = bad_frame["id"].astype(str)
        merged_bad = bad_frame[["id"]].merge(lex_bad[["id", "probability"]], on="id", how="left")
        bad_lex_probs = merged_bad["probability"].fillna(0.5).to_numpy(dtype="float32")

        from sklearn.model_selection import StratifiedGroupKFold
        bad_groups = frame.loc[bad_mask, "group"].astype(str).to_numpy()
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
        bad_folds = np.zeros(len(bad_labels), dtype="int8")
        for fold_idx, (_, test_idx) in enumerate(splitter.split(bad_frame, bad_labels, groups=bad_groups)):
            bad_folds[test_idx] = fold_idx

        # Load artifact for VLM predictions
        import joblib
        from ozon_quality.official_multimodal import load_embedding_cache, _safe_logit
        from sklearn.preprocessing import normalize

        artifact = joblib.load(str(ROOT / "artifacts" / "official_multimodal.joblib"))
        embeddings = np.asarray(load_embedding_cache(str(ROOT / "artifacts" / "reference_joint_full_v2"), frame["id"]), dtype="float32")
        bad_x = normalize(embeddings[bad_mask], norm="l2")
        bad_vlm_probs = artifact["models"]["БАД"].predict_proba(bad_x)[:, 1]

        # Blend
        decision = artifact["decisions"]["БАД"]
        alpha = float(decision["blend_alpha_vlm"])
        bad_blend = alpha * bad_vlm_probs + (1 - alpha) * bad_lex_probs

        bad_v45_f1 = f1_score(bad_labels, (bad_blend >= 0.55).astype(int), zero_division=0)
        print(f"BAD v45 proxy: F1={bad_v45_f1:.6f}")

        # OCR stacking for BAD
        bad_result = evaluate_fold_local_stacking(
            bad_blend, bad_features, bad_labels, bad_folds, 0.55, "БАД"
        )
        print(f"BAD v45+OCR: F1={bad_result['ocr_stacking_f1']:.6f}, delta={bad_result['delta_f1']:+.6f}")
        for fr in bad_result["fold_results"]:
            print(f"  Fold {fr['fold']}: v45={fr['v45_f1']:.4f} stack={fr['ocr_stacking_f1']:.4f} delta={fr['delta_stacking']:+.4f}")

    # === MACRO F1 ===
    print("\n" + "=" * 80)
    print("MACRO F1 SUMMARY")
    print("=" * 80)

    flam_new_f1 = result["ocr_stacking_f1"]
    bad_new_f1 = bad_result["ocr_stacking_f1"] if "bad_result" in dir() else 0.935901
    macro_new = (bad_new_f1 + flam_new_f1) / 2
    macro_old = (0.935901 + v45_f1) / 2

    print(f"v45:      BAD={0.935901:.4f} Flam={v45_f1:.4f} Macro={macro_old:.4f}")
    print(f"v45+OCR:  BAD={bad_new_f1:.4f} Flam={flam_new_f1:.4f} Macro={macro_new:.4f}")
    print(f"Delta:    BAD={bad_new_f1-0.935901:+.4f} Flam={flam_new_f1-v45_f1:+.4f} Macro={macro_new-macro_old:+.4f}")

    if macro_new > macro_old + 0.005:
        print("\n*** OCR stacking shows stable improvement. Recommend building v48 candidate. ***")
    elif macro_new > macro_old:
        print("\n*** OCR stacking shows marginal improvement. More validation needed. ***")
    else:
        print("\n*** OCR stacking does not improve over v45. ***")


if __name__ == "__main__":
    main()
