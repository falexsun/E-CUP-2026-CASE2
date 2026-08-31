"""INVALID — residual OCR char-TFIDF + title interaction experiment.

THIS SCRIPT IS INVALID AND MUST NOT BE RUN. Issues found by independent audit:

1. For outer fold k, base_prob on meta-train rows are single-level outer OOF
   predictions from base models that may have trained on fold k (leakage).
2. Line 228 (original) selects threshold on final train-fit probabilities,
   not meta-inner-OOF predictions.
3. Char TFIDF is fit on all outer-train before inner meta CV (leakage).
4. To fix: within each outer fold, generate base INNER-OOF predictions for
   outer-train and a base outer-test prediction from fit on outer-train,
   then for each meta inner fold fit char TFIDF only on meta-inner-train
   and choose meta threshold from meta-inner-OOF.

Status: INVALID. Do not run. Rewrite required before execution.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data
from ozon_quality.official import OFFICIAL_CATEGORIES
from ozon_quality.official_baseline import choose_positive_threshold

# ── OCR keyword features (predeclared, image-derived) ────────────────────

def _ocr_keyword_features(text: str) -> np.ndarray:
    """Extract keyword features from text that may contain OCR content.

    These keywords are predeclared based on domain knowledge about
    flammable/BAD product packaging — NOT derived from label inspection.
    """
    t = str(text).lower()
    return np.array([
        float(bool(re.search(r"пиротехник|петард|салют|бенгальск|дымов|шашк|фейерверк|18\+|громкий", t))),
        float(bool(re.search(r"уголь|брикет|charcoal|briquette", t))),
        float(bool(re.search(r"газ|баллон|топлив|fuel|бензин", t))),
        float(bool(re.search(r"спичк|зажигалк|розжиг|горелк|burner|lighter|сухое горюч", t))),
        float(bool(re.search(r"бад|биологически активн|dietary supplement", t))),
        float(bool(re.search(r"огнеопасн|легковоспламен|flammable|горюч|воспламен|18\+|класс опасност", t))),
        float(bool(re.search(r"в комплект|в набор|содержит|включает|внутри", t))),
        float(bool(re.search(r"конфетти|серпантин|confetti", t))),
    ], dtype="float32")


OCR_FEATURE_NAMES = [
    "ocr_pyro", "ocr_coal", "ocr_gas_fuel", "ocr_matches",
    "ocr_bad_marking", "ocr_warning", "ocr_set_contents", "ocr_confetti",
]


def _norm_title(v):
    v = unicodedata.normalize("NFKC", str(v)).casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", v).strip()


def _assert_no_overlap(train_groups, test_groups, label=""):
    overlap = set(train_groups) & set(test_groups)
    assert len(overlap) == 0, f"GROUP LEAKAGE ({label}): {len(overlap)} shared groups"


def run_residual_ocr(nested_csv_path: str, outer_seed: int = 2026):
    """Run residual OCR experiment on top of nested audit base predictions.

    Args:
        nested_csv_path: Path to nested_audit seed{outer_seed}.csv
        outer_seed: Must match the seed used in nested_audit.py
    """
    # Load nested audit results
    nested_df = pd.read_csv(nested_csv_path, dtype={"id": str})
    print(f"Loaded nested audit: {len(nested_df)} rows from {nested_csv_path}")

    # Verify it has the expected columns
    required_cols = {"id", "category", "label", "group", "fold", "blend_prob", "prediction"}
    missing = required_cols - set(nested_df.columns)
    assert not missing, f"Nested CSV missing columns: {missing}"

    # Load full data for OCR features
    frame, _ = load_data(
        str(ROOT / "data" / "full_grouped.csv"),
        str(ROOT / "configs" / "ozon_schema.json"),
        require_label=True,
    )
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()
    groups = frame["group"].astype(str).to_numpy()
    ids = frame["id"].astype(str).tolist()
    n = len(frame)

    # Load OCR texts if available
    ocr_dir = ROOT / "artifacts" / "ocr_features_full"
    ocr_available = (ocr_dir / "state.json").exists()
    if ocr_available:
        state = json.loads((ocr_dir / "state.json").read_text())
        assert state.get("complete"), "OCR extraction not complete"
        ocr_texts = pd.read_csv(str(ocr_dir / "texts.csv"), dtype=str)
        ocr_id_to_text = dict(zip(ocr_texts["id"], ocr_texts["ocr_text"]))
        print(f"OCR texts loaded: {len(ocr_id_to_text)} products")
    else:
        print("WARNING: OCR texts not available, using empty OCR features")
        ocr_id_to_text = {}

    # Build OCR keyword features for all rows
    ocr_features = np.zeros((n, len(OCR_FEATURE_NAMES)), dtype="float32")
    for i, id_ in enumerate(ids):
        ocr_text = ocr_id_to_text.get(id_, "")
        if ocr_text:
            ocr_features[i] = _ocr_keyword_features(ocr_text)

    # Build char TFIDF on product titles (for title-interaction features)
    # This is separate from OCR — it captures character patterns in titles
    titles = [_norm_title(t) for t in frame["title"].values]

    # Align nested audit fold assignments
    # nested_df rows match frame rows by id
    nested_id_to_fold = dict(zip(nested_df["id"].astype(str), nested_df["fold"]))
    nested_id_to_base_prob = dict(zip(nested_df["id"].astype(str), nested_df["blend_prob"]))
    nested_id_to_base_pred = dict(zip(nested_df["id"].astype(str), nested_df["prediction"]))

    oof_fold = np.array([nested_id_to_fold.get(id_, -1) for id_ in ids], dtype="int8")
    base_prob = np.array([nested_id_to_base_prob.get(id_, 0.0) for id_ in ids], dtype="float32")
    base_pred = np.array([nested_id_to_base_pred.get(id_, 0) for id_ in ids], dtype="int8")

    # Verify fold assignment matches
    processed_mask = oof_fold >= 0
    assert processed_mask.all(), f"Missing fold assignments for {(~processed_mask).sum()} rows"

    # Outer split (must match nested_audit.py)
    strat_key = (frame["category"] + "_" + frame["labels"].map(lambda v: str(v[0]))).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=outer_seed)
    fold_iter = list(outer.split(frame, strat_key, groups=groups))

    # OOF storage for meta-model
    meta_oof_prob = np.zeros(n, dtype="float32")
    meta_oof_pred = np.zeros(n, dtype="int8")

    for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
        train_g = groups[train_idx]
        test_g = groups[test_idx]
        _assert_no_overlap(train_g, test_g, f"meta_fold{fold_idx}")

        # Verify fold assignments match nested audit
        assert (oof_fold[test_idx] == fold_idx).all(), (
            f"Fold {fold_idx}: nested audit fold mismatch"
        )

        for category in OFFICIAL_CATEGORIES:
            tc_mask = categories[train_idx] == category
            tec_mask = categories[test_idx] == category
            if not tc_mask.any():
                continue

            tc_idx = train_idx[tc_mask]
            te_idx = test_idx[tec_mask]

            # Meta-features for outer-train: base prediction + OCR keywords + char TFIDF
            # Fit char TFIDF on outer-train only
            tc_titles = [titles[i] for i in tc_idx]
            te_titles = [titles[i] for i in te_idx]

            char_tfidf = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5),
                max_features=2000, sublinear_tf=True,
            )
            tc_char = char_tfidf.fit_transform(tc_titles).toarray().astype("float32")
            te_char = char_tfidf.transform(te_titles).toarray().astype("float32")

            # Build meta-feature matrix
            # [base_prob, ocr_keywords, char_tfidf]
            tc_meta = np.column_stack([
                base_prob[tc_idx].reshape(-1, 1),
                ocr_features[tc_idx],
                tc_char,
            ])
            te_meta = np.column_stack([
                base_prob[te_idx].reshape(-1, 1),
                ocr_features[te_idx],
                te_char,
            ])

            tc_y = labels[tc_idx]
            te_y = labels[te_idx]
            tc_groups_cat = groups[tc_idx]

            # Inner-OOF for meta-model C selection
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=outer_seed + fold_idx)
            best_score, best_c = -1, 0.1

            for c in (0.01, 0.03, 0.1, 0.3):
                inner_oof = np.zeros(len(tc_y), dtype="float32")
                for it, (ti, vi) in enumerate(inner.split(tc_meta, tc_y, groups=tc_groups_cat)):
                    _assert_no_overlap(tc_groups_cat[ti], tc_groups_cat[vi], f"meta_inner_{fold_idx}_{it}")
                    lr = LogisticRegression(C=c, class_weight="balanced", max_iter=1000,
                                            random_state=outer_seed + fold_idx)
                    lr.fit(tc_meta[ti], tc_y[ti])
                    inner_oof[vi] = lr.predict_proba(tc_meta[vi])[:, 1]
                _, score = choose_positive_threshold(tc_y, inner_oof)
                if score > best_score:
                    best_score, best_c = score, c

            # Refit on full outer-train
            lr_final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000,
                                          random_state=outer_seed + fold_idx)
            lr_final.fit(tc_meta, tc_y)
            threshold, _ = choose_positive_threshold(tc_y, lr_final.predict_proba(tc_meta)[:, 1])

            # Predict on outer-test
            te_meta_prob = lr_final.predict_proba(te_meta)[:, 1]
            te_meta_pred = (te_meta_prob >= threshold).astype("int8")

            meta_oof_prob[te_idx] = te_meta_prob
            meta_oof_pred[te_idx] = te_meta_pred

            # Per-fold metrics
            meta_f1 = f1_score(te_y, te_meta_pred, zero_division=0)
            base_f1 = f1_score(te_y, base_pred[te_idx], zero_division=0)
            print(f"  Fold {fold_idx} {category}: base={base_f1:.4f} meta={meta_f1:.4f} "
                  f"delta={meta_f1-base_f1:+.4f} C={best_c}")

    # Aggregate
    print(f"\n{'='*80}")
    print(f"RESIDUAL OCR AGGREGATE (seed={outer_seed}):")
    for c in OFFICIAL_CATEGORIES:
        m = categories == c
        y = labels[m]
        base_f1 = f1_score(y, base_pred[m], zero_division=0)
        meta_f1 = f1_score(y, meta_oof_pred[m], zero_division=0)
        print(f"  {c}: base={base_f1:.6f} meta={meta_f1:.6f} delta={meta_f1-base_f1:+.6f}")

    base_macro = np.mean([
        f1_score(labels[categories == c], base_pred[categories == c], zero_division=0)
        for c in OFFICIAL_CATEGORIES
    ])
    meta_macro = np.mean([
        f1_score(labels[categories == c], meta_oof_pred[categories == c], zero_division=0)
        for c in OFFICIAL_CATEGORIES
    ])
    print(f"  Macro: base={base_macro:.6f} meta={meta_macro:.6f} delta={meta_macro-base_macro:+.6f}")

    # Save
    out = ROOT / "artifacts" / "residual_ocr"
    out.mkdir(parents=True, exist_ok=True)
    df = frame[["id", "title", "category"]].copy()
    df["label"] = labels
    df["group"] = groups
    df["fold"] = oof_fold
    df["base_prob"] = base_prob
    df["base_pred"] = base_pred
    df["meta_prob"] = meta_oof_prob
    df["meta_pred"] = meta_oof_pred
    df.to_csv(out / f"residual_ocr_seed{outer_seed}.csv", index=False)
    print(f"Saved to {out}")
    return base_macro, meta_macro


def main():
    raise RuntimeError(
        "INVALID: residual_ocr_experiment.py has critical leakage issues. "
        "See module docstring. Do not run without rewrite."
    )


if __name__ == "__main__":
    main()
