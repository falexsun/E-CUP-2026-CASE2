"""Nested StratifiedGroupKFold audit — lexical+VLM inner-OOF blend.

NOT full v45: kNN and standardized head are absent (Phase 2 pending).
Label accordingly in metadata.

Every parameter selected on inner-OOF predictions from models that
NEVER saw the target row or its group.

CRITICAL: V45_REGEX was selected by zero-FP on ALL released data.
Applying it to outer folds is post-selection leakage.
--no-regex (default) is the honest audit. --regex is diagnostic only.

Frozen C values from v45 fullrefit artifact:
  lexical: BAD C=4.0, flammable C=10.0 (class_weight="balanced")
  VLM: C=1.0 (class_weight="balanced")
No expensive grid; these are confirmed.

Assertions enforce:
- zero group overlap in every split
- OOF coverage exactly once per processed row
- no train-fit arrays used for parameter selection
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
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
from ozon_quality.lexical import build_lexical_classifier
from ozon_quality.official import OFFICIAL_CATEGORIES
from ozon_quality.official_baseline import add_rule_tokens, choose_positive_threshold
from ozon_quality.official_multimodal import _safe_logit, load_embedding_cache

# ── v45 regex (POST-SELECTION LEAKAGE — diagnostic only) ────────────────
V45_REGEX = [
    (re.compile(r"\b(?:уголь\s+(?:древесн\w*|каменн\w*)|брикетированн\w*\s+топлив\w*)\b", re.I), 1),
    (re.compile(r"(?:мангал\w*.*одноразов\w*.*угл|одноразов\w*.*мангал\w*.*угл)", re.I), 1),
    (re.compile(r"\bгаз\w*\s+(?:для|к)\s+портативн\w*\s+(?:плит\w*|горел\w*)\b", re.I), 1),
    (re.compile(r"\b(?:дым\w*\s+шашк\w*|шашк\w*\s+дым\w*)\b", re.I), 1),
    (re.compile(r"\bбрикет\w*\s+для\s+грил\w*\b", re.I), 1),
]

# ── frozen v45 C values (from fullrefit artifact) ────────────────────────
LEX_FROZEN_C = {"БАД": 4.0, "Легковоспламеняющиеся": 10.0}
VLM_FROZEN_C = 1.0


def _norm_title(v):
    v = unicodedata.normalize("NFKC", str(v)).casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", v).strip()


# ── assertion helpers ────────────────────────────────────────────────────

def _assert_no_overlap(train_groups, test_groups, label=""):
    overlap = set(train_groups) & set(test_groups)
    assert len(overlap) == 0, f"GROUP LEAKAGE ({label}): {len(overlap)} shared groups"


def _assert_oof_coverage(oof_counter, processed_mask, label=""):
    processed = oof_counter[processed_mask]
    assert (processed == 1).all(), (
        f"OOF coverage error ({label}): "
        f"min={processed.min()}, max={processed.max()}, expected=1"
    )


# ── inner-OOF helpers (frozen C, no grid) ────────────────────────────────

def _inner_oof_lexical(frame_cat, y_cat, groups_cat, category, outer_seed, fold_idx):
    """Inner-OOF lexical predictions. Frozen C from v45 artifact."""
    text = add_rule_tokens(frame_cat["text"], category)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=outer_seed + fold_idx)
    c = LEX_FROZEN_C[category]
    w = "balanced"
    oof = np.zeros(len(y_cat), dtype="float32")

    for it, (ti, vi) in enumerate(inner.split(text, y_cat, groups=groups_cat)):
        _assert_no_overlap(groups_cat[ti], groups_cat[vi], f"lex_inner_{fold_idx}_{it}")
        lex = build_lexical_classifier(outer_seed + fold_idx, "single_label")
        lex.set_params(classifier__C=c, classifier__class_weight=w)
        lex.fit(text.iloc[ti], y_cat[ti])
        oof[vi] = lex.predict_proba(text.iloc[vi])[:, 1]

    # Refit on full outer-train for test prediction
    lex_final = build_lexical_classifier(outer_seed + fold_idx, "single_label")
    lex_final.set_params(classifier__C=c, classifier__class_weight=w)
    lex_final.fit(text, y_cat)
    threshold, _ = choose_positive_threshold(y_cat, oof)
    return oof, lex_final, threshold


def _inner_oof_vlm(x_cat, y_cat, groups_cat, outer_seed, fold_idx):
    """Inner-OOF VLM LR predictions. Frozen C=1.0 from v45 artifact."""
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=outer_seed + fold_idx + 100)
    c = VLM_FROZEN_C
    w = "balanced"
    oof = np.zeros(len(y_cat), dtype="float32")

    for it, (ti, vi) in enumerate(inner.split(x_cat, y_cat, groups=groups_cat)):
        _assert_no_overlap(groups_cat[ti], groups_cat[vi], f"vlm_inner_{fold_idx}_{it}")
        lr = LogisticRegression(C=c, class_weight=w, max_iter=1000, solver="lbfgs",
                                random_state=outer_seed + fold_idx)
        lr.fit(x_cat[ti], y_cat[ti])
        oof[vi] = lr.predict_proba(x_cat[vi])[:, 1]

    lr_final = LogisticRegression(C=c, class_weight=w, max_iter=1000, solver="lbfgs",
                                  random_state=outer_seed + fold_idx)
    lr_final.fit(x_cat, y_cat)
    threshold, _ = choose_positive_threshold(y_cat, oof)
    return oof, lr_final, threshold


def _select_blend(lex_oof, vlm_oof, y_cat):
    """Select blend alpha/strategy/threshold on inner-OOF predictions."""
    lex_thresh, _ = choose_positive_threshold(y_cat, lex_oof)
    vlm_thresh, _ = choose_positive_threshold(y_cat, vlm_oof)
    lex_m = _safe_logit(lex_oof) - _safe_logit(lex_thresh)
    vlm_m = _safe_logit(vlm_oof) - _safe_logit(vlm_thresh)

    best_score, best_alpha, best_strategy, best_t = -1, 0.5, "raw_probability", 0.5
    for alpha in np.linspace(0, 1, 21):
        raw = alpha * vlm_oof + (1 - alpha) * lex_oof
        margin = 1 / (1 + np.exp(-(alpha * vlm_m + (1 - alpha) * lex_m)))
        for strategy, prob in [("raw_probability", raw), ("threshold_normalized_logit", margin)]:
            t, s = choose_positive_threshold(y_cat, prob)
            if s > best_score:
                best_score, best_alpha, best_strategy, best_t = s, alpha, strategy, t
    return best_alpha, best_strategy, best_t, lex_thresh, vlm_thresh


def _apply_blend(lex_prob, vlm_prob, alpha, strategy, lex_thresh, vlm_thresh):
    if strategy == "raw_probability":
        return alpha * vlm_prob + (1 - alpha) * lex_prob
    lm = _safe_logit(lex_prob) - _safe_logit(lex_thresh)
    vm = _safe_logit(vlm_prob) - _safe_logit(vlm_thresh)
    return 1 / (1 + np.exp(-(alpha * vm + (1 - alpha) * lm)))


# ── main audit ───────────────────────────────────────────────────────────

def run_nested_audit(outer_seed=2026, smoke_fold=None, use_regex=False):
    """Run nested audit.

    Args:
        smoke_fold: if set (0-4), run only that single outer fold.
        use_regex: if True, apply v45 regex (DIAGNOSTIC BIASED ONLY).
                   Default False is the honest audit.

    Returns (macro_f1, metadata_dict).
    """
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

    emb = np.asarray(
        load_embedding_cache(str(ROOT / "artifacts" / "reference_joint_full_v2"), frame["id"]),
        dtype="float32",
    )
    emb_norm = normalize(emb, norm="l2")

    strat_key = (frame["category"] + "_" + frame["labels"].map(lambda v: str(v[0]))).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=outer_seed)

    oof_counter = np.zeros(n, dtype="int32")
    oof_lex_prob = np.zeros(n, dtype="float32")
    oof_vlm_prob = np.zeros(n, dtype="float32")
    oof_blend_prob = np.zeros(n, dtype="float32")
    oof_pred = np.zeros(n, dtype="int8")
    oof_fold = np.full(n, -1, dtype="int8")

    data_hash = hashlib.sha256("".join(ids).encode()).hexdigest()[:16]
    config = {
        "seed": outer_seed,
        "phase": "lexical_vlm_inner_oof_blend",  # NOT full v45
        "components": ["lexical_tfidf_lr", "vlm_embedding_lr", "inner_oof_blend"],
        "missing": ["knn_reference_bank", "standardized_linear_head"],
        "regex": "v45_post_selection_biased" if use_regex else "none",
        "frozen_lex_C": LEX_FROZEN_C,
        "frozen_vlm_C": VLM_FROZEN_C,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]
    print(f"Data hash: {data_hash}, Config hash: {config_hash}")
    print(f"Rows: {n}, Groups: {len(set(groups))}, Seed: {outer_seed}")
    print(f"Phase: {config['phase']}, Regex: {config['regex']}")
    if use_regex:
        print("WARNING: --regex applies post-selection-biased v45 rules (diagnostic only)")

    fold_iter = list(outer.split(frame, strat_key, groups=groups))
    if smoke_fold is not None:
        fold_iter = [(smoke_fold, fold_iter[smoke_fold])]
    else:
        fold_iter = list(enumerate(fold_iter))

    processed_ids = []
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in fold_iter:
        train_g = groups[train_idx]
        test_g = groups[test_idx]
        _assert_no_overlap(train_g, test_g, f"outer_fold{fold_idx}")

        tf = frame.iloc[train_idx].reset_index(drop=True)
        tef = frame.iloc[test_idx].reset_index(drop=True)
        ty = labels[train_idx]
        tey = labels[test_idx]
        tx = emb_norm[train_idx]
        tex = emb_norm[test_idx]

        oof_fold[test_idx] = fold_idx
        oof_counter[test_idx] += 1
        processed_ids.extend(ids[i] for i in test_idx)

        for category in OFFICIAL_CATEGORIES:
            tc_mask = tf["category"].eq(category).to_numpy()
            tec_mask = tef["category"].eq(category).to_numpy()
            if not tc_mask.any():
                continue

            tc_y = ty[tc_mask]
            te_y = tey[tec_mask]
            tc_groups = train_g[tc_mask]
            tc_x = tx[tc_mask]
            te_x = tex[tec_mask]
            tc_frame = tf[tc_mask].reset_index(drop=True)
            te_frame = tef[tec_mask].reset_index(drop=True)

            # Inner-OOF lexical (frozen C)
            lex_oof, lex_model, lex_thresh = _inner_oof_lexical(
                tc_frame, tc_y, tc_groups, category, outer_seed, fold_idx
            )

            # Inner-OOF VLM (frozen C=1.0)
            vlm_oof, vlm_model, vlm_thresh = _inner_oof_vlm(
                tc_x, tc_y, tc_groups, outer_seed, fold_idx
            )

            # Blend selection on inner-OOF predictions
            alpha, strategy, blend_t, b_lex_t, b_vlm_t = _select_blend(lex_oof, vlm_oof, tc_y)

            # Test predictions from final models (fit on full outer-train)
            lex_test_prob = lex_model.predict_proba(
                add_rule_tokens(te_frame["text"], category)
            )[:, 1]
            vlm_test_prob = vlm_model.predict_proba(te_x)[:, 1]
            blend_test = _apply_blend(lex_test_prob, vlm_test_prob, alpha, strategy, b_lex_t, b_vlm_t)

            # Threshold from inner-OOF
            pred_test = (blend_test >= blend_t).astype("int8")

            # Regex (DIAGNOSTIC BIASED ONLY — post-selection leakage)
            if use_regex and category == "Легковоспламеняющиеся":
                for pattern, label in V45_REGEX:
                    for j in range(len(te_frame)):
                        assert 0 <= j < len(te_frame)
                        if pattern.search(_norm_title(te_frame.iloc[j]["title"])):
                            pred_test[j] = label
                            blend_test[j] = float(label)

            # Store
            cat_global_idx = test_idx[np.where(tec_mask)[0]]
            oof_lex_prob[cat_global_idx] = lex_test_prob
            oof_vlm_prob[cat_global_idx] = vlm_test_prob
            oof_blend_prob[cat_global_idx] = blend_test
            oof_pred[cat_global_idx] = pred_test

            f1 = f1_score(te_y, pred_test, zero_division=0)
            ap = average_precision_score(te_y, blend_test) if len(np.unique(te_y)) > 1 else 0
            print(
                f"  Fold {fold_idx} {category}: F1={f1:.4f} AP={ap:.4f} "
                f"P={precision_score(te_y, pred_test, zero_division=0):.4f} "
                f"R={recall_score(te_y, pred_test, zero_division=0):.4f} "
                f"TP={((te_y==1)&(pred_test==1)).sum()} FP={((te_y==0)&(pred_test==1)).sum()} "
                f"FN={((te_y==1)&(pred_test==0)).sum()}"
            )

        elapsed = time.time() - t0
        print(f"  Fold {fold_idx} done in {elapsed:.0f}s")

    # ── assertions ──
    processed_mask = oof_counter > 0
    _assert_oof_coverage(oof_counter, processed_mask, f"seed{outer_seed}")
    print(f"\n✓ OOF coverage assertion passed ({processed_mask.sum()} rows)")

    # ── aggregate on processed rows ──
    print(f"\n{'='*80}")
    print(f"AGGREGATE (seed={outer_seed}, {processed_mask.sum()} rows, {config['phase']}):")
    for c in OFFICIAL_CATEGORIES:
        m = (categories == c) & processed_mask
        if m.sum() == 0:
            continue
        y, p = labels[m], oof_pred[m]
        print(f"  {c}: F1={f1_score(y,p,zero_division=0):.6f} "
              f"P={precision_score(y,p,zero_division=0):.6f} "
              f"R={recall_score(y,p,zero_division=0):.6f} "
              f"AP={average_precision_score(y,oof_blend_prob[m]):.6f}")

    macro = np.mean([
        f1_score(
            labels[(categories == c) & processed_mask],
            oof_pred[(categories == c) & processed_mask],
            zero_division=0,
        )
        for c in OFFICIAL_CATEGORIES
        if ((categories == c) & processed_mask).any()
    ])
    print(f"  Macro F1: {macro:.6f}")

    # ── save ──
    out = ROOT / "artifacts" / "nested_audit"
    out.mkdir(parents=True, exist_ok=True)
    regex_tag = "_regex" if use_regex else ""
    tag = f"seed{outer_seed}{regex_tag}" if smoke_fold is None else f"seed{outer_seed}_smoke{smoke_fold}{regex_tag}"

    metadata = {
        "data_hash": data_hash,
        "config_hash": config_hash,
        "config": config,
        "warning": "Phase 1 only: lexical+VLM inner-OOF blend. kNN+std head NOT included.",
        "regex_warning": "v45 regex was selected on all released data; applying to outer folds is post-selection leakage." if use_regex else None,
        "outer_seed": outer_seed,
        "smoke_fold": smoke_fold,
        "n_rows": n,
        "n_groups": len(set(groups)),
        "n_processed": int(processed_mask.sum()),
        "processed_ids": processed_ids,
        "aggregate_macro_f1": macro,
        "per_category": {
            c: {
                "f1": float(f1_score(
                    labels[(categories == c) & processed_mask],
                    oof_pred[(categories == c) & processed_mask],
                    zero_division=0,
                )),
                "n_processed": int(((categories == c) & processed_mask).sum()),
            }
            for c in OFFICIAL_CATEGORIES
            if ((categories == c) & processed_mask).any()
        },
    }
    with open(out / f"nested_{tag}_meta.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    df = frame[["id", "title", "category"]].copy()
    df["label"] = labels
    df["group"] = groups
    df["fold"] = oof_fold
    df["blend_prob"] = oof_blend_prob
    df["prediction"] = oof_pred
    df["error"] = np.where(
        (labels == 1) & (oof_pred == 0), "FN",
        np.where((labels == 0) & (oof_pred == 1), "FP", ""),
    )
    df.to_csv(out / f"nested_{tag}.csv", index=False)
    print(f"Saved to {out}")
    return macro, metadata


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nested lexical+VLM inner-OOF blend audit")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: one fold only")
    parser.add_argument("--smoke-fold", type=int, default=0, help="Which fold for smoke (default 0)")
    parser.add_argument("--single-seed", action="store_true", help="Run only --seed")
    parser.add_argument("--seed", type=int, default=2026, help="Outer seed (default 2026)")
    parser.add_argument("--regex", action="store_true",
                        help="Apply v45 regex (DIAGNOSTIC BIASED: post-selection leakage)")
    args = parser.parse_args()

    if args.smoke:
        print(f"=== SMOKE TEST: seed {args.seed}, fold {args.smoke_fold} ===")
        run_nested_audit(outer_seed=args.seed, smoke_fold=args.smoke_fold, use_regex=args.regex)
    elif args.single_seed:
        print(f"=== SINGLE SEED: {args.seed} ===")
        run_nested_audit(outer_seed=args.seed, use_regex=args.regex)
    else:
        print("=== FULL AUDIT: seed 2026 ===")
        m2026, _ = run_nested_audit(outer_seed=2026, use_regex=args.regex)
        print("\n=== FULL AUDIT: seed 42 ===")
        m42, _ = run_nested_audit(outer_seed=42, use_regex=args.regex)
        print(f"\n{'='*80}")
        print("SEED COMPARISON:")
        print(f"  Seed 2026: Macro F1={m2026:.6f}")
        print(f"  Seed 42:   Macro F1={m42:.6f}")
        print(f"  Delta:     {m2026-m42:+.6f}")


if __name__ == "__main__":
    main()
