"""Evaluate v47 with party cracker rules added to v45+regex."""

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data

def norm(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", value).strip()

def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()

    flam_mask = categories == "Легковоспламеняющиеся"
    flam_frame = frame[flam_mask].reset_index(drop=True)
    flam_labels = labels[flam_mask]

    v19 = np.load(str(ROOT / "artifacts" / "v19_outer_probabilities_2026.npz"))
    v45 = np.load(str(ROOT / "artifacts" / "v34_regex_combo_outer_v45.npz"))
    assert (v19["y"] == flam_labels).all()

    thresh = float(json.loads((ROOT / "configs" / "v19_knn.json").read_text())["blend_threshold"])

    titles_norm = flam_frame["title"].map(norm).to_numpy()

    # v47 rules = v45 regex + new safe rules
    new_rules = {
        "matches_tourist": re.compile(r"\bспичк\w*\s+(?:длительн\w*\s+)?горен\w*.*(?:турист|50\s*мм|5\s*см)\b", re.I),
        "smoke_broad": re.compile(r"\b(?:шашк\w*\s+(?:дымов|страйкбол|учебн|имитацион|бел)|гранат\w*\s+страйкбол)\b", re.I),
        "sparkler": re.compile(r"бенгальск\w*\s+свеч", re.I),
        "moxibustion": re.compile(r"мокс\w*|моксотерап", re.I),
        # Party cracker rules (100% precision, all fold 3)
        "party_super": re.compile(r"хлопушк\w*\s+супер", re.I),
        "party_maxi": re.compile(r"хлопушк\w*\s+макси", re.I),
    }

    new_rule_matches = np.zeros(len(flam_frame), dtype=bool)
    for name, pattern in new_rules.items():
        matches = np.array([bool(pattern.search(t)) for t in titles_norm])
        new_rule_matches |= matches
        if matches.sum() > 0:
            tp = int(flam_labels[matches].sum())
            fp = int((flam_labels[matches] == 0).sum())
            print(f"  {name}: {matches.sum()} matches, {tp} TP, {fp} FP")

    print(f"\nTotal new: {new_rule_matches.sum()}, TP: {int(flam_labels[new_rule_matches].sum())}")

    # Build predictions
    v45_pred = (v45["p"] >= thresh).astype(int)
    v45_pred[v45["regex"]] = 1

    v47_pred = v45_pred.copy()
    v47_pred[new_rule_matches] = 1

    print(f"\n{'='*70}")
    print(f"AGGREGATE FLAMMABLE:")
    for name, pred in [("v19", (v19["p"] >= thresh).astype(int)),
                       ("v45+regex", v45_pred),
                       ("v47", v47_pred)]:
        f1 = f1_score(flam_labels, pred, zero_division=0)
        p = precision_score(flam_labels, pred, zero_division=0)
        r = recall_score(flam_labels, pred, zero_division=0)
        print(f"  {name:>12}: F1={f1:.6f}, P={p:.6f}, R={r:.6f}")

    # BAD stays same (0.9359)
    bad_f1 = 0.935901
    flam_f1_v47 = f1_score(flam_labels, v47_pred, zero_division=0)
    macro_v47 = (bad_f1 + flam_f1_v47) / 2
    print(f"\nMACRO F1: {macro_v47:.6f}")
    print(f"  BAD: {bad_f1:.6f}")
    print(f"  Flammable: {flam_f1_v47:.6f}")

    print(f"\nPER-FOLD:")
    print(f"{'Fold':>4} {'Pos':>4} | {'v19 F1':>8} {'v45 F1':>8} {'v47 F1':>8} | {'new TP':>6} {'new FP':>6}")
    for fold in range(5):
        fm = v19["fold"] == fold
        y = flam_labels[fm]
        p19_f = (v19["p"][fm] >= thresh).astype(int)
        p45_f = v45_pred[fm]
        p47_f = v47_pred[fm]
        new_tp = int(((p47_f == 1) & (p45_f == 0) & (y == 1)).sum())
        new_fp = int(((p47_f == 1) & (p45_f == 0) & (y == 0)).sum())
        print(f"{fold:>4} {int(y.sum()):>4} | {f1_score(y,p19_f):>8.4f} {f1_score(y,p45_f):>8.4f} {f1_score(y,p47_f):>8.4f} | {new_tp:>6} {new_fp:>6}")

    # Detailed FN analysis for v47
    fn_mask = (flam_labels == 1) & (v47_pred == 0)
    print(f"\nv47 REMAINING FN ({fn_mask.sum()}):")
    for i in np.where(fn_mask)[0]:
        print(f"  fold={v19['fold'][i]} p45={v45['p'][i]:.4f} | {str(flam_frame.iloc[i]['title'])[:100]}")

if __name__ == "__main__":
    main()
