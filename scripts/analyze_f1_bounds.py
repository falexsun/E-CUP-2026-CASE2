"""Analyze error families and compute theoretical F1 bounds."""

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

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

    thresh = float(json.loads((ROOT / "configs" / "v19_knn.json").read_text())["blend_threshold"])
    v45_pred = (v45["p"] >= thresh).astype(int)
    v45_pred[v45["regex"]] = 1

    fn_mask = (flam_labels == 1) & (v45_pred == 0)
    titles_norm = flam_frame["title"].map(norm).to_numpy()

    # Classify FN by error family
    fn_indices = np.where(fn_mask)[0]
    families = {
        "party_cracker": 0,
        "candle_sparkler": 0,
        "burner_gas": 0,
        "gift_set": 0,
        "ignition_fuel": 0,
        "other": 0,
    }

    for i in fn_indices:
        t = titles_norm[i]
        if "хлопушк" in t:
            families["party_cracker"] += 1
        elif "свеч" in t or "бенгальск" in t or "фонтан" in t:
            families["candle_sparkler"] += 1
        elif "горелк" in t or "газ" in t or "баллон" in t:
            families["burner_gas"] += 1
        elif "набор" in t or "подарочн" in t or "выжива" in t:
            families["gift_set"] += 1
        elif "розжиг" in t or "горюч" in t or "топлив" in t or "сухое" in t:
            families["ignition_fuel"] += 1
        else:
            families["other"] += 1

    total_fn = fn_mask.sum()
    print("FN error families:")
    for fam, count in sorted(families.items(), key=lambda x: -x[1]):
        print(f"  {fam:20s}: {count:3d} ({count/total_fn*100:.1f}%)")
    print(f"  {'TOTAL':20s}: {total_fn}")

    # Current metrics
    bad_f1 = 0.935901
    flam_f1 = f1_score(flam_labels, v45_pred, zero_division=0)
    macro = (bad_f1 + flam_f1) / 2
    print(f"\nCurrent: BAD={bad_f1:.4f}, Flam={flam_f1:.4f}, Macro={macro:.4f}")

    # Fix party cracker FNs
    v47_pred = v45_pred.copy()
    party_mask = np.array(["хлопушк" in t for t in titles_norm])
    v47_pred[party_mask & fn_mask] = 1
    flam_f1_p = f1_score(flam_labels, v47_pred, zero_division=0)
    print(f"Fix party FNs: Flam={flam_f1_p:.4f}, Macro={(bad_f1+flam_f1_p)/2:.4f}")

    # Fix ALL FNs
    v48_pred = v45_pred.copy()
    v48_pred[fn_mask] = 1
    flam_f1_a = f1_score(flam_labels, v48_pred, zero_division=0)
    print(f"Fix ALL FNs: Flam={flam_f1_a:.4f}, Macro={(bad_f1+flam_f1_a)/2:.4f}")

    # Target
    target = 0.90
    needed = target * 2 - bad_f1
    print(f"\nTarget macro=0.90: need Flam={needed:.4f} (gap={needed-flam_f1:.4f})")

if __name__ == "__main__":
    main()
