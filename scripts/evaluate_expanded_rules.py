"""Evaluate expanded high-precision regex rules for flammable category.

Tests new rule candidates against the full dataset using outer group CV
to ensure fold-safe precision estimation.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data


def _norm_title(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", value).strip()


# Existing v45 rules (safe, confirmed)
V45_RULES = {
    "standalone_coal_or_fuel_briquette": re.compile(
        r"(?:^|\s)(уголь|брикет)\S*\s+(?:для\s+)?(?:барбекю|гриля|мангала|шашлык)", re.I
    ),
    "disposable_grill_with_coal": re.compile(
        r"(?:одноразов\S*\s+)?мангал\S*\s+.*(?:уголь|брикет|розжиг)", re.I
    ),
    "gas_for_portable_stove_or_burner": re.compile(
        r"(?:газ|баллон\S*)\s+.*(?:для\s+)?(?:портативн\S*\s+)?(?:плит|горелк)", re.I
    ),
    "smoke_bomb": re.compile(
        r"(?:дымов\S*\s+)?(?:шашк|бомб)\S*\s+(?:для\s+)?(?:сигнализаци|фейерверк|пейнтбол)", re.I
    ),
    "barbecue_fuel_briquette": re.compile(
        r"(?:^|\s)(?:топливн\S*\s+)?брикет\S*\s+(?:для\s+)?(?:розжиг|мангал|барбекю|гриль)", re.I
    ),
}

# New rule candidates (need fold-safe validation)
CANDIDATE_RULES = {
    # Matches and ignition
    "matches_durability": re.compile(
        r"спичк\S*\s+(?:длительн\S*\s+)?горени|спичк\S*\s+.*туристическ", re.I
    ),
    "fire_starters": re.compile(
        r"(?:набор|палочк|ролл)\S*\s+(?:для\s+)?(?:разведен\S*\s+огн|розжиг)", re.I
    ),
    "dry_fuel_tablets": re.compile(
        r"сухо\S*\s+горюч\S*|таблет\S*\s+(?:для\s+)?розжиг", re.I
    ),
    "fire_gel_starters": re.compile(
        r"(?:растопк|розжиг)\S*\s+(?:для\s+)?(?:костр|мангал|барбекю|грил|печ)", re.I
    ),
    "ignition_kits": re.compile(
        r"разогреват\S*\s+портативн\S*\s+.*(?:розжиг|спичк|горюч)", re.I
    ),
    # Pyrotechnics
    "smoke_grenade_broad": re.compile(
        r"(?:шашк|гранат)\S*\s+(?:дымов|страйкбол|учебн|имитацион)", re.I
    ),
    "colored_smoke": re.compile(
        r"(?:цветн\S*\s+дым|дым\S*\s+(?:фонтан|цветн))", re.I
    ),
    "party_cracker": re.compile(
        r"хлопушк\S*\s+(?:для\s+)?(?:праздн|серпантин|сюрприз)", re.I
    ),
    "sparkler": re.compile(
        r"бенгальск\S*\s+свеч", re.I
    ),
    "cake_candles": re.compile(
        r"(?:свеч\S*\s+(?:фонтан|для\s+торт)|фонтан\S*\s+для\s+торт)", re.I
    ),
    "fireworks": re.compile(
        r"салют|фейерверк|петард|пиротехник", re.I
    ),
    # Gas and fuel
    "gas_refill": re.compile(
        r"газ\S*\s+(?:для\s+)?(?:заправк|зажигалк|баллон)", re.I
    ),
    "gas_burner_set": re.compile(
        r"(?:горелк|плитк)\S*\s+(?:газов|.*с\s+баллон)", re.I
    ),
    # Charcoal
    "charcoal_standalone": re.compile(
        r"уголь\S*\s+(?:древесн|берёзов|березов|каменн)", re.I
    ),
    # Bio fireplace
    "bio_fireplace": re.compile(
        r"биокамин", re.I
    ),
    # Gift/tourist sets with fire items
    "gift_set_with_fire": re.compile(
        r"набор\S*\s+.*(?:зажигалк|спичк|горелк|розжиг|выжива)", re.I
    ),
    # Moxibustion
    "moxibustion": re.compile(
        r"мокс|моксотерап", re.I
    ),
}


def main():
    schema = str(ROOT / "configs" / "ozon_schema.json")
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"), schema, require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()
    groups = frame["group"].astype(str).to_numpy()

    flam_mask = categories == "Легковоспламеняющиеся"
    flam_frame = frame[flam_mask].copy()
    flam_labels = labels[flam_mask]
    flam_groups = groups[flam_mask]
    flam_titles = flam_frame["title"].map(_norm_title).to_numpy()

    print(f"Flammable rows: {len(flam_frame)}, positives: {flam_labels.sum()}")

    # Evaluate each rule on full data (for initial screening)
    print("\n=== Full-data rule statistics ===")
    all_rules = {**V45_RULES, **CANDIDATE_RULES}
    for name, pattern in all_rules.items():
        matches = np.array([bool(pattern.search(t)) for t in flam_titles])
        n_matches = matches.sum()
        if n_matches == 0:
            print(f"  {name:35s}: 0 matches")
            continue
        n_positive = flam_labels[matches].sum()
        precision = n_positive / n_matches
        recall_pos = n_positive / flam_labels.sum() if flam_labels.sum() > 0 else 0
        is_v45 = name in V45_RULES
        tag = " [v45]" if is_v45 else ""
        print(f"  {name:35s}: {n_matches:4d} matches, {n_positive:4d} positive, "
              f"P={precision:.4f}, R={recall_pos:.4f}{tag}")

    # Fold-safe evaluation for new candidates only
    print("\n=== Fold-safe evaluation (5-fold outer group CV) ===")
    n_folds = 5
    outer_splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=2026)

    for rule_name, pattern in CANDIDATE_RULES.items():
        matches = np.array([bool(pattern.search(t)) for t in flam_titles])
        if matches.sum() == 0:
            continue

        fold_precisions = []
        fold_recalls = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_splitter.split(flam_frame, flam_labels, groups=flam_groups)
        ):
            test_matches = matches[test_idx]
            test_labels = flam_labels[test_idx]
            n_test_matches = test_matches.sum()
            if n_test_matches == 0:
                continue
            n_tp = (test_labels[test_matches] == 1).sum()
            fold_precisions.append(n_tp / n_test_matches)
            fold_recalls.append(n_tp / test_labels.sum() if test_labels.sum() > 0 else 0)

        if fold_precisions:
            mean_p = np.mean(fold_precisions)
            std_p = np.std(fold_precisions)
            mean_r = np.mean(fold_recalls)
            min_p = min(fold_precisions)
            print(f"  {rule_name:35s}: mean_P={mean_p:.4f} (std={std_p:.4f}, min={min_p:.4f}), "
                  f"mean_R={mean_r:.4f}, folds={len(fold_precisions)}")
        else:
            print(f"  {rule_name:35s}: no matches in any test fold")

    # Combined rule evaluation (v45 + best new candidates)
    print("\n=== Combined v45 + candidates ===")
    v45_matches = np.zeros(len(flam_frame), dtype=bool)
    for name, pattern in V45_RULES.items():
        v45_matches |= np.array([bool(pattern.search(t)) for t in flam_titles])

    # Test adding each new rule to v45
    for rule_name, pattern in CANDIDATE_RULES.items():
        new_matches = np.array([bool(pattern.search(t)) for t in flam_titles])
        combined = v45_matches | new_matches
        n_new = (combined & ~v45_matches).sum()
        if n_new == 0:
            continue
        n_new_positive = flam_labels[combined & ~v45_matches].sum()
        new_precision = n_new_positive / n_new if n_new > 0 else 0
        print(f"  + {rule_name:35s}: {n_new:4d} new matches, {n_new_positive:4d} new positive, "
              f"new_P={new_precision:.4f}")

    # Also evaluate on BAD category to check for cross-category false positives
    print("\n=== Cross-category check (BAD should have 0 matches) ===")
    bad_mask = categories == "БАД"
    bad_frame = frame[bad_mask]
    bad_titles = bad_frame["title"].map(_norm_title).to_numpy()

    for rule_name, pattern in CANDIDATE_RULES.items():
        matches = np.array([bool(pattern.search(t)) for t in bad_titles])
        if matches.sum() > 0:
            print(f"  WARNING {rule_name:35s}: {matches.sum()} matches in BAD category!")
            for t in bad_titles[matches][:5]:
                print(f"    -> {t[:100]}")
        else:
            print(f"  {rule_name:35s}: 0 BAD matches (safe)")


if __name__ == "__main__":
    main()
