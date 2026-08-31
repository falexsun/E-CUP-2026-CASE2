"""Analyze non-party-cracker false negatives."""

import json
import re
import unicodedata
from pathlib import Path
import numpy as np
from ozon_quality.data import load_data

frame, _ = load_data('data/full_grouped.csv', 'configs/ozon_schema.json', require_label=True)
labels = frame['labels'].map(lambda v: int(v[0])).to_numpy()
categories = frame['category'].to_numpy()

flam_mask = categories == 'Легковоспламеняющиеся'
flam_frame = frame[flam_mask].reset_index(drop=True)
flam_labels = labels[flam_mask]

v19 = np.load('artifacts/v19_outer_probabilities_2026.npz')
v45 = np.load('artifacts/v34_regex_combo_outer_v45.npz')

thresh = float(json.loads(Path('configs/v19_knn.json').read_text())['blend_threshold'])
v45_pred = (v45['p'] >= thresh).astype(int)
v45_pred[v45['regex']] = 1

def norm(value):
    value = unicodedata.normalize('NFKC', str(value)).casefold().replace('ё', 'е')
    return re.sub(r'[^a-zа-я0-9]+', ' ', value).strip()

titles_norm = flam_frame['title'].map(norm).to_numpy()

# Find non-party-cracker FNs
fn_mask = (flam_labels == 1) & (v45_pred == 0)
party_mask = np.array(['хлопушк' in t for t in titles_norm])
non_party_fn = fn_mask & ~party_mask

print(f'Total FN: {fn_mask.sum()}')
print(f'Party cracker FN: (fn_mask & party_mask).sum() = {(fn_mask & party_mask).sum()}')
print(f'Non-party-cracker FN: {non_party_fn.sum()}')
print()

for i in np.where(non_party_fn)[0]:
    fold = v19['fold'][i]
    p = v45['p'][i]
    title = str(flam_frame.iloc[i]['title'])[:120]
    desc = str(flam_frame.iloc[i]['description'])[:200]
    print(f'  fold={fold} p={p:.4f}')
    print(f'    title: {title}')
    print(f'    desc: {desc}')
    print()
