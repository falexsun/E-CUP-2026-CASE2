"""Analyze party cracker distribution across folds."""

import re
import unicodedata
import numpy as np
from ozon_quality.data import load_data

frame, _ = load_data('data/full_grouped.csv', 'configs/ozon_schema.json', require_label=True)
labels = frame['labels'].map(lambda v: int(v[0])).to_numpy()
categories = frame['category'].to_numpy()

flam_mask = categories == 'Легковоспламеняющиеся'
flam_frame = frame[flam_mask].reset_index(drop=True)
flam_labels = labels[flam_mask]
flam_groups = frame.loc[flam_mask, 'group'].astype(str).to_numpy()

v19 = np.load('artifacts/v19_outer_probabilities_2026.npz')

def norm(value):
    value = unicodedata.normalize('NFKC', str(value)).casefold().replace('ё', 'е')
    return re.sub(r'[^a-zа-я0-9]+', ' ', value).strip()

titles_norm = flam_frame['title'].map(norm).to_numpy()
party_mask = np.array(['хлопушк' in t for t in titles_norm])

print('Party cracker distribution across folds:')
for fold in range(5):
    fm = v19['fold'] == fold
    total = (party_mask & fm).sum()
    pos = (party_mask & fm & (flam_labels == 1)).sum()
    neg = (party_mask & fm & (flam_labels == 0)).sum()
    print(f'  Fold {fold}: {total} total ({pos} pos, {neg} neg)')

pos_groups = set(flam_groups[(party_mask) & (flam_labels == 1)])
neg_groups = set(flam_groups[(party_mask) & (flam_labels == 0)])
print(f'\nPositive party cracker groups: {len(pos_groups)}')
print(f'Negative party cracker groups: {len(neg_groups)}')
print(f'Overlap: {len(pos_groups & neg_groups)}')

print('\nPositive party cracker group -> fold mapping:')
for group in sorted(pos_groups):
    gm = flam_groups == group
    folds = sorted(set(v19['fold'][gm]))
    print(f'  {group}: {gm.sum()} items, folds={folds}')
