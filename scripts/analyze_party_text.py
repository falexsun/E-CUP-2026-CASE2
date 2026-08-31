"""Analyze text features for party crackers."""

import re
import pandas as pd
from ozon_quality.data import load_data

frame, _ = load_data('data/full_grouped.csv', 'configs/ozon_schema.json', require_label=True)
labels = frame['labels'].map(lambda v: int(v[0])).to_numpy()
categories = frame['category'].to_numpy()

flam_mask = categories == 'Легковоспламеняющиеся'
flam = frame[flam_mask].copy()
flam['label'] = labels[flam_mask]

hl = flam[flam['title'].str.contains('хлопушк', case=False, na=False)]
print(f'Party crackers: {len(hl)}, positive: {hl["label"].sum()}')

# Text features
pyro_pat = r'пиротехник|громкий хлопок|18\+|пиро|fuse|wick'
pneumatic_pat = r'пневматическ|сжат.{0,5}воздух|pneumatic|compressed'

for name, pat in [('pyro', pyro_pat), ('pneumatic', pneumatic_pat)]:
    pos = hl[hl['label']==1]['text'].str.contains(pat, case=False, na=False, regex=True).sum()
    neg = hl[hl['label']==0]['text'].str.contains(pat, case=False, na=False, regex=True).sum()
    print(f'{name}: pos={pos}/{(hl["label"]==1).sum()}, neg={neg}/{(hl["label"]==0).sum()}')

# Brand/manufacturer patterns
print('\nBrands in positive party crackers:')
for _, row in hl[hl['label']==1].head(5).iterrows():
    text = str(row['text'])[:200]
    print(f'  {row["title"][:60]}')
    print(f'    text: {text}')
