"""Compare OCR features vs text features to assess OCR's unique value."""

import re
import numpy as np
import pandas as pd
from ozon_quality.data import load_data

frame, _ = load_data('data/full_grouped.csv', 'configs/ozon_schema.json', require_label=True)
labels = frame['labels'].map(lambda v: int(v[0])).to_numpy()
categories = frame['category'].to_numpy()

# Load OCR results
ocr_df = pd.read_csv('artifacts/ocr_features_sample.csv', on_bad_lines='skip')
ocr_df['id'] = ocr_df['id'].astype(str)

# Merge
frame['id'] = frame['id'].astype(str)
merged = frame.merge(ocr_df[['id', 'all_ocr']], on='id', how='inner')
merged['label'] = labels[:len(merged)]

print(f'Merged: {len(merged)} products')

# Compare text vs OCR for key keywords
keywords = {
    'pyro': r'пиротехник|хлопушк|петард|салют|бенгальск|дымов|шашк|фейерверк|громкий хлопок|18\+',
    'coal': r'уголь|брикет|charcoal|briquette',
    'gas_fuel': r'газ|баллон|топлив|fuel|бензин',
    'bad_marking': r'бад|биологически активн|dietary supplement',
}

for cat in ['Легковоспламеняющиеся', 'БАД']:
    cat_df = merged[merged['category'] == cat]
    pos = cat_df[cat_df['label'] == 1]
    neg = cat_df[cat_df['label'] == 0]
    
    print(f'\n{cat} ({len(cat_df)} products, {len(pos)} pos, {len(neg)} neg):')
    print(f'  {"keyword":20s} | {"text_pos":>8} {"text_neg":>8} {"text_diff":>9} | {"ocr_pos":>8} {"ocr_neg":>8} {"ocr_diff":>9} | {"ocr_unique":>10}')
    
    for name, pattern in keywords.items():
        text_pos = pos['text'].str.contains(pattern, case=False, na=False, regex=True).mean()
        text_neg = neg['text'].str.contains(pattern, case=False, na=False, regex=True).mean()
        ocr_pos = pos['all_ocr'].str.contains(pattern, case=False, na=False, regex=True).mean()
        ocr_neg = neg['all_ocr'].str.contains(pattern, case=False, na=False, regex=True).mean()
        
        # OCR unique: has keyword in OCR but not in text
        ocr_unique_pos = ((pos['all_ocr'].str.contains(pattern, case=False, na=False, regex=True)) & 
                         (~pos['text'].str.contains(pattern, case=False, na=False, regex=True))).mean()
        
        print(f'  {name:20s} | {text_pos:>8.3f} {text_neg:>8.3f} {text_pos-text_neg:>+9.3f} | {ocr_pos:>8.3f} {ocr_neg:>8.3f} {ocr_pos-ocr_neg:>+9.3f} | {ocr_unique_pos:>10.3f}')
