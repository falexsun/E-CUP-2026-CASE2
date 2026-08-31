"""Analyze OCR extraction results."""

import pandas as pd

df = pd.read_csv("artifacts/ocr_features_sample.csv", on_bad_lines="skip")
print(f"Total: {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print()

# Party crackers
hl = df[df["title"].str.contains("хлопушк", case=False, na=False)]
print(f"Party crackers in sample: {len(hl)}")
for _, row in hl.iterrows():
    ocr = str(row.get("all_ocr", ""))[:200].replace("\n", " ")
    print(f"  id={row['id']} label={row['label']} | {ocr}")

# Pyro keywords
print("\n\nPYRO keywords in OCR text:")
pyro_kw = r"пиротехник|хлопушк|петард|салют|бенгальск|дымов|шашк|фейерверк"
for cat in ["Легковоспламеняющиеся", "БАД"]:
    cat_df = df[df["category"] == cat]
    pos = cat_df[cat_df["label"] == 1]
    neg = cat_df[cat_df["label"] == 0]
    pos_rate = pos["all_ocr"].str.contains(pyro_kw, case=False, na=False, regex=True).mean() if len(pos) > 0 else 0
    neg_rate = neg["all_ocr"].str.contains(pyro_kw, case=False, na=False, regex=True).mean() if len(neg) > 0 else 0
    print(f"  {cat}: pos={pos_rate:.3f} neg={neg_rate:.3f} diff={pos_rate-neg_rate:+.3f}")

# BAD marking
print("\nBAD marking in OCR text:")
bad_kw = r"бад|биологически активн|dietary supplement"
for cat in ["Легковоспламеняющиеся", "БАД"]:
    cat_df = df[df["category"] == cat]
    pos = cat_df[cat_df["label"] == 1]
    neg = cat_df[cat_df["label"] == 0]
    pos_rate = pos["all_ocr"].str.contains(bad_kw, case=False, na=False, regex=True).mean() if len(pos) > 0 else 0
    neg_rate = neg["all_ocr"].str.contains(bad_kw, case=False, na=False, regex=True).mean() if len(neg) > 0 else 0
    print(f"  {cat}: pos={pos_rate:.3f} neg={neg_rate:.3f} diff={pos_rate-neg_rate:+.3f}")

# Show some interesting OCR extractions
print("\n\nSample OCR extractions (flammable positives with pyro keywords):")
pos_flam = df[(df["category"] == "Легковоспламеняющиеся") & (df["label"] == 1)]
pyro_mask = pos_flam["all_ocr"].str.contains(pyro_kw, case=False, na=False, regex=True)
for _, row in pos_flam[pyro_mask].head(10).iterrows():
    print(f"  {row['title'][:60]}")
    print(f"    OCR: {str(row.get('all_ocr', ''))[:200]}")
    print()
