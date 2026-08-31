"""Test Qwen3-VL-2B-Instruct as an independent binary classifier for flammable category.

Uses structured yes/no prompting with logprob extraction.
Runs on a stratified sample first, then can be scaled to full data.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data

FLAMMABLE_PROMPT = """Is this product flammable or does it contain flammable materials? Consider:
- Fire sources: matches, lighters, burners, torches
- Fuel: gas, gasoline, alcohol, solid fuel tablets
- Pyrotechnics: fireworks, smoke bombs, sparklers, party crackers with pyrotechnic elements
- Combustible materials: charcoal, briquettes, fire starters

Answer ONLY "Yes" or "No"."""

BAD_PROMPT = """Is this product a dietary supplement (БАД/биологически активная добавка)?
Look for markings: "БАД", "dietary supplement", "биологически активная добавка к пище".
Regular vitamins, sports nutrition without БАД marking = No.

Answer ONLY "Yes" or "No"."""


def _resize_image(image: Image.Image, max_pixels: int = 100352) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    new_w = max(28, int(w * scale) // 28 * 28)
    new_h = max(28, int(h * scale) // 28 * 28)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def classify_with_vlm(
    frame: pd.DataFrame,
    output_path: str,
    model_path: str = "models/Qwen/Qwen3-VL-2B-Instruct",
    sample_size: int = 500,
):
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # Stratified sample: all flammable positives, random negatives, random BAD
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()

    flam_pos = frame[(categories == "Легковоспламеняющиеся") & (labels == 1)]
    flam_neg = frame[(categories == "Легковоспламеняющиеся") & (labels == 0)].sample(
        n=min(200, len(frame[(categories == "Легковоспламеняющиеся") & (labels == 0)])), random_state=42
    )
    bad_pos = frame[(categories == "БАД") & (labels == 1)].sample(n=min(50, len(frame[(categories == "БАД") & (labels == 1)])), random_state=42)
    bad_neg = frame[(categories == "БАД") & (labels == 0)].sample(n=min(50, len(frame[(categories == "БАД") & (labels == 0)])), random_state=42)

    sample = pd.concat([flam_pos, flam_neg, bad_pos, bad_neg]).drop_duplicates(subset=["id"]).reset_index(drop=True)
    print(f"VLM classification sample: {len(sample)} products")
    print(f"  Flammable+: {(sample['category']=='Легковоспламеняющиеся').sum()}, BAD: {(sample['category']=='БАД').sum()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device).eval()

    tokenizer = processor.tokenizer
    yes_token = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token = tokenizer.encode("No", add_special_tokens=False)[0]

    results = []
    start_time = time.time()

    for idx, row in sample.iterrows():
        product_id = str(row["id"])
        category = str(row["category"])
        label = int(row["labels"][0]) if isinstance(row["labels"], list) else int(row["labels"])
        images_raw = row["images"]
        if isinstance(images_raw, str):
            import ast
            try:
                images_raw = ast.literal_eval(images_raw)
            except:
                images_raw = []

        # Use main image
        img = None
        if images_raw:
            try:
                with Image.open(images_raw[0]) as im:
                    img = _resize_image(im).copy()
            except:
                pass

        prompt = FLAMMABLE_PROMPT if category == "Легковоспламеняющиеся" else BAD_PROMPT
        task_text = "Определи, является ли товар легковоспламеняющимся." if category == "Легковоспламеняющиеся" else "Определи, является ли товар биологически активной добавкой (БАД)."

        messages = [
            {"role": "system", "content": "You are a product safety classifier. Answer only Yes or No."},
            {"role": "user", "content": [
                *([{"type": "image", "image": img}] if img is not None else []),
                {"type": "text", "text": f"{task_text}\n\nProduct: {str(row['title'])[:300]}\nDescription: {str(row['description'])[:500]}\n\n{prompt}"},
            ]},
        ]

        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img] if img else None, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = model(**inputs)
                # Get logits for next token
                next_logits = outputs.logits[0, -1, :]
                yes_logit = next_logits[yes_token].item()
                no_logit = next_logits[no_token].item()

                # Convert to probability
                max_logit = max(yes_logit, no_logit)
                yes_prob = np.exp(yes_logit - max_logit) / (np.exp(yes_logit - max_logit) + np.exp(no_logit - max_logit))

            if img:
                img.close()

            results.append({
                "id": product_id,
                "category": category,
                "label": label,
                "yes_prob": float(yes_prob),
                "yes_logit": float(yes_logit),
                "no_logit": float(no_logit),
                "title": str(row["title"])[:200],
            })
        except Exception as e:
            results.append({
                "id": product_id,
                "category": category,
                "label": label,
                "yes_prob": 0.5,
                "yes_logit": 0.0,
                "no_logit": 0.0,
                "title": str(row["title"])[:200],
                "error": str(e),
            })
            if img:
                img.close()

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(sample) - idx - 1) / rate
            print(f"  {idx+1}/{len(sample)} done, {rate:.1f}/s, ETA {remaining/60:.1f}m")

    # Analysis
    results_df = pd.DataFrame(results)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)
    print(f"\nSaved to {output}")

    from sklearn.metrics import f1_score, average_precision_score

    for cat in ["Легковоспламеняющиеся", "БАД"]:
        cat_df = results_df[results_df["category"] == cat]
        if len(cat_df) == 0:
            continue
        y = cat_df["label"].to_numpy()
        p = cat_df["yes_prob"].to_numpy()

        # Find best threshold
        best_f1, best_thresh = 0, 0.5
        for thresh in np.linspace(0.1, 0.9, 17):
            pred = (p >= thresh).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else 0
        pred = (p >= best_thresh).astype(int)
        print(f"\n{cat}:")
        print(f"  Best F1={best_f1:.4f} at threshold={best_thresh:.2f}")
        print(f"  AP={ap:.4f}")
        print(f"  TP={((y==1)&(pred==1)).sum()}, FP={((y==0)&(pred==1)).sum()}")
        print(f"  FN={((y==1)&(pred==0)).sum()}, TN={((y==0)&(pred==0)).sum()}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    classify_with_vlm(frame, str(ROOT / "artifacts" / "vlm_classifier_sample.csv"))


if __name__ == "__main__":
    main()
