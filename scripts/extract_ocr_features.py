"""Extract OCR text features from product images using Qwen3-VL-2B-Instruct.

Processes all 5 images per product and extracts text evidence relevant to
БАД and Легковоспламеняющиеся classification.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data

OCR_PROMPT = """Read all text visible in this product image. Focus on:
1. Any text markings like "БАД", "dietary supplement", "биологически активная добавка"
2. Warning labels, safety symbols, fire/flame icons
3. Product contents: gas, fuel, alcohol, flammable substances
4. Volume/weight indicators (ml, L, g, kg)
5. Category indicators: matches, lighters, fireworks, smoke, coal, briquettes
6. Brand name and product type
7. Any text about what's included in the set/kit

Return ONLY the extracted text, nothing else. If no text is visible, say "NO_TEXT"."""


def _resize_image(image: Image.Image, max_pixels: int = 100352) -> Image.Image:
    import math
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    new_w = max(28, int(w * scale) // 28 * 28)
    new_h = max(28, int(h * scale) // 28 * 28)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def extract_ocr_for_sample(
    frame: pd.DataFrame,
    output_path: str,
    model_path: str = "models/Qwen/Qwen3-VL-2B-Instruct",
    sample_size: int = 500,
    max_images: int = 3,
):
    """Extract OCR text for a sample of products."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    # Sample: all flammable positives + random flammable negatives + random BAD
    flam_pos = frame[(frame["category"] == "Легковоспламеняющиеся") & (
        frame["labels"].map(lambda v: int(v[0])) == 1
    )]
    flam_neg = frame[(frame["category"] == "Легковоспламеняющиеся") & (
        frame["labels"].map(lambda v: int(v[0])) == 0
    )].sample(n=min(200, len(flam_neg := frame[(frame["category"] == "Легковоспламеняющиеся") & (
        frame["labels"].map(lambda v: int(v[0])) == 0
    )])), random_state=42)
    bad_sample = frame[frame["category"] == "БАД"].sample(n=min(100, len(frame[frame["category"] == "БАД"])), random_state=42)

    sample = pd.concat([flam_pos, flam_neg, bad_sample]).drop_duplicates(subset=["id"]).reset_index(drop=True)
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"OCR extraction for {len(sample)} products ({len(flam_pos)} flam+, {len(flam_neg)} flam-, {len(bad_sample)} BAD)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    from transformers import Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    results = []
    start_time = time.time()

    for idx, row in sample.iterrows():
        product_id = str(row["id"])
        images_raw = row["images"]
        if isinstance(images_raw, str):
            import ast
            try:
                images_raw = ast.literal_eval(images_raw)
            except:
                images_raw = []

        ocr_texts = []
        for img_idx, img_path in enumerate(images_raw[:max_images]):
            try:
                with Image.open(img_path) as img:
                    img_resized = _resize_image(img)

                    messages = [
                        {"role": "system", "content": "You are an OCR assistant. Extract all text from product images accurately."},
                        {"role": "user", "content": [
                            {"type": "image", "image": img_resized},
                            {"type": "text", "text": OCR_PROMPT},
                        ]},
                    ]

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[img_resized], return_tensors="pt").to(device)

                    with torch.inference_mode():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            temperature=1.0,
                        )

                    generated = output[0][inputs["input_ids"].shape[1]:]
                    ocr_text = processor.decode(generated, skip_special_tokens=True).strip()
                    ocr_texts.append(ocr_text)
                    img_resized.close()
            except Exception as e:
                ocr_texts.append(f"ERROR: {e}")

        results.append({
            "id": product_id,
            "category": row["category"],
            "label": int(row["labels"][0]) if isinstance(row["labels"], list) else int(row["labels"]),
            "title": str(row["title"])[:200],
            "ocr_text_0": ocr_texts[0] if len(ocr_texts) > 0 else "",
            "ocr_text_1": ocr_texts[1] if len(ocr_texts) > 1 else "",
            "ocr_text_2": ocr_texts[2] if len(ocr_texts) > 2 else "",
            "all_ocr": " | ".join(ocr_texts),
        })

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(sample) - idx - 1) / rate
            print(f"  {idx+1}/{len(sample)} done, {rate:.1f} products/s, ETA {remaining/60:.1f}m")

    # Save results
    results_df = pd.DataFrame(results)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)
    print(f"\nSaved OCR results to {output}")

    # Quick analysis: keyword presence in OCR text
    keywords = {
        "BAD_marking": r"бад|биологически активн|dietary supplement",
        "flammable_warning": r"огнеопасн|легковоспламен|flammable|горюч|воспламен",
        "gas_fuel": r"газ|balloon|баллон|топлив|fuel",
        "matches_fire": r"спичк|зажигалк|розжиг|горелк|burner|lighter",
        "pyro": r"пиротехник|фейерверк|салют|бенгальск|дымов|шашк|хлопушк",
        "coal_briquette": r"уголь|брикет|charcoal|briquette",
        "volume": r"\d+\s*(?:мл|ml|л\b|l\b|г\b|g\b|кг|kg)",
        "set_contents": r"в комплект|в набор|содержит|включает",
    }

    print("\nKeyword analysis in OCR text:")
    for name, pattern in keywords.items():
        for cat in ["Легковоспламеняющиеся", "БАД"]:
            cat_df = results_df[results_df["category"] == cat]
            if len(cat_df) == 0:
                continue
            pos = cat_df[cat_df["label"] == 1]
            neg = cat_df[cat_df["label"] == 0]
            pos_rate = pos["all_ocr"].str.contains(pattern, case=False, na=False, regex=True).mean() if len(pos) > 0 else 0
            neg_rate = neg["all_ocr"].str.contains(pattern, case=False, na=False, regex=True).mean() if len(neg) > 0 else 0
            print(f"  {name:20s} | {cat:25s} | pos={pos_rate:.3f} neg={neg_rate:.3f} | diff={pos_rate-neg_rate:+.3f}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return results_df


def main():
    schema = str(ROOT / "configs" / "ozon_schema.json")
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"), schema, require_label=True)

    output_path = str(ROOT / "artifacts" / "ocr_features_sample.csv")
    extract_ocr_for_sample(frame, output_path, sample_size=500, max_images=3)


if __name__ == "__main__":
    main()
