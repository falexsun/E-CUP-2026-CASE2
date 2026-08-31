"""InternVL3.5-2B binary classifier for flammable/BAD.

Uses the model's official .chat() API and dynamic_preprocess image pipeline.
Requires isolated venv with transformers==4.56.2.
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
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_path, input_size=448, max_num=6):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def classify_with_internvl(
    frame: pd.DataFrame,
    model_path: str = "models/OpenGVLab/InternVL3_5-2B",
    sample_size: int = 498,
    smoke_size: int = 8,
):
    from transformers import AutoModel, AutoTokenizer

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading InternVL3.5 from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # Build sample
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()

    flam_pos = frame[(categories == "Легковоспламеняющиеся") & (labels == 1)]
    flam_neg = frame[(categories == "Легковоспламеняющиеся") & (labels == 0)].sample(
        n=min(200, len(frame[(categories == "Легковоспламеняющиеся") & (labels == 0)])), random_state=42
    )
    bad_pos = frame[(categories == "БАД") & (labels == 1)].sample(
        n=min(50, len(frame[(categories == "БАД") & (labels == 1)])), random_state=42
    )
    bad_neg = frame[(categories == "БАД") & (labels == 0)].sample(
        n=min(50, len(frame[(categories == "БАД") & (labels == 0)])), random_state=42
    )
    sample = pd.concat([flam_pos, flam_neg, bad_pos, bad_neg]).drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Smoke test first
    smoke = sample.head(smoke_size)
    print(f"\n--- SMOKE TEST ({smoke_size} items) ---")
    for _, row in smoke.iterrows():
        category = str(row["category"])
        label = int(row["labels"][0]) if isinstance(row["labels"], list) else int(row["labels"])
        images_raw = row["images"]
        if isinstance(images_raw, str):
            import ast
            try:
                images_raw = ast.literal_eval(images_raw)
            except:
                images_raw = []

        if not images_raw:
            print(f"  SKIP (no images): {row['title'][:60]}")
            continue

        try:
            pixel_values = load_image(images_raw[0], max_num=4).to(dtype).cuda()
            if category == "Легковоспламеняющиеся":
                question = "<image>\nIs this product flammable? Does it contain fire, gas, fuel, pyrotechnics, matches, or charcoal? Answer only Yes or No."
            else:
                question = "<image>\nIs this a dietary supplement (БАД)? Look for БАД marking. Answer only Yes or No."

            generation_config = dict(max_new_tokens=16, do_sample=False)
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            print(f"  label={label} | {str(row['title'])[:60]} | response: {response[:80]}")
            del pixel_values
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")

    # Full audit
    print(f"\n--- FULL AUDIT ({len(sample)} items) ---")
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

        if not images_raw:
            results.append({"id": product_id, "category": category, "label": label,
                           "response": "NO_IMAGE", "yes_prob": 0.5, "title": str(row["title"])[:200]})
            continue

        try:
            pixel_values = load_image(images_raw[0], max_num=4).to(dtype).cuda()
            if category == "Легковоспламеняющиеся":
                question = "<image>\nIs this product flammable? Does it contain fire, gas, fuel, pyrotechnics, matches, or charcoal? Answer only Yes or No."
            else:
                question = "<image>\nIs this a dietary supplement (БАД)? Look for БАД marking. Answer only Yes or No."

            generation_config = dict(max_new_tokens=16, do_sample=False)
            response = model.chat(tokenizer, pixel_values, question, generation_config)

            # Parse yes/no
            resp_lower = response.strip().lower()
            if resp_lower.startswith("yes"):
                yes_prob = 0.9
            elif resp_lower.startswith("no"):
                yes_prob = 0.1
            else:
                # Try to extract from response
                if "yes" in resp_lower[:20]:
                    yes_prob = 0.8
                elif "no" in resp_lower[:20]:
                    yes_prob = 0.2
                else:
                    yes_prob = 0.5

            results.append({"id": product_id, "category": category, "label": label,
                           "response": response[:200], "yes_prob": yes_prob,
                           "title": str(row["title"])[:200]})
            del pixel_values
            torch.cuda.empty_cache()
        except Exception as e:
            results.append({"id": product_id, "category": category, "label": label,
                           "response": f"ERROR: {e}", "yes_prob": 0.5,
                           "title": str(row["title"])[:200]})

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(sample) - idx - 1) / rate
            print(f"  {idx+1}/{len(sample)}, {rate:.1f}/s, ETA {remaining/60:.1f}m")

    # Analysis
    results_df = pd.DataFrame(results)
    output = ROOT / "artifacts" / "internvl35_classifier_sample.csv"
    results_df.to_csv(output, index=False)
    print(f"\nSaved to {output}")

    from sklearn.metrics import f1_score, average_precision_score

    for cat in ["Легковоспламеняющиеся", "БАД"]:
        cat_df = results_df[results_df["category"] == cat]
        if len(cat_df) == 0:
            continue
        y = cat_df["label"].to_numpy()
        p = cat_df["yes_prob"].to_numpy()

        best_f1, best_thresh = 0, 0.5
        for thresh in [0.3, 0.5, 0.7]:
            pred = (p >= thresh).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else 0
        pred = (p >= best_thresh).astype(int)
        print(f"\n{cat}:")
        print(f"  Best F1={best_f1:.4f} at t={best_thresh:.1f}, AP={ap:.4f}")
        print(f"  TP={((y==1)&(pred==1)).sum()}, FP={((y==0)&(pred==1)).sum()}")
        print(f"  FN={((y==1)&(pred==0)).sum()}, TN={((y==0)&(pred==0)).sum()}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    classify_with_internvl(frame)


if __name__ == "__main__":
    main()
