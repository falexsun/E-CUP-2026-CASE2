"""MiniCPM-V-4.6 binary classifier for flammable/BAD.

Uses transformers>=5.7.0 AutoModelForImageTextToText.
Smoke test first, then full 498-product audit.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data


def classify_with_minicpm(
    frame: pd.DataFrame,
    model_path: str = "models/openbmb/MiniCPM-V-4_6",
    sample_size: int = 498,
    smoke_size: int = 8,
):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading MiniCPM-V-4.6 from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
    ).eval()
    # Multi-slice processing in transformers 5.15 can produce inconsistent
    # reshape sizes for extreme product-image aspect ratios. A padded square,
    # single-slice input is deterministic and keeps the whole image visible.
    processor_kwargs = {"downsample_mode": "16x", "max_slice_nums": 1}

    # Build sample
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()

    flam_pos = frame[(categories == "Легковоспламеняющиеся") & (labels == 1)]
    flam_neg = frame[(categories == "Легковоспламеняющиеся") & (labels == 0)].sample(
        n=min(200, len(frame[(categories == "Легковоспламеняющиеся") & (labels == 0)])), random_state=42)
    bad_pos = frame[(categories == "БАД") & (labels == 1)].sample(
        n=min(50, len(frame[(categories == "БАД") & (labels == 1)])), random_state=42)
    bad_neg = frame[(categories == "БАД") & (labels == 0)].sample(
        n=min(50, len(frame[(categories == "БАД") & (labels == 0)])), random_state=42)

    sample = pd.concat([flam_pos, flam_neg, bad_pos, bad_neg]).drop_duplicates(subset=["id"]).reset_index(drop=True)

    def get_image(row):
        images_raw = row["images"]
        if isinstance(images_raw, str):
            import ast
            try:
                images_raw = ast.literal_eval(images_raw)
            except:
                images_raw = []
        if images_raw:
            try:
                image = Image.open(images_raw[0]).convert("RGB")
                return ImageOps.pad(image, (448, 448), color=(255, 255, 255))
            except:
                return None
        return None

    # Smoke test
    smoke = sample.head(smoke_size)
    print(f"\n--- SMOKE TEST ({smoke_size} items) ---")
    for _, row in smoke.iterrows():
        category = str(row["category"])
        label = int(row["labels"][0]) if isinstance(row["labels"], list) else int(row["labels"])
        img = get_image(row)
        if img is None:
            print(f"  SKIP (no image): {row['title'][:60]}")
            continue

        if category == "Легковоспламеняющиеся":
            question = "Is this product flammable? Does it contain fire, gas, fuel, pyrotechnics, matches, or charcoal? Answer only Yes or No."
        else:
            question = "Is this a dietary supplement (БАД)? Look for БАД marking. Answer only Yes or No."

        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
        inputs = None
        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
                processor_kwargs=processor_kwargs,
            ).to(device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs, downsample_mode="16x", max_new_tokens=16, do_sample=False,
                )
            generated = output[0][inputs["input_ids"].shape[1]:]
            response = processor.decode(generated, skip_special_tokens=True).strip()
            print(f"  label={label} | {str(row['title'])[:60]} | response: {response[:80]}")
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            img.close()
            if inputs is not None:
                del inputs
            torch.cuda.empty_cache()

    # Full audit
    print(f"\n--- FULL AUDIT ({len(sample)} items) ---")
    results = []
    start_time = time.time()

    for idx, row in sample.iterrows():
        product_id = str(row["id"])
        category = str(row["category"])
        label = int(row["labels"][0]) if isinstance(row["labels"], list) else int(row["labels"])
        img = get_image(row)

        if img is None:
            results.append({"id": product_id, "category": category, "label": label,
                           "response": "NO_IMAGE", "yes_prob": 0.5, "title": str(row["title"])[:200]})
            continue

        if category == "Легковоспламеняющиеся":
            question = "Is this product flammable? Does it contain fire, gas, fuel, pyrotechnics, matches, or charcoal? Answer only Yes or No."
        else:
            question = "Is this a dietary supplement (БАД)? Look for БАД marking. Answer only Yes or No."

        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
        inputs = None
        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
                processor_kwargs=processor_kwargs,
            ).to(device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs, downsample_mode="16x", max_new_tokens=16, do_sample=False,
                )
            generated = output[0][inputs["input_ids"].shape[1]:]
            response = processor.decode(generated, skip_special_tokens=True).strip()

            resp_lower = response.strip().lower()
            if resp_lower.startswith("yes"):
                yes_prob = 0.9
            elif resp_lower.startswith("no"):
                yes_prob = 0.1
            elif "yes" in resp_lower[:20]:
                yes_prob = 0.8
            elif "no" in resp_lower[:20]:
                yes_prob = 0.2
            else:
                yes_prob = 0.5

            results.append({"id": product_id, "category": category, "label": label,
                           "response": response[:200], "yes_prob": yes_prob,
                           "title": str(row["title"])[:200]})
        except Exception as e:
            results.append({"id": product_id, "category": category, "label": label,
                           "response": f"ERROR: {e}", "yes_prob": 0.5,
                           "title": str(row["title"])[:200]})
        finally:
            img.close()
            if inputs is not None:
                del inputs
            torch.cuda.empty_cache()

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(sample) - idx - 1) / rate
            print(f"  {idx+1}/{len(sample)}, {rate:.1f}/s, ETA {remaining/60:.1f}m")

    # Analysis
    results_df = pd.DataFrame(results)
    output = ROOT / "artifacts" / "minicpm_classifier_sample.csv"
    results_df.to_csv(output, index=False)
    print(f"\nSaved to {output}")

    invalid = results_df["response"].str.startswith(("ERROR:", "NO_IMAGE"), na=False)
    print(f"Invalid rows: {int(invalid.sum())}/{len(results_df)}")
    if invalid.any():
        raise RuntimeError("MiniCPM audit is invalid: inference errors or missing images remain")

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

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    classify_with_minicpm(frame)


if __name__ == "__main__":
    main()
