"""Extract embeddings from all product images (not just main).

Goal: test whether aggregating embeddings from multiple images provides
new signal beyond the main image alone, especially for items where
the main image doesn't show flammable content but secondary images do.

Approach:
1. Extract per-image embeddings for images 0-4
2. Aggregate: mean pooling across available images
3. Evaluate fold-local Logistic Regression on multi-image vs single-image
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


def _resize_image(image: Image.Image, max_pixels: int = 100352) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    new_w = max(28, int(w * scale) // 28 * 28)
    new_h = max(28, int(h * scale) // 28 * 28)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def extract_multi_image_embeddings(
    input_path: str,
    output: str,
    *,
    schema: str,
    model_path: str = "models/Qwen/Qwen3-VL-Embedding-2B",
    batch_size: int = 16,
    max_images: int = 5,
    max_pixels: int = 100_352,
):
    """Extract per-image embeddings for all images, then compute mean aggregation."""
    import torch
    from transformers import AutoModel, AutoProcessor

    from ozon_quality.data import load_data

    if not torch.cuda.is_available():
        raise RuntimeError("Requires CUDA device")

    frame, _ = load_data(input_path, schema, require_label=False)
    ids = frame["id"].astype(str).tolist()

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(
        model_path, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device).eval()

    system_prompt = (
        "Represent this marketplace product image for category compliance classification. "
        "Pay attention to package markings, product contents, warning labels, "
        "flammable substances and safety symbols."
    )

    # Process one product at a time (multiple images per product)
    all_embeddings = []  # list of (n_images, dim) arrays per product
    start_time = time.time()

    for row_idx, row in frame.iterrows():
        images_raw = row["images"]
        if isinstance(images_raw, str):
            import ast
            try:
                images_raw = ast.literal_eval(images_raw)
            except:
                images_raw = []

        # Open and resize all images
        opened = []
        for img_path in images_raw[:max_images]:
            try:
                with Image.open(img_path) as im:
                    opened.append(_resize_image(im, max_pixels).copy())
            except:
                continue

        if not opened:
            dim = 2048  # default for Qwen3-VL-Embedding-2B
            all_embeddings.append(np.zeros((1, dim), dtype="float16"))
            continue

        # Extract embedding for each image separately
        per_image_embs = []
        for img in opened:
            try:
                conversations = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Category: {row['category']}\nProduct: {row['title']}"},
                            {"type": "image", "image": img},
                        ],
                    },
                ]
                try:
                    model_inputs = processor.apply_chat_template(
                        conversations, tokenize=True, return_dict=True,
                        return_tensors="pt", add_generation_prompt=True,
                    )
                except TypeError:
                    model_inputs = processor.apply_chat_template(
                        conversations, tokenize=True, return_dict=True,
                        return_tensors="pt", padding=True, truncation="longest_first",
                        add_generation_prompt=True,
                    )
                model_inputs = model_inputs.to(device)
                with torch.inference_mode():
                    hidden = model(**model_inputs).last_hidden_state
                attention_mask = model_inputs["attention_mask"]
                last_pos = attention_mask.shape[1] - attention_mask.flip(1).argmax(1) - 1
                rows_t = torch.arange(hidden.shape[0], device=hidden.device)
                emb = torch.nn.functional.normalize(
                    hidden[rows_t, last_pos].float(), p=2, dim=-1
                ).cpu().numpy().astype("float16")
                per_image_embs.append(emb[0])
            except Exception:
                continue

        for img in opened:
            img.close()

        if per_image_embs:
            all_embeddings.append(np.stack(per_image_embs))
        else:
            dim = all_embeddings[0].shape[1] if all_embeddings else 2048
            all_embeddings.append(np.zeros((1, dim), dtype="float16"))

        if (row_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (row_idx + 1) / elapsed
            remaining = (len(frame) - row_idx - 1) / rate
            print(f"  {row_idx+1}/{len(frame)} products, {rate:.1f}/s, ETA {remaining/60:.1f}m")

    # Determine embedding dimension from first non-empty entry
    dim = next(e.shape[1] for e in all_embeddings if len(e) > 0)

    # Save per-image embeddings
    # For memory efficiency, save as mean-pooled and also per-image max
    mean_embeddings = np.zeros((len(frame), dim), dtype="float16")
    max_embeddings = np.zeros((len(frame), dim), dtype="float16")

    for i, embs in enumerate(all_embeddings):
        mean_embeddings[i] = embs.mean(axis=0).astype("float16")
        max_embeddings[i] = embs.max(axis=0).astype("float16")

    np.save(output_dir / "mean_embeddings.npy", mean_embeddings)
    np.save(output_dir / "max_embeddings.npy", max_embeddings)

    # Save metadata
    state = {
        "rows": len(frame),
        "dimensions": dim,
        "max_images": max_images,
        "model": model_path,
        "aggregations": ["mean", "max"],
    }
    (output_dir / "state.json").write_text(json.dumps(state, indent=2))
    pd.DataFrame({"id": ids}).to_csv(output_dir / "ids.csv", index=False)

    print(f"\nSaved to {output_dir}")
    print(f"Mean embeddings: {mean_embeddings.shape}")
    print(f"Max embeddings: {max_embeddings.shape}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def main():
    extract_multi_image_embeddings(
        input_path=str(ROOT / "data" / "full_grouped.csv"),
        output=str(ROOT / "artifacts" / "multi_image_embeddings"),
        schema=str(ROOT / "configs" / "ozon_schema.json"),
    )


if __name__ == "__main__":
    main()
