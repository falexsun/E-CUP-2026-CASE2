"""Fast batch OCR extraction for all products using Qwen3-VL-2B-Instruct.

Processes one image per product (main image) in batch mode for efficiency.
Extracts text and keyword features for downstream use.
"""

from __future__ import annotations

import gc
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data


def _resize_image(image: Image.Image, max_pixels: int = 80_000) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    new_w = max(28, int(w * scale) // 28 * 28)
    new_h = max(28, int(h * scale) // 28 * 28)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


OCR_PROMPT = "Extract all text visible in this image. Be concise. If no text, say NONE."


def extract_ocr_features(text: str) -> dict[str, float]:
    t = text.lower()
    return {
        "ocr_pyro": float(bool(re.search(r"пиротехник|хлопушк|петард|салют|бенгальск|дымов|шашк|фейерверк|18\+|громкий", t))),
        "ocr_coal": float(bool(re.search(r"уголь|брикет|charcoal|briquette", t))),
        "ocr_gas_fuel": float(bool(re.search(r"газ|баллон|топлив|fuel|бензин", t))),
        "ocr_matches": float(bool(re.search(r"спичк|зажигалк|розжиг|горелк|burner|lighter|сухое горюч", t))),
        "ocr_bad_marking": float(bool(re.search(r"бад|биологически активн|dietary supplement", t))),
        "ocr_warning": float(bool(re.search(r"огнеопасн|легковоспламен|flammable|горюч|воспламен|18\+|класс опасност", t))),
        "ocr_set_contents": float(bool(re.search(r"в комплект|в набор|содержит|включает|внутри", t))),
        "ocr_confetti": float(bool(re.search(r"конфетти|серпантин|confetti", t))),
    }


def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=False)
    ids = frame["id"].astype(str).tolist()

    output_dir = ROOT / "artifacts" / "ocr_features_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for partial results
    state_path = output_dir / "state.json"
    start = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        start = int(state.get("next_index", 0))
        print(f"Resuming from index {start}")

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_path = "models/Qwen/Qwen3-VL-2B-Instruct"
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # Decoder-only batched generation requires left padding; right padding can
    # change generated OCR tokens for shorter examples in the same batch.
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device).eval()

    feature_names = list(extract_ocr_features("").keys())
    all_features = np.zeros((len(frame), len(feature_names)), dtype="float32")
    all_texts = [""] * len(frame)

    if start > 0:
        prev = np.load(str(output_dir / "features.npy"), mmap_mode="r")
        all_features[:start] = prev[:start]
        prev_texts = pd.read_csv(str(output_dir / "texts.csv"), nrows=start)
        for i, t in enumerate(prev_texts["ocr_text"]):
            all_texts[i] = str(t)

    batch_size = 8  # Process 8 products at a time
    start_time = time.time()

    for batch_start in range(start, len(frame), batch_size):
        batch_end = min(batch_start + batch_size, len(frame))
        batch_frame = frame.iloc[batch_start:batch_end]

        # Prepare batch
        images = []
        texts = []
        valid_indices = []

        for i, (_, row) in enumerate(batch_frame.iterrows()):
            img_path = None
            if isinstance(row["images"], list) and row["images"]:
                img_path = row["images"][0]
            elif isinstance(row["images"], str):
                import ast
                try:
                    paths = ast.literal_eval(row["images"])
                    if paths:
                        img_path = paths[0]
                except:
                    pass

            img = None
            if img_path:
                try:
                    with Image.open(img_path) as im:
                        img = _resize_image(im).copy()
                except:
                    pass

            images.append(img)
            texts.append(OCR_PROMPT)
            valid_indices.append(i)

        # Build conversations
        messages_batch = []
        for img, text in zip(images, texts):
            content = []
            if img is not None:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": text})
            messages_batch.append([
                {"role": "system", "content": "You are an OCR assistant."},
                {"role": "user", "content": content},
            ])

        try:
            template = processor.apply_chat_template(
                messages_batch, tokenize=False, add_generation_prompt=True
            )
            if isinstance(template, str):
                template = [template]

            batch_images = [img for img in images if img is not None]
            inputs = processor(
                text=template,
                images=batch_images if batch_images else None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )

            # Decode outputs
            for i in range(len(messages_batch)):
                generated = outputs[i][inputs["input_ids"].shape[1]:]
                ocr_text = processor.decode(generated, skip_special_tokens=True).strip()
                idx = batch_start + i
                all_texts[idx] = ocr_text
                feats = extract_ocr_features(ocr_text)
                all_features[idx] = [feats[k] for k in feature_names]

        except Exception as e:
            print(f"  Error at batch {batch_start}: {e}")
            # Fill with zeros
            for i in range(len(messages_batch)):
                idx = batch_start + i
                all_texts[idx] = "ERROR"

        # Close images
        for img in images:
            if img is not None:
                img.close()

        if (batch_end) % 100 == 0 or batch_end == len(frame):
            elapsed = time.time() - start_time
            rate = (batch_end - start) / elapsed
            remaining = (len(frame) - batch_end) / rate if rate > 0 else 0
            print(f"  {batch_end}/{len(frame)} products, {rate:.1f}/s, ETA {remaining/60:.1f}m")

        # Save periodically
        if batch_end % 500 == 0 or batch_end == len(frame):
            np.save(str(output_dir / "features.npy"), all_features)
            pd.DataFrame({"id": ids, "ocr_text": all_texts}).to_csv(str(output_dir / "texts.csv"), index=False)
            state = {"next_index": batch_end, "total": len(frame), "feature_names": feature_names}
            state_path.write_text(json.dumps(state, indent=2))

    # Final save
    np.save(str(output_dir / "features.npy"), all_features)
    pd.DataFrame({"id": ids, "ocr_text": all_texts}).to_csv(str(output_dir / "texts.csv"), index=False)
    state = {"next_index": len(frame), "total": len(frame), "feature_names": feature_names, "complete": True}
    state_path.write_text(json.dumps(state, indent=2))

    print(f"\nSaved to {output_dir}")
    print(f"Features: {all_features.shape}")
    print(f"Feature sums: {all_features.sum(axis=0)}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
