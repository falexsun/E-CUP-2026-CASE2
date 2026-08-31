"""Resumable zero-shot Qwen3-VL compliance scores from first-token logits."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from ozon_quality.data import load_data

DEFAULT_VQA_MODEL: Final[str] = "Qwen/Qwen3-VL-2B-Instruct"
RULE_PROMPTS: Final[dict[str, str]] = {
    "БАД": (
        "Верни 1 только если в тексте или на упаковке есть прямая маркировка БАД, "
        "биологически активная добавка или dietary supplement. Спортивное питание, явное "
        "указание что товар не БАД и отсутствие прямой маркировки означают 0."
    ),
    "Легковоспламеняющиеся": (
        "Верни 1, если сам товар является источником открытого огня, содержит горючее "
        "вещество или газ либо такой предмет входит в комплект. Верни 0 для пустой "
        "конструкции, встроенного источника, горючего компонента изделия или отсутствующего "
        "в комплекте содержимого."
    ),
}


def _fingerprint(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _open_main_image(paths: list[str], max_pixels: int) -> Image.Image | None:
    if not paths:
        return None
    try:
        with Image.open(paths[0]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            if width * height > max_pixels:
                scale = math.sqrt(max_pixels / (width * height))
                width = max(28, int(width * scale) // 28 * 28)
                height = max(28, int(height * scale) // 28 * 28)
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            return image.copy()
    except (OSError, ValueError):
        return None


def _render_prompt(processor: Any, row: Any, with_image: bool) -> str:
    content: list[dict[str, str]] = []
    if with_image:
        content.append({"type": "image", "image": "placeholder"})
    content.append(
        {
            "type": "text",
            "text": (
                f"Категория: {row.category}\nПравила: {RULE_PROMPTS[str(row.category)]}\n"
                f"Название: {row.title}\nОписание: {row.description[:2500]}\n\n"
                "Ответь строго одним символом: 1 если товар соответствует правилам указанной "
                "категории (не бан), иначе 0 (бан)."
            ),
        }
    )
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Ты строгий модератор маркетплейса. Не рассуждай и верни только 0 или 1.",
                }
            ],
        },
        {"role": "user", "content": content},
    ]
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def extract_vqa_scores(
    input_path: str,
    output: str,
    *,
    schema: str,
    model_path: str = DEFAULT_VQA_MODEL,
    batch_size: int = 32,
    max_pixels: int = 100_352,
    limit: int | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    frame, _ = load_data(input_path, schema, require_label=False)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    ids = frame["id"].astype(str).tolist()
    fingerprint = _fingerprint(ids)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    scores_path = output_dir / "scores.npy"
    ids_path = output_dir / "ids.csv"
    start = 0
    state: dict[str, Any] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint or state.get("rows") != len(frame):
            raise ValueError("Existing VQA cache belongs to different rows/order")
        start = int(state.get("next_index", 0))
    else:
        pd.DataFrame({"id": ids}).to_csv(ids_path, index=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    token_zero = processor.tokenizer.encode("0", add_special_tokens=False)
    token_one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(token_zero) != 1 or len(token_one) != 1:
        raise ValueError("VQA verdicts 0/1 must each map to a single tokenizer token")
    scores = (
        np.lib.format.open_memmap(scores_path, mode="r+")
        if start
        else np.lib.format.open_memmap(
            scores_path, mode="w+", dtype="float32", shape=(len(frame),)
        )
    )
    broken = int(state.get("missing_or_broken_images", 0))
    for batch_start in range(start, len(frame), batch_size):
        batch_end = min(batch_start + batch_size, len(frame))
        images: list[Image.Image] = []
        prompts: list[str] = []
        for row in frame.iloc[batch_start:batch_end].itertuples(index=False):
            image = _open_main_image(row.images, max_pixels)
            prompts.append(_render_prompt(processor, row, image is not None))
            if image is not None:
                images.append(image)
            else:
                broken += 1
        kwargs: dict[str, Any] = {
            "text": prompts,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if images:
            kwargs["images"] = images
        inputs = processor(**kwargs).to("cuda")
        with torch.inference_mode():
            output_logits = model(**inputs, logits_to_keep=1).logits[:, -1]
        verdict_logits = output_logits[:, [token_zero[0], token_one[0]]].float()
        scores[batch_start:batch_end] = torch.softmax(verdict_logits, dim=-1)[
            :, 1
        ].cpu().numpy()
        scores.flush()
        for image in images:
            image.close()
        state = {
            "artifact_version": 1,
            "fingerprint": fingerprint,
            "rows": len(frame),
            "next_index": batch_end,
            "complete": batch_end == len(frame),
            "model": model_path,
            "max_pixels": max_pixels,
            "missing_or_broken_images": broken,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"vqa scores: {batch_end}/{len(frame)}", flush=True)
    del model, processor, scores
    gc.collect()
    torch.cuda.empty_cache()
    return state
