#!/usr/bin/env python3
"""Extract leakage-safe few-shot Qwen VLM scores for an evaluation split."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data  # noqa: E402
from ozon_quality.official_vqa import (  # noqa: E402
    RULE_PROMPTS,
    _fingerprint,
    _open_main_image,
)


def render_prompt(
    processor: Any,
    row: Any,
    examples: pd.DataFrame,
    *,
    with_image: bool,
) -> str:
    content: list[dict[str, str]] = []
    if with_image:
        content.append({"type": "image", "image": "placeholder"})
    demonstrations = []
    # Interleave classes to avoid a positional block-label bias.
    for number, index in enumerate((0, 2, 1, 3), start=1):
        example = examples.iloc[index]
        demonstrations.append(
            f"Пример {number}\nНазвание: {example['title']}\n"
            f"Описание: {str(example['description'])[:500]}\n"
            f"Правильный ответ: {int(example['label'])}"
        )
    content.append(
        {
            "type": "text",
            "text": (
                f"Категория: {row.category}\nПравила: {RULE_PROMPTS[str(row.category)]}\n\n"
                "Ниже приведены похожие товары с правильными ответами. Учитывай различия "
                "в составе и комплектности, не копируй ответ только из-за похожего названия.\n\n"
                + "\n\n".join(demonstrations)
                + f"\n\nНовый товар\nНазвание: {row.title}\n"
                f"Описание: {row.description[:2500]}\n\n"
                "Ответь строго одним символом: 1, если новый товар соответствует категории, "
                "иначе 0."
            ),
        }
    )
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Ты строгий модератор маркетплейса. Верни только 0 или 1.",
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
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference-data", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=100_352)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    frame, _ = load_data(args.input, args.schema, require_label=False)
    reference, _ = load_data(args.reference_data, args.schema, require_label=True)
    example_indices = np.load(args.examples)["examples"]
    if len(example_indices) != len(frame):
        raise ValueError("Example mapping and input must have equal row counts")
    if args.limit is not None:
        frame = frame.iloc[: args.limit].copy()
        example_indices = example_indices[: args.limit]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    score_path = output / "scores.npy"
    ids = frame["id"].astype(str).tolist()
    fingerprint = _fingerprint(ids)
    start = 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise ValueError("Existing cache belongs to different rows")
        start = int(state["next_index"])
    else:
        pd.DataFrame({"id": ids}).to_csv(output / "ids.csv", index=False)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    token_zero = processor.tokenizer.encode("0", add_special_tokens=False)
    token_one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(token_zero) != 1 or len(token_one) != 1:
        raise ValueError("Verdicts must map to one token each")
    scores = (
        np.lib.format.open_memmap(score_path, mode="r+")
        if start
        else np.lib.format.open_memmap(score_path, mode="w+", dtype="float32", shape=(len(frame),))
    )

    for batch_start in range(start, len(frame), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(frame))
        images = []
        prompts = []
        for local_index, row in enumerate(
            frame.iloc[batch_start:batch_end].itertuples(index=False)
        ):
            image = _open_main_image(row.images, args.max_pixels)
            images.append(image)
            examples = reference.iloc[example_indices[batch_start + local_index]]
            prompts.append(render_prompt(processor, row, examples, with_image=image is not None))
        kwargs: dict[str, Any] = {
            "text": prompts,
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        valid_images = [image for image in images if image is not None]
        if valid_images:
            kwargs["images"] = valid_images
        inputs = processor(**kwargs).to("cuda")
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits[:, -1]
        verdict_logits = logits[:, [token_zero[0], token_one[0]]].float()
        scores[batch_start:batch_end] = torch.softmax(verdict_logits, dim=-1)[:, 1].cpu()
        scores.flush()
        for image in valid_images:
            image.close()
        state_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "rows": len(frame),
                    "next_index": batch_end,
                    "complete": batch_end == len(frame),
                    "model": args.model,
                    "examples": args.examples,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"retrieval VQA: {batch_end}/{len(frame)}", flush=True)

    del model, processor, scores
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
