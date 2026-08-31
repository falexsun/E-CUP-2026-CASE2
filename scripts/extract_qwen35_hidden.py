#!/usr/bin/env python3
"""Cache frozen Qwen3.5 multimodal last-token hidden states."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data  # noqa: E402
from ozon_quality.official_vqa import (  # noqa: E402
    _fingerprint,
    _open_main_image,
    _render_prompt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=25_088)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    frame, _ = load_data(args.input, args.schema, require_label=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ids = frame["id"].astype(str).tolist()
    fingerprint = _fingerprint(ids)
    state_path = output / "state.json"
    embeddings_path = output / "embeddings.npy"
    start = 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint or state["rows"] != len(frame):
            raise ValueError("Existing cache belongs to different rows/order")
        start = int(state["next_index"])
    else:
        pd.DataFrame({"id": ids}).to_csv(output / "ids.csv", index=False)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .to("cuda")
        .eval()
    )
    dimensions = int(model.config.text_config.hidden_size)
    embeddings = (
        np.lib.format.open_memmap(embeddings_path, mode="r+")
        if start
        else np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype="float16",
            shape=(len(frame), dimensions),
        )
    )

    for batch_start in range(start, len(frame), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(frame))
        images = []
        prompts = []
        for row in frame.iloc[batch_start:batch_end].itertuples(index=False):
            image = _open_main_image(row.images, args.max_pixels)
            if image is None:
                raise ValueError(f"Unreadable main image for id={row.id}")
            images.append(image)
            prompts.append(_render_prompt(processor, row, True))
        inputs = processor(
            text=prompts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            hidden = model.model(**inputs, use_cache=False).last_hidden_state
        positions = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(prompts), device="cuda")
        embeddings[batch_start:batch_end] = (
            hidden[batch_indices, positions].float().cpu().numpy().astype("float16")
        )
        embeddings.flush()
        for image in images:
            image.close()
        state_path.write_text(
            json.dumps(
                {
                    "artifact_version": 1,
                    "fingerprint": fingerprint,
                    "rows": len(frame),
                    "dimensions": dimensions,
                    "next_index": batch_end,
                    "complete": batch_end == len(frame),
                    "model": args.model,
                    "max_pixels": args.max_pixels,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Qwen3.5 hidden: {batch_end}/{len(frame)}", flush=True)

    del model, processor, embeddings
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
