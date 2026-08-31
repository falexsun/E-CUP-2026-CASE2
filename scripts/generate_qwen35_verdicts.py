#!/usr/bin/env python3
"""Generate and parse actual greedy Qwen3.5 verdicts instead of restricted-token logits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data  # noqa: E402
from ozon_quality.official_vqa import _fingerprint, _open_main_image, _render_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=100_352)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    frame, _ = load_data(args.input, args.schema, require_label=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    ids = frame["id"].astype(str).tolist()
    fingerprint = _fingerprint(ids)
    state_path = output / "state.json"
    start = 0
    rows = []
    csv_path = output / "verdicts.csv"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise ValueError("Existing cache belongs to different input")
        start = int(state["next_index"])
        rows = pd.read_csv(csv_path, dtype={"id": str}).to_dict("records")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
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
    for batch_start in range(start, len(frame), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(frame))
        images = []
        prompts = []
        for row in frame.iloc[batch_start:batch_end].itertuples(index=False):
            image = _open_main_image(row.images, args.max_pixels)
            if image is None:
                raise ValueError(f"Unreadable image for id={row.id}")
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
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        prefix = inputs["input_ids"].shape[1]
        texts = processor.tokenizer.batch_decode(generated[:, prefix:], skip_special_tokens=True)
        for row, text in zip(
            frame.iloc[batch_start:batch_end].itertuples(index=False), texts, strict=True
        ):
            match = re.search(r"(?<!\d)([01])(?!\d)", text)
            rows.append(
                {
                    "id": str(row.id),
                    "generated": text.replace("\n", " ").strip(),
                    "prediction": int(match.group(1)) if match else -1,
                }
            )
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        state_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "rows": len(frame),
                    "next_index": batch_end,
                    "complete": batch_end == len(frame),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        for image in images:
            image.close()
        valid = sum(int(row["prediction"]) >= 0 for row in rows)
        print(f"generated verdicts: {batch_end}/{len(frame)}, parsed={valid}", flush=True)


if __name__ == "__main__":
    main()
