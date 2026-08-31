#!/usr/bin/env python3
"""LoRA-tune Qwen3.5-4B as a multimodal flammable-product classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data  # noqa: E402
from ozon_quality.official_vqa import RULE_PROMPTS, _open_main_image  # noqa: E402

CATEGORY = "Легковоспламеняющиеся"


def render_prompt(processor: Any, row: Any, description_chars: int) -> str:
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
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {
                    "type": "text",
                    "text": (
                        f"Категория: {row.category}\nПравила: {RULE_PROMPTS[CATEGORY]}\n"
                        f"Название: {row.title}\nОписание: {row.description[:description_chars]}\n\n"
                        "Ответь строго одним символом: 1, если товар соответствует категории, "
                        "иначе 0."
                    ),
                },
            ],
        },
    ]
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class RowDataset(Dataset):
    def __init__(self, frame: Any, indices: list[int]) -> None:
        self.frame = frame
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.frame.iloc[self.indices[index]]


class TrainCollator:
    def __init__(self, processor: Any, max_pixels: int, description_chars: int) -> None:
        self.processor = processor
        self.max_pixels = max_pixels
        self.description_chars = description_chars

    def __call__(self, rows: list[Any]) -> dict[str, torch.Tensor]:
        images = [_open_main_image(row.images, self.max_pixels) for row in rows]
        if any(image is None for image in images):
            raise ValueError("Training requires a readable main image for every row")
        prompts = [
            render_prompt(self.processor, row, self.description_chars) + str(int(row.label))
            for row in rows
        ]
        inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = torch.full_like(inputs["input_ids"], -100)
        final_positions = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(rows))
        labels[batch_indices, final_positions] = inputs["input_ids"][batch_indices, final_positions]
        inputs["labels"] = labels
        for image in images:
            image.close()
        return inputs


class EvalCollator:
    def __init__(self, processor: Any, max_pixels: int, description_chars: int) -> None:
        self.processor = processor
        self.max_pixels = max_pixels
        self.description_chars = description_chars

    def __call__(self, rows: list[Any]) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        images = [_open_main_image(row.images, self.max_pixels) for row in rows]
        if any(image is None for image in images):
            raise ValueError("Evaluation requires a readable main image for every row")
        inputs = self.processor(
            text=[render_prompt(self.processor, row, self.description_chars) for row in rows],
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = np.asarray([int(row.label) for row in rows], dtype="int8")
        for image in images:
            image.close()
        return inputs, labels


def best_f1(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(values.argmax())
    threshold = float(thresholds[min(index, len(thresholds) - 1)])
    return float(values[index]), threshold


@torch.inference_mode()
def evaluate(
    model: Any,
    loader: DataLoader,
    token_zero: int,
    token_one: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    labels = []
    for inputs, batch_labels in loader:
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        logits = model(**inputs, logits_to_keep=1).logits[:, -1]
        verdict = torch.softmax(logits[:, [token_zero, token_one]].float(), dim=-1)[:, 1]
        probabilities.append(verdict.cpu().numpy())
        labels.append(batch_labels)
    return np.concatenate(labels), np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--positive-repeat", type=int, default=10)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=50_176)
    parser.add_argument("--description-chars", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, _ = load_data(args.train, args.schema, require_label=True)
    valid, _ = load_data(args.valid, args.schema, require_label=True)
    train = train[train["category"].eq(CATEGORY)].reset_index(drop=True)
    valid = valid[valid["category"].eq(CATEGORY)].reset_index(drop=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    target_pattern = (
        r".*language_model.*\.(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|"
        r"in_proj_z|out_proj)$"
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            bias="none",
            target_modules=target_pattern,
        ),
    )
    model.print_trainable_parameters()

    negative = train.index[train["label"].eq(0)].tolist()
    positive = train.index[train["label"].eq(1)].tolist()
    train_indices = negative + positive * args.positive_repeat
    random.shuffle(train_indices)
    train_loader = DataLoader(
        RowDataset(train, train_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=TrainCollator(processor, args.max_pixels, args.description_chars),
    )
    valid_loader = DataLoader(
        RowDataset(valid, list(range(len(valid)))),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=EvalCollator(processor, args.max_pixels, args.description_chars),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    planned_updates = int(np.ceil(len(train_loader) / args.gradient_accumulation)) * args.epochs
    total_updates = min(planned_updates, args.max_updates or planned_updates)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)
    token_zero = processor.tokenizer.encode("0", add_special_tokens=False)[0]
    token_one = processor.tokenizer.encode("1", add_special_tokens=False)[0]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    update = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        rolling_loss = 0.0
        for step, inputs in enumerate(train_loader, start=1):
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            loss = model(**inputs).loss / args.gradient_accumulation
            loss.backward()
            rolling_loss += float(loss.detach()) * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update % 5 == 0:
                    print(
                        f"epoch={epoch + 1} update={update}/{total_updates} "
                        f"loss={rolling_loss / 5:.5f}",
                        flush=True,
                    )
                    rolling_loss = 0.0
                if update >= total_updates:
                    break

        if update >= total_updates:
            print(f"stopped after {update} optimizer updates", flush=True)

        y_valid, probability = evaluate(model, valid_loader, token_zero, token_one)
        score, threshold = best_f1(y_valid, probability)
        fixed_score = f1_score(y_valid, probability >= 0.5)
        np.savez(output / f"valid_epoch{epoch + 1}.npz", y=y_valid, p=probability)
        print(
            f"epoch={epoch + 1} valid_f1={score:.6f} threshold={threshold:.6f} "
            f"f1_at_0.5={fixed_score:.6f}",
            flush=True,
        )
        model.save_pretrained(output / f"epoch{epoch + 1}")
        if update >= total_updates:
            break

    (output / "training_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
