#!/usr/bin/env python3
"""Train a small multimodal LoRA head and score one untouched outer fold."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from transformers import AutoModel, AutoProcessor

from ozon_quality.data import load_data
from ozon_quality.official_multimodal import (
    REFERENCE_INSTRUCTION,
    _open_product_images,
    build_reference_text,
)

CATEGORY = "Легковоспламеняющиеся"


def best_f1(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probability)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(f1.argmax())
    return float(f1[index]), float(thresholds[min(index, len(thresholds) - 1)])


def prepare_batch(
    processor: Any,
    frame: pd.DataFrame,
    texts: list[str],
    indices: np.ndarray,
    *,
    max_pixels: int,
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    conversations = []
    opened = []
    for index in indices:
        row = frame.iloc[int(index)]
        images = _open_product_images(row["images"][:1], max_images=1, max_pixels=max_pixels)
        image = images[0] if images else None
        if image is not None:
            opened.append(image)
        content: list[dict[str, Any]] = [{"type": "text", "text": texts[int(index)]}]
        if image is not None:
            content.append({"type": "image", "image": image})
        conversations.append(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": REFERENCE_INSTRUCTION}],
                },
                {"role": "user", "content": content},
            ]
        )
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "text_kwargs": {"padding": True, "truncation": "longest_first"},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {"return_tensors": "pt"},
        },
        add_generation_prompt=True,
    )
    return inputs, opened


def pooled_output(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = model(**inputs).last_hidden_state
    attention_mask = inputs["attention_mask"]
    last_positions = attention_mask.shape[1] - attention_mask.flip(1).argmax(1) - 1
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return torch.nn.functional.normalize(hidden[rows, last_positions].float(), p=2, dim=-1)


def score_rows(
    model: nn.Module,
    head: nn.Module,
    processor: Any,
    frame: pd.DataFrame,
    texts: list[str],
    indices: np.ndarray,
    *,
    batch_size: int,
    max_pixels: int,
) -> np.ndarray:
    model.eval()
    head.eval()
    scores = np.zeros(len(indices), dtype="float32")
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            inputs, opened = prepare_batch(
                processor, frame, texts, batch_indices, max_pixels=max_pixels
            )
            try:
                inputs = inputs.to("cuda")
                scores[start : start + len(batch_indices)] = torch.sigmoid(
                    head(pooled_output(model, inputs)).squeeze(1)
                ).cpu().numpy()
            finally:
                for image in opened:
                    image.close()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--outer", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--initial-embedding-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=50_176)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    full_frame, _ = load_data(args.data, args.schema, require_label=True)
    category_mask = full_frame["category"].eq(CATEGORY).to_numpy()
    frame = full_frame[category_mask].reset_index(drop=True)
    labels = frame["labels"].map(lambda values: int(values[0])).to_numpy(dtype="int8")
    outer = np.load(args.outer)
    np.testing.assert_array_equal(outer["y"], labels)
    score_indices = np.flatnonzero(outer["fold"] == args.fold)
    fit_indices = np.flatnonzero(outer["fold"] != args.fold)

    # Exclude contradictory groups and sample groups rather than rows so large duplicate
    # product families cannot dominate the rare positive class.
    fit_frame = frame.iloc[fit_indices].copy()
    fit_frame["label"] = labels[fit_indices]
    group_labels = fit_frame.groupby("group")["label"].nunique()
    clean_groups = set(group_labels[group_labels.eq(1)].index.astype(str))
    positive_groups: dict[str, np.ndarray] = {}
    negative_groups: dict[str, np.ndarray] = {}
    for group, rows in fit_frame.groupby("group"):
        if str(group) not in clean_groups:
            continue
        target = positive_groups if int(rows["label"].iloc[0]) == 1 else negative_groups
        target[str(group)] = rows.index.to_numpy(dtype="int64")
    if not positive_groups or not negative_groups:
        raise ValueError("Both clean positive and negative groups are required")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    base = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    ).to("cuda")
    try:
        model.gradient_checkpointing_enable()
    except (AttributeError, ValueError):
        pass
    hidden_size = int(base.config.text_config.hidden_size)
    head = nn.Linear(hidden_size, 1).to("cuda", dtype=torch.float32)
    embedding_cache = Path(args.initial_embedding_cache)
    cache_ids = pd.read_csv(embedding_cache / "ids.csv", dtype={"id": str})["id"].tolist()
    if cache_ids != full_frame["id"].astype(str).tolist():
        raise ValueError("Initial embedding cache is not aligned")
    frozen_x = np.load(embedding_cache / "embeddings.npy").astype("float32")[category_mask]
    scaler = StandardScaler().fit(frozen_x[fit_indices])
    initializer = LogisticRegression(
        C=0.03, class_weight="balanced", max_iter=3000
    ).fit(scaler.transform(frozen_x[fit_indices]), labels[fit_indices])
    initial_probability = initializer.predict_proba(scaler.transform(frozen_x))[:, 1]
    initial_weight = initializer.coef_[0] / scaler.scale_
    initial_bias = initializer.intercept_[0] - np.dot(initializer.coef_[0], scaler.mean_ / scaler.scale_)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(initial_weight).to("cuda", dtype=torch.float32))
        head.bias.fill_(float(initial_bias))
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": 2e-5},
            {"params": head.parameters(), "lr": 2e-4},
        ],
        weight_decay=0.01,
    )
    texts = build_reference_text(frame)
    rng = np.random.default_rng(args.seed)
    positive_names = np.asarray(list(positive_groups), dtype=object)
    negative_names = np.asarray(list(negative_groups), dtype=object)
    positive_weight = np.asarray(
        [
            max(0.02, float(np.mean(1 - initial_probability[positive_groups[str(group)]])))
            for group in positive_names
        ]
    )
    negative_weight = np.asarray(
        [
            max(0.02, float(np.mean(initial_probability[negative_groups[str(group)]])))
            for group in negative_names
        ]
    )
    positive_weight /= positive_weight.sum()
    negative_weight /= negative_weight.sum()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    head.train()
    losses = []
    micro_steps = args.updates * args.grad_accum
    positives_per_batch = max(1, args.batch_size // 2)
    for micro_step in range(micro_steps):
        selected_groups = list(
            rng.choice(
                positive_names,
                size=positives_per_batch,
                replace=True,
                p=positive_weight,
            )
        ) + list(
            rng.choice(
                negative_names,
                size=args.batch_size - positives_per_batch,
                replace=True,
                p=negative_weight,
            )
        )
        batch_indices = np.asarray(
            [rng.choice((positive_groups | negative_groups)[str(group)]) for group in selected_groups]
        )
        order = rng.permutation(len(batch_indices))
        batch_indices = batch_indices[order]
        targets = torch.from_numpy(labels[batch_indices].astype("float32")).to("cuda")
        inputs, opened = prepare_batch(
            processor, frame, texts, batch_indices, max_pixels=args.max_pixels
        )
        try:
            inputs = inputs.to("cuda")
            logits = head(pooled_output(model, inputs)).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach()))
        finally:
            for image in opened:
                image.close()
        if (micro_step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + list(head.parameters()), 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update = (micro_step + 1) // args.grad_accum
            if update % 10 == 0 or update == args.updates:
                print(
                    f"update={update}/{args.updates} loss={np.mean(losses[-40:]):.5f}",
                    flush=True,
                )

    probability = score_rows(
        model,
        head,
        processor,
        frame,
        texts,
        score_indices,
        batch_size=max(args.batch_size, 8),
        max_pixels=args.max_pixels,
    )
    score_labels = labels[score_indices]
    oracle_f1, oracle_threshold = best_f1(score_labels, probability)
    base_probability = outer["p"][score_indices]
    base_f1, base_oracle_threshold = best_f1(score_labels, base_probability)
    metrics = {
        "fold": args.fold,
        "updates": args.updates,
        "train_rows": int(len(fit_indices)),
        "valid_rows": int(len(score_indices)),
        "valid_positives": int(score_labels.sum()),
        "lora_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "loss_last40": float(np.mean(losses[-40:])),
        "lora_ap": float(average_precision_score(score_labels, probability)),
        "lora_auc": float(roc_auc_score(score_labels, probability)),
        "lora_oracle_f1": oracle_f1,
        "lora_oracle_threshold": oracle_threshold,
        "v19_ap": float(average_precision_score(score_labels, base_probability)),
        "v19_auc": float(roc_auc_score(score_labels, base_probability)),
        "v19_oracle_f1": base_f1,
        "v19_oracle_threshold": base_oracle_threshold,
        "v19_fixed_f1": float(f1_score(score_labels, base_probability >= 0.26)),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "adapter")
    torch.save(head.state_dict(), output / "head.pt")
    pd.DataFrame(
        {
            "id": frame.iloc[score_indices]["id"].astype(str).to_numpy(),
            "label": score_labels,
            "probability": probability,
            "v19_probability": base_probability,
        }
    ).to_csv(output / "validation_predictions.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
