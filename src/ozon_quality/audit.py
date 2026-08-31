"""Dataset and split audits that make unknown-data assumptions explicit."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ozon_quality.data import load_data, read_table
from ozon_quality.encoders import load_image


def _quantiles(values: pd.Series) -> str:
    if values.empty:
        return "n/a"
    result = values.quantile([0, 0.5, 0.9, 0.99, 1]).to_dict()
    return ", ".join(f"p{int(key * 100)}={value:.1f}" for key, value in result.items())


def _image_audit(image_lists: pd.Series, limit: int) -> dict[str, int | float]:
    paths = [path for paths in image_lists for path in paths][:limit]
    broken = low_resolution = 0
    widths: list[int] = []
    heights: list[int] = []
    hashes: list[str] = []
    for path in paths:
        try:
            image = load_image(path)
            widths.append(image.width)
            heights.append(image.height)
            low_resolution += min(image.size) < 224
            thumbnail = image.convert("L").resize((9, 8))
            pixels = np.asarray(thumbnail)
            hashes.append(hashlib.sha256((pixels[:, 1:] > pixels[:, :-1]).tobytes()).hexdigest())
        except Exception:  # audit must count arbitrary decoder/network failures
            broken += 1
    return {
        "sampled": len(paths),
        "broken": broken,
        "low_resolution_lt_224": low_resolution,
        "duplicate_perceptual_hashes": len(hashes) - len(set(hashes)),
        "median_width": float(np.median(widths)) if widths else 0,
        "median_height": float(np.median(heights)) if heights else 0,
    }


def write_data_audit(
    input_path: str,
    output_path: str,
    *,
    schema: str | None,
    image_sample: int = 500,
) -> None:
    raw = read_table(input_path)
    frame, config = load_data(input_path, schema, require_label=False)
    text_lengths = frame["text"].str.len()
    image_counts = frame["images"].map(len)
    image_stats = _image_audit(frame["images"], image_sample)
    lines = [
        "# DATA AUDIT",
        "",
        f"Source: `{Path(input_path).resolve()}`",
        "",
        "## Schema",
        "",
        f"- Shape: **{raw.shape[0]} × {raw.shape[1]}**",
        f"- Source columns: `{list(map(str, raw.columns))}`",
        f"- Canonical mapping: `{json.dumps(config.get('columns', {}), ensure_ascii=False)}`",
        f"- Duplicate canonical IDs: **{int(frame['id'].duplicated().sum())}**",
        f"- Missing/empty text: **{int(frame['text'].str.strip().eq('').sum())}**",
        f"- Products without declared images: **{int(image_counts.eq(0).sum())}**",
        f"- Images per product: {_quantiles(image_counts)}",
        "",
        "## Text",
        "",
        f"- Character lengths: {_quantiles(text_lengths)}",
        f"- Exact duplicate composed texts: **{int(frame['text'].duplicated().sum())}**",
        "- Languages, HTML/markup, URLs/phones, articles and templates: **MANUAL AUDIT REQUIRED**",
        "",
        "## Images",
        "",
    ]
    lines.extend(f"- {key}: **{value}**" for key, value in image_stats.items())
    if "labels" in frame:
        counts = Counter(label for labels in frame["labels"] for label in labels)
        multi_rows = int(frame["labels"].map(len).gt(1).sum())
        lines.extend(
            [
                "",
                "## Target",
                "",
                f"- Inferred task: **{'multi-label' if multi_rows else 'single-label'}**",
                f"- Classes: **{len(counts)}**",
                f"- Rows with multiple labels: **{multi_rows}**",
                "",
                "| class | count | rate |",
                "|---|---:|---:|",
            ]
        )
        for label, count in counts.most_common():
            lines.append(f"| {label} | {count} | {count / len(frame):.6f} |")
    else:
        lines.extend(["", "## Target", "", "Target is not present in this file."])
    lines.extend(
        [
            "",
            "## Required manual multimodal audit",
            "",
            "Review at least 100 stratified examples and assign: `TEXT_ONLY`, `IMAGE_ONLY`, "
            "`TEXT_AND_IMAGE`, `AMBIGUOUS`, `POSSIBLE_LABEL_NOISE`.",
            "",
            "This automatic report does not infer legal violation types or treat model explanations as ground truth.",
        ]
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _local_content_hashes(image_lists: pd.Series, first_only: bool = True) -> set[str]:
    result: set[str] = set()
    for paths in image_lists:
        for value in paths[:1] if first_only else paths:
            path = Path(value)
            if path.is_file():
                result.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def _local_perceptual_hashes(image_lists: pd.Series, first_only: bool = True) -> set[str]:
    result: set[str] = set()
    for paths in image_lists:
        for value in paths[:1] if first_only else paths:
            path = Path(value)
            if not path.is_file():
                continue
            try:
                image = load_image(str(path)).convert("L").resize((9, 8))
                pixels = np.asarray(image)
                result.add(
                    hashlib.sha256((pixels[:, 1:] > pixels[:, :-1]).tobytes()).hexdigest()
                )
            except Exception:
                continue
    return result


def _near_text_overlap(train_text: pd.Series, valid_text: pd.Series, limit: int = 5000) -> int:
    train_values = train_text.str.strip().loc[lambda values: values.str.len().ge(20)].head(limit)
    valid_values = valid_text.str.strip().loc[lambda values: values.str.len().ge(20)].head(limit)
    exact = set(train_values)
    valid_values = valid_values[~valid_values.isin(exact)]
    if train_values.empty or valid_values.empty:
        return 0
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=50_000, min_df=2
    )
    try:
        train_matrix = vectorizer.fit_transform(train_values)
    except ValueError:
        return 0
    valid_matrix = vectorizer.transform(valid_values)
    matches = 0
    for start in range(0, valid_matrix.shape[0], 256):
        similarity = valid_matrix[start : start + 256] @ train_matrix.T
        maxima = similarity.max(axis=1).toarray().ravel()
        matches += int((maxima >= 0.97).sum())
    return matches


def write_leakage_report(
    train_path: str,
    valid_path: str,
    output_path: str,
    *,
    schema: str | None,
) -> bool:
    train, _ = load_data(train_path, schema, require_label=False)
    valid, _ = load_data(valid_path, schema, require_label=False)
    id_overlap = set(train["id"]) & set(valid["id"])
    train_text = set(train["text"].str.casefold().str.replace(r"\s+", " ", regex=True)) - {""}
    valid_text = set(valid["text"].str.casefold().str.replace(r"\s+", " ", regex=True)) - {""}
    text_overlap = train_text & valid_text
    declared_images = {path for paths in train["images"] for path in paths} & {
        path for paths in valid["images"] for path in paths
    }
    content_overlap = _local_content_hashes(train["images"]) & _local_content_hashes(valid["images"])
    perceptual_overlap = _local_perceptual_hashes(train["images"]) & _local_perceptual_hashes(
        valid["images"]
    )
    near_text_overlap = _near_text_overlap(
        train["text"].str.casefold(), valid["text"].str.casefold()
    )
    group_overlap: set[str] = set()
    if "group" in train and "group" in valid:
        group_overlap = (set(train["group"]) - {""}) & (set(valid["group"]) - {""})
    blocked = bool(
        id_overlap
        or text_overlap
        or near_text_overlap
        or declared_images
        or content_overlap
        or perceptual_overlap
        or group_overlap
    )
    lines = [
        "# VALIDATION",
        "",
        f"Status: **{'BLOCKED' if blocked else 'READY (exact checks only)'}**",
        "",
        f"- Train rows: **{len(train)}**",
        f"- Validation rows: **{len(valid)}**",
        f"- Overlapping product IDs: **{len(id_overlap)}**",
        f"- Exact normalized text overlaps: **{len(text_overlap)}**",
        f"- Near-text overlaps ≥0.97 (sampled, excluding exact): **{near_text_overlap}**",
        f"- Reused declared image paths/URLs: **{len(declared_images)}**",
        f"- Identical first-image contents: **{len(content_overlap)}**",
        f"- Perceptually duplicate first images (dHash): **{len(perceptual_overlap)}**",
        f"- Overlapping configured groups: **{len(group_overlap)}**",
        "",
        "Near-title checking is sampled at up to 5,000 rows per split; confirm borderline cases manually.",
        "Official metric, split version and grouping entity must be recorded here after data release.",
    ]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return not blocked
