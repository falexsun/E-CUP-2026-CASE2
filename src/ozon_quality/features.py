"""Feature extraction, fusion and reusable cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ozon_quality.encoders import TextEncoder, VisionEncoder

FEATURE_VERSION = 4


def _categorical_hash(values: pd.Series, dimension: int = 64) -> np.ndarray:
    output = np.zeros((len(values), dimension), dtype="float32")
    for row, value in enumerate(values.fillna("").astype(str).str.casefold()):
        if not value:
            continue
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest)
        output[row, number % dimension] = 1.0 if number & 1 else -1.0
    return output


def _fingerprint(
    frame: pd.DataFrame,
    text_model: str,
    vision_model: str,
    text_revision: str | None,
    vision_revision: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"feature-version:{FEATURE_VERSION}".encode())
    digest.update(text_model.encode())
    digest.update(vision_model.encode())
    digest.update(str(text_revision).encode())
    digest.update(str(vision_revision).encode())
    for row in frame[["id", "text", "images"]].itertuples(index=False):
        digest.update(json.dumps(list(row), ensure_ascii=False, sort_keys=True).encode())
    return digest.hexdigest()[:20]


def build_features(
    frame: pd.DataFrame,
    *,
    text_model: str,
    vision_model: str,
    device: str,
    batch_size: int,
    cache_dir: str | Path | None = None,
    text_revision: str | None = None,
    vision_revision: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    cache_path = None
    if cache_dir:
        fingerprint = _fingerprint(
            frame, text_model, vision_model, text_revision, vision_revision
        )
        cache_path = Path(cache_dir) / f"features-{fingerprint}.npz"
        if cache_path.exists():
            payload = np.load(cache_path)
            return payload["features"], json.loads(str(payload["metadata"]))
    text = TextEncoder(
        text_model,
        device=device,
        batch_size=batch_size,
        revision=text_revision,
    ).encode(frame["text"].tolist())
    vision_language = VisionEncoder(
        vision_model,
        device=device,
        batch_size=batch_size,
        revision=vision_revision,
    )
    vl_text = vision_language.encode_texts(frame["text"].tolist())
    image, image_present, image_stats = vision_language.encode_products(
        frame["images"].tolist()
    )
    if vl_text.shape[1] != image.shape[1]:
        raise ValueError(
            f"VLM text/image dimensions differ: {vl_text.shape[1]} vs {image.shape[1]}"
        )
    stripped = frame["text"].str.strip()
    text_present = stripped.ne("").to_numpy(dtype="float32")
    joint_present = text_present * image_present
    vl_text = vl_text * text_present[:, None]
    difference = np.abs(vl_text - image) * joint_present[:, None]
    product = (vl_text * image) * joint_present[:, None]
    cosine = (vl_text * image).sum(axis=1, keepdims=True) * joint_present[:, None]
    numeric_metadata = np.column_stack(
        [
            text_present,
            image_present,
            np.log1p(stripped.str.len().to_numpy(dtype="float32")),
            np.log1p(stripped.str.split().map(len).to_numpy(dtype="float32")),
            np.log1p(frame["images"].map(len).to_numpy(dtype="float32")),
            image_stats,
        ]
    ).astype("float32")
    empty = pd.Series("", index=frame.index, dtype="object")
    category_hash = _categorical_hash(frame["category"] if "category" in frame else empty)
    brand_hash = _categorical_hash(frame["brand"] if "brand" in frame else empty)
    metadata_features = np.concatenate(
        [numeric_metadata, category_hash, brand_hash], axis=1
    )
    blocks = [text, vl_text, image, difference, product, cosine, metadata_features]
    names = [
        "llm_text",
        "vlm_text",
        "vlm_image",
        "vlm_abs_difference",
        "vlm_product",
        "vlm_cosine",
        "metadata",
    ]
    features = np.concatenate(blocks, axis=1)
    boundaries: dict[str, list[int]] = {}
    offset = 0
    for name, block in zip(names, blocks, strict=True):
        boundaries[name] = [offset, offset + block.shape[1]]
        offset += block.shape[1]
    metadata: dict[str, object] = {
        "feature_version": FEATURE_VERSION,
        "text_model": text_model,
        "vision_model": vision_model,
        "text_revision": text_revision,
        "vision_revision": vision_revision,
        "text_dimensions": text.shape[1],
        "vlm_text_dimensions": vl_text.shape[1],
        "image_dimensions": image.shape[1],
        "metadata_dimensions": metadata_features.shape[1],
        "feature_dimensions": features.shape[1],
        "rows": len(frame),
        "blocks": boundaries,
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=features, metadata=json.dumps(metadata))
    return features, metadata
