"""Official Qwen3-VL embedding extraction and leakage-safe classifier training."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

from ozon_quality.data import load_data
from ozon_quality.lexical import build_lexical_classifier
from ozon_quality.official import OFFICIAL_CATEGORIES, official_category_f1
from ozon_quality.official_baseline import add_rule_tokens, choose_positive_threshold

DEFAULT_QWEN_EMBED_MODEL: Final[str] = "Qwen/Qwen3-VL-Embedding-2B"
REFERENCE_INSTRUCTION: Final[str] = (
    "Represent this marketplace product for category compliance classification. Pay special "
    "attention to explicit package markings, product contents, ignition sources, flammable "
    "substances and whether an item is merely an empty construction."
)

TASK_CONTEXT: Final[dict[str, str]] = {
    "БАД": (
        "Определи, соответствует ли товар категории БАД. Положительный класс требует прямой "
        "маркировки БАД, биологически активная добавка или dietary supplement. Спортивное "
        "питание и товары без такой маркировки относятся к отрицательному классу."
    ),
    "Легковоспламеняющиеся": (
        "Определи, является ли сам товар или содержимое комплекта легковоспламеняющимся: "
        "источник открытого огня, горючее вещество или газ — положительный класс. Пустая "
        "конструкция, встроенный источник и горючий компонент изделия — отрицательный класс."
    ),
}


def build_embedding_text(frame: pd.DataFrame) -> list[str]:
    """Build task-aware text without using labels or split-specific information."""
    texts: list[str] = []
    for row in frame.itertuples(index=False):
        category = str(row.category)
        title = str(getattr(row, "title", ""))
        description = str(getattr(row, "description", ""))
        # Leave enough context for visual tokens. Product descriptions contain long repeated
        # HTML/template tails, for which the beginning is normally the most informative part.
        description = description[:5000]
        texts.append(
            f"Инструкция: {TASK_CONTEXT[category]}\n"
            f"Категория: {category}\nНазвание: {title}\nОписание: {description}"
        )
    return texts


def build_reference_text(frame: pd.DataFrame) -> list[str]:
    """Focused text for the reference VLM contract; TF-IDF keeps the full description."""
    return [
        f"Category: {row.category}\nProduct name: {row.title}\nDescription: {row.description[:2500]}"
        for row in frame.itertuples(index=False)
    ]


def _fingerprint(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _resize_image(image: Image.Image, max_pixels: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    new_width = max(28, int(width * scale) // 28 * 28)
    new_height = max(28, int(height * scale) // 28 * 28)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _open_product_images(paths: list[str], max_images: int, max_pixels: int) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths[:max_images]:
        try:
            with Image.open(path) as source:
                images.append(_resize_image(source, max_pixels).copy())
        except (OSError, ValueError):
            continue
    return images


def _masked_mean(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)


def _embed_batch(
    processor: Any, model: Any, texts: list[str], images: list[list[Image.Image]], device: str
) -> np.ndarray:
    import torch

    vision_start = getattr(processor, "vision_start_token", "<|vision_start|>")
    vision_end = getattr(processor, "vision_end_token", "<|vision_end|>")
    image_token = getattr(processor, "image_token", "<|image|>")
    placeholder = vision_start + image_token + vision_end
    rendered = [
        text + placeholder * len(sample_images)
        for text, sample_images in zip(texts, images, strict=True)
    ]
    kwargs: dict[str, Any] = {
        "text": rendered,
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }
    if any(images):
        kwargs["images"] = images
    inputs = processor(**kwargs).to(device)
    with torch.inference_mode():
        output = model(**inputs)
    pooled = _masked_mean(output.last_hidden_state, inputs["attention_mask"])
    return pooled.cpu().numpy().astype("float32")


def extract_qwen_embeddings(
    input_path: str,
    output: str,
    *,
    schema: str,
    model_path: str = DEFAULT_QWEN_EMBED_MODEL,
    batch_size: int = 32,
    max_images: int = 5,
    max_pixels: int = 100_352,
    limit: int | None = None,
) -> dict[str, Any]:
    """Extract resumable joint text/image embeddings into a memory-mapped cache."""
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL extraction requires a CUDA device")
    frame, _ = load_data(input_path, schema, require_label=False)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    ids = frame["id"].astype(str).tolist()
    fingerprint = _fingerprint(ids)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    ids_path = output_dir / "ids.csv"
    embedding_path = output_dir / "embeddings.npy"
    start = 0
    state: dict[str, Any] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint or state.get("rows") != len(frame):
            raise ValueError("Existing embedding cache belongs to a different row set/order")
        start = int(state.get("next_index", 0))
    else:
        pd.DataFrame({"id": ids}).to_csv(ids_path, index=False)

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )
    texts = build_embedding_text(frame)
    embeddings: np.memmap | None = None
    if start:
        embeddings = np.lib.format.open_memmap(embedding_path, mode="r+")
    failures = int(state.get("image_failures", 0))
    for batch_start in range(start, len(frame), batch_size):
        batch_end = min(batch_start + batch_size, len(frame))
        batch_images = [
            _open_product_images(paths, max_images=max_images, max_pixels=max_pixels)
            for paths in frame.iloc[batch_start:batch_end]["images"]
        ]
        try:
            batch_embeddings = _embed_batch(
                processor, model, texts[batch_start:batch_end], batch_images, device
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows = []
            for text, images in zip(texts[batch_start:batch_end], batch_images, strict=True):
                try:
                    rows.append(_embed_batch(processor, model, [text], [images], device)[0])
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    torch.cuda.empty_cache()
                    rows.append(_embed_batch(processor, model, [text], [[]], device)[0])
                    failures += 1
            batch_embeddings = np.stack(rows)
        finally:
            for images in batch_images:
                for image in images:
                    image.close()
        if embeddings is None:
            embeddings = np.lib.format.open_memmap(
                embedding_path,
                mode="w+",
                dtype="float16",
                shape=(len(frame), batch_embeddings.shape[1]),
            )
        embeddings[batch_start:batch_end] = batch_embeddings.astype("float16")
        embeddings.flush()
        state = {
            "artifact_version": 1,
            "fingerprint": fingerprint,
            "rows": len(frame),
            "dimensions": int(batch_embeddings.shape[1]),
            "next_index": batch_end,
            "complete": batch_end == len(frame),
            "model": model_path,
            "max_images": max_images,
            "max_pixels": max_pixels,
            "image_failures": failures,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"embedded {batch_end}/{len(frame)}", flush=True)
    del model, processor, embeddings
    gc.collect()
    torch.cuda.empty_cache()
    return state


def extract_reference_embeddings(
    input_path: str,
    output: str,
    *,
    schema: str,
    model_path: str = DEFAULT_QWEN_EMBED_MODEL,
    mode: str = "joint",
    image_index: int = 0,
    batch_size: int = 32,
    max_pixels: int = 100_352,
    limit: int | None = None,
) -> dict[str, Any]:
    """Use the model-card chat contract and last-token pooling.

    ``mode`` is one of ``text``, ``image`` or ``joint``. Image/joint modes use the main
    product image. The implementation deliberately uses only ``transformers`` because the
    official runtime image does not contain the optional ``sentence_transformers`` package.
    Separate caches can later be concatenated for dense fusion.
    """
    if mode not in {"text", "image", "joint"}:
        raise ValueError("mode must be one of: text, image, joint")
    if image_index < 0:
        raise ValueError("image_index must be non-negative")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModel, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("Reference Qwen3-VL extraction requires a CUDA device")
    frame, _ = load_data(input_path, schema, require_label=False)
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    ids = frame["id"].astype(str).tolist()
    fingerprint = _fingerprint(ids)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    ids_path = output_dir / "ids.csv"
    embedding_path = output_dir / "embeddings.npy"
    start = 0
    state: dict[str, Any] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("fingerprint") != fingerprint
            or state.get("rows") != len(frame)
            or state.get("mode") != mode
            or int(state.get("image_index", 0)) != image_index
        ):
            raise ValueError("Existing reference cache belongs to different rows or mode")
        start = int(state.get("next_index", 0))
    else:
        pd.DataFrame({"id": ids}).to_csv(ids_path, index=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    texts = build_reference_text(frame)
    embeddings: np.memmap | None = None
    if start:
        embeddings = np.lib.format.open_memmap(embedding_path, mode="r+")
    missing_or_broken = int(state.get("missing_or_broken_images", 0))
    for batch_start in range(start, len(frame), batch_size):
        batch_end = min(batch_start + batch_size, len(frame))
        conversations: list[list[dict[str, Any]]] = []
        opened: list[Image.Image] = []
        for local_index, paths in enumerate(frame.iloc[batch_start:batch_end]["images"]):
            image = None
            if mode in {"image", "joint"} and paths:
                images = _open_product_images(
                    paths[image_index : image_index + 1],
                    max_images=1,
                    max_pixels=max_pixels,
                )
                if images:
                    image = images[0]
                    opened.append(image)
            content: list[dict[str, Any]] = []
            if mode in {"text", "joint"} or image is None:
                content.append({"type": "text", "text": texts[batch_start + local_index]})
            if mode in {"image", "joint"} and image is not None:
                # SentenceTransformers preserves the input dict order. Training used
                # {"text": ..., "image": ...}, so the image must follow the text here.
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
            if mode in {"image", "joint"} and image is None:
                missing_or_broken += 1
        try:
            try:
                model_inputs = processor.apply_chat_template(
                    conversations,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={
                        "text_kwargs": {
                            "padding": True,
                            "truncation": "longest_first",
                        },
                        "images_kwargs": {},
                        "audio_kwargs": {},
                        "videos_kwargs": {},
                        "common_kwargs": {"return_tensors": "pt"},
                    },
                    add_generation_prompt=True,
                )
            except TypeError:
                # Compatibility path for the older ProcessorMixin API.
                model_inputs = processor.apply_chat_template(
                    conversations,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                    truncation="longest_first",
                    add_generation_prompt=True,
                )
            model_inputs = model_inputs.to("cuda")
            with torch.inference_mode():
                hidden = model(**model_inputs).last_hidden_state
            attention_mask = model_inputs["attention_mask"]
            last_positions = attention_mask.shape[1] - attention_mask.flip(1).argmax(1) - 1
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            batch_embeddings = (
                torch.nn.functional.normalize(hidden[rows, last_positions].float(), p=2, dim=-1)
                .cpu()
                .numpy()
                .astype("float32")
            )
        finally:
            for image in opened:
                image.close()
        if embeddings is None:
            embeddings = np.lib.format.open_memmap(
                embedding_path,
                mode="w+",
                dtype="float16",
                shape=(len(frame), batch_embeddings.shape[1]),
            )
        embeddings[batch_start:batch_end] = batch_embeddings.astype("float16")
        embeddings.flush()
        state = {
            "artifact_version": 2,
            "contract": "model_card_chat_lasttoken_transformers",
            "fingerprint": fingerprint,
            "rows": len(frame),
            "dimensions": int(batch_embeddings.shape[1]),
            "next_index": batch_end,
            "complete": batch_end == len(frame),
            "model": model_path,
            "mode": mode,
            "image_index": image_index,
            "max_pixels": max_pixels,
            "instruction": REFERENCE_INSTRUCTION,
            "missing_or_broken_images": missing_or_broken,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"reference {mode}: {batch_end}/{len(frame)}", flush=True)
    del model, processor, embeddings
    gc.collect()
    torch.cuda.empty_cache()
    return state


def load_embedding_cache(path: str, expected_ids: pd.Series | list[str]) -> np.ndarray:
    cache = Path(path)
    state = json.loads((cache / "state.json").read_text(encoding="utf-8"))
    if not state.get("complete"):
        raise ValueError(
            f"Embedding cache is incomplete: {state.get('next_index')}/{state.get('rows')}"
        )
    ids = pd.read_csv(cache / "ids.csv", dtype=str)["id"].astype(str).tolist()
    expected = [str(value) for value in expected_ids]
    if ids != expected:
        raise ValueError("Embedding IDs/order do not match the requested dataframe")
    return np.load(cache / "embeddings.npy", mmap_mode="r")


def combine_embedding_caches(caches: list[str], output: str) -> dict[str, Any]:
    """Concatenate aligned frozen caches without loading the full matrices into RAM."""
    if len(caches) < 2:
        raise ValueError("At least two embedding caches are required")
    cache_paths = [Path(path) for path in caches]
    ids = pd.read_csv(cache_paths[0] / "ids.csv", dtype=str)["id"].astype(str).tolist()
    matrices = [load_embedding_cache(str(path), ids) for path in cache_paths]
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids}).to_csv(output_dir / "ids.csv", index=False)
    dimensions = [int(matrix.shape[1]) for matrix in matrices]
    combined = np.lib.format.open_memmap(
        output_dir / "embeddings.npy",
        mode="w+",
        dtype="float16",
        shape=(len(ids), sum(dimensions)),
    )
    for start in range(0, len(ids), 1024):
        end = min(start + 1024, len(ids))
        combined[start:end] = np.concatenate(
            [np.asarray(matrix[start:end], dtype="float16") for matrix in matrices], axis=1
        )
    combined.flush()
    states = [json.loads((path / "state.json").read_text(encoding="utf-8")) for path in cache_paths]
    state = {
        "artifact_version": 2,
        "contract": "concatenated_frozen_embeddings",
        "fingerprint": _fingerprint(ids),
        "rows": len(ids),
        "dimensions": int(sum(dimensions)),
        "component_dimensions": dimensions,
        "next_index": len(ids),
        "complete": True,
        "model": DEFAULT_QWEN_EMBED_MODEL,
        "sources": [str(path) for path in cache_paths],
        "source_states": states,
    }
    (output_dir / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return state


def _align_probabilities(path: Path, ids: pd.Series, column: str = "probability") -> np.ndarray:
    source = pd.read_csv(path, dtype={"id": str})
    if source["id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in {path}")
    aligned = pd.DataFrame({"id": ids.astype(str)}).merge(
        source[["id", column]], on="id", how="left", validate="one_to_one"
    )
    if aligned[column].isna().any():
        raise ValueError(f"Missing IDs while aligning {path}")
    return aligned[column].to_numpy(dtype="float32")


def _safe_logit(values: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(values, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def _normalize_lookup_title(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", " ", value).strip()


def _cosine_reference_features(
    query: np.ndarray,
    references: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return class-difference, nearest cosine and nearest reference label."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = torch.from_numpy(normalize(np.asarray(references, dtype="float32"), norm="l2")).to(
        device
    )
    reference_labels = np.asarray(labels, dtype="int8")
    positive = torch.from_numpy(np.flatnonzero(reference_labels == 1)).to(device)
    negative = torch.from_numpy(np.flatnonzero(reference_labels == 0)).to(device)
    reference_label_tensor = torch.from_numpy(reference_labels).to(device)
    if not len(positive) or not len(negative):
        raise ValueError("KNN override requires both positive and negative references")
    differences: list[np.ndarray] = []
    nearest_similarities: list[np.ndarray] = []
    nearest_labels: list[np.ndarray] = []
    for start in range(0, len(query), batch_size):
        batch = torch.from_numpy(np.asarray(query[start : start + batch_size], dtype="float32")).to(
            device
        )
        similarity = batch @ reference.T
        difference = (
            similarity[:, positive].max(dim=1).values - similarity[:, negative].max(dim=1).values
        )
        nearest_similarity, nearest_index = similarity.max(dim=1)
        differences.append(difference.cpu().numpy())
        nearest_similarities.append(nearest_similarity.cpu().numpy())
        nearest_labels.append(reference_label_tensor[nearest_index].cpu().numpy())
    del reference, positive, negative, reference_label_tensor
    if device == "cuda":
        torch.cuda.empty_cache()
    return (
        np.concatenate(differences).astype("float32"),
        np.concatenate(nearest_similarities).astype("float32"),
        np.concatenate(nearest_labels).astype("int8"),
    )


def _max_cosine_label_difference(
    query: np.ndarray,
    references: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """Return max cosine to positive prototypes minus max cosine to negatives."""
    difference, _, _ = _cosine_reference_features(query, references, labels, batch_size=batch_size)
    return difference


def predict_official_multimodal(
    frame: pd.DataFrame, embeddings: np.ndarray, artifact: dict[str, Any]
) -> pd.DataFrame:
    """Apply the frozen category models, OOF thresholds and lexical/VLM blend."""
    if len(frame) != len(embeddings):
        raise ValueError("frame and embeddings must have the same number of rows")
    raw_embeddings = np.asarray(embeddings, dtype="float32")
    x = normalize(raw_embeddings, norm="l2")
    lexical_probability = np.zeros(len(frame), dtype="float32")
    vlm_probability = np.zeros(len(frame), dtype="float32")
    blend_probability = np.zeros(len(frame), dtype="float32")
    prediction = np.zeros(len(frame), dtype="int8")
    for category in OFFICIAL_CATEGORIES:
        mask = frame["category"].eq(category).to_numpy()
        if not mask.any():
            continue
        text = add_rule_tokens(frame.loc[mask, "text"], category)
        lexical_probability[mask] = artifact["lexical"]["models"][category].predict_proba(text)[
            :, 1
        ]
        vlm_probability[mask] = artifact["models"][category].predict_proba(x[mask])[:, 1]
        decision = artifact["decisions"][category]
        alpha = float(decision["blend_alpha_vlm"])
        if decision.get("blend_strategy", "raw_probability") == "raw_probability":
            blend_probability[mask] = (
                alpha * vlm_probability[mask] + (1 - alpha) * lexical_probability[mask]
            )
        else:
            lexical_threshold = float(artifact["lexical"]["thresholds"][category])
            margin = alpha * (
                _safe_logit(vlm_probability[mask]) - _safe_logit(float(decision["vlm_threshold"]))
            ) + (1 - alpha) * (
                _safe_logit(lexical_probability[mask]) - _safe_logit(lexical_threshold)
            )
            blend_probability[mask] = 1 / (1 + np.exp(-margin))
        prediction[mask] = blend_probability[mask] >= float(decision["blend_threshold"])
    unknown = sorted(set(frame["category"].astype(str)) - set(OFFICIAL_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown official categories: {unknown}")
    for category, override in artifact.get("knn_overrides", {}).items():
        mask = frame["category"].eq(category).to_numpy()
        if not mask.any():
            continue
        nearest_threshold = override.get("nearest_override_threshold")
        if nearest_threshold is None:
            score = _max_cosine_label_difference(
                x[mask],
                override["reference_embeddings"],
                override["reference_labels"],
                batch_size=int(override.get("batch_size", 512)),
            )
            nearest_similarity = nearest_label = None
        else:
            score, nearest_similarity, nearest_label = _cosine_reference_features(
                x[mask],
                override["reference_embeddings"],
                override["reference_labels"],
                batch_size=int(override.get("batch_size", 512)),
            )
        alpha = float(override["knn_alpha"])
        margin = (1 - alpha) * (
            _safe_logit(blend_probability[mask]) - _safe_logit(float(override["base_threshold"]))
        ) + alpha * (score - float(override["knn_threshold"])) / float(override["margin_scale"])
        blend_probability[mask] = 1 / (1 + np.exp(-margin))
        prediction[mask] = blend_probability[mask] >= float(override["blend_threshold"])
        if nearest_similarity is not None and nearest_label is not None:
            confident = nearest_similarity >= float(nearest_threshold)
            indices = np.flatnonzero(mask)[confident]
            prediction[indices] = nearest_label[confident]
            blend_probability[indices] = nearest_label[confident].astype("float32")
    for category, override in artifact.get("linear_head_overrides", {}).items():
        mask = frame["category"].eq(category).to_numpy()
        if not mask.any():
            continue
        head_probability = override["model"].predict_proba(
            override["scaler"].transform(raw_embeddings[mask])
        )[:, 1]
        alpha = float(override["alpha_head"])
        blended_logit = (1 - alpha) * _safe_logit(blend_probability[mask])
        blended_logit += alpha * _safe_logit(head_probability)
        blend_probability[mask] = 1 / (1 + np.exp(-blended_logit))
        prediction[mask] = blend_probability[mask] >= float(override["blend_threshold"])
    for category, mapping in artifact.get("exact_title_overrides", {}).items():
        mask = frame["category"].eq(category).to_numpy()
        indices = np.flatnonzero(mask)
        for index in indices:
            label = mapping.get(_normalize_lookup_title(frame.iloc[index]["title"]))
            if label is None:
                continue
            prediction[index] = int(label)
            blend_probability[index] = 1.0 if int(label) else 0.0
    for category, rules in artifact.get("regex_title_overrides", {}).items():
        mask = frame["category"].eq(category).to_numpy()
        indices = np.flatnonzero(mask)
        compiled = [(re.compile(str(rule["pattern"])), int(rule["label"])) for rule in rules]
        for index in indices:
            title = _normalize_lookup_title(frame.iloc[index]["title"])
            for pattern, label in compiled:
                if pattern.search(title) is None:
                    continue
                prediction[index] = label
                blend_probability[index] = float(label)
    return pd.DataFrame(
        {
            "lexical_probability": lexical_probability,
            "vlm_probability": vlm_probability,
            "probability": blend_probability,
            "prediction": prediction,
        },
        index=frame.index,
    )


def train_official_multimodal(
    train_path: str,
    valid_path: str,
    output: str,
    *,
    schema: str,
    embedding_cache: str,
    embedding_input: str,
    lexical_artifacts: str,
    seed: int = 42,
    folds: int = 5,
) -> dict[str, Any]:
    """Train category models and select VLM/text blend from train OOF only."""
    full, _ = load_data(embedding_input, schema, require_label=False)
    full_embeddings = load_embedding_cache(embedding_cache, full["id"])
    id_to_index = {value: index for index, value in enumerate(full["id"].astype(str))}
    train, config = load_data(train_path, schema, require_label=True)
    valid, _ = load_data(valid_path, schema, require_label=True)

    def select(frame: pd.DataFrame) -> np.ndarray:
        indices = [id_to_index.get(value) for value in frame["id"].astype(str)]
        if any(index is None for index in indices):
            raise ValueError("Split contains IDs absent from the embedding cache")
        return normalize(np.asarray(full_embeddings[indices], dtype="float32"), norm="l2")

    train_x = select(train)
    valid_x = select(valid)
    train_y = train["labels"].map(lambda values: int(values[0])).to_numpy()
    valid_y = valid["labels"].map(lambda values: int(values[0])).to_numpy()
    lexical_dir = Path(lexical_artifacts)
    lexical_oof = _align_probabilities(lexical_dir / "oof_predictions.csv", train["id"])
    lexical_valid = _align_probabilities(lexical_dir / "validation_predictions.csv", valid["id"])
    lexical_artifact = joblib.load(lexical_dir / "official_text_baseline.joblib")

    vlm_oof = np.zeros(len(train), dtype="float32")
    vlm_valid = np.zeros(len(valid), dtype="float32")
    blend_oof = np.zeros(len(train), dtype="float32")
    blend_valid = np.zeros(len(valid), dtype="float32")
    models: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for offset, category in enumerate(OFFICIAL_CATEGORIES):
        train_mask = train["category"].eq(category).to_numpy()
        valid_mask = valid["category"].eq(category).to_numpy()
        x = train_x[train_mask]
        y = train_y[train_mask]
        groups = train.loc[train_mask, "group"].astype(str).to_numpy()
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + offset)
        splits = list(splitter.split(x, y, groups=groups))
        candidates = [
            (c, weight) for c in (0.01, 0.03, 0.1, 0.3, 1.0) for weight in (None, "balanced")
        ]
        best: tuple[float, float, float, str | None, np.ndarray, float] | None = None
        for c, weight in candidates:
            candidate_oof = np.zeros(len(x), dtype="float32")
            for fold, (fit_idx, holdout_idx) in enumerate(splits):
                model = LogisticRegression(
                    C=c,
                    class_weight=weight,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=seed + offset * 100 + fold,
                )
                model.fit(x[fit_idx], y[fit_idx])
                candidate_oof[holdout_idx] = model.predict_proba(x[holdout_idx])[:, 1]
            threshold, score = choose_positive_threshold(y, candidate_oof)
            candidate = (score, -abs(threshold - 0.5), c, weight, candidate_oof, threshold)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        score, _, best_c, best_weight, category_oof, vlm_threshold = best
        final_model = LogisticRegression(
            C=best_c,
            class_weight=best_weight,
            max_iter=1000,
            solver="lbfgs",
            random_state=seed + offset,
        ).fit(x, y)
        category_valid = final_model.predict_proba(valid_x[valid_mask])[:, 1]
        category_indices = np.flatnonzero(train_mask)
        vlm_oof[category_indices] = category_oof
        vlm_valid[valid_mask] = category_valid

        best_blend: tuple[float, float, int, str, float, np.ndarray, float] | None = None
        lexical_threshold = float(lexical_artifact["thresholds"][category])

        lexical_margin = _safe_logit(lexical_oof[train_mask]) - _safe_logit(lexical_threshold)
        vlm_margin = _safe_logit(category_oof) - _safe_logit(vlm_threshold)
        for alpha in np.linspace(0, 1, 21):
            raw_probability = alpha * category_oof + (1 - alpha) * lexical_oof[train_mask]
            margin = alpha * vlm_margin + (1 - alpha) * lexical_margin
            margin_probability = 1 / (1 + np.exp(-margin))
            for strategy, probability in (
                ("raw_probability", raw_probability),
                ("threshold_normalized_logit", margin_probability),
            ):
                threshold, blend_score = choose_positive_threshold(y, probability)
                candidate = (
                    blend_score,
                    -abs(alpha - 0.5),
                    int(strategy == "raw_probability"),
                    strategy,
                    float(alpha),
                    probability,
                    threshold,
                )
                if best_blend is None or candidate[:3] > best_blend[:3]:
                    best_blend = candidate
        assert best_blend is not None
        blend_score, _, _, strategy, alpha, category_blend_oof, blend_threshold = best_blend
        if strategy == "raw_probability":
            category_blend_valid = alpha * category_valid + (1 - alpha) * lexical_valid[valid_mask]
        else:
            valid_margin = alpha * (_safe_logit(category_valid) - _safe_logit(vlm_threshold)) + (
                1 - alpha
            ) * (_safe_logit(lexical_valid[valid_mask]) - _safe_logit(lexical_threshold))
            category_blend_valid = 1 / (1 + np.exp(-valid_margin))
        blend_oof[category_indices] = category_blend_oof
        blend_valid[valid_mask] = category_blend_valid
        models[category] = final_model
        decisions[category] = {
            "C": best_c,
            "class_weight": best_weight,
            "vlm_oof_f1": score,
            "vlm_threshold": vlm_threshold,
            "blend_alpha_vlm": alpha,
            "blend_strategy": strategy,
            "blend_oof_f1": blend_score,
            "blend_threshold": blend_threshold,
        }

    def predict(probability: np.ndarray, threshold_key: str) -> np.ndarray:
        return np.asarray(
            [
                probability[index] >= decisions[category][threshold_key]
                for index, category in enumerate(valid["category"])
            ],
            dtype="int8",
        )

    vlm_prediction = predict(vlm_valid, "vlm_threshold")
    blend_prediction = predict(blend_valid, "blend_threshold")
    lexical_prediction = np.asarray(
        [
            lexical_valid[index] >= lexical_artifact["thresholds"][category]
            for index, category in enumerate(valid["category"])
        ],
        dtype="int8",
    )
    metrics = {
        "lexical": official_category_f1(valid_y, lexical_prediction, valid["category"]),
        "vlm": official_category_f1(valid_y, vlm_prediction, valid["category"]),
        "blend": official_category_f1(valid_y, blend_prediction, valid["category"]),
        "decisions": decisions,
    }
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_version": 1,
        "models": models,
        "decisions": decisions,
        "schema": config,
        "embedding_model": json.loads((Path(embedding_cache) / "state.json").read_text())["model"],
        "lexical": lexical_artifact,
    }
    joblib.dump(artifact, output_dir / "official_multimodal.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    predictions = valid[["id", "title", "description", "category"]].copy()
    predictions["label"] = valid_y
    predictions["lexical_probability"] = lexical_valid
    predictions["vlm_probability"] = vlm_valid
    predictions["blend_probability"] = blend_valid
    predictions["prediction"] = blend_prediction
    predictions["error"] = np.where(
        (valid_y == 1) & (blend_prediction == 0),
        "FN",
        np.where((valid_y == 0) & (blend_prediction == 1), "FP", ""),
    )
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    predictions[predictions["error"].ne("")].to_csv(output_dir / "errors.csv", index=False)
    oof = train[["id", "title", "category", "group"]].copy()
    oof["label"] = train_y
    oof["lexical_probability"] = lexical_oof
    oof["vlm_probability"] = vlm_oof
    oof["blend_probability"] = blend_oof
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    return metrics


def refit_official_multimodal(
    full_path: str,
    output: str,
    *,
    schema: str,
    embedding_cache: str,
    frozen_artifact: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Refit lexical and dense heads on all released labels with frozen OOF decisions."""
    frame, config = load_data(full_path, schema, require_label=True)
    embeddings = normalize(
        np.asarray(load_embedding_cache(embedding_cache, frame["id"]), dtype="float32"),
        norm="l2",
    )
    labels = frame["labels"].map(lambda values: int(values[0])).to_numpy()
    frozen = joblib.load(frozen_artifact)
    lexical_models: dict[str, Any] = {}
    dense_models: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    for offset, category in enumerate(OFFICIAL_CATEGORIES):
        mask = frame["category"].eq(category).to_numpy()
        text = add_rule_tokens(frame.loc[mask, "text"], category)
        lexical = build_lexical_classifier(seed + offset, "single_label")
        frozen_lexical_estimator = frozen["lexical"]["models"][category].named_steps["classifier"]
        lexical.set_params(
            classifier__C=float(frozen_lexical_estimator.C),
            classifier__class_weight=frozen_lexical_estimator.class_weight,
        )
        lexical.fit(text, labels[mask])
        decision = frozen["decisions"][category]
        dense = LogisticRegression(
            C=float(decision["C"]),
            class_weight=decision["class_weight"],
            max_iter=1000,
            solver="lbfgs",
            random_state=seed + offset,
        ).fit(embeddings[mask], labels[mask])
        lexical_models[category] = lexical
        dense_models[category] = dense
        stats[category] = {
            "rows": int(mask.sum()),
            "positives": int(labels[mask].sum()),
            "C": float(decision["C"]),
            "class_weight": decision["class_weight"],
            "lexical_C": float(frozen_lexical_estimator.C),
            "lexical_class_weight": frozen_lexical_estimator.class_weight,
        }
    artifact = {
        **frozen,
        "artifact_version": int(frozen.get("artifact_version", 1)) + 1,
        "models": dense_models,
        "lexical": {
            **frozen["lexical"],
            "models": lexical_models,
        },
        "schema": config,
        "refit": {
            "full_data": True,
            "rows": len(frame),
            "frozen_decisions_source": str(frozen_artifact),
            "stats": stats,
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return artifact["refit"]


def attach_reference_knn(
    full_path: str,
    output: str,
    *,
    schema: str,
    embedding_cache: str,
    base_artifact: str,
    config_path: str,
) -> dict[str, Any]:
    """Attach the frozen v19 nearest-reference stage to a full-data refit artifact."""
    frame, _ = load_data(full_path, schema, require_label=True)
    embeddings = normalize(
        np.asarray(load_embedding_cache(embedding_cache, frame["id"]), dtype="float32"),
        norm="l2",
    )
    labels = frame["labels"].map(lambda values: int(values[0])).to_numpy(dtype="int8")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required = {
        "category",
        "knn_threshold",
        "base_threshold",
        "margin_scale",
        "knn_alpha",
        "blend_threshold",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"KNN config is missing required keys: {missing}")
    category = str(config["category"])
    mask = frame["category"].eq(category).to_numpy()
    if not mask.any():
        raise ValueError(f"KNN category is absent from full data: {category}")
    if set(np.unique(labels[mask])) != {0, 1}:
        raise ValueError("KNN reference bank requires both binary labels")

    artifact = joblib.load(base_artifact)
    override = {
        **config,
        "reference_embeddings": embeddings[mask].astype("float16"),
        "reference_labels": labels[mask],
        "reference_ids": frame.loc[mask, "id"].astype(str).tolist(),
        "batch_size": int(config.get("batch_size", 512)),
    }
    artifact["knn_overrides"] = {category: override}
    artifact["knn_provenance"] = {
        "config": str(config_path),
        "embedding_cache": str(embedding_cache),
        "full_data": str(full_path),
        "rows": int(mask.sum()),
        "positives": int(labels[mask].sum()),
        "id_fingerprint": _fingerprint(frame.loc[mask, "id"].astype(str).tolist()),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    return artifact["knn_provenance"]
