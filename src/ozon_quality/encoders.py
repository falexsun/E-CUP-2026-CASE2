"""Lazy model adapters and deterministic debug encoders."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class TextEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 16,
        revision: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.revision = revision
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model_name.startswith("debug-hash"):
            return np.stack([_hash_vector(text, 64) for text in texts]).astype("float32")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install model dependencies with: uv sync --extra models") from error
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                revision=self.revision,
            )
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self.batch_size,
            ),
            dtype="float32",
        )


class VisionEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 16,
        revision: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.revision = revision
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Install model dependencies with: uv sync --extra models") from error
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, revision=self.revision
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            revision=self.revision,
        ).to(self.device)
        self._model.eval()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode product text in the same SigLIP space as images."""
        if self.model_name.startswith("debug-pixels"):
            return np.stack([_hash_vector(text, 9) for text in texts]).astype("float32")
        self._load_model()
        import torch

        chunks: list[np.ndarray] = []
        tokenizer_limit = int(getattr(self._processor.tokenizer, "model_max_length", 64))
        max_length = min(tokenizer_limit, 64)
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                inputs = self._processor(
                    text=list(texts[start : start + self.batch_size]),
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                    if key in {"input_ids", "attention_mask"}
                }
                features = self._model.get_text_features(**inputs)
                features = torch.nn.functional.normalize(features, dim=-1)
                chunks.append(features.float().cpu().numpy())
        return np.concatenate(chunks).astype("float32")

    def encode_products(
        self, image_lists: Sequence[Sequence[str]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model_name.startswith("debug-pixels"):
            return self._debug_encode(image_lists)
        self._load_model()
        import torch

        flat: list[Image.Image] = []
        owners: list[int] = []
        sizes: list[tuple[int, int]] = []
        for owner, paths in enumerate(image_lists):
            for path in paths:
                try:
                    image = load_image(path)
                    flat.append(image)
                    owners.append(owner)
                    sizes.append(image.size)
                except (OSError, requests.RequestException, ValueError):
                    continue
        dimension = int(getattr(self._model.config, "projection_dim", 768))
        if not flat:
            return (
                np.zeros((len(image_lists), dimension), dtype="float32"),
                np.zeros(len(image_lists), dtype="float32"),
                np.zeros((len(image_lists), 4), dtype="float32"),
            )
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(flat), self.batch_size):
                inputs = self._processor(images=flat[start : start + self.batch_size], return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items() if key == "pixel_values"}
                features = self._model.get_image_features(**inputs)
                features = torch.nn.functional.normalize(features, dim=-1)
                chunks.append(features.float().cpu().numpy())
        all_features = np.concatenate(chunks)
        output = np.zeros((len(image_lists), all_features.shape[1]), dtype="float32")
        counts = np.zeros(len(image_lists), dtype="float32")
        image_stats = np.zeros((len(image_lists), 4), dtype="float32")
        for owner, feature, (width, height) in zip(owners, all_features, sizes, strict=True):
            output[owner] += feature
            counts[owner] += 1
            image_stats[owner] += [
                np.log1p(width),
                np.log1p(height),
                width / max(height, 1),
                float(min(width, height) < 224),
            ]
        present = counts > 0
        output[present] /= counts[present, None]
        norms = np.linalg.norm(output[present], axis=1, keepdims=True).clip(min=1e-12)
        output[present] /= norms
        image_stats[present] /= counts[present, None]
        return output, present.astype("float32"), image_stats

    def _debug_encode(
        self, image_lists: Sequence[Sequence[str]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        output = np.zeros((len(image_lists), 9), dtype="float32")
        present = np.zeros(len(image_lists), dtype="float32")
        image_stats = np.zeros((len(image_lists), 4), dtype="float32")
        for index, paths in enumerate(image_lists):
            features = []
            for path in paths:
                try:
                    original = load_image(path)
                    width, height = original.size
                    image = original.resize((16, 16)).convert("RGB")
                    pixels = np.asarray(image, dtype="float32") / 255.0
                    features.append(np.r_[pixels.mean((0, 1)), pixels.std((0, 1)), np.quantile(pixels, [0.1, 0.5, 0.9])])
                    image_stats[index] += [
                        np.log1p(width),
                        np.log1p(height),
                        width / max(height, 1),
                        float(min(width, height) < 224),
                    ]
                except (OSError, requests.RequestException, ValueError):
                    continue
            if features:
                output[index] = np.mean(features, axis=0)
                present[index] = 1
                image_stats[index] /= len(features)
        return output, present, image_stats


def _hash_vector(text: str, dimension: int) -> np.ndarray:
    output = np.zeros(dimension, dtype="float32")
    tokens = text.casefold().split()
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest)
        output[number % dimension] += 1.0 if number & 1 else -1.0
    norm = np.linalg.norm(output)
    return output / norm if norm else output


def load_image(path: str, timeout: float = 10.0) -> Image.Image:
    if path.startswith(("http://", "https://")):
        response = requests.get(path, timeout=timeout)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return Image.open(Path(path)).convert("RGB")
