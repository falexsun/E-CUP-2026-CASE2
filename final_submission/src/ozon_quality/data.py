"""Conservative schema adapter for unknown hackathon tables."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "id": ("id", "item_id", "product_id", "offer_id", "sku", "ozon_id"),
    "title": ("title", "name", "product_name", "item_name", "offer_name"),
    "description": ("description", "desc", "product_description", "annotation"),
    "brand": ("brand", "brand_name", "manufacturer", "vendor", "trademark"),
    "category": ("category", "category_name", "cat", "type_name"),
    "attributes": ("attributes", "attrs", "characteristics", "properties"),
    "images": (
        "images",
        "image",
        "image_url",
        "image_urls",
        "picture",
        "picture_url",
        "photo",
        "photos",
    ),
    "label": ("label", "target", "class", "quality", "decision", "is_valid", "moderation_label"),
    "group": ("group", "group_id", "seller_id", "shop_id", "vendor_id"),
}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9а-яё]+", "_", value.casefold()).strip("_")


def infer_columns(columns: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    normalized: dict[str, list[str]] = {}
    for column in columns:
        normalized.setdefault(_norm(str(column)), []).append(str(column))
    mapping: dict[str, str] = {}
    ambiguous: list[str] = []
    used: set[str] = set()
    for role, aliases in ALIASES.items():
        matches = list(
            dict.fromkeys(
                source
                for alias in aliases
                for source in normalized.get(_norm(alias), [])
                if source not in used
            )
        )
        exact = [source for source in matches if _norm(source) == role]
        if len(exact) == 1:
            mapping[role] = exact[0]
            used.add(exact[0])
        elif len(matches) == 1:
            mapping[role] = matches[0]
            used.add(matches[0])
        elif matches:
            ambiguous.append(f"{role}: {matches}")
    return mapping, ambiguous


def schema_suggestion(columns: Sequence[str]) -> dict[str, Any]:
    mapping, ambiguous = infer_columns(columns)
    unresolved = []
    if "title" not in mapping and "description" not in mapping:
        unresolved.append("title or description")
    if "images" not in mapping:
        unresolved.append("images (VLM branch will otherwise be empty)")
    return {
        "schema_version": 1,
        "columns": mapping,
        "text_fields": ["title", "brand", "description", "category", "attributes"],
        "image_separator": "|",
        "label_separator": "|",
        "task_type": "auto",
        "label_values": {},
        "metadata": {
            "ambiguous": ambiguous,
            "unresolved": unresolved,
            "source_columns": list(map(str, columns)),
        },
    }


def read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(source, nrows=nrows)
    if suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(source, lines=True)
        return frame.head(nrows) if nrows else frame
    if suffix == ".json":
        frame = pd.read_json(source)
        return frame.head(nrows) if nrows else frame
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
        return frame.head(nrows) if nrows else frame
    raise ValueError(f"Unsupported table format {suffix!r}; use CSV, Parquet, JSON or JSONL")


def load_schema(path: str | Path | None, columns: Sequence[str]) -> dict[str, Any]:
    if path is None:
        return schema_suggestion(columns)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version", 1) != 1 or not isinstance(payload.get("columns"), dict):
        raise ValueError("Schema must have schema_version=1 and a columns object")
    return payload


def _stringify(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def parse_images(value: Any, separator: str = "|") -> list[str]:
    if value is None or value is pd.NA:
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split(separator) if item.strip()]


def parse_labels(value: Any, separator: str = "|") -> list[str]:
    """Parse a scalar, JSON/list or delimiter-separated target into canonical labels."""
    if value is None or value is pd.NA:
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split(separator) if item.strip()]


def _is_positive(value: Any) -> bool:
    if value is None or value is pd.NA:
        return False
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "no", "нет", "nan", "none"}
    return bool(value)


def adapt_frame(
    raw: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    source_dir: str | Path,
    require_label: bool,
) -> pd.DataFrame:
    columns = dict(config.get("columns", {}))
    label_source = columns.get("label")
    label_columns = list(config.get("label_columns", []))
    labels_available = label_source in raw or (label_columns and all(source in raw for source in label_columns))
    if require_label and not labels_available:
        raise ValueError("A configured label column is required for training")
    # Keep every source column for audit/error analysis. Canonical names are backed up
    # before normalization when a source already uses the same name.
    result = raw.copy()
    for canonical in (
        "id",
        "title",
        "description",
        "brand",
        "category",
        "attributes",
        "text",
        "images",
        "labels",
        "group",
    ):
        if canonical in result:
            result[f"raw__{canonical}"] = result[canonical]
    result["id"] = raw[columns["id"]].map(_stringify) if columns.get("id") in raw else [str(i) for i in raw.index]
    text_parts: list[pd.Series] = []
    text_fields = config.get(
        "text_fields", ["title", "brand", "description", "category", "attributes"]
    )
    forbidden = {"id", "images", "label", "group"} & set(text_fields)
    if forbidden:
        raise ValueError(f"text_fields contains leakage/metadata roles: {sorted(forbidden)}")
    for role in text_fields:
        source = columns.get(role)
        if source in raw:
            values = raw[source].map(_stringify)
            result[role] = values
            text_parts.append(values.where(values.eq(""), f"[{role.upper()}] " + values))
    if not text_parts:
        raise ValueError("No configured text field exists in the input")
    result["text"] = text_parts[0]
    for part in text_parts[1:]:
        result["text"] = (result["text"] + " " + part).str.strip()
    image_source = columns.get("images")
    separator = str(config.get("image_separator", "|"))
    base = Path(source_dir)
    if image_source in raw:
        result["images"] = raw[image_source].map(
            lambda value: parse_images(value, separator)
        )
    elif config.get("images_directory"):
        image_root = base / str(config["images_directory"])
        extensions = {
            str(value).casefold()
            for value in config.get(
                "image_extensions", [".jpg", ".jpeg", ".png", ".webp"]
            )
        }

        def discover_images(product_id: str) -> list[str]:
            directory = image_root / product_id
            if not directory.is_dir():
                return []
            return [
                str(path.resolve())
                for path in sorted(directory.iterdir(), key=lambda path: path.name)
                if path.is_file() and path.suffix.casefold() in extensions
            ]

        result["images"] = result["id"].map(discover_images)
    else:
        result["images"] = [[] for _ in range(len(raw))]
    result["images"] = result["images"].map(
        lambda paths: [str((base / p).resolve()) if not re.match(r"^(https?://|/)", p) else p for p in paths]
    )
    mapping = {str(key): str(value) for key, value in config.get("label_values", {}).items()}
    if label_source in raw:
        separator = str(config.get("label_separator", "|"))
        result["labels"] = raw[label_source].map(
            lambda value: [mapping.get(label, label) for label in parse_labels(value, separator)]
        )
    elif label_columns and all(source in raw for source in label_columns):
        result["labels"] = raw[label_columns].apply(
            lambda row: [mapping.get(str(source), str(source)) for source in label_columns if _is_positive(row[source])],
            axis=1,
        )
    group_source = columns.get("group")
    if group_source in raw:
        result["group"] = raw[group_source].map(_stringify)
    return result.reset_index(drop=True)


def load_data(path: str | Path, schema: str | Path | None, require_label: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = read_table(path)
    config = load_schema(schema, raw.columns)
    return adapt_frame(raw, config, source_dir=Path(path).resolve().parent, require_label=require_label), config
