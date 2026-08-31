"""Strict structured-output contract for zero-shot VLM and teacher audits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ozon_quality.data import load_data


@dataclass(frozen=True)
class VLMDecision:
    predicted_labels: list[str]
    confidence: dict[str, float]
    evidence_text: list[str]
    evidence_image: list[str]
    reason_short: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_vlm_prompt(product_text: str, allowed_labels: list[str], rules: str = "") -> str:
    labels_json = json.dumps(allowed_labels, ensure_ascii=False)
    return f"""You audit a marketplace product using its text and attached images.
Allowed labels: {labels_json}
Rules/context (may be empty): {rules}
Product text: {product_text}

Return JSON only with exactly this schema:
{{
  "predicted_labels": [],
  "confidence": {{}},
  "evidence_text": [],
  "evidence_image": [],
  "reason_short": ""
}}
Use only allowed labels. Confidence values must be numbers from 0 to 1. Evidence must be short observations, not claims of ground truth. Do not invent a violation when evidence is absent."""


def parse_vlm_decision(raw: str, allowed_labels: set[str]) -> VLMDecision:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"VLM output is not valid JSON: {error}") from error
    required = {
        "predicted_labels",
        "confidence",
        "evidence_text",
        "evidence_image",
        "reason_short",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"VLM JSON must contain exactly {sorted(required)}")
    labels = payload["predicted_labels"]
    confidence = payload["confidence"]
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("predicted_labels must be a list of strings")
    unknown = sorted(set(labels) - allowed_labels)
    if unknown:
        raise ValueError(f"Unknown predicted labels: {unknown}")
    if not isinstance(confidence, dict) or set(confidence) - allowed_labels:
        raise ValueError("confidence must map allowed labels to probabilities")
    normalized_confidence: dict[str, float] = {}
    for label, value in confidence.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Invalid confidence for {label!r}: {value!r}")
        normalized_confidence[label] = float(value)
    for field in ("evidence_text", "evidence_image"):
        if not isinstance(payload[field], list) or not all(
            isinstance(value, str) for value in payload[field]
        ):
            raise ValueError(f"{field} must be a list of strings")
    if not isinstance(payload["reason_short"], str):
        raise ValueError("reason_short must be a string")
    return VLMDecision(
        predicted_labels=labels,
        confidence=normalized_confidence,
        evidence_text=payload["evidence_text"],
        evidence_image=payload["evidence_image"],
        reason_short=payload["reason_short"],
    )


def write_vlm_prompt_batch(
    input_path: str,
    output_path: str,
    *,
    schema: str | None,
    allowed_labels: list[str] | None,
    rules: str,
    limit: int,
) -> int:
    """Write auditable JSONL requests without running or trusting a teacher model."""
    frame, _ = load_data(input_path, schema, require_label=False)
    if allowed_labels is None:
        if "labels" not in frame:
            raise ValueError("Provide --labels-json when input has no target")
        allowed_labels = sorted({label for labels in frame["labels"] for label in labels})
    if not allowed_labels:
        raise ValueError("Allowed label list is empty")
    # Round-robin by primary label gives a cheap reproducible stratified audit sample.
    if "labels" in frame:
        buckets: dict[str, list[int]] = {}
        for index, labels in enumerate(frame["labels"]):
            key = labels[0] if labels else "__EMPTY__"
            buckets.setdefault(key, []).append(index)
        selected: list[int] = []
        while len(selected) < min(limit, len(frame)) and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(selected) < limit:
                    selected.append(buckets[key].pop(0))
    else:
        selected = list(range(min(limit, len(frame))))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for index in selected:
            row = frame.iloc[index]
            payload = {
                "id": row["id"],
                "images": row["images"],
                "allowed_labels": allowed_labels,
                "prompt": build_vlm_prompt(row["text"], allowed_labels, rules),
            }
            if "labels" in frame:
                payload["ground_truth_for_offline_eval_only"] = row["labels"]
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(selected)
