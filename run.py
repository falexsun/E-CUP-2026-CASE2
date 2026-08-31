"""Offline E-CUP 2026 task 2 submission entry point."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data  # noqa: E402
from ozon_quality.official import format_result, validate_submission  # noqa: E402
from ozon_quality.official_explanations import (  # noqa: E402
    deterministic_comment,
    generate_comments,
)
from ozon_quality.official_multimodal import (  # noqa: E402
    extract_reference_embeddings,
    load_embedding_cache,
    predict_official_multimodal,
)

SHARED_MODELS = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))
EMBED_MODEL = SHARED_MODELS / "Qwen/Qwen3-VL-Embedding-2B"
LLM_MODEL = SHARED_MODELS / "Qwen/Qwen3.5-4B"
SCHEMA = ROOT / "configs/ozon_schema.json"
CLASSIFIER = ROOT / "artifacts/official_multimodal.joblib"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--test_data_path", "--test-data-path", required=True)
    parser.add_argument("-o", "--output_path", "--output-path", required=True)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--llm-batch-size", type=int, default=64)
    args = parser.parse_args()
    frame, _ = load_data(args.test_data_path, SCHEMA, require_label=False)
    with tempfile.TemporaryDirectory(prefix="ozon_quality_") as cache:
        extract_reference_embeddings(
            args.test_data_path,
            cache,
            schema=str(SCHEMA),
            model_path=str(EMBED_MODEL),
            mode="joint",
            batch_size=args.embed_batch_size,
            max_pixels=100_352,
        )
        embeddings = load_embedding_cache(cache, frame["id"])
        artifact = joblib.load(CLASSIFIER)
        scores = predict_official_multimodal(frame, embeddings, artifact)
    predictions = scores["prediction"].astype(int).tolist()
    try:
        comments = generate_comments(
            str(LLM_MODEL), frame, predictions, batch_size=args.llm_batch_size
        )
    except Exception as error:  # A valid deterministic result is preferable to a failed run.
        print(f"LLM comment generation failed, using grounded fallback: {error}", flush=True)
        comments = [
            deterministic_comment(frame.iloc[index], label)
            for index, label in enumerate(predictions)
        ]
    result = pd.DataFrame(
        {
            "id": frame["id"],
            "result": [
                format_result(comment, label)
                for comment, label in zip(comments, predictions, strict=True)
            ],
        }
    )
    validate_submission(result, frame["id"])
    destination = Path(args.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)


if __name__ == "__main__":
    main()
