#!/usr/bin/env python3
"""Extract a rule-aware Qwen3-VL reference embedding cache."""

from __future__ import annotations

import argparse

from ozon_quality import official_multimodal

RULE_AWARE_INSTRUCTION = (
    "Represent this marketplace product for a binary category-compliance decision. "
    "For category Dietary Supplement, positive requires an explicit BAD, biologically active "
    "supplement, or dietary supplement marking in the text or package image; sports nutrition, "
    "an explicit denial, and products without such marking are negative. For category Flammable, "
    "positive means the sold product is a standalone ignition source, contains fuel, a flammable "
    "substance or combustible gas, or the included kit contains a flammable item. Empty grills, "
    "stoves, burners or constructions, absent contents, built-in ignition sources, combustible "
    "material used only as a component, and flammable accessories not included in the kit are "
    "negative. Read visible package markings and distinguish included contents from compatibility."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=100_352)
    args = parser.parse_args()

    official_multimodal.REFERENCE_INSTRUCTION = RULE_AWARE_INSTRUCTION
    result = official_multimodal.extract_reference_embeddings(
        args.input,
        args.output,
        schema=args.schema,
        model_path=args.model,
        mode="joint",
        batch_size=args.batch_size,
        max_pixels=args.max_pixels,
    )
    print(result)


if __name__ == "__main__":
    main()
