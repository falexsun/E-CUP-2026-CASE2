"""CLI for schema onboarding, training and prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ozon_quality.audit import write_data_audit, write_leakage_report
from ozon_quality.data import read_table, schema_suggestion
from ozon_quality.official_baseline import train_official_text_baseline
from ozon_quality.official_multimodal import (
    DEFAULT_QWEN_EMBED_MODEL,
    attach_reference_knn,
    combine_embedding_caches,
    extract_qwen_embeddings,
    extract_reference_embeddings,
    refit_official_multimodal,
    train_official_multimodal,
)
from ozon_quality.official_splitting import official_holdout, write_grouped_official_data
from ozon_quality.official_vqa import DEFAULT_VQA_MODEL, extract_vqa_scores
from ozon_quality.oof import run_group_oof
from ozon_quality.pipeline import predict, train
from ozon_quality.splitting import group_holdout
from ozon_quality.vlm_contract import write_vlm_prompt_batch

DEFAULT_TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_VISION_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_TEXT_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_VISION_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ozon multimodal product quality-control pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("schema-probe", help="Create an editable schema mapping")
    probe.add_argument("--input", required=True)
    probe.add_argument("--output", required=True)
    audit = commands.add_parser("audit", help="Generate DATA_AUDIT.md from actual data")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--schema")
    audit.add_argument("--image-sample", type=int, default=500)
    leakage = commands.add_parser("leakage-check", help="Check exact leakage between fixed splits")
    leakage.add_argument("--train", required=True)
    leakage.add_argument("--valid", required=True)
    leakage.add_argument("--output", required=True)
    leakage.add_argument("--schema")
    split = commands.add_parser("group-split", help="Create a label-balanced group holdout")
    split.add_argument("--input", required=True)
    split.add_argument("--train-output", required=True)
    split.add_argument("--valid-output", required=True)
    split.add_argument("--schema")
    split.add_argument("--valid-size", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--candidates", type=int, default=64)
    official_split = commands.add_parser(
        "official-split", help="E-CUP category/label-balanced entity holdout"
    )
    official_split.add_argument("--input", required=True)
    official_split.add_argument("--train-output", required=True)
    official_split.add_argument("--valid-output", required=True)
    official_split.add_argument("--schema", required=True)
    official_split.add_argument("--valid-size", type=float, default=0.2)
    official_split.add_argument("--seed", type=int, default=42)
    official_split.add_argument("--candidates", type=int, default=512)
    official_group = commands.add_parser(
        "official-group-data", help="Write full official data with leakage-safe entity groups"
    )
    official_group.add_argument("--input", required=True)
    official_group.add_argument("--output", required=True)
    official_group.add_argument("--schema", required=True)
    official_text = commands.add_parser(
        "official-text-baseline", help="Category-specific rule-aware TF-IDF baseline"
    )
    official_text.add_argument("--train", required=True)
    official_text.add_argument("--valid", required=True)
    official_text.add_argument("--output", required=True)
    official_text.add_argument("--schema", required=True)
    official_text.add_argument("--seed", type=int, default=42)
    official_text.add_argument("--oof-folds", type=int, default=5)
    official_text.add_argument("--tune", action="store_true")
    official_embed = commands.add_parser(
        "official-embed", help="Cache joint Qwen3-VL text/image embeddings"
    )
    official_embed.add_argument("--input", required=True)
    official_embed.add_argument("--output", required=True)
    official_embed.add_argument("--schema", required=True)
    official_embed.add_argument("--model", default=DEFAULT_QWEN_EMBED_MODEL)
    official_embed.add_argument("--batch-size", type=int, default=32)
    official_embed.add_argument("--max-images", type=int, default=5)
    official_embed.add_argument("--max-pixels", type=int, default=100_352)
    official_embed.add_argument("--limit", type=int)
    official_reference = commands.add_parser(
        "official-reference-embed",
        help="Cache model-card Qwen3-VL last-token text/image/joint embeddings",
    )
    official_reference.add_argument("--input", required=True)
    official_reference.add_argument("--output", required=True)
    official_reference.add_argument("--schema", required=True)
    official_reference.add_argument("--model", default=DEFAULT_QWEN_EMBED_MODEL)
    official_reference.add_argument("--mode", choices=("text", "image", "joint"), required=True)
    official_reference.add_argument("--image-index", type=int, default=0)
    official_reference.add_argument("--batch-size", type=int, default=32)
    official_reference.add_argument("--max-pixels", type=int, default=100_352)
    official_reference.add_argument("--limit", type=int)
    official_combine = commands.add_parser(
        "official-combine-embeddings", help="Concatenate aligned frozen embedding caches"
    )
    official_combine.add_argument("--cache", action="append", required=True)
    official_combine.add_argument("--output", required=True)
    official_vqa = commands.add_parser(
        "official-vqa-scores", help="Cache Qwen3-VL-Instruct first-token compliance scores"
    )
    official_vqa.add_argument("--input", required=True)
    official_vqa.add_argument("--output", required=True)
    official_vqa.add_argument("--schema", required=True)
    official_vqa.add_argument("--model", default=DEFAULT_VQA_MODEL)
    official_vqa.add_argument("--batch-size", type=int, default=32)
    official_vqa.add_argument("--max-pixels", type=int, default=100_352)
    official_vqa.add_argument("--limit", type=int)
    official_mm = commands.add_parser(
        "official-multimodal", help="Train OOF Qwen3-VL classifier and lexical blend"
    )
    official_mm.add_argument("--train", required=True)
    official_mm.add_argument("--valid", required=True)
    official_mm.add_argument("--embedding-input", required=True)
    official_mm.add_argument("--embedding-cache", required=True)
    official_mm.add_argument("--lexical-artifacts", required=True)
    official_mm.add_argument("--output", required=True)
    official_mm.add_argument("--schema", required=True)
    official_mm.add_argument("--seed", type=int, default=42)
    official_mm.add_argument("--folds", type=int, default=5)
    official_refit = commands.add_parser(
        "official-refit", help="Refit frozen official heads on all released labels"
    )
    official_refit.add_argument("--input", required=True)
    official_refit.add_argument("--embedding-cache", required=True)
    official_refit.add_argument("--frozen-artifact", required=True)
    official_refit.add_argument("--output", required=True)
    official_refit.add_argument("--schema", required=True)
    official_refit.add_argument("--seed", type=int, default=42)
    official_knn = commands.add_parser(
        "official-attach-knn", help="Attach a frozen nearest-reference stage to full refit"
    )
    official_knn.add_argument("--input", required=True)
    official_knn.add_argument("--embedding-cache", required=True)
    official_knn.add_argument("--base-artifact", required=True)
    official_knn.add_argument("--config", required=True)
    official_knn.add_argument("--output", required=True)
    official_knn.add_argument("--schema", required=True)
    vlm_prompt = commands.add_parser(
        "vlm-prompt-batch", help="Build a stratified structured-output VLM audit batch"
    )
    vlm_prompt.add_argument("--input", required=True)
    vlm_prompt.add_argument("--output", required=True)
    vlm_prompt.add_argument("--schema")
    vlm_prompt.add_argument("--labels-json", help="JSON file containing the official label list")
    vlm_prompt.add_argument("--rules", help="Optional UTF-8 rules/context file")
    vlm_prompt.add_argument("--limit", type=int, default=100)
    oof = commands.add_parser("oof", help="Group-aware OOF for robust ensemble selection")
    oof.add_argument("--input", required=True)
    oof.add_argument("--output", required=True)
    oof.add_argument("--schema")
    oof.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    oof.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    oof.add_argument("--text-revision", default=DEFAULT_TEXT_REVISION)
    oof.add_argument("--vision-revision", default=DEFAULT_VISION_REVISION)
    oof.add_argument("--device", default="auto")
    oof.add_argument("--batch-size", type=int, default=16)
    oof.add_argument("--seed", type=int, default=42)
    oof.add_argument("--folds", type=int, default=5)
    fit = commands.add_parser("train", help="Fit fusion classifier on fixed train/validation")
    fit.add_argument("--train", required=True)
    fit.add_argument("--valid", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--schema")
    fit.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    fit.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    fit.add_argument("--text-revision", default=DEFAULT_TEXT_REVISION, help="Pinned Hugging Face commit/tag")
    fit.add_argument("--vision-revision", default=DEFAULT_VISION_REVISION, help="Pinned Hugging Face commit/tag")
    fit.add_argument("--device", default="auto")
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--blend-config", help="Frozen blend_config.json produced by group OOF")
    fit.add_argument(
        "--refit-all",
        action="store_true",
        help="After evaluation, refit artifact on train+valid with frozen decisions",
    )
    infer = commands.add_parser("predict", help="Run frozen artifact on unlabeled data")
    infer.add_argument("--input", required=True)
    infer.add_argument("--model", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--schema")
    infer.add_argument("--device", default="auto")
    infer.add_argument("--batch-size", type=int, default=16)
    infer.add_argument("--cache-dir")
    infer.add_argument("--prediction-column", default="prediction")
    args = parser.parse_args()
    if args.command == "schema-probe":
        suggestion = schema_suggestion(read_table(args.input, nrows=100).columns)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(suggestion, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if args.command == "audit":
        write_data_audit(
            args.input,
            args.output,
            schema=args.schema,
            image_sample=args.image_sample,
        )
        return
    if args.command == "leakage-check":
        ready = write_leakage_report(
            args.train, args.valid, args.output, schema=args.schema
        )
        if not ready:
            raise SystemExit(2)
        return
    if args.command == "group-split":
        result = group_holdout(
            args.input,
            args.train_output,
            args.valid_output,
            schema=args.schema,
            valid_size=args.valid_size,
            seed=args.seed,
            candidates=args.candidates,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "official-split":
        result = official_holdout(
            args.input,
            args.train_output,
            args.valid_output,
            schema=args.schema,
            valid_size=args.valid_size,
            seed=args.seed,
            candidates=args.candidates,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "official-group-data":
        result = write_grouped_official_data(
            args.input,
            args.output,
            schema=args.schema,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "official-text-baseline":
        result = train_official_text_baseline(
            args.train,
            args.valid,
            args.output,
            schema=args.schema,
            seed=args.seed,
            oof_folds=args.oof_folds,
            tune_hyperparameters=args.tune,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "official-embed":
        result = extract_qwen_embeddings(
            args.input,
            args.output,
            schema=args.schema,
            model_path=args.model,
            batch_size=args.batch_size,
            max_images=args.max_images,
            max_pixels=args.max_pixels,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "official-reference-embed":
        result = extract_reference_embeddings(
            args.input,
            args.output,
            schema=args.schema,
            model_path=args.model,
            mode=args.mode,
            image_index=args.image_index,
            batch_size=args.batch_size,
            max_pixels=args.max_pixels,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "official-combine-embeddings":
        result = combine_embedding_caches(args.cache, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "official-vqa-scores":
        result = extract_vqa_scores(
            args.input,
            args.output,
            schema=args.schema,
            model_path=args.model,
            batch_size=args.batch_size,
            max_pixels=args.max_pixels,
            limit=args.limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "official-multimodal":
        result = train_official_multimodal(
            args.train,
            args.valid,
            args.output,
            schema=args.schema,
            embedding_cache=args.embedding_cache,
            embedding_input=args.embedding_input,
            lexical_artifacts=args.lexical_artifacts,
            seed=args.seed,
            folds=args.folds,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "official-refit":
        result = refit_official_multimodal(
            args.input,
            args.output,
            schema=args.schema,
            embedding_cache=args.embedding_cache,
            frozen_artifact=args.frozen_artifact,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "official-attach-knn":
        result = attach_reference_knn(
            args.input,
            args.output,
            schema=args.schema,
            embedding_cache=args.embedding_cache,
            base_artifact=args.base_artifact,
            config_path=args.config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "vlm-prompt-batch":
        labels = None
        if args.labels_json:
            labels = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
            if not isinstance(labels, list) or not all(isinstance(value, str) for value in labels):
                raise ValueError("--labels-json must contain a JSON list of strings")
        rules = Path(args.rules).read_text(encoding="utf-8") if args.rules else ""
        rows = write_vlm_prompt_batch(
            args.input,
            args.output,
            schema=args.schema,
            allowed_labels=labels,
            rules=rules,
            limit=args.limit,
        )
        print(f"Wrote {rows} VLM audit requests to {args.output}")
        return
    if args.command == "oof":
        metrics = run_group_oof(
            args.input,
            args.output,
            schema=args.schema,
            text_model=args.text_model,
            vision_model=args.vision_model,
            text_revision=args.text_revision,
            vision_revision=args.vision_revision,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            folds=args.folds,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    if args.command == "train":
        metrics = train(
            args.train,
            args.valid,
            args.output,
            schema=args.schema,
            text_model=args.text_model,
            vision_model=args.vision_model,
            text_revision=args.text_revision,
            vision_revision=args.vision_revision,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            blend_config=args.blend_config,
            refit_all=args.refit_all,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    result = predict(
        args.input,
        args.model,
        args.output,
        schema=args.schema,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        prediction_column=args.prediction_column,
    )
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
