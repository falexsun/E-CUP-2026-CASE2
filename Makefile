UV ?= uv
DATA ?= data/data.csv
SCHEMA ?= configs/ozon_schema.json
EMBED_MODEL ?= models/Qwen/Qwen3-VL-Embedding-2B

.PHONY: help install test lint verify-final quality prepare-data train-text embed train-base refit attach-knn build-head build-v45 reproduce-v19 reproduce-v45

help:
	@echo "make install       Install dev/model dependencies"
	@echo "make quality       Run lint and tests"
	@echo "make verify-final  Verify hashes of the submitted v45 snapshot"
	@echo "make reproduce-v19 Rebuild the public-confirmed v19 foundation (GPU required)"
	@echo "make reproduce-v45 Rebuild the final public-confirmed v45 artifact"

install:
	$(UV) sync --extra dev --extra models

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests \
		scripts/build_standardized_head_artifact.py \
		scripts/build_regex_only_artifact.py \
		scripts/verify_final_submission.py

verify-final:
	$(UV) run python scripts/verify_final_submission.py

quality: lint test verify-final

prepare-data:
	$(UV) run ozon-quality official-group-data \
		--input $(DATA) --schema $(SCHEMA) --output data/full_grouped.csv
	$(UV) run ozon-quality official-split \
		--input $(DATA) --schema $(SCHEMA) \
		--train-output data/train.csv --valid-output data/valid.csv --seed 42

train-text:
	$(UV) run ozon-quality official-text-baseline \
		--train data/train.csv --valid data/valid.csv --schema $(SCHEMA) \
		--output artifacts/text_v2 --seed 42 --oof-folds 5

embed:
	$(UV) run ozon-quality official-reference-embed \
		--input data/full_grouped.csv --schema $(SCHEMA) --model $(EMBED_MODEL) \
		--output artifacts/reference_joint_full --mode joint \
		--batch-size 64 --max-pixels 100352

train-base:
	$(UV) run ozon-quality official-multimodal \
		--train data/train.csv --valid data/valid.csv \
		--embedding-input data/full_grouped.csv \
		--embedding-cache artifacts/reference_joint_full \
		--lexical-artifacts artifacts/text_v2 --schema $(SCHEMA) \
		--output artifacts/multimodal_v8

refit:
	$(UV) run ozon-quality official-refit \
		--input data/full_grouped.csv --embedding-cache artifacts/reference_joint_full \
		--frozen-artifact artifacts/multimodal_v8/official_multimodal.joblib \
		--schema $(SCHEMA) --output artifacts/v10_fullrefit.joblib

attach-knn:
	$(UV) run ozon-quality official-attach-knn \
		--input data/full_grouped.csv --embedding-cache artifacts/reference_joint_full \
		--base-artifact artifacts/v10_fullrefit.joblib --config configs/v19_knn.json \
		--schema $(SCHEMA) --output artifacts/v19_fullrefit.joblib

build-head:
	$(UV) run python scripts/build_standardized_head_artifact.py \
		--base-artifact artifacts/v19_fullrefit.joblib \
		--data data/full_grouped.csv --embedding-cache artifacts/reference_joint_full \
		--config configs/v45.json --output artifacts/v45_linear_intermediate.joblib

build-v45:
	$(UV) run python scripts/build_regex_only_artifact.py \
		--base-artifact artifacts/v45_linear_intermediate.joblib \
		--config configs/v45.json --output artifacts/official_multimodal.joblib

reproduce-v19: prepare-data train-text embed train-base refit attach-knn

reproduce-v45: reproduce-v19 build-head build-v45
