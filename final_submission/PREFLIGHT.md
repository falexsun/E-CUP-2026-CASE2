# Submission preflight

- Entry point: `python -u run.py`
- Input aliases: `--test_data_path`, `--test-data-path`, `-i`
- Output aliases: `--output_path`, `--output-path`, `-o`
- Output columns: exactly `id,result`, in input order
- Shared models: `Qwen/Qwen3-VL-Embedding-2B`, `Qwen/Qwen3.5-4B`
- Licenses: Apache-2.0 according to the competition model list
- Classifier training rows: 12 971 (all released labels)
- Embedding contract: model-card chat prompt, last-token pooling, main image
- Runtime dependencies: only packages from the official baseline image; no `sentence_transformers`
- Image training cache: 12 971/12 971 complete, 0 unreadable images
- Offline smoke: 10 input rows passed output validation
- Local checks: Ruff and 24 pytest cases passed on 2026-08-11

The archive intentionally does not contain model weights supplied by the competition image.
If Qwen3.5 comment generation fails, the runner uses a deterministic grounded comment while
preserving the classifier verdict and output contract.
