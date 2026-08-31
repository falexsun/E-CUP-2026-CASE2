# Submission preflight

## Контракт

- entry point: `python -u run.py`;
- input aliases: `--test_data_path`, `--test-data-path`, `-i`;
- output aliases: `--output_path`, `--output-path`, `-o`;
- output columns: строго `id,result` в исходном порядке;
- verdict: `бан` или `не бан`;
- модели: `Qwen/Qwen3-VL-Embedding-2B`, `Qwen/Qwen3.5-4B`;
- model weights не включены в архив;
- runtime offline, модели читаются из `SHARED_MODELS_PATH`.

## Artifact v45

- training rows: 12 971;
- entity groups: 6 418;
- embedding contract: model-card chat + main image + last-token pooling;
- reference category rows: 5 502;
- reference positives: 198;
- frozen decisions: `configs/v19_knn.json`;
- v19 foundation SHA-256: `3ce98ab04a35030b7c641917e9252b01b46be72b9c7ff3aba5592c8f41fbf4ab`;
- standardized head: balanced LR, `C=0.03`, `alpha=0.25`, threshold `0.26`;
- узкие rules: 5, без exact-title lookup и hard-nearest override;
- final decisions: `configs/v45.json`;
- artifact SHA-256: `e7604e6868e46f673428aebf8439eb87e3875fdfbd8c3c2e4cfd8c7ac7d9d53a`;
- submitted ZIP SHA-256: `1361b836b6c926e0be0c99689b29238a82d8add53adc7a4e17e537ddbfe12b07`;
- public Macro F1: `0.8142217631`.

## Проверки

- полный image cache: 12 971/12 971, 0 failures;
- ID/order fingerprint: включён;
- exact ZIP A100 smoke: пройден;
- deterministic comment fallback: покрыт;
- local Ruff: пройден;
- local pytest: 49 passed, 1 skipped без локального `torch` (2026-08-31).

## Failure policy

Ошибка explanation LLM не должна приводить к отсутствию submission: runner использует deterministic comment. Ограничения этого fallback зафиксированы в `COMMENT_QUALITY.md`. Ошибка embedding/classifier является fatal, поскольку молчаливый text-only fallback уже однажды создавал train/runtime mismatch.
