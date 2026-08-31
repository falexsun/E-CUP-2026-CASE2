# Воспроизводимость

## Что считается воспроизведением

Есть три уровня проверки:

1. **CPU checks** — установка, unit-тесты, metric/split/output contracts.
2. **Training reproduction** — построение grouped data, OOF, embeddings, v19 foundation, standardized head и v45 artifact.
3. **Official runtime smoke** — запуск точного ZIP в baseline-окружении с моделями из `SHARED_MODELS_PATH`.

## Окружение

```bash
uv sync --extra dev --extra models
uv run python --version
uv run pytest
uv run ruff check src tests \
  scripts/build_standardized_head_artifact.py \
  scripts/build_regex_only_artifact.py \
  scripts/verify_final_submission.py
uv run python scripts/verify_final_submission.py
```

Поддерживается Python `>=3.11,<3.13`. Training embeddings выполнялся на NVIDIA A100 80 GB. Сам classifier refit работает на CPU после подготовки embeddings.

## Данные

Ожидаемая структура:

```text
data/
├── data.csv
└── images/
    ├── <product_id>/
    │   ├── 0.jpg
    │   └── ...
    └── ...
```

`data.csv` содержит `id,name,description,category,label`; лишний индексный столбец безопасно сохраняется как raw field. Generated `train.csv`, `valid.csv` и `full_grouped.csv` не являются исходными данными и пересоздаются CLI-командами из README.

Контрольные свойства released train:

- 12 971 строк;
- 6 418 entity groups;
- 7 469 строк категории `БАД`;
- 5 502 строки категории `Легковоспламеняющиеся`;
- 198 положительных воспламеняемых.

## Модели

Training layout:

```text
models/
└── Qwen/
    ├── Qwen3-VL-Embedding-2B/
    └── Qwen3.5-4B/
```

Official inference layout задаётся переменной:

```bash
export SHARED_MODELS_PATH=/shared_models
```

Архив не содержит веса моделей. IDs, revisions и лицензии перечислены в [MODELS.md](MODELS.md).

## Полная сборка

Исполняемая последовательность приведена в разделе «Полное воспроизведение v45» файла [README.md](README.md). Критические frozen decisions находятся не в тексте отчёта, а в [configs/v19_knn.json](configs/v19_knn.json) и [configs/v45.json](configs/v45.json). Полная цепочка запускается `make reproduce-v45`.

После `official-reference-embed` файл `state.json` должен содержать:

```json
{
  "rows": 12971,
  "next_index": 12971,
  "complete": true,
  "missing_or_broken_images": 0,
  "contract": "model_card_chat_lasttoken_transformers"
}
```

`load_embedding_cache` дополнительно требует полного совпадения списка и порядка ID.

Builder `official-attach-knn` был повторно запущен из repo-команд на сохранённом cache и сравнён с public v19:

- frozen параметры совпали;
- 5 502/5 502 reference IDs совпали в том же порядке;
- labels совпали, positives = 198;
- `max_abs_difference` reference embeddings = `0.0`;
- fingerprint reference IDs: `69edc70fbe31d54a5a37fe625b4ecce7bd84a12832d2f7e3ce6ad3c117170b7d`.

Поверх воспроизведённой v19 были повторно запущены v45 builders с [configs/v45.json](configs/v45.json). Сравнение с реально отправленным артефактом на всех 12 971 released строках показало:

- scaler mean/scale и коэффициенты Logistic Regression совпали точно;
- reference IDs, labels и embeddings совпали точно;
- список и порядок пяти regex совпали;
- 12 971/12 971 predictions совпали;
- `max_abs_difference` итогового decision score = `0.0`.

## Контрольные хэши

Локальные generated-файлы не обязаны храниться в Git, но их можно проверить:

| Файл | SHA-256 |
|---|---|
| `artifacts/official_multimodal_v19_knn_fullrefit.joblib` | `3ce98ab04a35030b7c641917e9252b01b46be72b9c7ff3aba5592c8f41fbf4ab` |
| `submissions/qwen3vl_v10_knn_v19_FULL_REFIT_RUNTIME_FIXED_20260811.zip` | `21f9ac0f1cb19b75ccf434ac28643da36df1055d2e5838ce78cf2555751a413f` |
| `configs/ozon_schema.json` | `b3419a6210d2b64e5a9345484246d823f009ecdddcb995b48912e5f6e42c2897` |
| `configs/v19_knn.json` | `930fbb2b35a354b204b8442a7d511af02d152c561628c7d137189d64fc360081` |
| `configs/v45.json` | `7e78ccfc222ebfb7069542a8b20fcffc67e55704b69031a27ab604cd9de946b4` |
| `artifacts/official_multimodal_v45_standardized_regex_fullrefit.joblib` | `e7604e6868e46f673428aebf8439eb87e3875fdfbd8c3c2e4cfd8c7ac7d9d53a` |
| `submissions/qwen3vl_v45_STANDARDIZED_REGEX_FULL_REFIT_20260812.zip` | `1361b836b6c926e0be0c99689b29238a82d8add53adc7a4e17e537ddbfe12b07` |

Хэши v19 сохранены как контроль воспроизведения foundation. Канонический submission — v45. Final artifact size: 28 603 061 байт; submission ZIP size: 28 630 485 байт.

## Inference smoke

```bash
SHARED_MODELS_PATH=/shared_models python -u run.py \
  --test_data_path /path/to/smoke.csv \
  --output_path /tmp/predictions.csv
```

Проверяется:

- ровно две колонки `id,result`;
- исходный порядок и уникальность ID;
- комментарий длиной 50–300 символов;
- единственный вердикт `бан` или `не бан`;
- отсутствие сетевых загрузок;
- работа deterministic fallback при ошибке explanation LLM.

Exact-ZIP smoke v45 выполнялся на A100: 10/10 embeddings, 10/10 комментариев, 11 строк CSV с header.

Исследовательские LoRA/retrieval scripts требуют дополнительный набор зависимостей:

```bash
uv sync --extra dev --extra models --extra research
```

## Что не коммитится

`data/`, промежуточные `artifacts/`, `.venv/`, caches и веса Qwen исключены из Git. В репозитории остаются:

- исходный код;
- конфиги frozen decisions;
- tests;
- журнал экспериментов;
- документация происхождения артефактов.
- точный `final_submission/` с classifier artifact размером 28.6 MB.

Для архивов submission рекомендуется GitHub Release или конкурсное хранилище, а не Git history.
