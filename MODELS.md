# Модели

Все модели финального runtime входят в разрешённый организаторами список моделей до 4B и имеют Apache-2.0. Веса загружаются из `SHARED_MODELS_PATH` и не включаются в submission.

| Роль | Model ID | Проверенный revision | Параметры | Режим | В финале |
|---|---|---|---:|---|---|
| joint text/image encoder | [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | [`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/commit/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda) | 2B | frozen, last-token | да |
| explanation LLM | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | [`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`](https://huggingface.co/Qwen/Qwen3.5-4B/commit/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | 4B | frozen generation | да, с fallback |

## Измеренный runtime

| Компонент | Условия | Наблюдение |
|---|---|---|
| Qwen3-VL embedding | A100, batch 64 | около 29.3 GB VRAM в раннем extractor; оптимизированный runtime smoke около 10.4 GB |
| Qwen3-VL reference cache | 12 971 карточка | cache 12 971/12 971, 2 048 dimensions, 0 broken images |
| Qwen3.5 comments | exact ZIP smoke | 10/10 комментариев; fallback покрывает ошибку генерации |

## Проверенные, но отклонённые роли моделей

| Модель/вариант | Гипотеза | Причина отказа |
|---|---|---|
| `Qwen3-VL-2B-Instruct` first-token VQA | прямое применение правил | слабая ранжировка и смещение к positive |
| `Qwen3.5-4B` retrieval few-shot | reasoning по соседним примерам | flammable AP 0.094 |
| `Qwen3.5-4B` LoRA | обучаемый multimodal verdict | train collapse, valid F1 0.074 |
| `Qwen3.5-4B` hidden-state head | второй dense classifier | outer gain +0.0019 не оправдывает runtime |
| `Qwen3-VL-Embedding-2B` LoRA | адаптация embedding space | hard-fold AP 0.209 против 0.363 у v19 |

## Лицензии и provenance

Model IDs и лицензии сверялись с перечнем, опубликованным в условии соревнования,
и с model cards: обе страницы указывают `apache-2.0`, а ссылки на commit выше
подтверждают существование зафиксированных revisions. Локальный revision фиксирует
использованный snapshot, но official image может предоставлять эквивалентный model
directory без Git metadata. Pipeline не выполняет сетевую загрузку во время
official inference.
