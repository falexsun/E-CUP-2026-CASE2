# Сторонние модели и ПО

Финальный runtime не вызывает закрытые API и не выполняет сетевые загрузки.

## Модели

| Model ID | Revision | Лицензия | Роль |
|---|---|---|---|
| [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | [`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/commit/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda) | Apache-2.0 | joint text/image embeddings |
| [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | [`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`](https://huggingface.co/Qwen/Qwen3.5-4B/commit/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | Apache-2.0 | комментарии модератору |

Обе модели входят в опубликованный организаторами перечень допустимых моделей
до 4B. Веса предоставляет official runtime через `SHARED_MODELS_PATH`.

## Основные библиотеки

Используются открытые библиотеки: NumPy (BSD-3-Clause), pandas
(BSD-3-Clause), scikit-learn (BSD-3-Clause), joblib (BSD-3-Clause), PyArrow
(Apache-2.0), Pillow (HPND), Requests (Apache-2.0), PyTorch/torchvision
(BSD-style), Transformers/Accelerate/Sentence Transformers (Apache-2.0).

Точные разрешённые версии и транзитивные зависимости зафиксированы в
[`uv.lock`](../uv.lock). Research extras не входят в официальный inference.
