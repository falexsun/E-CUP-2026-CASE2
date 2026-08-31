# E-CUP 2026 · Контроль качества товаров Ozon

[![quality](https://github.com/falexsun/E-CUP-2026-CASE2/actions/workflows/quality.yml/badge.svg)](https://github.com/falexsun/E-CUP-2026-CASE2/actions/workflows/quality.yml)

Воспроизводимое мультимодальное решение задачи №2. По названию, описанию и
главному изображению товара система определяет соответствие правилам категории
`БАД` или `Легковоспламеняющиеся`, затем формирует комментарий для модератора и
строгий вердикт `бан / не бан`.

| Итог | Значение |
|---|---|
| Финальная версия | `v45` |
| Public Macro F1 | **0.8142217631** |
| Private Macro F1 | **0.8644397759 · топ-8** |
| Runtime-модели | Qwen3-VL-Embedding-2B + Qwen3.5-4B |
| Лицензии моделей | Apache-2.0 |
| Точный submission | [`final_submission/`](final_submission/) |
| Проверка репозитория | `make quality` |

## Маршрут для жюри

### За 3 минуты

1. [Матрица соответствия требованиям второго этапа](docs/REPOSITORY_AUDIT.md).
2. [Точный отправленный v45 и его SHA-256](docs/FINAL_SUBMISSION.md).
3. [Архитектура, модели и лицензии](docs/SOLUTION.md), [MODELS.md](docs/MODELS.md).

### За 10 минут

4. [Как развивалось решение и почему идеи отклонялись](docs/EXPERIMENT_JOURNEY.md).
5. [Validation, leakage checks и честные ограничения](docs/VALIDATION.md).
6. [Качество комментариев для модератора](docs/COMMENT_QUALITY.md).
7. [Полное воспроизведение](docs/REPRODUCIBILITY.md).

Все материалы собраны в [индексе документации](docs/README.md).

## Почему решение устроено именно так

```mermaid
flowchart LR
    T["Название + описание"] --> L["word/char TF-IDF"]
    T --> V["Qwen3-VL joint embedding"]
    I["Главное изображение"] --> V
    L --> B["Category-specific blend"]
    V --> B
    V --> K["Reference kNN редкого класса"]
    V --> H["Standardized linear head"]
    B --> D["Frozen decision"]
    K --> D
    H --> D
    R["5 узких rules"] --> D
    D --> C["Qwen3.5-4B comment"]
    D --> F["Deterministic fallback"]
```

- В категории `БАД` 5 564 положительных примера: устойчиво работает сочетание
  lexical и joint VLM признаков.
- В категории `Легковоспламеняющиеся` только 198 положительных строк и 113
  независимых positive groups: поэтому используется отдельный reference kNN.
- Пороги и веса фиксируются по group OOF до full-data refit.
- Генератор комментария получает уже готовый label и не может изменить метрику
  классификатора.

## Масштабирование на новые категории

Финальное решение обучено на двух конкурсных направлениях, но его основа не
является монолитным классификатором только для БАД и воспламеняемых. Общими для
всех направлений остаются:

- schema adapter карточки товара и поиск изображений по `id`;
- entity grouping, group OOF и контроль утечек;
- TF-IDF и joint Qwen3-VL feature extraction;
- кеширование embeddings, full-data refit и offline runtime;
- генерация комментария после фиксированного classifier verdict;
- единый input/output contract и тесты.

Каждая категория получает собственные heads, fusion weights и threshold. Поэтому
новое направление модерации можно добавлять независимо: определить task context,
подготовить размеченные примеры, обучить category head и откалибровать порог, не
переписывая инфраструктуру и не переобучая уже работающие головы.

Это инженерная расширяемость, а не обещание zero-shot качества: новая категория
требует собственных данных, правил и отдельной leakage-aware проверки. Подробная
граница переиспользования описана в [SOLUTION.md](docs/SOLUTION.md#11-масштабирование-на-новые-категории).

## Подтверждённая эволюция

| Версия | Проверка | Macro F1 | Вывод |
|---|---|---:|---|
| text v2 | fixed holdout | 0.756857 | сильный дешёвый baseline |
| reference VLM v8 | fixed holdout | 0.788047 | исправлен image/model contract |
| full OOF v9 | public | 0.745090 | отклонён: нестабильный новый threshold |
| frozen refit v10 | public | 0.793388 | решения заморожены до refit |
| reference kNN v19 | public | 0.808892 | основа финальной версии |
| title/rules v27 | public | 0.800964 | отклонён: optimistic lookup не перенёсся |
| **standardized + rules v45** | **public** | **0.814222** | **финальный submission** |
| **standardized + rules v45** | **private** | **0.864440** | **итоговое 8-е место** |

Полный машинно-читаемый журнал: [`experiments.csv`](experiments.csv).

## Точный конкурсный runtime

Каталог [`final_submission/`](final_submission/) извлечён из реально отправленного
ZIP. Он содержит код и обученный classifier artifact, но не содержит базовые
веса Qwen: организаторы предоставляют их через `SHARED_MODELS_PATH`.

```bash
uv run python scripts/verify_final_submission.py
```

Проверяемые SHA-256:

- submission ZIP: `1361b836b6c926e0be0c99689b29238a82d8add53adc7a4e17e537ddbfe12b07`;
- classifier artifact: `e7604e6868e46f673428aebf8439eb87e3875fdfbd8c3c2e4cfd8c7ac7d9d53a`.

## Быстрая проверка репозитория

Требования: Python 3.11/3.12 и [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
make quality
```

Ожидаемый результат:

```text
ruff: all checks passed
pytest: 49 passed, 1 skipped
final v45 snapshot: OK
```

GPU и веса моделей для этих CPU-проверок не нужны. Полное обучение и inference
описаны в [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Официальный inference

```bash
SHARED_MODELS_PATH=/shared_models python -u final_submission/run.py \
  --test_data_path /path/to/test.csv \
  --output_path /path/to/predictions.csv
```

Runner гарантирует:

- две колонки `id,result` в исходном порядке;
- формат `<комментарий>...<вердикт>бан|не бан`;
- комментарий длиной 50–300 символов;
- отсутствие сетевых загрузок;
- deterministic fallback при сбое explanation LLM.

## Структура

| Путь | Назначение |
|---|---|
| [`final_submission/`](final_submission/) | точный код и classifier artifact отправленного v45 |
| [`src/ozon_quality/`](src/ozon_quality/) | подготовка данных, split, обучение и inference |
| [`configs/`](configs/) | schema и frozen параметры v19/v45 |
| [`scripts/`](scripts/) | финальные builders и архив абляций |
| [`tests/`](tests/) | metric, leakage, cache, artifact и output-contract tests |
| [`docs/`](docs/) | решение, путь экспериментов, validation и воспроизводимость |
| [`experiments.csv`](experiments.csv) | журнал принятых и отклонённых запусков |

Датасет, изображения, промежуточные caches и базовые веса Qwen не коммитятся.

## Соответствие правилам

- Используются только разрешённые модели до 4B:
  `Qwen/Qwen3-VL-Embedding-2B` и `Qwen/Qwen3.5-4B`.
- Обе модели имеют Apache-2.0; Model IDs и revisions приведены в
  [MODELS.md](docs/MODELS.md).
- Закрытые API и проприетарные модели не используются.
- Inference полностью автономный и рассчитан на официальный baseline image.
- Обученный classifier artifact опубликован; вся цепочка его восстановления
  описана командами и versioned configs.

## Честно зафиксированные ограничения

- Для редкого класса всего 113 positive entity groups, поэтому variance между
  folds остаётся высокой.
- Пять regex v45 отбирались по released train; их outer gain является
  model-selection proxy, а не независимой оценкой. Перенос подтверждён public.
- Explanation LLM видит название и описание, но не пиксели. У fallback есть
  известная зона риска необоснованной ссылки на изображения; она раскрыта в
  [COMMENT_QUALITY.md](docs/COMMENT_QUALITY.md), а точный submission не
  переписывался задним числом.
- `joblib`-артефакт следует загружать только из этого доверенного репозитория.
