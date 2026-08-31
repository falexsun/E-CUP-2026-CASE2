# Аудит данных

Источник: `data/data.csv`. Аудит выполнен до выбора финальной архитектуры.

## Схема

- 12 971 строк, 6 исходных колонок;
- `id` уникален, дубликатов ID нет;
- target бинарный, single-label;
- пустых title/description нет;
- у каждой карточки объявлено хотя бы одно изображение;
- число изображений: median 5, p90 5, max 5.

Canonical mapping зафиксирован в [configs/ozon_schema.json](../configs/ozon_schema.json):

```text
id → id
name → title
description → description
category → category
label → label
entity_group → group (для generated split files)
```

## Target и дисбаланс

| Категория | label=0 | label=1 | Всего | Positive rate |
|---|---:|---:|---:|---:|
| БАД | 1 905 | 5 564 | 7 469 | 74.50% |
| Легковоспламеняющиеся | 5 304 | 198 | 5 502 | 3.60% |
| **Всего** | **7 209** | **5 762** | **12 971** | **44.42%** |

Глобальный баланс скрывает проблему: воспламеняемые — редкий positive class. После группировки остаётся только 113 положительных entity groups, median размер положительной группы равен 1.

Следствия для pipeline:

- отдельные category heads и thresholds;
- positive F1, а не accuracy;
- group OOF вместо random split;
- retrieval/reference signal для редкой категории;
- осторожность с LoRA и threshold search.

## Текст

- длина составного текста: median 1 439 символов, p90 2 946, max 6 170;
- 3 467 точных дубликатов составного текста;
- встречаются HTML, seller templates, повторяющиеся хвосты, артикулы и смешение русского/английского;
- title обычно информативнее шаблонного хвоста description, но title-only ablation не улучшила F1.

Для grouping используются normalised title, full text и консервативные head/tail signatures. TF-IDF сохраняет полный текст, а VLM prompt ограничивает description, чтобы шаблонный хвост не вытеснял image tokens.

## Изображения

Автоматический sample 1 000 изображений:

- broken: 0;
- low resolution `<224`: 1;
- duplicate perceptual hashes: 57;
- median width: 903;
- median height: 1 200.

### Инцидент целостности GPU-копии

Первая удалённая копия занимала 387 МБ вместо ожидаемых 9.2 ГБ; многие файлы были усечены до 262 144 байт. Это не было missing-path ошибкой: filenames существовали, поэтому ранний extractor молча терял visual signal.

После обнаружения:

1. image corpus досинхронизирован;
2. локальные и remote SHA-256 проверены на выборке;
3. reference cache пересчитан полностью;
4. `state.json` фиксирует rows/next_index/complete/failures;
5. загрузка cache проверяет точный порядок ID.

Этот инцидент объясняет, почему ранний public score нельзя интерпретировать только как качество модели.

## Ручной и error-driven audit

Проводился targeted разбор FP/FN, особенно для воспламеняемых: газовые баллоны в комплекте, одноразовые мангалы с углём, пиротехника, сухое горючее, пустые конструкции и встроенный поджиг. Отдельный формальный файл со 100 строками и тегами `TEXT_ONLY/IMAGE_ONLY/...` в ходе спринта не был сохранён; это указано явно, а не задним числом реконструировано.

Наблюдения из error analysis:

- часть решений определяется прямой фразой в title/description;
- часть требует различить содержимое комплекта и совместимость устройства;
- package markings важны для `БАД`;
- встречаются противоречивые labels внутри близких товарных семейств.

## Проверяемые generated outputs

- `data/full_grouped.csv`: 12 971 строк, 6 418 groups;
- `data/train.csv`: 10 428 строк;
- `data/valid.csv`: 2 543 строки;
- embedding cache: 12 971 × 2 048, complete, zero failures.

Команды пересоздания находятся в [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
