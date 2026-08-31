# Скрипты экспериментов

Каталог хранит исследовательскую историю, включая отклонённые гипотезы. Это не
единый production package: многие скрипты являются одноразовыми воспроизводимыми
запусками и требуют сохранённых embedding caches или research extras.

## Финальная воспроизводимая цепочка

- `build_standardized_head_artifact.py` — standardized head v45;
- `build_regex_only_artifact.py` — пять frozen regex и финальный artifact;
- `verify_final_submission.py` — хэши точного конкурсного snapshot.

Эти файлы входят в обязательный `make quality`. Основные стадии v19 реализованы
как CLI в `src/ozon_quality/` и запускаются через `make reproduce-v19`.

## Исследовательский архив

Остальные скрипты соответствуют строкам `experiments.csv`: OCR, LoRA, VQA,
multi-image, alternative heads, nested/outer audits. Они сохранены как evidence
пути к решению, но не вызываются финальным runtime и не входят в основной lint
gate. Результат и причина принятия/отклонения каждой ветки описаны в
`EXPERIMENT_JOURNEY.md` и `RESULTS.md`.

