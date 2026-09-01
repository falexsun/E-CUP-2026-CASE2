# Матрица проверки репозитория

Документ сводит требования второго этапа к конкретным проверяемым объектам.

| Требование | Где проверить | Статус |
|---|---|---|
| Репозиторий соответствует финальному submission | [`final_submission/`](../final_submission/) и [FINAL_SUBMISSION.md](FINAL_SUBMISSION.md) | выполнено |
| Итоговый результат зафиксирован | private Macro F1 `0.8644397759`, 8-е место | выполнено |
| Обученный classifier доступен | `final_submission/artifacts/official_multimodal.joblib`, 28 603 061 байт | выполнено |
| Базовые веса не дублируются | Qwen читаются из `SHARED_MODELS_PATH` | выполнено |
| Только разрешённые модели до 4B | [MODELS.md](MODELS.md) | выполнено |
| Открытые лицензии моделей | обе runtime-модели Apache-2.0 | выполнено |
| Нет закрытых inference API | offline `transformers`, scikit-learn и локальные artifacts | выполнено |
| Нет внешних train labels или закрытых объяснений | только released train и его labels | выполнено |
| Training pipeline предоставлен | [REPRODUCIBILITY.md](REPRODUCIBILITY.md), `make reproduce-v45` | exact reassembly проверена; полный второй raw-data rebuild не выполнялся |
| Архитектура имеет точки расширения | общие schema/VLM/OOF/refit stages + отдельные category heads; [SOLUTION.md](SOLUTION.md#11-расширение-на-новые-категории) | качество на третьей категории не проверялось |
| Frozen decisions versioned | [`configs/v19_knn.json`](../configs/v19_knn.json), [`configs/v45.json`](../configs/v45.json) | выполнено |
| История экспериментов сохранена | [EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md), [`experiments.csv`](../experiments.csv) | выполнено |
| Validation leakage описан честно | [VALIDATION.md](VALIDATION.md) | выполнено |
| Комментарии и их ограничения проверены | [COMMENT_QUALITY.md](COMMENT_QUALITY.md) | выполнено |
| Данные и изображения не опубликованы | `.gitignore`, только [`data/README.md`](../data/README.md) | выполнено |
| CPU quality gate | `49 passed, 1 skipped`; Ruff clean; SHA-256 OK | выполнено |
| A100 exact-runtime smoke | 10/10 rows, 10/10 comments, 10/10 output contract | выполнено 2026-08-31 |

## Provenance

Git-репозиторий собран для этапа проверки 31 августа. Более ранняя Git-история
не реконструировалась искусственно. Происхождение конкурсного решения
подтверждается:

- датированным submission ZIP `qwen3vl_v45_STANDARDIZED_REGEX_FULL_REFIT_20260812.zip`;
- SHA-256 ZIP и classifier artifact;
- журналом запусков 11–15 августа;
- exact snapshot исполняемых файлов из ZIP;
- повторной сборкой v45 с совпадением 12 971/12 971 train predictions.

Competition baseline использовался как разрешённое окружение и пример контракта.
Финальный training/inference pipeline не использует закрытые модели, внешние
закрытые API или дополнительные размеченные данные.

## Последняя проверка

Локально:

```text
ruff: all checks passed
pytest: 49 passed, 1 skipped
final v45 snapshot: OK
markdown local links: OK
```

На NVIDIA A100 80 GB с локальными Qwen snapshots:

```text
reference joint: 10/10
generated comments: 10/10
output rows: 10
valid results: 10/10
```

Предупреждения Transformers о документации `min_frames/max_frames` и deprecated
`torch_dtype` не влияют на загрузку моделей или результат smoke.
