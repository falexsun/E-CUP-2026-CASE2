# Документация решения

Этот индекс организован по вопросам, которые обычно возникают при проверке
конкурсного репозитория.

## Что именно было отправлено?

- [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md) — единая compliance-матрица для
  проверки требований второго этапа.
- [FINAL_SUBMISSION.md](FINAL_SUBMISSION.md) — канонический v45, hashes и граница
  между точным runtime и исследовательским кодом.
- [PREFLIGHT.md](PREFLIGHT.md) — официальный input/output/runtime contract.

## Как работает решение?

- [TASK_AND_DATA.md](TASK_AND_DATA.md) — постановка, правила категорий и данные.
- [SOLUTION.md](SOLUTION.md) — архитектура, признаки, fusion и inference.
- [PIPELINE_REVIEW.md](PIPELINE_REVIEW.md) — краткий design review.
- [MODELS.md](MODELS.md) — роли моделей, revisions, параметры и лицензии.

## Как автор пришёл к v45?

- [EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md) — последовательная история от
  baseline до v45, включая ошибки и отклонённые ветки.
- [RESULTS.md](RESULTS.md) — public, holdout и outer числа без смешения источников.
- [`experiments.csv`](../experiments.csv) — первичный журнал 33 запусков.

## Насколько оценке можно доверять?

- [VALIDATION.md](VALIDATION.md) — entity grouping, OOF, leakage checks и границы
  outer proxy.
- [DATA_AUDIT.md](DATA_AUDIT.md) — распределения, дубликаты и инцидент с image
  corpus.
- [COMMENT_QUALITY.md](COMMENT_QUALITY.md) — польза объяснений для модератора и
  известная factual limitation fallback.

## Как воспроизвести?

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) — окружение, полная цепочка обучения,
  hashes и A100 smoke.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) — сторонние модели и ПО.

## Как защищать решение?

- [JURY_PITCH.md](JURY_PITCH.md) — сценарий рассказа на 5–7 минут и ответы на
  вероятные вопросы.

Вернуться на [главную страницу репозитория](../README.md).
