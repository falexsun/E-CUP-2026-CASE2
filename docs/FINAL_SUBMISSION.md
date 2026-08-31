# Финальное конкурсное решение

Каноническая отправка — `qwen3vl_v45_STANDARDIZED_REGEX_FULL_REFIT_20260812.zip`.
Она получила public Macro F1 `0.8142217630853994` и итоговый private Macro F1
`0.8644397759`, заняв 8-е место private leaderboard.

## Неизменяемый snapshot

Каталог [`final_submission/`](../final_submission/) извлечён непосредственно из
отправленного ZIP без изменений исполняемых файлов. Он нужен, чтобы жюри могло
проверить ровно тот runtime, который участвовал в оценке, независимо от более
поздних исследовательских файлов в корне репозитория.

Ключевые контрольные суммы:

| Объект | SHA-256 |
|---|---|
| исходный ZIP | `1361b836b6c926e0be0c99689b29238a82d8add53adc7a4e17e537ddbfe12b07` |
| classifier artifact | `e7604e6868e46f673428aebf8439eb87e3875fdfbd8c3c2e4cfd8c7ac7d9d53a` |
| `final_submission/run.py` | `fccacca5a818dfa4c8b3a7c3bf7241c681f62289d93b86958716bd3f63d2152f` |
| `official_explanations.py` | `aeae933175685b873c2302ee48cb5b0637b10b281b517462e1de71ee65057c35` |

Проверка snapshot:

```bash
uv run python scripts/verify_final_submission.py
```

## Два слоя репозитория

- `final_submission/` — точный конкурсный runtime и обученный classifier artifact;
- `src/`, `scripts/`, `configs/`, `Makefile` — воспроизводимая цепочка обучения,
  абляции и проверки, из которой был собран v45.

Веса `Qwen/Qwen3-VL-Embedding-2B` и `Qwen/Qwen3.5-4B` не входят в репозиторий:
в официальном окружении они предоставляются через `SHARED_MODELS_PATH`.
Идентификаторы, revisions и лицензии зафиксированы в [`MODELS.md`](MODELS.md).
