# Результаты

Числа разделены по источнику оценки. Public, fixed holdout и outer CV нельзя смешивать в одну ранжированную таблицу.

## Public leaderboard

| Версия | Изменение | Macro F1 | Δ | Статус |
|---|---|---:|---:|---|
| v2 | ранний multimodal, повреждённый image corpus | 0.748643 | — | отклонён |
| v9 | full-data OOF с заново выбранными thresholds | 0.745090 | −0.003554 | отклонён |
| v10 | full refit с frozen v8 decisions | 0.793388 | +0.048299 | принят |
| v19 | v10 + repeated-OOF reference kNN | 0.808892 | +0.015504 | предыдущий лучший; основа v45 |
| v27 | exact-title + rules + nearest hard override | 0.800964 | −0.007928 | отклонён |
| **v45** | **v19 + standardized head + 5 узких rules** | **0.814222** | **+0.005329** | **лучший подтверждённый** |

## Fixed entity-group holdout

Размер: 10 428 train / 2 543 validation. В holdout только 40 положительных воспламеняемых, поэтому он используется для проверки переноса, но не для тонкой настройки их порога.

| Эксперимент | БАД F1 | Воспламеняемые F1 | Mean F1 |
|---|---:|---:|---:|
| Rule-aware TF-IDF | 0.9137 | 0.6000 | 0.756857 |
| Старый masked-mean VLM blend | 0.9157 | 0.6000 | 0.757864 |
| Reference VLM без lexical | 0.8705 | 0.6842 | 0.7774 |
| **Lexical + reference joint VLM v8** | **0.9094** | **0.6667** | **0.788047** |
| Reference text-only | — | — | 0.781309 |
| Reference image-only | — | — | 0.752093 |
| Concatenated text/image | — | — | 0.787115 |
| MLP-head ablation | — | — | 0.790494 |

MLP улучшал holdout только на `+0.0024` и был чувствителен к seed, поэтому не заменил стабильную ветку.

## Full-data OOF v9

| Категория | Lexical OOF F1 | VLM OOF F1 | Blend OOF F1 | VLM weight |
|---|---:|---:|---:|---:|
| БАД | 0.9326 | 0.9109 | 0.9345 | 0.15 |
| Легковоспламеняющиеся | 0.6045 | 0.4533 | 0.6081 | 0.25 |

Несмотря на приемлемый OOF, threshold воспламеняемых сдвинулся с `0.331` до `0.630`, и public v9 упал. Поэтому v10 сохраняет v8 decisions при full refit.

## Outer proxy для исследований после v19

Outer — пятифолдовая stratified group оценка по всем released labels. Для v19:

- `БАД F1 = 0.929681`;
- `Легковоспламеняющиеся F1 = 0.674221`;
- mean `0.801951`.

| Кандидат | Flammable F1/AP | Сравнение с v19 | Решение |
|---|---:|---:|---|
| standardized linear head | F1 0.681818 | +0.007597 | слишком малый standalone gain |
| Qwen3.5 hidden head | F1 0.676140 | +0.001919 | не оправдывает вторую модель |
| RBF SVC | F1 0.674221 | 0 | отклонён |
| PCA metric | F1 0.641400 | −0.032821 | отклонён |
| title-only TF-IDF | F1 0.670554 | −0.003667 | отклонён |
| Qwen3-VL LoRA, hard fold | AP 0.209187 | v19 AP 0.363317 | отклонён |
| regex-only v44 | F1 0.768421 | research-only | отдельно не отправлялся |
| **standardized + regex v45** | **F1 0.775726** | **+0.101505** | **public 0.814222; принят** |

v27 показал, что outer proxy может быть оптимистичным, если lookup построен по полному labeled dataset. Поэтому из v45 исключены exact-title и hard-nearest механизмы. Regex v45 тоже отбирались по всему released train, поэтому их outer gain является model-selection proxy, а не независимой оценкой. Public-прирост `+0.005329` заметно скромнее локального proxy, но имеет правильный знак и подтверждает перенос консервативной надстройки.

## Главные исправления

1. Восстановлен полный image corpus: удалённая копия была усечена до 387 МБ вместо 9.2 ГБ.
2. Введены cache completeness, zero-failure и ID fingerprint checks.
3. Masked mean заменён на model-card chat contract и last-token pooling.
4. Full refit отделён от выбора thresholds.
5. Для редкого класса добавлен repeated-OOF reference kNN.
6. Добавлена небольшая standardized head и только пять высокоточных правил.
7. Generative explanation отделён от classifier verdict.

Полный список 28 запусков с runtime, VRAM и причиной принятия/отклонения: [experiments.csv](experiments.csv).
