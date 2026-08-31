# Validation protocol

## Почему random split запрещён

В released data много дубликатов текста, повторных товарных серий и похожих изображений. Строки одной сущности в train и validation завысили бы качество title lookup, TF-IDF и nearest-neighbour веток.

`entity_group` строится до split и соединяет карточки по exact normalized title/full text/head/tail и консервативной сигнатуре главного изображения. Все splitters сохраняют группу целиком.

## Уровни оценки

| Уровень | Назначение | Можно выбирать параметры? |
|---|---|---|
| 5-fold group OOF на train | выбор C, class weight, blend и thresholds | да |
| fixed entity holdout | проверка переноса замороженного решения | нет |
| outer 5-fold group proxy | исследования после v19, особенно редкий класс | только если feature строится внутри fold |
| public leaderboard | внешняя проверка submission | нет, только принять/отклонить версию |

## Fixed holdout

- seed: 42;
- train: 10 428 строк, 5 134 groups;
- validation: 2 543 строки, 1 284 groups;
- overlap ID: 0;
- overlap entity groups: 0.

Распределение validation:

| Категория | label=0 | label=1 |
|---|---:|---:|
| БАД | 382 | 1 072 |
| Легковоспламеняющиеся | 1 049 | 40 |

Leakage audit:

- exact normalized text overlap: 0;
- near-text overlap `≥0.97` в sampled check: 151;
- reused declared image path: 0;
- identical first-image contents: 2;
- perceptual first-image collisions: 38.

Две exact image collisions относятся к частым generic placeholders. Их объединение создало бы искусственные группы на десятки несвязанных товаров; исключение задокументировано.

## Почему одного holdout недостаточно

Всего 40 положительных воспламеняемых делают category F1 высокодисперсным. На outer folds v19 получил:

| Fold | Positives | F1 |
|---:|---:|---:|
| 0 | 40 | 0.8000 |
| 1 | 39 | 0.8861 |
| 2 | 40 | 0.7595 |
| 3 | 40 | 0.3607 |
| 4 | 39 | 0.4688 |
| **OOF aggregate** | **198** | **0.6742** |

Поэтому после v19 новые признаки проверялись не только на лёгком fixed holdout, но и на aggregate outer predictions.

## Repeated OOF для v19 kNN

Nearest-reference feature вычислялся leakage-safe: для каждой score-строки reference bank содержал только training folds. Использованы пять seeds `41,43,47,51,59`; усреднённый OOF signal определил frozen параметры [configs/v19_knn.json](../configs/v19_knn.json).

После выбора параметров full-data refit разрешён: он не меняет архитектуру, blend или threshold, а только обучает heads и формирует reference bank на всех released labels.

## Зафиксированная ошибка v27

В v27 exact-title mappings были построены по полному labeled dataset, а затем оценены на outer predictions. Это делало оценку lookup-части оптимистичной, несмотря на корректные base OOF probabilities. Public score снизился с `0.808892` до `0.800964`.

После этого:

- v27 отклонён;
- exact lookup не считается честным outer evidence;
- v19 восстановлен как безопасная основа;
- из следующей ветки удалены exact-title и hard-nearest механизмы;
- очищенная v45 затем отдельно подтвердила улучшение на public: `0.8142217631` против `0.8088923057`.

## Ограничение outer proxy для v45 rules

Пять regex v45 были сформулированы после анализа released train и отобраны по
совпадениям `39/39` на всём labeled наборе. Поэтому reported outer gain для
regex-части является model-selection proxy, а не полностью независимой оценкой:
правила не переобучались внутри каждого outer fold. Это не является test leakage
и допустимо для full-data обучения, но делает локальную оценку оптимистичной.

Доказательством переноса считается не outer `0.8527`, а внешнее сравнение уже
замороженных submission: public v19 `0.8088923057` против v45 `0.8142217631` и
итоговый private score v45 `0.8644397759`. Private test не использовался для
выбора параметров и был раскрыт только после фиксации submission.

## Official metric

```text
score = (F1_positive(БАД) + F1_positive(Легковоспламеняющиеся)) / 2
```

Реализация покрыта unit-тестом. Accuracy и micro-F1 не используются для выбора финальной модели.
