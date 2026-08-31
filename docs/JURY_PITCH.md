# Сценарий защиты решения

Ниже — структура рассказа на 5–7 минут. Это не отдельная «маркетинговая» версия: все числа и выводы ссылаются на воспроизводимые файлы репозитория.

## 0:00–0:30 — задача и результат

> Я решал бинарную модерацию товаров в двух категориях по названию, описанию и изображениям. Главная особенность данных — противоположный баланс классов: для БАД 74.5% положительных карточек, а для воспламеняемых только 3.6%. Поэтому я строил отдельные category heads и оптимизировал официальный positive F1. Лучший подтверждённый public score — 0.814222 у версии v45; эта же версия вошла в топ-8 private leaderboard.

## 0:30–1:15 — почему validation был первой частью решения

> В датасете 3 467 дубликатов составного текста и повторяющиеся изображения. Random split дал бы утечку через товарные семьи. Я объединил exact title/full text/head-tail templates и консервативные image signatures в entity groups. Все OOF и holdout-разбиения сохраняют группу целиком. Порог выбирается на group OOF, holdout только проверяет уже замороженное решение.

Показать: [VALIDATION.md](VALIDATION.md), таблицу fold variance редкого класса.

## 1:15–2:15 — архитектура

> Первая ветка — word/char TF-IDF: она сохраняет прямые маркировки, артикулы и опечатки. Вторая — joint embedding текста и главного изображения через Qwen3-VL-Embedding-2B по model-card chat contract с last-token pooling. Их вероятности объединяются отдельно для каждой категории. Для редких воспламеняемых добавляется разность cosine similarity до ближайшего положительного и отрицательного reference. Это улучшило public v10 с 0.793388 до 0.808892. Финальная v45 добавляет небольшую standardized linear head и пять узких, однозначных по условию правил; public вырос до 0.814222.

Показать: диаграмму в [README.md](../README.md) и формулу в [SOLUTION.md](SOLUTION.md).

## 2:15–3:15 — важнейший инженерный поворот

> Ранний public был 0.748643. Причина оказалась не только в модели: копия изображений на GPU занимала 387 МБ вместо 9.2 ГБ, многие файлы были усечены до 256 КБ. Имена существовали, поэтому extractor молча терял visual signal. Я восстановил corpus, добавил completeness, zero-failure и ID fingerprint checks, затем заменил masked mean на рекомендованный last-token contract. Fixed holdout вырос до 0.788047.

Главный тезис: проверенный data contract дал больший эффект, чем усложнение classifier.

## 3:15–4:15 — неудачи и дисциплина экспериментов

> Я сохранял отрицательные результаты. Full-data OOF v9 самовольно сдвинул rare-class threshold и получил public 0.745090. В v10 я заморозил decisions до refit и получил 0.793388. LoRA, RBF, explicit-rule prompts и Qwen3.5 few-shot либо переобучались, либо не улучшали thresholded F1. v27 локально выглядел очень сильным, но exact-title lookup был оптимистично оценён и public снизился до 0.800964. Я удалил lookup и hard-nearest, оставил только устойчивую standardized head и пять правил с 39/39 precision на released data. Эта очищенная v45 получила 0.814222.

Показать: [EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md) или [experiments.csv](../experiments.csv).

## 4:15–5:00 — воспроизводимость

> Полный путь оформлен CLI-командами: group data, split, lexical OOF, Qwen embeddings, multimodal selection, frozen full refit, attach kNN, standardized head и пять узких rules из versioned JSON. Я повторно собрал reference bank этим builder: совпали все 5 502 ID, labels, frozen параметры, а максимальная разница embeddings равна нулю. Tests проверяют metric, grouping, cache alignment, artifact assembly и output contract. Точный отправленный runtime и classifier artifact лежат в `final_submission/` и проверяются по SHA-256.

Показать: [REPRODUCIBILITY.md](REPRODUCIBILITY.md) и `make quality`.

## 5:00–5:30 — объяснения и безопасность runtime

> Qwen3.5 генерирует комментарий только после фиксированного verdict и не может изменить класс. При ошибке используется детерминированный fallback, поэтому генеративная часть не создаёт недетерминированность classifier metric. Explanation LLM видит название и описание, но не пиксели; у fallback есть задокументированная зона риска необоснованной ссылки на изображения. Я не скрываю и не исправляю её задним числом в точном конкурсном snapshot.

## 5:30–5:50 — масштабирование

> Я строил не монолитный бинарный классификатор, а общий multimodal pipeline с отдельными heads, fusion weights и thresholds для направлений модерации. Schema adapter, обработка изображений, embeddings, group OOF, refit, output contract и explanation stage переиспользуются. Для новой категории нужны её правила, размеченные карточки и отдельная калибровка, но не переписывание всей системы и не переобучение уже работающих heads. Это расширяемая архитектура, а не zero-shot обещание.

## Завершение

> Итог — не самая большая обучаемая модель, а система, в которой lexical, joint VLM, retrieval и маленькая calibrated head компенсируют разные ошибки. Самые важные решения — честная группировка, восстановление image contract, заморозка thresholds до refit и консервативное исправление ошибок для 198 редких positives.

## Вероятные вопросы жюри

### Почему kNN только для воспламеняемых?

У БАД достаточно positives и прямых маркировок; linear blend стабилен. У воспламеняемых 198 строк и 113 positive groups, а редкие товарные семьи теряются линейной границей. Category-specific kNN дал подтверждённый public gain `+0.015504`.

### Не является ли full reference bank утечкой?

Нет на test inference: используются только released train labels. Параметры kNN выбирались на repeated OOF, где score fold отсутствовал в reference bank. Full bank создавался после заморозки параметров, как обычный full-data refit.

### Почему не использовали LoRA?

Использовал как ablation. Qwen3.5 LoRA дал valid F1 0.074 после почти нулевого train loss; Qwen3-VL LoRA ухудшил AP сложного fold с 0.363 до 0.209. При таком числе независимых positives frozen retrieval переносился лучше.

### Почему только главное изображение?

Главное изображение давало лучший trade-off качества и runtime. Image-only ветка была слабее joint, а дополнительные image views/concat не дали стабильного OOF/outer выигрыша.

### Почему v27 локально вырос, а public упал?

Exact-title mappings формировались по всему labeled dataset, поэтому outer proxy этой части был оптимистичным. Это задокументированная ошибка evaluation; версия отклонена. Финальная v45 основана на v19, но не содержит ни exact-title lookup, ни hard-nearest override.

### Как гарантируется лицензия?

Обе runtime-модели взяты из разрешённого условием списка и имеют Apache-2.0. Их IDs и использованные revisions записаны в [MODELS.md](MODELS.md); веса не входят в архив.

### Что произойдёт при падении LLM?

Classifier verdict уже рассчитан. Runner формирует deterministic comment и сохраняет валидный CSV. Падение embedding/classifier, напротив, является fatal: silent modality fallback запрещён после обнаруженного image mismatch.

### Что бы вы улучшили дальше?

С новым независимым validation я бы сначала сохранил формальный 100-example evidence audit, затем проверил OCR на заранее определённом slice и subtype-aware rare-class head с nested group CV. Уже использованный public нельзя превращать в validation.

### Почему ручные правила не являются подгонкой под public?

Они выбраны до отправки по двум критериям: прямо следуют из определения класса и не имеют false positive на released data. Правила не используют public labels, exact-title словарь или соседей из test. При этом их outer gain нельзя считать полностью независимым: паттерны отбирались на всём released train. Поэтому я опираюсь на внешний public результат уже замороженной v45, а не выдаю локальный proxy за unbiased estimate.

### Как добавить новое направление модерации?

Добавить его task context и размеченные карточки, зарегистрировать категорию,
затем тем же group OOF pipeline обучить её lexical/VLM heads и заморозить
threshold. Общие schema, image processing, embeddings, cache checks, refit,
runtime и комментарии сохраняются. Если error analysis докажет необходимость,
для нового направления отдельно подключается retrieval или узкое правило.

Нельзя обещать zero-shot качество: новая категория должна пройти собственный
holdout и leakage audit до включения в общий artifact.

## Что не стоит говорить

- Не выдавать локальный outer `0.8527` за leaderboard score; подтверждённый public v45 — `0.814222`.
- Не сравнивать outer `0.8527` напрямую с public leaderboard.
- Не утверждать, что formal 100-example audit сохранён: был targeted error analysis, но отдельный файл не создан.
- Не скрывать v9/v27 — именно их разбор демонстрирует validation discipline.
- Не описывать Qwen3.5 как classifier: в финале она только объясняет frozen verdict.
