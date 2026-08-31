# Design review

## Почему финальная система именно такая

| Решение | Evidence |
|---|---|
| category-specific heads | positive rate 74.5% против 3.6% |
| lexical + VLM late fusion | joint v8 0.7880 против text-only 0.7813 и image-only 0.7521 |
| model-card last-token contract | исправил ранний masked-mean/image mismatch |
| frozen decisions при full refit | v10 public 0.7934 против v9 0.7451 |
| reference kNN только для rare category | v19 public +0.0155 |
| standardized head + 5 узких rules | v45 public 0.8142, +0.0053 к v19 |
| frozen encoders | LoRA не переносился на hard groups |
| explanation после verdict | комментарий не влияет на classifier metric |

## Сильные инженерные свойства

- schema adapter сохраняет raw fields и запрещает label leakage в text;
- grouping и split являются явными CLI-стадиями;
- embedding cache resumable и проверяет completeness/order;
- OOF predictions сохраняются для повторной калибровки;
- full refit не переоткрывает thresholds;
- final runner offline и использует только shared models;
- deterministic explanation fallback сохраняет валидный output; его factual limitations раскрыты в `COMMENT_QUALITY.md`;
- metric, grouping, cache alignment и output contract покрыты тестами.

## Осознанные компромиссы

- Используется только главное изображение в reference branch: дополнительные изображения увеличивали runtime и не дали стабильного прироста.
- kNN bank увеличивает artifact примерно до 28 МБ, но не требует второй neural model.
- Probability не является идеально откалиброванной вероятностью; это decision score, оптимизированный под positive F1.
- Формальный 100-example manual audit не был сохранён как отдельный артефакт — это пробел процесса, отмеченный в DATA_AUDIT.

## Что я бы делал при дополнительном независимом validation

1. Создал бы versioned manual audit с типами evidence и label-noise flags.
2. Проверил бы OCR только на заранее определённом text-on-image slice.
3. Обучал бы subtype-aware rare-class head с nested group CV.
4. Калибровал бы ensemble на новом закрытом split, не используя уже просмотренный public.

Развёрнутая история экспериментов: [EXPERIMENT_JOURNEY.md](EXPERIMENT_JOURNEY.md).
