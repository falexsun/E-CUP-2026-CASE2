# Локальные данные

Исходные данные и изображения не коммитятся.

```text
data/
├── data.csv
└── images/<id>/*.{jpg,jpeg,png,webp}
```

Generated-файлы:

- `full_grouped.csv` — полный train с `entity_group`;
- `train.csv`, `valid.csv` — фиксированный leakage-aware split.

Они пересоздаются командами `official-group-data` и `official-split` из [README.md](../README.md). Относительные image paths разрешаются от директории входной таблицы.
