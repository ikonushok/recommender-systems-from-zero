# Retailrocket

Сюда нужно положить локальные CSV-файлы датасета `Retailrocket ecommerce dataset`.

Источник:

- [Retailrocket dataset overview](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

Где взять код загрузки:

- `notebooks/advanced/04_sequence_recommender_intro.ipynb`

Что делает код:

- helper проекта скачивает публичные CSV-файлы Retailrocket напрямую;
- сохраняет их в эту директорию;
- дальше notebook читает уже локальные `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv`.

Как скачать:

```python
import importlib
import recsys_basics.data as data_module

data_module = importlib.reload(data_module)
print(f"Retailrocket target dir: {data_module.get_retailrocket_data_dir()}")

data_module.download_retailrocket_files()
```

Ожидаемый результат:

```text
data/raw/retailrocket/
├── README.md
├── category_tree.csv
├── events.csv
├── item_properties_part1.csv
└── item_properties_part2.csv
```

Важно:

- notebook ожидает файлы прямо в `data/raw/retailrocket/`, без дополнительной вложенной папки;
- отдельный Kaggle token для текущего helper не нужен;
- если файлы уже есть локально, helper не скачивает их повторно без `overwrite=True`.
