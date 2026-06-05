# Amazon Reviews 2023

Сюда нужно положить локальные raw-файлы датасета `Amazon Reviews 2023` для категории `All_Beauty`.

Источник:

- [McAuley Lab: Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)

Где взять код загрузки:

- `notebooks/advanced/03_two_tower_model_intro.ipynb`

Что делает код:

- helper проекта скачивает два официальных `jsonl.gz`-файла;
- распаковывает их при скачивании;
- сохраняет локально уже готовые `.jsonl` в эту директорию;
- дальше notebook читает локальные файлы без повторной сетевой загрузки.

Как скачать:

```python
import importlib
import recsys_basics.data as data_module

data_module = importlib.reload(data_module)
print("Download URLs:")
print("\n".join(data_module.get_amazon_reviews_2023_download_urls("All_Beauty")))

data_module.download_amazon_reviews_2023_files(category="All_Beauty")
```

Ожидаемый результат:

```text
data/raw/amazon_reviews_2023/
├── README.md
├── All_Beauty.jsonl
└── meta_All_Beauty.jsonl
```

Важно:

- notebook ожидает файлы прямо в `data/raw/amazon_reviews_2023/`, без дополнительной вложенной папки;
- для текущей advanced-главы используется категория `All_Beauty`;
- если файлы уже есть локально, helper не скачивает их повторно без `overwrite=True`.
