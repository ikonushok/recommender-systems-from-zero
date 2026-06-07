# MovieLens Latest Small

Сюда нужно положить локальные CSV-файлы датасета `MovieLens latest small`.

Источник:

- [GroupLens MovieLens datasets](https://grouplens.org/datasets/movielens/)

Где взять код загрузки:

- `notebooks/basic/01_intro_dataset.ipynb`

Что делает код:

- helper проекта скачивает официальный архив `ml-latest-small.zip`;
- распаковывает из него `links.csv`, `movies.csv`, `ratings.csv`, `tags.csv`;
- сохраняет файлы прямо в эту директорию;
- дальше notebook читает уже локальные CSV без повторной сетевой загрузки.

Как скачать:

```python
import importlib
import recsys_basics.data as data_module

data_module = importlib.reload(data_module)
print(f"MovieLens target dir: {data_module.get_movielens_data_dir()}")
print(f"Download URL: {data_module.get_movielens_download_url()}")

data_module.download_movielens_latest_small()
```

Ожидаемый результат:

```text
data/raw/movielens/
├── README.md
├── links.csv
├── movies.csv
├── ratings.csv
└── tags.csv
```

Важно:

- файлы должны лежать прямо в `data/raw/movielens/`, без дополнительной вложенной папки `ml-latest-small/`;
- для первого notebook используются в первую очередь `ratings.csv` и `movies.csv`;
- `links.csv` и `tags.csv` нужны для расширения анализа и следующих материалов.
- если файлы уже есть локально, helper не скачивает их повторно без `overwrite=True`.
