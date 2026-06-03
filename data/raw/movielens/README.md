# MovieLens Latest Small

Сюда нужно положить распакованные файлы датасета `MovieLens latest small`.

Источник:

- [GroupLens MovieLens datasets](https://grouplens.org/datasets/movielens/)

Что сделать:

1. Скачать архив `ml-latest-small.zip`.
2. Распаковать архив.
3. Скопировать файлы из распакованной папки в эту директорию.

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
