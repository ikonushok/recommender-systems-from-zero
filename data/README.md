# Data

В этой директории хранятся raw и processed данные для учебного проекта.

Сейчас в проекте уже используются:

- `data/raw/movielens/`
- `data/raw/amazon_reviews_2023/`
- `data/raw/retailrocket/`

Для каждого датасета в `data/raw/...` есть свой README с источником, ожидаемой раскладкой и примером одноразового скачивания.

Рекомендуемые датасеты:

- `MovieLens latest small` как основной датасет для `core`;
- `Retailrocket` как датасет для `implicit feedback` и e-commerce сценариев;
- `Amazon Reviews 2023` в виде одной небольшой категории, по умолчанию `All_Beauty`, как text-rich расширение.

Текущий рекомендуемый режим работы такой:

- один раз скачать raw-файлы в `data/raw/...`;
- дальше читать уже локально сохранённые файлы из notebook'ов и модулей проекта;
- processed-слой использовать для явно подготовленных таблиц, если отдельная глава или preprocessing-шаг этого требует.

Куда смотреть по каждому датасету:

- `MovieLens`: [data/raw/movielens/README.md](../data/raw/movielens/README.md)
- `Amazon Reviews 2023`: [data/raw/amazon_reviews_2023/README.md](../data/raw/amazon_reviews_2023/README.md)
- `Retailrocket`: [data/raw/retailrocket/README.md](../data/raw/retailrocket/README.md)

Для `MovieLens` в проекте уже есть helper, который скачивает официальный архив и раскладывает CSV в нужную директорию.

Для `Retailrocket` в проекте предпочтителен режим:

- один раз скачать публичные CSV-файлы Retailrocket в `data/raw/retailrocket/`;
- дальше читать уже локальные `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv`.

Для `Amazon Reviews 2023` в проекте предпочтителен режим:

- один раз скачать нужную категорию в `data/raw/amazon_reviews_2023/`;
- дальше читать уже локально сохранённые файлы из notebook'ов и модулей.

## Текущая структура

Рабочая раскладка данных сейчас такая:

```text
data/
├── raw/
│   ├── movielens/
│   ├── retailrocket/
│   └── amazon_reviews_2023/
└── processed/
    ├── movielens/
    ├── retailrocket/
    └── amazon_reviews_2023/
```

То есть `raw`-директории уже есть и используются в notebook'ах и helper-функциях проекта.

`processed/` пока остаётся более лёгким слоем: он зарезервирован под явно подготовленные таблицы и не обязан быть одинаково заполнен для всех датасетов на каждом этапе курса.

## Правило для processed-слоя

Независимо от исходного формата, в processed-таблицах interactions нужно придерживаться схемы:

- `user_id`
- `item_id`
- `event` или `rating`
- `timestamp`

Дополнительно:

- для `MovieLens` исходные `userId` и `movieId` приводятся к `user_id` и `item_id`;
- для `Retailrocket` события `view`, `addtocart`, `transaction` сохраняют явный тип события;
- для `Amazon Reviews 2023` нужно заранее зафиксировать одну категорию и не смешивать категории без отдельного объяснения.

Практическая граница такая:

- `raw/` хранит исходные локальные файлы датасетов;
- `processed/` хранит уже приведённые к учебной задаче таблицы, если они нужны для следующего шага маршрута;
- canonical interaction table в проекте всё равно должна сводиться к `user_id`, `item_id`, `event/rating`, `timestamp`.
