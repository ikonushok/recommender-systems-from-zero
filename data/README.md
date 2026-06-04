# Data

В этой директории планируется хранить raw и processed данные для учебного проекта.

Рекомендуемые датасеты:

- `MovieLens latest small` как основной датасет для `core`;
- `Retailrocket` как датасет для `implicit feedback` и e-commerce сценариев;
- `Amazon Reviews 2023` в виде одной небольшой категории, по умолчанию `All_Beauty`, как text-rich расширение.

Для `Retailrocket` в проекте предпочтителен режим:

- один раз скачать публичные CSV-файлы Retailrocket в `data/raw/retailrocket/`;
- дальше читать уже локальные `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv`.

Для `Amazon Reviews 2023` в проекте предпочтителен режим:

- один раз скачать нужную категорию в `data/raw/amazon_reviews_2023/`;
- дальше читать уже локально сохранённые файлы из notebook'ов и модулей.

## Ожидаемая структура

Пример целевой раскладки:

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

Эта структура пока описывает целевую организацию данных, а не гарантирует, что все директории уже созданы.

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
