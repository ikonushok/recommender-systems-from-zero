# Датасеты

Этот проект не завязан на один-единственный датасет.
Для учебного маршрута рекомендуется набор из трёх открытых датасетов, где у каждого своя роль:

1. `MovieLens latest small` как основной датасет для `core`;
2. `Retailrocket` как датасет для `implicit feedback` и e-commerce сценариев;
3. `Amazon Reviews 2023` в виде одной небольшой категории как text-rich расширение для hybrid и advanced-блоков.

## Рекомендуемые датасеты

### 1. MovieLens latest small

Источник: [GroupLens MovieLens](https://grouplens.org/datasets/movielens/).

Роль в проекте:

- основной учебный датасет для первых notebook'ов;
- подходит для глав `data -> baseline -> content-based -> collaborative filtering -> metrics -> hybrid`.

Почему он подходит:

- маленький и воспроизводимый;
- имеет понятную схему `userId, movieId, rating, timestamp`;
- содержит метаданные фильмов и пользовательские теги;
- позволяет объяснить `explicit feedback` без лишней инфраструктуры.

Ограничения:

- это не e-commerce и не `implicit feedback`;
- для экспериментов с бинарными взаимодействиями нужно явно объяснять, как рейтинг переводится в событие.

### 2. Retailrocket

Источник: [Retailrocket ecommerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

Роль в проекте:

- второй учебный датасет после MovieLens;
- нужен для тем про `implicit feedback`, временной split, leakage и candidate universe;
- подходит для intro в session/sequential сценарии.

Почему он подходит:

- содержит события `view`, `addtocart`, `transaction`;
- имеет `timestamp`;
- ближе к реальному e-commerce потоку, чем MovieLens;
- есть item properties и category tree, что позволяет показать content/hybrid признаки.

Ограничения:

- данные и свойства товаров анонимизированы и частично захешированы;
- для новичка он сложнее MovieLens;
- скачивание идёт через Kaggle, значит воспроизводимость зависит от доступа к Kaggle-аккаунту.

### 3. Amazon Reviews 2023

Источник: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/main.html).

Рекомендуемый режим использования:

- не весь корпус;
- одна небольшая категория, с которой можно работать в ноутбуке без тяжёлой инфраструктуры;
- стартовый кандидат: `All_Beauty`;
- резервные кандидаты: `Digital_Music` или `Amazon_Fashion`, если понадобится другой домен.

Роль в проекте:

- text-rich датасет для content-based, hybrid и advanced-рекомендаций;
- полезен для глав про признаки объектов, текстовые описания и более реалистичные metadata.

Почему он подходит:

- содержит `rating` и временную информацию;
- имеет отзывы, текст и item metadata;
- даёт более современный пример, чем старые Amazon-срезы;
- подходит для аккуратного введения в retrieval/ranking-пайплайн.

Ограничения:

- весь датасет слишком большой для учебного проекта;
- raw-формат сложнее, чем у MovieLens;
- без жёсткого выбора одной категории курс быстро станет тяжелее, чем нужно новичку.

## Каноническая схема interactions

Независимо от исходного формата датасета, в processed-слое проекта нужно приводить данные к общей схеме:

- `user_id`
- `item_id`
- `event` или `rating`
- `timestamp`

Правила:

- `explicit feedback` и `implicit feedback` не смешиваются без отдельного пояснения;
- отсутствие взаимодействия не становится явным negative label;
- split должен быть временным там, где задача зависит от порядка событий;
- mapping между исходными ID и внутренними индексами модели должен быть явным и воспроизводимым.

## Рекомендуемое распределение по учебному маршруту

- `MovieLens latest small`: основной датасет для `core`.
- `Retailrocket`: датасет для `implicit feedback`, ошибок со split и intro в sequential/use-case.
- `Amazon Reviews 2023 / All_Beauty`: дополнительный датасет для text-rich примеров и advanced-расширения.
