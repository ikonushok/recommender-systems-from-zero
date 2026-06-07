# Recommender Systems From Zero

Когда делал проект по рекомендательной системе для одного туристического сайта, обнаружил, что простой и понятной теории по ним не так уж и много.
Решил заполнить этот пробел.

Учебный git-проект по рекомендательным системам: от данных, baseline-моделей и метрик до hybrid-подходов и введения в neural recommenders.

Проект задуман как курс-практикум для начинающих и продолжающих: с понятным учебным маршрутом, воспроизводимыми notebook'ами и минимальными Python-модулями без преждевременного ухода в production-инфраструктуру.

## Что это за проект

`recommender-systems-from-zero` — это учебный проект по основам рекомендательных систем.

Цель проекта:

- последовательно объяснить базовые идеи рекомендательных систем;
- показать минимальные рабочие реализации основных подходов;
- научить правильно оценивать качество рекомендаций;
- разобрать типичные ошибки, которые делают новички;
- дать аккуратное введение в более современные подходы, включая neural recommenders.

Это не production-платформа и не готовый recommender service. Проект ориентирован на обучение, а не на industrial-grade инфраструктуру.

## Project status

Проект уже можно использовать как учебный tutorial для новичка по `Core path`.

В репозитории уже есть:

- core-документация;
- core-notebooks;
- минимальный учебный код в `src/recsys_basics`;
- advanced-материалы в виде docs, notebooks и supporting modules.

`Advanced`-часть уже полезна как второй проход после базы. Её лучше трактовать как optional-расширение: runnable-главы имеют docs, notebooks и supporting code, а обзорные разделы остаются в первую очередь концептуальными.

На `7 июня 2026 года` все `6` core-notebooks и `9` advanced-notebooks проверены через `restart kernel + run all` в окружении проекта на Python `3.12.1`.

Если вы впервые изучаете recommender systems, начинайте с `Core path`, а не с `Advanced`.

| Part | Docs | Notebook | Code | Status | Recommended for beginners |
|---|---:|---:|---:|---|---:|
| 01. Intro + dataset and interactions | ✅ | ✅ | ✅ | Ready | ✅ |
| 02. Popularity baseline | ✅ | ✅ | ✅ | Ready | ✅ |
| 03. Content-based TF-IDF | ✅ | ✅ | ✅ | Ready | ✅ |
| 04. Item-item collaborative filtering | ✅ | ✅ | ✅ | Ready | ✅ |
| 05. Metrics | ✅ | ✅ | ✅ | Ready | ✅ |
| 06. Hybrid recommendations | ✅ | ✅ | ✅ | Ready | ✅ |
| 07. Common mistakes | ✅ | — | — | Conceptual guide | ✅ |
| 08. Cold-start | ✅ | — | — | Conceptual guide | ✅ |
| Advanced: ALS | ✅ | ✅ | ✅ | Ready | Optional |
| Advanced: LightFM hybrid factorization | ✅ | ✅ | ✅ | Ready | Optional |
| Advanced: LightFM feature engineering | ✅ | ✅ | ✅ | Ready | Optional |
| Advanced: LightFM hyperparameter tuning | ✅ | ✅ | ✅ | Ready | Optional |
| Advanced: Neural Collaborative Filtering | ✅ | ✅ | ✅ | Ready | Optional |
| Advanced: Two-Tower retrieval | ✅ | ✅ | ✅ | Ready | Optional |
| Advanced: Sequential recommendations | ✅ | ✅ | ✅ | Ready | Optional |
| Advanced: Retrieval vs ranking | ✅ | ✅ | — | Conceptual guide | Optional |
| Advanced: Production overview | ✅ | — | — | Conceptual guide | Optional |

Status legend:

- `Ready` — runnable-материал завершён и проверен сверху вниз в заявленном окружении.
- `Conceptual guide` — завершённая концептуальная глава без обязательного notebook или отдельного модуля.
- `Recommended for beginners = ✅` — входит в обязательный `Core path`.
- `Recommended for beginners = Optional` — изучается после `Core path`; это не статус готовности.

Статусов `Draft` и `Planned` в текущей таблице нет: все перечисленные материалы либо проверены как `Ready`, либо завершены как `Conceptual guide`.

Если вы новичок в рекомендательных системах, не начинайте с `Advanced` notebooks.
Сначала пройдите `Core path`: dataset -> popularity baseline -> content-based -> item-item CF -> metrics -> hybrid.

## Для кого этот проект

Проект рассчитан на:

- начинающих ML-инженеров;
- аналитиков, которые хотят понять, как работают рекомендации;
- backend/data-разработчиков, которым нужен практический вход в recommender systems;
- студентов, которым нужен структурированный путь от простого к сложному.

Минимальный вход:

- базовый Python;
- базовый `pandas`;
- понимание таблиц и joins;
- базовое представление о train/test split;
- желательно общее знакомство с machine learning.

## Что вы поймёте после прохождения

После прохождения core-части проекта должно быть понятно:

- какие данные нужны рекомендательной системе;
- чем отличаются `explicit feedback` и `implicit feedback`;
- почему baseline-модель обязательна;
- как работают content-based и collaborative filtering подходы;
- как правильно делать split без leakage;
- как считать и интерпретировать top-K метрики;
- зачем нужны hybrid-рекомендации;
- что такое `cold-start` и почему система должна уметь работать и с тёплыми, и с новыми пользователями / объектами;
- какие ограничения есть у простых моделей;
- какие типичные ошибки искажают выводы.

После `Core path` уже можно самостоятельно собрать простую учебную рекомендательную систему, сравнить её с baseline и честно оценить через top-K метрики.

После advanced-части должно появиться общее понимание:

- что дают matrix factorization и ALS;
- как hybrid factorization связывает collaborative-сигнал и признаки объектов;
- как neural recommenders связаны с классическими подходами;
- где применяются two-tower и sequential модели;
- как recsys-пайплайн выглядит ближе к production-среде.

## Учебный маршрут

### Core path

Обязательный маршрут для новичка:

1. Concepts and problem framing
2. Data and interaction table
3. Popularity baseline
4. Content-based recommendations
5. Collaborative filtering
6. Evaluation and metrics
7. Hybrid recommendations
8. Common mistakes
9. Cold-start

### Advanced path

Расширение после освоения базы:

1. Matrix factorization and ALS
2. LightFM Hybrid Factorization
3. LightFM feature engineering
4. LightFM hyperparameter tuning
5. Neural Collaborative Filtering
6. Two-Tower retrieval models
7. Sequential recommenders
8. Ranking / Learning-to-Rank intro
9. Production overview

## Как проходить advanced

В `advanced`-части проекта важен не только порядок тем, но и порядок роста сложности.

Здесь действует более жёсткое правило:

- сначала меняем тип модели, а не всё сразу;
- не вводим новый датасет одновременно с новой сложной постановкой, если это можно избежать;
- сохраняем baseline comparison;
- не ломаем split policy и ID mapping, уже зафиксированные в `core`;
- каждая глава должна отвечать на вопрос: что именно стало понятно после неё, чего не было после `core`.

Логика advanced-маршрута такая:

1. `ALS` даёт переход от neighbourhood-подходов к latent factors.
2. `LightFM` показывает, как latent factor модель может использовать item-features и hybrid-логику.
3. Отдельный шаг `feature engineering` позволяет усилить hybrid-модель, не меняя её класс.
4. Отдельный шаг `hyperparameter tuning` отделяет gain от признаков и gain от настройки модели.
5. `Neural CF` меняет класс модели, но держит постановку близкой к `ALS` и `LightFM`.
6. `Two-Tower` переводит разговор из pair scoring в retrieval-мышление.
7. `Sequential` добавляет порядок событий и next-item логику.
8. `Ranking / LTR` объясняет, почему реальная система обычно многостадийная.
9. `Production overview` собирает всё в одну карту системы.

## Структура курса

### Core

Core-часть — это обязательный минимум, который должен пройти каждый, кто хочет понять рекомендательные системы с нуля.

План тем:

1. **Введение**
   Что такое рекомендательная система, какие бывают постановки задачи, почему рекомендации отличаются от обычной классификации.
   [doc](docs/basic/01_intro.md) [notebook](notebooks/basic/01_intro_dataset.ipynb)

2. **Данные и interactions**
   Что такое interaction table, какие поля в ней обязательны, как устроены user-item взаимодействия.
   [doc](docs/basic/02_data_and_interactions.md) [notebook](notebooks/basic/01_intro_dataset.ipynb)

3. **Popularity baseline**
   Самая простая модель рекомендаций и зачем она нужна как точка отсчёта.
   [doc](docs/basic/03_popularity_baseline.md) [notebook](notebooks/basic/02_popularity_baseline.ipynb)

4. **Content-based recommendations**
   Как рекомендовать похожие объекты по признакам самих объектов.
   [doc](docs/basic/04_content_based.md) [notebook](notebooks/basic/03_content_based_tfidf.ipynb)

5. **Collaborative filtering**
   Как использовать поведение пользователей для рекомендаций.
   [doc](docs/basic/05_collaborative_filtering.md) [notebook](notebooks/basic/04_item_item_cf.ipynb)

6. **Метрики качества**
   Почему `accuracy` здесь почти бесполезна и как считать `Precision@K`, `Recall@K`, `MAP@K`, `NDCG@K`.
   [doc](docs/basic/06_metrics.md) [notebook](notebooks/basic/05_metrics.ipynb)

7. **Hybrid recommendations**
   Как комбинировать несколько источников сигналов.
   [doc](docs/basic/07_hybrid_recommendations.md) [notebook](notebooks/basic/06_hybrid_intro.ipynb)

8. **Типичные ошибки**
   Leakage, неправильный split, неверная интерпретация implicit feedback, ошибки в candidate universe, сломанный ID mapping.
   [doc](docs/basic/08_common_mistakes.md)

9. **Cold-start**
   Что делать с новыми пользователями и новыми объектами, почему разные модели по-разному ведут себя в холодном старте и какие fallback-подходы нужны даже в учебном проекте.
   [doc](docs/basic/09_cold_start.md)

### Advanced

Advanced-часть нужна не для первого знакомства, а для расширения кругозора и постепенного перехода к более сильным подходам.

План тем:

1. **Matrix factorization и ALS**
   [doc](docs/advanced/01_matrix_factorization_als.md) [notebook](notebooks/advanced/01_als_implicit.ipynb)
2. **LightFM Hybrid Factorization**
   [doc](docs/advanced/02a_lightfm_hybrid_factorization.md) [notebook](notebooks/advanced/02a_lightfm_hybrid_factorization.ipynb)
3. **LightFM Feature Engineering**
   [doc](docs/advanced/02b_lightfm_feature_engineering.md) [notebook](notebooks/advanced/02b_lightfm_feature_engineering.ipynb)
4. **LightFM Hyperparameter Tuning**
   [doc](docs/advanced/02c_lightfm_hyperparameter_tuning.md) [notebook](notebooks/advanced/02c_lightfm_hyperparameter_tuning.ipynb)
5. **Neural Collaborative Filtering**
   [doc](docs/advanced/02d_neural_collaborative_filtering.md) [notebook](notebooks/advanced/02d_neural_collaborative_filtering.ipynb)
6. **Two-Tower models**
   [doc](docs/advanced/03_two_tower_models.md) [notebook](notebooks/advanced/03_two_tower_model_intro.ipynb)
7. **Sequential recommenders**
   [doc](docs/advanced/04_sequence_based_recommendations.md) [notebook](notebooks/advanced/04_sequence_recommender_intro.ipynb)
8. **Ranking / LTR intro**
   [doc](docs/advanced/05_ranking_and_ltr_intro.md) [notebook](notebooks/advanced/05_retrieval_vs_ranking_toy.ipynb)
9. **Production overview**
   [doc](docs/advanced/06_production_overview.md)

Рекомендуемый режим разработки `advanced`:

1. Сначала доводится до внятного состояния одна глава целиком: `docs` + `notebook` + минимальный модульный каркас.
2. Только после этого открывается следующая глава.
3. Для первых пяти advanced-глав желательно сохранять максимально похожую offline-постановку, чтобы новичок видел эффект именно смены модели, признаков и настройки.
4. Для глав `Ranking / LTR intro` и `Production overview` основной артефакт — сильная документация; тяжёлый production-код для проекта не обязателен.

Привязка датасетов к advanced-пути:

- `ALS`: старт на `MovieLens latest small` в упрощённой implicit-постановке.
- `LightFM 02a`: тот же базовый `MovieLens`, но уже с item-features, чтобы показать hybrid factorization без смены домена.
- `LightFM 02b`: тот же `MovieLens`, но с richer-features, чтобы отдельно показать эффект feature engineering.
- `LightFM 02c`: тот же `MovieLens` и тот же engineered feature set, чтобы отдельно показать эффект tuning.
- `Neural CF`: та же базовая постановка, что и в `ALS` и `LightFM`, чтобы сравнение было честным.
- `Two-Tower`: text-rich сценарий, по умолчанию `Amazon Reviews 2023 / All_Beauty`.
- `Sequential`: сценарий с естественным порядком событий, по умолчанию `Retailrocket`.
- `Ranking / LTR intro`: в основном концептуальная глава с toy-примерами признаков и многостадийного пайплайна.
- `Production overview`: обзорная глава без обязательного отдельного датасета.

## Данные и формат interactions

Во всём проекте базовый формат interaction table фиксируется явно.

Минимальная схема:

- `user_id`
- `item_id`
- `event` или `rating`
- `timestamp`

Ключевые правила:

- `explicit feedback` и `implicit feedback` не смешиваются без пояснения;
- отсутствие взаимодействия не трактуется автоматически как явный negative label;
- train/test split должен соответствовать задаче и не допускать leakage из будущего;
- отображение между исходными `user_id`, `item_id` и внутренними индексами модели должно быть консистентным.

Отдельный раздел по датасету должен отвечать на вопросы:

- какой датасет используется;
- почему он подходит для обучения;
- где лежат raw и processed данные;
- как именно строится split.

На текущем этапе для проекта рекомендуется не один датасет, а небольшой учебный набор:

- `MovieLens latest small` как основной датасет для `core`;
- `Retailrocket` как датасет для `implicit feedback`, временного split и e-commerce сценариев;
- `Amazon Reviews 2023` в виде одной небольшой категории, по умолчанию `All_Beauty`, как text-rich расширение для hybrid и advanced-тем.

Подробности и ограничения описываются в [docs/datasets.md](docs/datasets.md).
Отдельный список источников и краткий разбор покрытия тем вынесены в [docs/bibliography.md](docs/bibliography.md).

## Метрики

В `core` уже разобраны основные offline top-K метрики:

### Core metrics

- `Precision@K`
- `Recall@K`
- `HitRate@K`
- `MAP@K`
- `NDCG@K`

### Extended metrics

- `Coverage`
- `Diversity`
- `Novelty`

`Coverage` уже встречается в базовых notebook'ах. Более широкие свойства вроде `Diversity` и `Novelty` относятся к следующим итерациям расширения проекта и не должны восприниматься как полностью закрытая часть текущего beginner-маршрута.

Принцип проекта: нет метрики без объяснения её смысла и ограничений.

## Принципы проекта

- сначала простая модель, потом сложная;
- каждая новая модель сравнивается с baseline;
- нет метрики без объяснения;
- нельзя допускать leakage из будущего;
- split должен соответствовать постановке задачи;
- notebooks должны запускаться сверху вниз;
- код должен быть минимальным, читаемым и учебным;
- выводы не должны быть сильнее, чем позволяют данные;
- advanced-темы не должны ломать понятность core-маршрута.

## Структура репозитория

Ниже показана рабочая структура репозитория на текущем этапе. Внутри некоторых директорий материалы ещё будут расширяться, но основные пути уже используются именно так.

```text
recommender-systems-from-zero/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── docs/
│   ├── basic/
│   │   ├── README.md
│   │   ├── 01_intro.md
│   │   ├── 02_data_and_interactions.md
│   │   ├── 03_popularity_baseline.md
│   │   ├── 04_content_based.md
│   │   ├── 05_collaborative_filtering.md
│   │   ├── 06_metrics.md
│   │   ├── 07_hybrid_recommendations.md
│   │   ├── 08_common_mistakes.md
│   │   └── 09_cold_start.md
│   │
│   ├── advanced/
│   │   ├── README.md
│   │   ├── 01_matrix_factorization_als.md
│   │   ├── 02a_lightfm_hybrid_factorization.md
│   │   ├── 02b_lightfm_feature_engineering.md
│   │   ├── 02c_lightfm_hyperparameter_tuning.md
│   │   ├── 02d_neural_collaborative_filtering.md
│   │   ├── 03_two_tower_models.md
│   │   ├── 04_sequence_based_recommendations.md
│   │   ├── 05_ranking_and_ltr_intro.md
│   │   └── 06_production_overview.md
│   │
│   ├── bibliography.md
│   └── datasets.md
│
├── notebooks/
│   ├── basic/
│   │   ├── README.md
│   │   ├── 01_intro_dataset.ipynb
│   │   ├── 02_popularity_baseline.ipynb
│   │   ├── 03_content_based_tfidf.ipynb
│   │   ├── 04_item_item_cf.ipynb
│   │   ├── 05_metrics.ipynb
│   │   └── 06_hybrid_intro.ipynb
│   │
│   └── advanced/
│       ├── README.md
│       ├── 00_latent_factors_intuition.ipynb
│       ├── 01_als_implicit.ipynb
│       ├── 02a_lightfm_hybrid_factorization.ipynb
│       ├── 02b_lightfm_feature_engineering.ipynb
│       ├── 02c_lightfm_hyperparameter_tuning.ipynb
│       ├── 02d_neural_collaborative_filtering.ipynb
│       ├── 03_two_tower_model_intro.ipynb
│       ├── 04_sequence_recommender_intro.ipynb
│       └── 05_retrieval_vs_ranking_toy.ipynb
│
├── src/
│   └── recsys_basics/
│       ├── __init__.py
│       ├── data.py
│       ├── split.py
│       ├── metrics.py
│       │
│       ├── basic/
│       │   ├── popularity.py
│       │   ├── content_based.py
│       │   ├── item_item.py
│       │   └── hybrid.py
│       │
│       └── advanced/
│           ├── als.py
│           ├── lightfm.py
│           ├── neural_cf.py
│           ├── two_tower.py
│           └── sequence_model.py
│
└──  data/
    ├── README.md
    ├── raw/
    └── processed/
```

## Текущее состояние проекта

Сейчас это не “план будущего курса”, а уже рабочий учебный репозиторий.

Что уже можно использовать прямо сейчас:

- весь `Core path`: docs + notebooks + базовые модули в `src/`;
- загрузку учебных датасетов `MovieLens`, `Amazon Reviews 2023` и `Retailrocket`;
- весь заявленный `Advanced path` как optional-продолжение после базы.

Что всё ещё расширяется:

- глубина conceptual / production-oriented разделов advanced-пути;
- вспомогательные пояснения и дополнительные учебные примеры.

## Статус развития

Практический статус на текущем этапе:

- `Core path` — рекомендуемый beginner-маршрут и основная готовая часть проекта;
- `Advanced path` — уже содержит runnable optional-материалы; наиболее концептуальные части здесь — `Ranking / LTR` и `Production overview`;
- production-ориентированные темы даны как обзор и карта следующего шага, а не как обещание production-ready системы.

## Установка

Проект предназначен для запуска на `Linux` и `macOS`.

`Windows` не поддерживается: используемая в advanced-разделах библиотека `LightFM` на этой платформе работать не будет.

Окружение описано через `requirements.txt` и `pyproject.toml`. В `requirements.txt` уже включены зависимости для core и runnable advanced-notebooks, включая `implicit`, `lightfm-next`, `torch` и `optuna`.

README пока не пытается превратиться в большой install-manual для всех advanced-сценариев сразу, но сам core-маршрут уже не является “планом на будущее” и доступен для прохождения.

## Validation philosophy

Минимальная проверка зависит от типа изменения:

- `.py` файл: `python -m py_compile`
- метрики: tiny fixture с известным expected result
- split: проверка дат, размеров и отсутствия leakage
- модель: smoke `fit -> recommend -> no duplicates`
- notebook: restart kernel + run all cells
- docs: сверка путей, команд и порядка глав

Если полная проверка дорогая, нужно явно отделять:

- что проверено;
- что не проверено;
- какой остаётся риск.

## Что не входит в scope первой версии

Первая версия проекта не ставит целью:

- production-ready online API;
- feature store;
- A/B testing platform;
- real-time serving;
- сложную orchestration-инфраструктуру;
- полноценный deep learning курс по recommender systems;
- industrial monitoring stack.

Эти темы могут появиться позже как отдельные advanced-расширения.

## Лицензия

В репозитории используется разделение лицензий по типу материалов:

- код в `src/` и служебные файлы проекта лицензируются под `MIT`, см. [LICENSE](LICENSE);
- учебные материалы в `README.md`, `docs/` и `notebooks/` лицензируются под `CC BY 4.0`, см. [LICENSE-docs](LICENSE-docs).

Если для конкретного файла позже понадобится другое правило, оно должно быть указано явно рядом с этим файлом или в соответствующей директории.

## Repository Description

Короткое описание для GitHub:

```text
Учебный проект по рекомендательным системам: от baseline, content-based и collaborative filtering до метрик, hybrid-подходов и введения в neural recommenders.
```

English version:

```text
A course-style recommender systems project: from baselines, content-based filtering, and collaborative filtering to metrics, hybrid methods, and an introduction to neural recommenders.
```
