# Recommender Systems From Zero

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
- какие ограничения есть у простых моделей;
- какие типичные ошибки искажают выводы.

После advanced-части должно появиться общее понимание:

- что дают matrix factorization и ALS;
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

### Advanced path

Расширение после освоения базы:

1. Matrix factorization and ALS
2. Neural Collaborative Filtering
3. Two-Tower retrieval models
4. Sequential recommenders
5. Ranking / Learning-to-Rank intro
6. Production overview

## Структура курса

### Core

Core-часть — это обязательный минимум, который должен пройти каждый, кто хочет понять рекомендательные системы с нуля.

План тем:

1. **Введение**
   Что такое рекомендательная система, какие бывают постановки задачи, почему рекомендации отличаются от обычной классификации.

2. **Данные и interactions**
   Что такое interaction table, какие поля в ней обязательны, как устроены user-item взаимодействия.

3. **Popularity baseline**
   Самая простая модель рекомендаций и зачем она нужна как точка отсчёта.

4. **Content-based recommendations**
   Как рекомендовать похожие объекты по признакам самих объектов.

5. **Collaborative filtering**
   Как использовать поведение пользователей для рекомендаций.

6. **Метрики качества**
   Почему `accuracy` здесь почти бесполезна и как считать `Precision@K`, `Recall@K`, `MAP@K`, `NDCG@K`.

7. **Hybrid recommendations**
   Как комбинировать несколько источников сигналов.

8. **Типичные ошибки**
   Leakage, неправильный split, неверная интерпретация implicit feedback, ошибки в candidate universe, сломанный ID mapping.

### Advanced

Advanced-часть нужна не для первого знакомства, а для расширения кругозора и постепенного перехода к более сильным подходам.

План тем:

1. **Matrix factorization и ALS**
2. **Neural Collaborative Filtering**
3. **Two-Tower models**
4. **Sequential recommenders**
5. **Ranking / LTR intro**
6. **Production landscape**

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

## Метрики

В проекте планируется разобрать:

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

## Планируемая структура репозитория

Ниже не текущее состояние репозитория, а целевая структура курса.

```text
recommender-systems-from-zero/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── docs/
│   ├── basic/
│   │   ├── 01_intro.md
│   │   ├── 02_data_and_interactions.md
│   │   ├── 03_popularity_baseline.md
│   │   ├── 04_content_based.md
│   │   ├── 05_collaborative_filtering.md
│   │   ├── 06_metrics.md
│   │   ├── 07_hybrid_recommendations.md
│   │   └── 08_common_mistakes.md
│   │
│   ├── advanced/
│   │   ├── 01_matrix_factorization_als.md
│   │   ├── 02_neural_collaborative_filtering.md
│   │   ├── 03_two_tower_models.md
│   │   ├── 04_sequence_based_recommendations.md
│   │   ├── 05_ranking_and_ltr_intro.md
│   │   └── 06_production_overview.md
│   │
│   └── datasets.md
│
├── notebooks/
│   ├── basic/
│   │   ├── 01_intro_dataset.ipynb
│   │   ├── 02_popularity_baseline.ipynb
│   │   ├── 03_content_based_tfidf.ipynb
│   │   ├── 04_item_item_cf.ipynb
│   │   ├── 05_metrics.ipynb
│   │   └── 06_hybrid_intro.ipynb
│   │
│   └── advanced/
│       ├── 01_als_implicit.ipynb
│       ├── 02_neural_collaborative_filtering.ipynb
│       ├── 03_two_tower_model_intro.ipynb
│       └── 04_sequence_recommender_intro.ipynb
│
├── src/
│   └── recsys_basics/
│       ├── __init__.py
│       ├── data.py
│       ├── split.py
│       ├── metrics.py
│       ├── evaluation.py
│       │
│       ├── basic/
│       │   ├── popularity.py
│       │   ├── content_based.py
│       │   ├── item_item.py
│       │   └── hybrid.py
│       │
│       └── advanced/
│           ├── als.py
│           ├── neural_cf.py
│           ├── two_tower.py
│           └── sequence_model.py
│
├── tests/
│   ├── test_metrics.py
│   ├── test_split.py
│   ├── test_popularity.py
│   └── test_no_leakage.py
│
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
│
└── agents/
    ├── architect.md
    ├── tutorial_writer.md
    ├── notebook_reviewer.md
    ├── recommender_reviewer.md
    ├── metrics_reviewer.md
    ├── data_quality_reviewer.md
    ├── red_team.md
    ├── docs_handoff.md
    └── task_spec_short.md
```

## Текущее состояние проекта

Сейчас проект находится на этапе проектирования структуры курса и учебного маршрута.

Что уже есть:

- базовый `README`;
- правила для AI-assisted разработки;
- заготовка под `docs/`;
- структура reviewer/agent-ролей для развития проекта.

Что планируется дальше:

- выбрать учебный датасет;
- зафиксировать split policy;
- подготовить core docs;
- собрать первые notebook'и;
- реализовать baseline и базовые метрики;
- постепенно добавить advanced-блоки.

## Статус развития

План развития:

- [ ] выбрать и описать учебный датасет;
- [ ] оформить раздел про data and interactions;
- [ ] реализовать popularity baseline;
- [ ] реализовать content-based recommender;
- [ ] реализовать collaborative filtering baseline;
- [ ] добавить top-K metrics;
- [ ] добавить hybrid intro;
- [ ] добавить ALS;
- [ ] добавить Neural Collaborative Filtering;
- [ ] добавить Two-Tower intro;
- [ ] добавить sequence-based intro;
- [ ] оформить сравнение моделей и типичные ошибки.

## Установка

Раздел установки будет заполнен после появления минимального рабочего окружения проекта.

Пока репозиторий находится на этапе проектирования структуры и учебного плана, поэтому команды установки и запуска намеренно не фиксируются как окончательные.

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

- код в `src/`, `tests/` и служебные файлы проекта лицензируются под `MIT`, см. [LICENSE](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/LICENSE);
- учебные материалы в `README.md`, `docs/` и `notebooks/` лицензируются под `CC BY 4.0`, см. [LICENSE-docs](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/LICENSE-docs).

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
