# Advanced Path

## Зачем нужен advanced-блок

`Advanced`-часть не заменяет `core` и не делает вид, что после неё проект становится production-системой.

Её задача:

- аккуратно перейти от простых и понятных моделей к более сильным;
- показать, как меняется постановка задачи по мере усложнения recsys;
- не потерять учебную ясность по дороге.

Главный принцип блока:

- в одной главе меняем одну главную сложность, а не три сразу.

Это значит:

- сначала короткий toy-мост к latent factors;
- сначала latent factors на знакомой постановке;
- потом hybrid factorization на тех же данных, но уже с признаками объектов;
- потом отдельный шаг с feature engineering на той же hybrid-модели;
- потом отдельный шаг с tuning этой же hybrid-модели;
- потом neural scoring на максимально похожем сценарии;
- потом retrieval;
- потом последовательности;
- потом toy-мост от retrieval к ranking;
- потом многостадийный ranking;
- потом обзор production-ландшафта.

## Карта advanced-маршрута

### 1. ALS

- роль: мост от коллаборативной модели к latent factor моделям;
- основная мысль: поведенческий сигнал можно представить через скрытые факторы пользователя и объекта;
- материалы: [doc](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/01_matrix_factorization_als.md), [notebook](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/01_als_implicit.ipynb), [prep notebook](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/00_latent_factors_intuition.ipynb);
- датасет по умолчанию: `MovieLens latest small` в упрощённой implicit-постановке;
- обязательное сравнение: `popularity baseline` и, по возможности, коллаборативная модель.

### 02a. LightFM Hybrid Factorization

- роль: показать мост между pure collaborative factorization и feature-aware hybrid-моделями;
- основная мысль: latent factors можно обучать не только на взаимодействиях, но и с помощью item-features;
- материалы: [doc](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/02a_lightfm_hybrid_factorization.md), [notebook](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/02a_lightfm_hybrid_factorization.ipynb);
- датасет по умолчанию: `MovieLens latest small` с базовыми `genres`;
- обязательное сравнение: против `ALS` на максимально близкой offline-постановке.

### 02b. LightFM Feature Engineering

- роль: показать, что hybrid-модель можно усиливать не только сменой класса модели, но и качеством признаков;
- основная мысль: сначала держим модель фиксированной и отдельно улучшаем features;
- материалы: [doc](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/02b_lightfm_feature_engineering.md), [notebook](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/02b_lightfm_feature_engineering.ipynb);
- датасет по умолчанию: тот же, что в `ALS`;
- обязательное сравнение: `lightfm_genres` против `lightfm_engineered` на том же split.

### 02c. LightFM Hyperparameter Tuning

- роль: показать, что после feature engineering следующий шаг — настройка самой модели;
- основная мысль: сначала держим признаки фиксированными и отдельно тюним гиперпараметры;
- материалы: [doc](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/02c_lightfm_hyperparameter_tuning.md), [notebook](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/02c_lightfm_hyperparameter_tuning.ipynb);
- датасет по умолчанию: тот же, что в `ALS`;
- обязательное сравнение: лучшая tuned-конфигурация против `ALS` и baseline-версии `LightFM`.

### 02d. Neural Collaborative Filtering

- роль: показать, что neural-подход не отменяет классическую постановку, а расширяет `ALS` и `LightFM`;
- основная мысль: сначала сравниваем новый класс модели, а не меняем одновременно данные, split и candidate universe;
- материалы: [doc](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/02d_neural_collaborative_filtering.md), [notebook](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/02d_neural_collaborative_filtering.ipynb);
- датасет по умолчанию: тот же, что в `ALS`;
- обязательное сравнение: против `ALS` на той же offline-задаче.

### 3. Two-Tower Models

- роль: перейти от pair scoring к retrieval-мышлению;
- основная мысль: candidate generation и full ranking — не одно и то же;
- материалы: [doc](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/03_two_tower_models.md), [notebook](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/03_two_tower_model_intro.ipynb);
- датасет по умолчанию: `Amazon Reviews 2023 / All_Beauty`;
- обязательное ограничение: без тяжёлой distributed/ANN-инфраструктуры.

### 4. Sequential Recommenders

- роль: добавить порядок событий и next-item постановку;
- основная мысль: история пользователя — это не только множество объектов, но и последовательность;
- материалы: [doc](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/04_sequence_based_recommendations.md), [notebook](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/04_sequence_recommender_intro.ipynb);
- датасет по умолчанию: `Retailrocket`;
- обязательный мост: сначала простая sequential baseline-интуиция, потом более сильные модели.

### 5. Ranking / LTR Intro

- роль: показать, как retrieval и ranking соединяются в одну систему;
- основная мысль: хороший retrieval ещё не равен хорошему финальному ранжированию;
- материалы: [doc](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/05_ranking_and_ltr_intro.md), [notebook](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/05_retrieval_vs_ranking_toy.ipynb);
- датасет по умолчанию: toy-примеры или уже знакомые признаки из предыдущих глав;
- формат: в первую очередь документация, а не тяжёлый инженерный pipeline.

### 6. Production Overview

- роль: собрать карту реальной рекомендательной системы;
- основная мысль: онлайн-система почти всегда многостадийная и ограничена latency, cold-start, monitoring и feedback loops;
- материалы: [doc](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/06_production_overview.md);
- формат: обзорная завершающая глава.

## Правила для всех advanced-глав

- baseline comparison обязателен;
- split должен соответствовать постановке;
- ID mapping должен быть воспроизводимым;
- отсутствие взаимодействия не становится явным negative label без пояснения;
- выводы не должны быть сильнее, чем позволяют данные;
- notebook должен быть запустим сверху вниз;
- если глава в основном концептуальная, это надо написать прямо.

## Что должно быть в каждой главе

Минимальный набор для технически внятной advanced-главы:

1. Короткий мост от `core`: зачем эта глава нужна сейчас.
2. Явная постановка задачи.
3. Явный выбор датасета и объяснение, почему он подходит.
4. Baseline и условия сравнения.
5. Что именно делает notebook.
6. Ограничения и чего глава пока не решает.
7. Что станет следующим шагом в маршруте.

Для двух особенно абстрактных переходов полезны отдельные теоретические notebook'и:

- `00_latent_factors_intuition.ipynb` перед `ALS`;
- `05_retrieval_vs_ranking_toy.ipynb` рядом с `Ranking / LTR intro`.

Для `LightFM` отдельный toy-notebook не обязателен, потому что его задача — не новая абстракция, а аккуратное расширение уже понятного factorization-сценария через признаки объектов.

## Что считается готовностью главы

- есть осмысленный `docs`-раздел, а не заглушка;
- есть notebook-заготовка с понятным сценарием;
- ссылки на датасет, модуль и соседние главы согласованы;
- для кодовой главы определён минимальный smoke-path: `fit -> recommend -> no duplicate items`;
- для обзорной главы явно сказано, почему она обзорная и что сознательно не реализуется в коде.
