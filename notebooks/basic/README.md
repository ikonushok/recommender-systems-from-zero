# Core Notebooks

Эта папка содержит практические notebook'и для `core`-маршрута.

Правила для них:

- один notebook = одна основная учебная идея;
- сначала разбираем данные и baseline, потом персонализацию;
- notebook должен быть воспроизводимым сверху вниз;
- каждый следующий notebook должен опираться на уже понятную постановку, а не вводить сразу несколько новых сложностей.

Текущий план:

- [01_intro_dataset.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/basic/01_intro_dataset.ipynb): разделы [01_intro.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/basic/01_intro.md) и [02_data_and_interactions.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/basic/02_data_and_interactions.md); знакомство с `MovieLens latest small`, каноническая `interaction table`, базовая диагностика данных;
- [02_popularity_baseline.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/basic/02_popularity_baseline.ipynb): раздел [03_popularity_baseline.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/basic/03_popularity_baseline.md); первая baseline-модель и честная точка отсчёта;
- [03_content_based_tfidf.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/basic/03_content_based_tfidf.ipynb): раздел [04_content_based.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/basic/04_content_based.md); первый персонализированный подход через признаки объектов;
- [04_item_item_cf.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/basic/04_item_item_cf.ipynb): раздел [05_collaborative_filtering.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/basic/05_collaborative_filtering.md); коллаборативная модель и поведенческий сигнал;
- [05_metrics.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/basic/05_metrics.ipynb): раздел [06_metrics.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/basic/06_metrics.md); top-K метрики и их интерпретация;
- [06_hybrid_intro.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/basic/06_hybrid_intro.ipynb): раздел [07_hybrid_recommendations.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/basic/07_hybrid_recommendations.md); базовая логика hybrid-рекомендаций.

Главы `Intro`, `Common mistakes` и `Cold-start` в `core` остаются в основном документационными:

- [docs/basic/01_intro.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/basic/01_intro.md)
- [docs/basic/08_common_mistakes.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/basic/08_common_mistakes.md)
- [docs/basic/09_cold_start.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/basic/09_cold_start.md)

Это сделано специально, чтобы не раздувать `core` лишними notebook'ами там, где важнее аккуратное объяснение, чем дополнительный код.
