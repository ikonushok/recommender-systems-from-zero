# Advanced Notebooks

Эта папка содержит практические notebook'и для `advanced`-маршрута.

Правила для них:

- один notebook = одна основная учебная идея;
- notebook не должен одновременно объяснять новую сложную модель и новый сложный датасет без необходимости;
- первые пять advanced-глав держат максимально похожую offline-постановку, чтобы сравнение моделей, роли признаков и тюнинга было честным;
- каждый notebook должен быть воспроизводимым сверху вниз.

Текущий план:

- [00_latent_factors_intuition.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/00_latent_factors_intuition.ipynb): подготовительный notebook для раздела [01_matrix_factorization_als.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/01_matrix_factorization_als.md); toy-мост от коллаборативной модели к latent factors;
- [01_als_implicit.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/01_als_implicit.ipynb): раздел [01_matrix_factorization_als.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/01_matrix_factorization_als.md); latent factors и `ALS` в упрощённой implicit-постановке;
- [02a_lightfm_hybrid_factorization.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/02a_lightfm_hybrid_factorization.ipynb): раздел [02a_lightfm_hybrid_factorization.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/02a_lightfm_hybrid_factorization.md); bridge-шаг между `ALS` и hybrid factorization, где к factorization добавляются item-features;
- [02b_lightfm_feature_engineering.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/02b_lightfm_feature_engineering.ipynb): раздел [02b_lightfm_feature_engineering.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/02b_lightfm_feature_engineering.md); продолжение `LightFM` через усиление item-features;
- [02c_lightfm_hyperparameter_tuning.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/02c_lightfm_hyperparameter_tuning.ipynb): раздел [02c_lightfm_hyperparameter_tuning.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/02c_lightfm_hyperparameter_tuning.md); компактный tuning `LightFM` на тех же engineered-features;
- [02d_neural_collaborative_filtering.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/02d_neural_collaborative_filtering.ipynb): раздел [02d_neural_collaborative_filtering.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/02d_neural_collaborative_filtering.md); `Neural CF` на максимально похожей задаче;
- [03_two_tower_model_intro.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/03_two_tower_model_intro.ipynb): раздел [03_two_tower_models.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/03_two_tower_models.md); retrieval-интуиция и two-tower подход;
- [04_sequence_recommender_intro.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/04_sequence_recommender_intro.ipynb): раздел [04_sequence_based_recommendations.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/04_sequence_based_recommendations.md); next-item / sequential постановка;
- [05_retrieval_vs_ranking_toy.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/05_retrieval_vs_ranking_toy.ipynb): раздел [05_ranking_and_ltr_intro.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/05_ranking_and_ltr_intro.md); toy-мост между retrieval и `LTR`.

Главы `Ranking / LTR intro` и `Production overview` на текущем этапе остаются в основном документационными, но для `LTR` добавляется отдельный toy-notebook как поддерживающий учебный мост.
