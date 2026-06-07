# Список литературы

Этот файл собран на основе `archive/recsys bibliography and todo.docx`, но без части с TODO и peer-review заметками.

Цель списка не в том, чтобы превратить проект в академический курс, а в том, чтобы дать:

- минимальный набор опорных источников для `core`;
- более сильные статьи для `advanced`;
- понятную привязку литературы к главам и notebook'ам проекта.

## Как читать этот список

Уровни сложности:

- `▶` - вводный источник;
- `▶▶` - средний уровень;
- `▶▶▶` - продвинутый источник.

Практическое правило:

- если вы идёте по `Core path`, не пытайтесь читать всё подряд;
- сначала достаточно `MovieLens`, `item-item CF`, метрик и одного обзорного учебника;
- к `advanced`-статьям имеет смысл переходить только после core-notebooks.

## Короткий разбор покрытия

Что покрыто хорошо:

- классические основы collaborative filtering;
- базовые датасеты проекта;
- offline-оценка top-K рекомендаций;
- matrix factorization, ALS, LightFM, Neural CF;
- retrieval, sequential и ranking/LTR как следующий уровень сложности.

Что покрыто уже, но точечно:

- cold-start;
- text-rich retrieval;
- hybrid factorization.

Что почти не покрыто отдельными источниками:

- production architecture как самостоятельная литература;
- monitoring, serving и experimentation around recommender systems;
- обзорные beginner-friendly источники именно по sequential/retrieval без сильного research уклона.

Вывод для проекта:

- для `core` список уже достаточно сильный и сбалансированный;
- для `advanced` он скорее research-oriented, чем beginner-oriented;
- для будущего расширения проекта логично отдельно добавить 1-2 обзорных systems-источника про production recsys.

## Минимум для первого прохода

Если нужен короткий маршрут чтения вместе с `Core path`, начните с этого:

1. Harper, Konstan - *The MovieLens Datasets: History and Context*.
2. Sarwar et al. - *Item-based Collaborative Filtering Recommendation Algorithms*.
3. Herlocker et al. - *Evaluating Collaborative Filtering Recommender Systems*.
4. Pazzani, Billsus - *Content-Based Recommendation Systems*.
5. Aggarwal - *Recommender Systems: The Textbook*.

Этого достаточно, чтобы понимать:

- откуда берутся interactions;
- как работает базовый collaborative filtering;
- почему метрики в recsys отличаются от обычной `accuracy`;
- как выглядит общая карта методов.

## Источники по темам

### 1. Основы collaborative filtering

1. Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., & Riedl, J. (1994). *GroupLens: An Open Architecture for Collaborative Filtering of Netnews*. Proceedings of CSCW 1994. `▶`
   Связано с: [docs/basic/05_collaborative_filtering.md](../docs/basic/05_collaborative_filtering.md), [notebooks/basic/04_item_item_cf.ipynb](../notebooks/basic/04_item_item_cf.ipynb)
   Ссылка: [ACM DOI](https://dl.acm.org/doi/10.1145/192844.192905)

2. Goldberg, D., Nichols, D., Oki, B. M., & Terry, D. (1992). *Using Collaborative Filtering to Weave an Information Tapestry*. Communications of the ACM, 35(12). `▶`
   Связано с: [docs/basic/05_collaborative_filtering.md](../docs/basic/05_collaborative_filtering.md), [notebooks/basic/04_item_item_cf.ipynb](../notebooks/basic/04_item_item_cf.ipynb)

3. Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). *Item-based Collaborative Filtering Recommendation Algorithms*. Proceedings of WWW 2001. `▶▶`
   Связано с: [docs/basic/05_collaborative_filtering.md](../docs/basic/05_collaborative_filtering.md), [notebooks/basic/04_item_item_cf.ipynb](../notebooks/basic/04_item_item_cf.ipynb)
   Ссылка: [ACM DOI](https://dl.acm.org/doi/10.1145/371920.372071)

### 2. Датасеты проекта

4. Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4). `▶`
   Связано с: [docs/basic/01_intro.md](../docs/basic/01_intro.md), [docs/basic/02_data_and_interactions.md](../docs/basic/02_data_and_interactions.md), [docs/datasets.md](../docs/datasets.md)
   Ссылка: [ACM DOI](https://dl.acm.org/doi/10.1145/2827872)

5. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval and Recommendation*. `▶▶`
   Связано с: [docs/advanced/03_two_tower_models.md](../docs/advanced/03_two_tower_models.md), [notebooks/advanced/03_two_tower_model_intro.ipynb](../notebooks/advanced/03_two_tower_model_intro.ipynb)
   Ссылка: [arXiv](https://arxiv.org/abs/2403.03952)

6. Retailrocket (2015). *Retailrocket Recommender System Dataset*. `▶`
   Связано с: [docs/advanced/04_sequence_based_recommendations.md](../docs/advanced/04_sequence_based_recommendations.md), [notebooks/advanced/04_sequence_recommender_intro.ipynb](../notebooks/advanced/04_sequence_recommender_intro.ipynb)
   Ссылка: [Kaggle dataset page](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

### 3. Метрики и evaluation

7. Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). *Evaluating Collaborative Filtering Recommender Systems*. ACM Transactions on Information Systems, 22(1). `▶▶`
   Связано с: [docs/basic/06_metrics.md](../docs/basic/06_metrics.md), [notebooks/basic/05_metrics.ipynb](../notebooks/basic/05_metrics.ipynb)
   Ссылка: [ACM DOI](https://dl.acm.org/doi/10.1145/963770.963772)

8. Cremonesi, P., Koren, Y., & Turrin, R. (2010). *Performance of Recommender Algorithms on Top-N Recommendation Tasks*. Proceedings of RecSys 2010. `▶▶`
   Связано с: [docs/basic/06_metrics.md](../docs/basic/06_metrics.md), [notebooks/basic/05_metrics.ipynb](../notebooks/basic/05_metrics.ipynb)
   Ссылка: [ACM DOI](https://dl.acm.org/doi/10.1145/1864708.1864721)

### 4. Content-based и TF-IDF

9. Pazzani, M. J., & Billsus, D. (2007). *Content-Based Recommendation Systems*. In *The Adaptive Web*. `▶`
   Связано с: [docs/basic/04_content_based.md](../docs/basic/04_content_based.md), [notebooks/basic/03_content_based_tfidf.ipynb](../notebooks/basic/03_content_based_tfidf.ipynb)

10. Salton, G., & Buckley, C. (1988). *Term-Weighting Approaches in Automatic Text Retrieval*. Information Processing & Management, 24(5). `▶`
    Связано с: [docs/basic/04_content_based.md](../docs/basic/04_content_based.md), [notebooks/basic/03_content_based_tfidf.ipynb](../notebooks/basic/03_content_based_tfidf.ipynb)

### 5. Matrix factorization и ALS

11. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer, 42(8). `▶▶`
    Связано с: [docs/advanced/01_matrix_factorization_als.md](../docs/advanced/01_matrix_factorization_als.md), [notebooks/advanced/01_als_implicit.ipynb](../notebooks/advanced/01_als_implicit.ipynb)
    Ссылка: [IEEE](https://ieeexplore.ieee.org/document/5197422)

12. Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets*. Proceedings of ICDM 2008. `▶▶`
    Связано с: [docs/advanced/01_matrix_factorization_als.md](../docs/advanced/01_matrix_factorization_als.md), [notebooks/advanced/01_als_implicit.ipynb](../notebooks/advanced/01_als_implicit.ipynb)
    Ссылка: [IEEE](https://ieeexplore.ieee.org/document/4781121)

### 6. LightFM и гибридная факторизация

13. Kula, M. (2015). *Metadata Embeddings for User and Item Cold-start Recommendations*. Proceedings of CBRecSys@RecSys 2015. `▶▶`
    Связано с: [docs/advanced/02a_lightfm_hybrid_factorization.md](../docs/advanced/02a_lightfm_hybrid_factorization.md), [docs/advanced/02b_lightfm_feature_engineering.md](../docs/advanced/02b_lightfm_feature_engineering.md), [docs/advanced/02c_lightfm_hyperparameter_tuning.md](../docs/advanced/02c_lightfm_hyperparameter_tuning.md)
    Ссылка: [arXiv](https://arxiv.org/abs/1507.08439)

### 7. Neural CF и retrieval

14. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). *Neural Collaborative Filtering*. Proceedings of WWW 2017. `▶▶▶`
    Связано с: [docs/advanced/02d_neural_collaborative_filtering.md](../docs/advanced/02d_neural_collaborative_filtering.md), [notebooks/advanced/02d_neural_collaborative_filtering.ipynb](../notebooks/advanced/02d_neural_collaborative_filtering.ipynb)
    Ссылка: [arXiv](https://arxiv.org/abs/1708.05031)

15. Yi, X., Yang, J., Hong, L., Cheng, D. Z., Heldt, L., Kumthekar, A., Zhao, Z., Wei, L., & Chi, E. (2019). *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations*. Proceedings of RecSys 2019. `▶▶▶`
    Связано с: [docs/advanced/03_two_tower_models.md](../docs/advanced/03_two_tower_models.md), [notebooks/advanced/03_two_tower_model_intro.ipynb](../notebooks/advanced/03_two_tower_model_intro.ipynb)
    Ссылка: [Google Research](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)

### 8. Sequential recommendations

16. Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2016). *Session-based Recommendations with Recurrent Neural Networks*. Proceedings of ICLR 2016. `▶▶▶`
    Связано с: [docs/advanced/04_sequence_based_recommendations.md](../docs/advanced/04_sequence_based_recommendations.md), [notebooks/advanced/04_sequence_recommender_intro.ipynb](../notebooks/advanced/04_sequence_recommender_intro.ipynb)
    Ссылка: [arXiv](https://arxiv.org/abs/1511.06939)

17. Kang, W.-C., & McAuley, J. (2018). *Self-Attentive Sequential Recommendation*. Proceedings of ICDM 2018. `▶▶▶`
    Связано с: [docs/advanced/04_sequence_based_recommendations.md](../docs/advanced/04_sequence_based_recommendations.md), [notebooks/advanced/04_sequence_recommender_intro.ipynb](../notebooks/advanced/04_sequence_recommender_intro.ipynb)
    Ссылка: [arXiv](https://arxiv.org/abs/1808.09781)

### 9. Ranking и Learning-to-Rank

18. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. Proceedings of UAI 2009. `▶▶▶`
    Связано с: [docs/advanced/05_ranking_and_ltr_intro.md](../docs/advanced/05_ranking_and_ltr_intro.md), [notebooks/advanced/05_retrieval_vs_ranking_toy.ipynb](../notebooks/advanced/05_retrieval_vs_ranking_toy.ipynb)
    Ссылка: [arXiv](https://arxiv.org/abs/1205.2618)

19. Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Hullender, G. (2005). *Learning to Rank using Gradient Descent*. Proceedings of ICML 2005. `▶▶▶`
    Связано с: [docs/advanced/05_ranking_and_ltr_intro.md](../docs/advanced/05_ranking_and_ltr_intro.md), [notebooks/advanced/05_retrieval_vs_ranking_toy.ipynb](../notebooks/advanced/05_retrieval_vs_ranking_toy.ipynb)
    Ссылка: [ACM DOI](https://dl.acm.org/doi/10.1145/1102351.1102363)

### 10. Учебники и обзорные книги

20. Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. `▶▶`
    Ссылка: [Springer](https://link.springer.com/book/10.1007/978-3-319-29659-3)

21. Ricci, F., Rokach, L., & Shapira, B. (Eds.) (2022). *Recommender Systems Handbook (3rd ed.)*. `▶▶`
    Ссылка: [Springer](https://link.springer.com/book/10.1007/978-1-0716-2197-4)

22. Jannach, D., Zanker, M., Felfernig, A., & Friedrich, G. (2010). *Recommender Systems: An Introduction*. `▶`

### 11. Cold-start

23. Schein, A. I., Popescul, A., Ungar, L. H., & Pennock, D. M. (2002). *Methods and Metrics for Cold-Start Recommendations*. Proceedings of SIGIR 2002. `▶▶`
    Связано с: [docs/basic/09_cold_start.md](../docs/basic/09_cold_start.md)
    Ссылка: [ACM DOI](https://dl.acm.org/doi/10.1145/564376.564421)

## Что можно добавить позже

Если расширять список в следующей итерации, полезнее всего добавить:

1. 1-2 системных источника про production recommender architecture;
2. 1 обзорный источник по retrieval/ranking pipeline без сильного LTR-формализма;
3. 1 beginner-friendly обзор по sequential recommenders;
4. 1 источник по diversification / novelty / coverage beyond accuracy.
