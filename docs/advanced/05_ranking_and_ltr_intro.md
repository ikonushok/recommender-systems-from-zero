# Ranking и LTR Intro

## Что изучим

- почему retrieval и final ranking — это разные стадии;
- чем `pointwise`, `pairwise` и `listwise` постановки отличаются друг от друга;
- какие признаки обычно приходят в ranking-слой;
- почему leakage в ranking-features особенно опасен.

## Роль главы в маршруте

К этому моменту уже пройдены:

- latent factors;
- neural scoring;
- retrieval;
- sequence-aware логика.

Теперь нужно показать, что в реальной системе это часто не “одна волшебная модель”, а несколько стадий подряд.

Глава про `LTR` нужна как интеллектуальный мост:

- от отдельных моделей;
- к многостадийному recommender pipeline.

## Формат главы

Это в первую очередь концептуальная глава.

Для текущего учебного проекта не требуется:

- строить industrial `LTR` пайплайн;
- поднимать feature store;
- реализовывать сложный online serving.

Нужно другое:

- объяснить саму идею ranking-слоя;
- показать, какие сигналы туда могут входить;
- объяснить, как retrieval и ranking связаны между собой.

Чтобы эта глава не оставалась слишком абстрактной, рядом с ней нужен отдельный toy-notebook:

- [notebooks/advanced/05_retrieval_vs_ranking_toy.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/05_retrieval_vs_ranking_toy.ipynb).

Его задача:

- показать двухстадийный пайплайн на маленьком примере;
- отделить candidate generation от финального ranking;
- сделать признаки и reranking-интуицию визуально понятными.

## Что должно быть в главе

- короткая схема многостадийного пайплайна;
- объяснение `pointwise`, `pairwise`, `listwise`;
- примеры ranking-features: user features, item features, context features, retrieval scores;
- предупреждение про leakage и offline/online mismatch;
- честное описание границы проекта: без тяжёлой инженерной реализации.

## Основные риски

- превратить главу в абстрактную теорию без связи с предыдущими шагами;
- или наоборот, попытаться реализовать слишком большой production-stack;
- не объяснить, почему хорошие кандидаты ещё не гарантируют хороший финальный список.

## Что считается готовностью главы

- читателю понятно, зачем нужен ranking поверх retrieval;
- разница между `pointwise`, `pairwise`, `listwise` сформулирована простым языком;
- границы учебного проекта обозначены прямо;
- глава естественно ведёт к `production overview`.

## Что дальше

Следующая глава: [docs/advanced/06_production_overview.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/06_production_overview.md).
