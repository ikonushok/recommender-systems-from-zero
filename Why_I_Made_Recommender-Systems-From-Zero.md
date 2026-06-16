# Почему я сделал Recommender Systems From Zero

Когда я начал глубже разбираться с рекомендательными системами, я заметил один неприятный разрыв.

С одной стороны, есть сильные академические материалы: обзоры, книги, статьи, формулы, классификации подходов.

С другой — много коротких notebook’ов в стиле:

> загрузил датасет → вызвал `.fit()` → получил рекомендации → конец.

И то, и другое полезно.

Но если ты software developer или ML engineer, который уже умеет писать код, но впервые системно входит в recommender systems, этого часто недостаточно.

Проблема не только в том, что модели бывают сложными.

Проблема в том, что в RecSys очень легко пропустить базовые инженерные вещи:

- данные;
- interaction table;
- train/test split;
- leakage;
- baseline;
- top-K метрики;
- cold-start;
- candidate universe;
- честное сравнение моделей.

А без этого даже архитектурно безупречная модель может давать красивые, но неправильные результаты.

Именно поэтому я сделал **Recommender Systems From Zero**.

Это не попытка показать, как сразу построить production recommender service.

Это структурированный практикум для разработчиков и ML-инженеров, которым нужен понятный вход в recommender systems: от данных, baseline-моделей и метрик до hybrid-подходов и введения в neural recommenders.

Я хотел, чтобы проект отвечал не только на вопрос:

> Как создать модель рекомендательной системы?

А на более важный вопрос:

> Как понять, что эта модель действительно работает и сравнивается честно?

В проекте я сознательно иду от простого к сложному:

- popularity baseline;
- content-based recommendations;
- collaborative filtering;
- evaluation через Precision@K / Recall@K / MAP@K / NDCG@K;
- hybrid recommendations;
- типичные ошибки;
- cold-start;
- ALS;
- LightFM;
- Neural CF;
- Two-Tower;
- sequential recommenders.

Это не учебник, а методическое пособие, которое облегчит вход в RecSys для ML-инженеров.

Тут последовательно показываются:
- baseline-first;
- leakage-aware evaluation;
- reproducible notebooks;
- аккуратная работа с данными;
- постепенное усложнение моделей.

Ссылка на проект:  
https://github.com/ikonushok/recommender-systems-from-zero

#recommendersystems #machinelearning #mlengineering #datascience #python #portfolio