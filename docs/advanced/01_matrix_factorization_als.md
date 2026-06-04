# Matrix Factorization и ALS

## Что изучим

- зачем после `item-item collaborative filtering` переходить к latent factors;
- как `ALS` связан с matrix factorization;
- почему в advanced-части мы начинаем не с новой инфраструктуры, а с новой идеи модели;
- как аккуратно ввести `implicit feedback` на знакомом датасете;
- какие ограничения у этого шага остаются даже после роста качества.

## Мост от core

После `core` уже понятно:

- зачем нужен baseline;
- как работает `item-item CF`;
- почему split и leakage важнее красивых цифр;
- почему cold-start никуда не исчезает сам по себе.

Следующий естественный шаг:

`перейти от похожести объектов к скрытым представлениям пользователей и объектов`.

Именно это и делает глава про `ALS`.

Перед основным notebook здесь полезен короткий теоретический мост:

- [notebooks/advanced/00_latent_factors_intuition.ipynb](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/notebooks/advanced/00_latent_factors_intuition.ipynb).

Его задача:

- на toy-матрице показать интуицию latent factors;
- отделить идею скрытых представлений от деталей конкретной реализации `ALS`.

## Постановка главы

Это первая advanced-глава, поэтому здесь важно не менять слишком много вещей сразу.

Выбранный режим такой:

- берём знакомый `MovieLens latest small`;
- строим упрощённый `implicit` сигнал из положительных рейтингов;
- сохраняем понятную offline-логику;
- сравниваем новую модель с уже известными baseline-подходами.

Цель главы не в том, чтобы показать “production ALS”, а в том, чтобы объяснить:

- что такое latent factors;
- почему factorization может ловить структуру предпочтений лучше простого neighbourhood-подхода;
- как честно сравнивать его с уже пройденными моделями.

## Рекомендуемый датасет

По умолчанию:

- `MovieLens latest small`;
- положительный сигнал: явный порог по `rating`, который нужно объяснить прямо в notebook;
- split: `train / validation / test`, где у каждого пользователя последнее positive interaction уходит в `test`, предпоследнее в `validation`, а более ранние остаются в `train`.

Почему не `Retailrocket` сразу:

- иначе новичок одновременно получает новую модель, новый тип данных и новый домен;
- для первой advanced-главы это лишняя когнитивная нагрузка.

`Retailrocket` можно упоминать как следующий естественный implicit-сценарий, но не делать его обязательным первым шагом.

## Что должно быть в notebook

- короткая связь с toy-intuition notebook;
- подготовка бинарного `implicit` сигнала из `MovieLens`;
- объяснение, чем этот сигнал отличается от исходных `explicit ratings`;
- обучение модели `ALS` из библиотеки `implicit`;
- генерация top-K рекомендаций;
- сравнение с `popularity baseline` и, по возможности, с `item-item CF`;
- короткий sanity-check по рекомендациям и отсутствию дублей.

## Что должно быть в кодовом каркасе

- тонкая обёртка над реальной библиотекой `implicit`, а не самодельная замена;
- минимальный интерфейс `fit` / `recommend`;
- аккуратная работа с ID mapping;
- понятная остановка notebook, если `implicit` не установлена в окружение.

## Основные риски

- неявно перепутать `explicit` и `implicit` постановки;
- сломать честность сравнения из-за другого candidate universe;
- сделать вывод “ALS лучше”, когда на самом деле изменился не только класс модели;
- скрыть за новой моделью старые проблемы со split и seen-item filtering.

## Что считается готовностью главы

- документация объясняет, зачем нужен latent-factor шаг;
- notebook делает осмысленный run-through, а не просто импортирует библиотеку;
- baseline comparison зафиксирован;
- ограничения метода и cold-start описаны явно;
- есть понятный переход к следующей главе: `LightFM Hybrid Factorization`.

## Что дальше

Следующая глава: [docs/advanced/02a_lightfm_hybrid_factorization.md](/Users/bobrsubr/PycharmProjects/_researches/recommender-systems-from-zero/docs/advanced/02a_lightfm_hybrid_factorization.md).
