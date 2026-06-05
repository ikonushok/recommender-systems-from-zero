# Production Overview

## Что изучим

- как отдельные модели из курса складываются в реальную рекомендательную систему;
- почему в production важны latency, freshness, monitoring и fallback;
- где в системе живут retrieval, ranking, re-ranking и бизнес-правила;
- почему offline-ноутбук и online-сервис — это разные уровни задачи;
- почему хорошая метрика в ноутбуке ещё не равна продуктовому эффекту.

## Роль главы

Это не глава про `переписать проект в production`.

Это глава про честную карту мира:

- что из уже пройденного действительно используется в реальной системе;
- какие дополнительные инженерные слои появляются вокруг модели;
- где заканчивается учебный notebook и начинается online-инфраструктура.

Эта глава завершает курс. Её задача не открыть ещё одну модель, а собрать всё пройденное в одну систему.

## От одной модели к системе

В notebook чаще всего кажется, что есть одна таблица interactions, одна модель и один `top_k`.

В production обычно так не бывает.

Чаще система выглядит так:

```text
user/request
    ->
candidate generation (retrieval)
    ->
light filtering / business constraints
    ->
final ranking
    ->
re-ranking / diversification / rules
    ->
response to product surface
    ->
logging, feedback, monitoring
```

Главная мысль:

- retrieval отвечает за скорость и ширину candidate universe;
- ranking отвечает за точный порядок внутри уже ограниченного shortlist;
- post-processing отвечает за продуктовые ограничения, разнообразие и safety;
- вокруг всего этого живут данные, логи, обновления модели и мониторинг.

## Где здесь живут главы курса

Ниже не строгая классификация, а рабочая карта.

### Core-часть

- `popularity baseline` часто живёт как fallback;
- `content-based` полезен для cold-start и metadata-aware сценариев;
- коллаборативные модели полезны там, где уже накопился поведенческий сигнал;
- метрики, split и leakage-проверки остаются важными и в advanced, и в production.

### Advanced-часть

- `ALS`, `LightFM`, `NeuMF` можно мыслить как scoring-модели для ranking или candidate scoring;
- `two-tower` естественно ложится в retrieval-слой;
- sequence-aware модели могут жить либо как отдельный candidate generator, либо как часть ranking-features;
- `LTR` связывает retrieval и финальный ranking в многостадийный pipeline.

То есть модели из курса не конкурируют за одно и то же место в системе. Часто они отвечают за разные стадии.

## Минимальная production-карта

### 1. Candidate Generation

Это первый слой, который быстро сужает огромный каталог до manageable shortlist.

Типичные источники кандидатов:

- popularity и trending;
- collaborative retrieval;
- `two-tower` embedding retrieval;
- content-based retrieval;
- sequence-aware кандидаты;
- editorial или business-selected наборы.

На этом этапе важны:

- latency;
- recall по кандидатам;
- coverage;
- устойчивость к cold-start.

Плохой retrieval опасен тем, что ranking уже не увидит многие релевантные объекты.

### 2. Filtering и Rules

Ещё до финального ranking система часто убирает заведомо неподходящие объекты.

Примеры:

- нет в наличии;
- недоступен в регионе;
- запрещён по возрасту;
- уже куплен недавно, если повтор не нужен;
- не проходит policy или safety rule.

Это не “грязный хак”, а нормальная часть production-логики.

### 3. Final Ranking

Здесь система берёт shortlist и пытается выстроить лучший порядок.

Именно сюда естественно ложатся:

- richer user-features;
- item-features;
- context-features;
- retrieval scores;
- sequence signals;
- business priors.

Именно здесь особенно полезны `pointwise / pairwise / listwise` постановки из `LTR`.

### 4. Re-ranking

После ranking часто появляется ещё один слой.

Его цель не “улучшить модель математически”, а соблюсти продуктовые ограничения.

Примеры:

- diversification;
- novelty injection;
- fairness constraints;
- cap на одинаковые бренды или категории;
- смешивание sponsored и organic объектов;
- exploration.

Это важно, потому что лучший offline-score не всегда даёт лучший пользовательский опыт.

## Online-ограничения, которых нет в notebook

### Latency

В notebook можно считать рекомендации секунды или минуты. В продукте у запроса обычно есть жёсткий бюджет ответа.

Поэтому production почти всегда многостадийный:

- грубый и быстрый retrieval;
- потом более дорогой ranking на малом числе кандидатов.

### Freshness

В notebook данные обычно статичны на время эксперимента.

В продукте каталог и поведение меняются постоянно:

- появляются новые объекты;
- меняются цены и availability;
- пользователь только что кликнул или купил что-то;
- популярность и сезонность сдвигаются.

Поэтому в системе почти всегда есть вопрос:

- что обновляется offline батчами;
- что обновляется чаще;
- что вычисляется прямо во время запроса.

### Cold-Start

В core мы уже обсуждали `cold-start`, но в production он особенно заметен.

Проблемы:

- новый пользователь приходит без истории;
- новый объект ещё не попал в поведенческий сигнал;
- новая витрина или surface вообще имеет другой контекст выдачи.

Поэтому fallback и metadata не “дополнение”, а обязательный operational слой.

## Offline и Online — не одно и то же

Хороший notebook нужен, но он не равен production-качеству.

Почему:

- offline candidate universe может отличаться от реального;
- в online есть задержки логирования и обновления фичей;
- часть сигналов в продукте доступна только в request-time;
- интерфейс, позиция блока и UX влияют на итог не меньше модели;
- feedback loop меняет сами данные, на которых будет учиться следующая версия модели.

Поэтому правильная цепочка обычно такая:

1. Сначала корректный offline-эксперимент без leakage.
2. Потом sanity-check на serving-ограничениях.
3. Потом online-эксперимент или хотя бы guarded rollout.

## Почему offline-метрики не гарантируют продуктовый эффект

Даже если `NDCG@K` или `MAP@K` выросли, это ещё не обещает:

- рост CTR;
- рост conversion;
- рост retention;
- рост revenue;
- рост long-term satisfaction.

Причины простые:

- метрика может не совпадать с product objective;
- модель может переусиливать head-объекты;
- интерфейс может скрывать выигрыш;
- система может стать менее разнообразной;
- улучшение на “тёплых” пользователях может ухудшить cold-start experience.

То есть offline-метрики нужны, но сами по себе они не завершают работу.

## Feedback Loop

Production-система не просто рекомендует. Она ещё и создаёт будущие данные.

Если система часто показывает одни и те же объекты:

- именно по ним накапливается больше новых взаимодействий;
- head усиливается;
- хвост каталога получает меньше шансов;
- модель может всё сильнее закреплять собственные старые решения.

Это один из главных скрытых рисков recommender systems.

Именно поэтому важны:

- coverage;
- diversification;
- controlled exploration;
- separate monitoring по новым и старым объектам.

## Monitoring: что обычно смотрят

Минимально полезный production-набор мыслится в трёх группах.

### 1. Техническое здоровье

- latency по стадиям;
- availability;
- error rate;
- пустые выдачи;
- время загрузки артефактов.

### 2. Data quality

- доля missing-features;
- доля unknown users/items;
- сдвиг распределений фичей;
- размер candidate pool;
- доля fallback-ответов.

### 3. Quality proxies

- CTR;
- add-to-cart / save / purchase rate;
- long-click или dwell proxies;
- coverage;
- novelty/diversity;
- сегментные метрики по new users / new items / active users.

Без этих трёх групп monitoring легко оказаться в ситуации, где модель формально “запущена”, но никто не понимает, полезна ли она и не деградировала ли система.

## Что почти всегда должно быть в системе кроме модели

Даже самая хорошая модель не отменяет:

- fallback;
- feature pipelines;
- candidate source orchestration;
- кэширование;
- id mapping и consistency между train и serving;
- логирование показов, кликов и downstream events;
- мониторинг и rollback path.

Это важный финальный вывод курса: production recsys — это не только model training, а вся система вокруг принятия решения.

## Что сознательно не делаем в этом проекте

В рамках этого учебного проекта мы не будем:

- поднимать полноценный online inference service;
- строить feature store;
- реализовывать ANN-индексы и distributed retrieval;
- симулировать real-time event bus;
- делать полноценный A/B experimentation platform.

Это было бы архитектурно тяжело и учебно вредно для текущей цели проекта.

Задача курса другая:

- научиться корректно формулировать задачу;
- понимать, где какая модель уместна;
- не путать scoring, retrieval, ranking и production-слои;
- видеть ограничения своих offline-результатов.

## Что должно остаться после этой главы

После `Production Overview` должно быть понятно:

- почему реальная рекомендательная система почти всегда многостадийная;
- где в ней живут retrieval, ranking, re-ranking и fallback;
- как главы курса отображаются на разные части этой системы;
- почему latency, freshness, monitoring и cold-start нельзя “добавить потом”;
- почему сильный notebook ещё не означает production-ready решение.

## Что дальше

На этом advanced-маршрут завершается.

Логичное продолжение после курса уже зависит от цели:

- углубляться в `LTR` и multi-stage ranking;
- идти в retrieval infrastructure и ANN;
- изучать sequence-модели глубже;
- разбирать online experiments и recommender monitoring;
- строить минимальный service layer поверх одной из уже изученных моделей.
