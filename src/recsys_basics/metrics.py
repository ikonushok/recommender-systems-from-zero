"""Учебные top-K метрики для рекомендательных систем."""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def precision_at_k(relevant_items: Iterable[int], recommended_items: Iterable[int], k: int) -> float:
    """Precision@K: доля релевантных объектов среди первых K рекомендаций."""

    if k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    recommended_top_k = list(recommended_items)[:k]
    if not recommended_top_k:
        return 0.0

    hits = sum(item in relevant_set for item in recommended_top_k)
    return hits / k


def recall_at_k(relevant_items: Iterable[int], recommended_items: Iterable[int], k: int) -> float:
    """Recall@K: доля релевантных объектов, найденных в первых K рекомендациях."""

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    recommended_top_k = list(recommended_items)[:k]
    hits = sum(item in relevant_set for item in recommended_top_k)
    return hits / len(relevant_set)


def hit_rate_at_k(relevant_items: Iterable[int], recommended_items: Iterable[int], k: int) -> float:
    """HitRate@K: был ли хотя бы один релевантный объект в первых K рекомендациях."""

    relevant_set = set(relevant_items)
    recommended_top_k = list(recommended_items)[:k]
    if not relevant_set or not recommended_top_k:
        return 0.0
    return float(any(item in relevant_set for item in recommended_top_k))


def average_precision_at_k(
    relevant_items: Iterable[int],
    recommended_items: Iterable[int],
    k: int,
) -> float:
    """Average Precision@K для одного пользователя."""

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    recommended_top_k = list(recommended_items)[:k]
    precision_sum = 0.0
    hits = 0
    for index, item in enumerate(recommended_top_k, start=1):
        if item in relevant_set:
            hits += 1
            precision_sum += hits / index

    denominator = min(len(relevant_set), k)
    if denominator == 0:
        return 0.0
    return precision_sum / denominator


def dcg_at_k(relevant_items: Iterable[int], recommended_items: Iterable[int], k: int) -> float:
    """Discounted Cumulative Gain@K для бинарной релевантности."""

    relevant_set = set(relevant_items)
    recommended_top_k = list(recommended_items)[:k]
    dcg = 0.0
    for index, item in enumerate(recommended_top_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(index + 1)
    return dcg


def ndcg_at_k(relevant_items: Iterable[int], recommended_items: Iterable[int], k: int) -> float:
    """Normalized DCG@K для бинарной релевантности."""

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    actual_dcg = dcg_at_k(relevant_set, recommended_items, k)
    ideal_recommendations = list(relevant_set)[: min(len(relevant_set), k)]
    ideal_dcg = dcg_at_k(relevant_set, ideal_recommendations, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def build_user_recommendation_lists(recommendations: pd.DataFrame) -> dict[int, list[int]]:
    """Собирает рекомендации в словарь `user_id -> ordered item_id list`."""

    required_columns = {"user_id", "item_id", "rank"}
    missing_columns = required_columns.difference(recommendations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Не хватает колонок для recommendation lists: {missing}")

    ordered = recommendations.sort_values(["user_id", "rank", "item_id"])
    grouped = ordered.groupby("user_id")["item_id"].agg(list)
    return {int(user_id): [int(item_id) for item_id in item_ids] for user_id, item_ids in grouped.items()}


def build_user_relevant_items(test_interactions: pd.DataFrame) -> dict[int, list[int]]:
    """Собирает релевантные test-объекты в словарь `user_id -> relevant item_id list`."""

    required_columns = {"user_id", "item_id"}
    missing_columns = required_columns.difference(test_interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Не хватает колонок для relevant items: {missing}")

    grouped = test_interactions.groupby("user_id")["item_id"].agg(list)
    return {int(user_id): [int(item_id) for item_id in item_ids] for user_id, item_ids in grouped.items()}


def evaluate_ranking_metrics(
    test_interactions: pd.DataFrame,
    recommendations: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Считает средние top-K метрики по пользователям."""

    user_relevant = build_user_relevant_items(test_interactions)
    user_recommendations = build_user_recommendation_lists(recommendations)

    rows: list[dict[str, float | int]] = []
    for user_id, relevant_items in user_relevant.items():
        recommended_items = user_recommendations.get(int(user_id), [])
        rows.append(
            {
                "user_id": int(user_id),
                f"precision@{k}": precision_at_k(relevant_items, recommended_items, k),
                f"recall@{k}": recall_at_k(relevant_items, recommended_items, k),
                f"hit_rate@{k}": hit_rate_at_k(relevant_items, recommended_items, k),
                f"map@{k}": average_precision_at_k(relevant_items, recommended_items, k),
                f"ndcg@{k}": ndcg_at_k(relevant_items, recommended_items, k),
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "average_precision_at_k",
    "build_user_recommendation_lists",
    "build_user_relevant_items",
    "dcg_at_k",
    "evaluate_ranking_metrics",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
