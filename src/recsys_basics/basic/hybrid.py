"""Простой hybrid recommender для учебного курса."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[int]],
    weights: dict[str, float] | None = None,
    k_constant: int = 60,
) -> pd.DataFrame:
    """Объединяет несколько ранжированных списков через weighted RRF."""

    if not ranked_lists:
        return pd.DataFrame(columns=["item_id", "hybrid_score"])

    weights = weights or {name: 1.0 for name in ranked_lists}
    scores: dict[int, float] = {}

    for model_name, item_ids in ranked_lists.items():
        weight = float(weights.get(model_name, 1.0))
        for rank, item_id in enumerate(item_ids, start=1):
            scores[int(item_id)] = scores.get(int(item_id), 0.0) + weight / (k_constant + rank)

    fused = (
        pd.DataFrame(
            {
                "item_id": list(scores.keys()),
                "hybrid_score": list(scores.values()),
            }
        )
        .sort_values(["hybrid_score", "item_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return fused


@dataclass
class HybridRankFusionRecommender:
    """Объединяет несколько списков рекомендаций на уровне рангов."""

    weights: dict[str, float] | None = None
    k_constant: int = 60

    def recommend(
        self,
        ranked_lists: dict[str, list[int]],
        top_k: int = 10,
    ) -> list[int]:
        if top_k <= 0:
            return []

        fused = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            weights=self.weights,
            k_constant=self.k_constant,
        )
        return fused["item_id"].head(top_k).astype(int).tolist()

    def recommend_many(
        self,
        user_ranked_lists: dict[int, dict[str, list[int]]],
        top_k: int = 10,
    ) -> pd.DataFrame:
        rows: list[dict[str, int | float]] = []
        for user_id, ranked_lists in user_ranked_lists.items():
            fused = reciprocal_rank_fusion(
                ranked_lists=ranked_lists,
                weights=self.weights,
                k_constant=self.k_constant,
            ).head(top_k)
            rows.extend(
                {
                    "user_id": int(user_id),
                    "item_id": int(row.item_id),
                    "rank": rank,
                    "hybrid_score": float(row.hybrid_score),
                }
                for rank, row in enumerate(fused.itertuples(index=False), start=1)
            )

        return pd.DataFrame(rows)


def weighted_interleave(
    ranked_lists: dict[str, list[int]],
    weights: dict[str, float] | None = None,
    top_k: int = 10,
) -> list[int]:
    """Объединяет списки через weighted interleaving с удалением дубликатов.

    Более тяжёлые модели получают больший quota и обрабатываются раньше
    в каждом раунде, чтобы слабый источник сигнала не занимал верх списка
    только из-за порядка обхода словаря.
    """

    if top_k <= 0 or not ranked_lists:
        return []

    weights = weights or {name: 1.0 for name in ranked_lists}
    positive_weights = {name: max(float(weights.get(name, 1.0)), 0.0) for name in ranked_lists}
    total_weight = sum(positive_weights.values())
    if total_weight == 0:
        positive_weights = {name: 1.0 for name in ranked_lists}
        total_weight = float(len(positive_weights))

    quotas = {
        name: max(1, math.ceil(top_k * positive_weights[name] / total_weight))
        for name in ranked_lists
    }
    model_order = sorted(
        ranked_lists,
        key=lambda name: (-positive_weights[name], name),
    )
    pointers = {name: 0 for name in ranked_lists}
    recommendations: list[int] = []
    selected_items: set[int] = set()

    while len(recommendations) < top_k:
        progress = False
        for model_name in model_order:
            item_ids = ranked_lists[model_name]
            quota = quotas[model_name]
            taken = 0
            while taken < quota and pointers[model_name] < len(item_ids) and len(recommendations) < top_k:
                item_id = int(item_ids[pointers[model_name]])
                pointers[model_name] += 1
                if item_id in selected_items:
                    continue
                recommendations.append(item_id)
                selected_items.add(item_id)
                taken += 1
                progress = True
        if not progress:
            break

    return recommendations[:top_k]


@dataclass
class HybridInterleavingRecommender:
    """Объединяет несколько списков рекомендаций через weighted interleaving."""

    weights: dict[str, float] | None = None

    def recommend(
        self,
        ranked_lists: dict[str, list[int]],
        top_k: int = 10,
    ) -> list[int]:
        return weighted_interleave(ranked_lists=ranked_lists, weights=self.weights, top_k=top_k)

    def recommend_many(
        self,
        user_ranked_lists: dict[int, dict[str, list[int]]],
        top_k: int = 10,
    ) -> pd.DataFrame:
        rows: list[dict[str, int]] = []
        for user_id, ranked_lists in user_ranked_lists.items():
            recommendations = weighted_interleave(
                ranked_lists=ranked_lists,
                weights=self.weights,
                top_k=top_k,
            )
            rows.extend(
                {
                    "user_id": int(user_id),
                    "item_id": int(item_id),
                    "rank": rank,
                }
                for rank, item_id in enumerate(recommendations, start=1)
            )
        return pd.DataFrame(rows)


__all__ = [
    "HybridInterleavingRecommender",
    "HybridRankFusionRecommender",
    "reciprocal_rank_fusion",
    "weighted_interleave",
]
