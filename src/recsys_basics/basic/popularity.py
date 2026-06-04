"""Popularity baseline для учебного курса."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def filter_positive_explicit_feedback(
    interactions: pd.DataFrame,
    min_rating: float = 4.0,
) -> pd.DataFrame:
    """Оставляет только положительные explicit interactions."""

    required_columns = {"user_id", "item_id", "rating", "timestamp"}
    missing_columns = required_columns.difference(interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Не хватает колонок для фильтрации positive feedback: {missing}")

    positive = interactions[interactions["rating"] >= min_rating].copy()
    positive["event"] = "positive_rating"
    positive = positive.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    return positive


def build_seen_items_map(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> dict[Any, set[Any]]:
    """Строит словарь просмотренных/оценённых объектов для каждого пользователя."""

    required_columns = {user_col, item_col}
    missing_columns = required_columns.difference(interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Не хватает колонок для seen-items map: {missing}")

    seen_items = interactions.groupby(user_col)[item_col].agg(lambda values: set(values.tolist()))
    return dict(seen_items.items())


@dataclass
class PopularityRecommender:
    """Рекомендует самые популярные объекты из train."""

    user_col: str = "user_id"
    item_col: str = "item_id"
    popularity_col: str = "popularity_score"
    popularity_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    ranked_items_: list[Any] = field(init=False, default_factory=list)
    is_fitted_: bool = field(init=False, default=False)

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        required_columns = {self.user_col, self.item_col}
        missing_columns = required_columns.difference(interactions.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Не хватает колонок для fit popularity model: {missing}")

        popularity = (
            interactions.groupby(self.item_col)
            .size()
            .rename(self.popularity_col)
            .reset_index()
            .sort_values([self.popularity_col, self.item_col], ascending=[False, True])
            .reset_index(drop=True)
        )

        self.popularity_ = popularity
        self.ranked_items_ = popularity[self.item_col].tolist()
        self.is_fitted_ = True
        return self

    def recommend(
        self,
        user_id: int | None = None,
        seen_items: set[Any] | None = None,
        k: int = 10,
    ) -> list[Any]:
        if not self.is_fitted_:
            raise ValueError("Сначала вызовите fit()")

        if k <= 0:
            return []

        seen_items = seen_items or set()
        recommendations: list[Any] = []
        for item_id in self.ranked_items_:
            if item_id in seen_items:
                continue
            recommendations.append(item_id)
            if len(recommendations) == k:
                break
        return recommendations

    def recommend_many(
        self,
        user_ids: list[Any],
        seen_items_map: dict[Any, set[Any]] | None = None,
        k: int = 10,
    ) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("Сначала вызовите fit()")

        seen_items_map = seen_items_map or {}
        rows: list[dict[str, int]] = []
        for user_id in user_ids:
            user_recommendations = self.recommend(
                user_id=user_id,
                seen_items=seen_items_map.get(user_id, set()),
                k=k,
            )
            rows.extend(
                {
                    self.user_col: user_id,
                    self.item_col: item_id,
                    "rank": rank,
                }
                for rank, item_id in enumerate(user_recommendations, start=1)
            )

        return pd.DataFrame(rows)

    def get_popularity_table(self, top_n: int | None = None) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("Сначала вызовите fit()")

        if top_n is None:
            return self.popularity_.copy()
        return self.popularity_.head(top_n).copy()


__all__ = [
    "PopularityRecommender",
    "build_seen_items_map",
    "filter_positive_explicit_feedback",
]
