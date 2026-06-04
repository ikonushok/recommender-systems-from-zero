"""Простые sequential baseline-модели для advanced-части курса."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class LastItemTransitionRecommender:
    """Рекомендует объекты по переходам `previous_item -> next_item` из train."""

    user_col: str = "user_id"
    item_col: str = "item_id"
    time_col: str = "timestamp"
    transition_counts_: dict[Any, Counter[Any]] = field(init=False, default_factory=dict)
    global_counts_: Counter[Any] = field(init=False, default_factory=Counter)
    is_fitted_: bool = field(init=False, default=False)

    def fit(self, interactions: pd.DataFrame) -> "LastItemTransitionRecommender":
        required_columns = {self.user_col, self.item_col, self.time_col}
        missing_columns = required_columns.difference(interactions.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Не хватает колонок для fit transition model: {missing}")

        ordered = interactions.sort_values([self.user_col, self.time_col, self.item_col])
        transition_counts: dict[Any, Counter[Any]] = defaultdict(Counter)
        for _, group in ordered.groupby(self.user_col, sort=False):
            item_sequence = group[self.item_col].tolist()
            self.global_counts_.update(item_sequence)
            for previous_item, next_item in zip(item_sequence[:-1], item_sequence[1:], strict=False):
                if previous_item == next_item:
                    continue
                transition_counts[previous_item][next_item] += 1

        self.transition_counts_ = dict(transition_counts)
        self.is_fitted_ = True
        return self

    def recommend(
        self,
        last_item_id: Any | None,
        seen_items: set[Any] | None = None,
        k: int = 10,
    ) -> list[Any]:
        if not self.is_fitted_:
            raise ValueError("Сначала вызовите fit()")
        if k <= 0:
            return []

        seen_items = seen_items or set()
        ranked_items: list[Any] = []
        if last_item_id in self.transition_counts_:
            ranked_items.extend(
                item_id
                for item_id, _ in self.transition_counts_[last_item_id].most_common()
                if item_id not in seen_items
            )

        ranked_items.extend(
            item_id
            for item_id, _ in self.global_counts_.most_common()
            if item_id not in seen_items and item_id not in ranked_items
        )
        return ranked_items[:k]

    def recommend_many(
        self,
        user_last_items: dict[Any, Any],
        seen_items_map: dict[Any, set[Any]] | None = None,
        k: int = 10,
    ) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("Сначала вызовите fit()")

        seen_items_map = seen_items_map or {}
        rows: list[dict[str, Any]] = []
        for user_id, last_item_id in user_last_items.items():
            recommendations = self.recommend(
                last_item_id=last_item_id,
                seen_items=seen_items_map.get(user_id, set()),
                k=k,
            )
            rows.extend(
                {
                    self.user_col: user_id,
                    self.item_col: item_id,
                    "rank": rank,
                }
                for rank, item_id in enumerate(recommendations, start=1)
            )
        return pd.DataFrame(rows)


def build_user_last_items(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
) -> dict[Any, Any]:
    """Возвращает последний item в train-истории пользователя."""

    required_columns = {user_col, item_col, time_col}
    missing_columns = required_columns.difference(interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Не хватает колонок для last-item map: {missing}")

    ordered = interactions.sort_values([user_col, time_col, item_col])
    last_items = ordered.groupby(user_col)[item_col].last()
    return dict(last_items.items())


__all__ = [
    "LastItemTransitionRecommender",
    "build_user_last_items",
]
