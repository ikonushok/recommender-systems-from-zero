"""Item-based collaborative filtering для учебного курса."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ItemItemRecommender:
    """Простой item-based collaborative filtering на cosine similarity."""

    user_col: str = "user_id"
    item_col: str = "item_id"
    item_similarity_: np.ndarray = field(init=False)
    interactions_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    user_to_index_: dict[int, int] = field(init=False, default_factory=dict)
    item_to_index_: dict[int, int] = field(init=False, default_factory=dict)
    index_to_item_: dict[int, int] = field(init=False, default_factory=dict)
    user_histories_: dict[int, list[int]] = field(init=False, default_factory=dict)
    is_fitted_: bool = field(init=False, default=False)

    def fit(self, interactions: pd.DataFrame) -> "ItemItemRecommender":
        required_columns = {self.user_col, self.item_col}
        missing_columns = required_columns.difference(interactions.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Не хватает колонок для fit item-based collaborative filtering model: {missing}"
            )

        prepared = interactions[[self.user_col, self.item_col]].drop_duplicates().copy()
        prepared = prepared.sort_values([self.user_col, self.item_col]).reset_index(drop=True)

        user_ids = sorted(prepared[self.user_col].unique().tolist())
        item_ids = sorted(prepared[self.item_col].unique().tolist())

        self.user_to_index_ = {int(user_id): index for index, user_id in enumerate(user_ids)}
        self.item_to_index_ = {int(item_id): index for index, item_id in enumerate(item_ids)}
        self.index_to_item_ = {index: int(item_id) for item_id, index in self.item_to_index_.items()}

        row_indices = prepared[self.user_col].map(self.user_to_index_).to_numpy()
        col_indices = prepared[self.item_col].map(self.item_to_index_).to_numpy()
        values = np.ones(len(prepared), dtype=np.float32)

        user_item = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(user_ids), len(item_ids)),
        )
        item_user = user_item.transpose()
        self.item_similarity_ = cosine_similarity(item_user, dense_output=True)
        np.fill_diagonal(self.item_similarity_, 0.0)

        self.user_histories_ = (
            prepared.groupby(self.user_col)[self.item_col].agg(list).to_dict()
        )
        self.interactions_ = prepared
        self.is_fitted_ = True
        return self

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError("Сначала вызовите fit()")

    def get_similar_items(self, item_id: int, k: int = 10) -> pd.DataFrame:
        self._check_is_fitted()
        if int(item_id) not in self.item_to_index_:
            raise KeyError(f"item_id={item_id} отсутствует в fitted interactions")

        item_index = self.item_to_index_[int(item_id)]
        similarities = self.item_similarity_[item_index]
        ranked_indices = np.argsort(-similarities)

        rows: list[dict[str, float | int]] = []
        for index in ranked_indices:
            similarity = float(similarities[index])
            if similarity <= 0:
                continue
            rows.append(
                {
                    self.item_col: self.index_to_item_[int(index)],
                    "similarity": similarity,
                }
            )
            if len(rows) == k:
                break

        return pd.DataFrame(rows)

    def recommend(
        self,
        history_item_ids: list[int],
        seen_items: set[int] | None = None,
        k: int = 10,
    ) -> list[int]:
        self._check_is_fitted()
        if k <= 0:
            return []

        seen_items = seen_items or set()
        valid_history = [
            int(item_id) for item_id in history_item_ids if int(item_id) in self.item_to_index_
        ]
        if not valid_history:
            return []

        scores = np.zeros(len(self.item_to_index_), dtype=np.float32)
        for item_id in valid_history:
            item_index = self.item_to_index_[int(item_id)]
            scores += self.item_similarity_[item_index]

        ranked_indices = np.argsort(-scores)
        recommendations: list[int] = []
        for index in ranked_indices:
            item_id = self.index_to_item_[int(index)]
            if item_id in seen_items:
                continue
            if scores[index] <= 0:
                continue
            recommendations.append(item_id)
            if len(recommendations) == k:
                break
        return recommendations

    def recommend_many(
        self,
        user_histories: dict[int, list[int]],
        seen_items_map: dict[int, set[int]] | None = None,
        k: int = 10,
    ) -> pd.DataFrame:
        self._check_is_fitted()

        seen_items_map = seen_items_map or {}
        rows: list[dict[str, int]] = []
        for user_id, history_item_ids in user_histories.items():
            recommendations = self.recommend(
                history_item_ids=history_item_ids,
                seen_items=seen_items_map.get(int(user_id), set()),
                k=k,
            )
            rows.extend(
                {
                    self.user_col: int(user_id),
                    self.item_col: int(item_id),
                    "rank": rank,
                }
                for rank, item_id in enumerate(recommendations, start=1)
            )

        return pd.DataFrame(rows)

ItemBasedCFRecommender = ItemItemRecommender
ItemBasedCollaborativeRecommender = ItemItemRecommender


__all__ = [
    "ItemItemRecommender",
    "ItemBasedCFRecommender",
    "ItemBasedCollaborativeRecommender",
]
