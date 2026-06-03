"""Content-based recommender для учебного курса."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def prepare_genre_text_features(
    movies: pd.DataFrame,
    item_col: str = "item_id",
    genres_col: str = "genres",
) -> pd.DataFrame:
    """Готовит простое текстовое поле для TF-IDF по жанрам."""

    required_columns = {item_col, genres_col}
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Не хватает колонок для genre features: {missing}")

    prepared = movies.copy()
    prepared["genres_text"] = (
        prepared[genres_col]
        .fillna("")
        .str.replace("|", " ", regex=False)
        .str.replace("(no genres listed)", "no_genres", regex=False)
        .str.strip()
    )
    return prepared


@dataclass
class ContentBasedRecommender:
    """Простой content-based recommender на TF-IDF признаках объектов."""

    item_col: str = "item_id"
    text_col: str = "genres_text"
    title_col: str = "title"
    vectorizer_: TfidfVectorizer = field(init=False)
    items_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    item_matrix_: object = field(init=False, default=None)
    item_to_index_: dict[int, int] = field(init=False, default_factory=dict)
    feature_names_: list[str] = field(init=False, default_factory=list)
    is_fitted_: bool = field(init=False, default=False)

    def fit(
        self,
        items: pd.DataFrame,
        min_df: int = 1,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> "ContentBasedRecommender":
        required_columns = {self.item_col, self.text_col}
        missing_columns = required_columns.difference(items.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Не хватает колонок для fit content-based model: {missing}")

        prepared_items = items.copy().reset_index(drop=True)
        prepared_items[self.text_col] = prepared_items[self.text_col].fillna("")

        self.vectorizer_ = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
        self.item_matrix_ = self.vectorizer_.fit_transform(prepared_items[self.text_col])
        self.items_ = prepared_items
        self.item_to_index_ = {
            int(item_id): index
            for index, item_id in enumerate(prepared_items[self.item_col].tolist())
        }
        self.feature_names_ = self.vectorizer_.get_feature_names_out().tolist()
        self.is_fitted_ = True
        return self

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError("Сначала вызовите fit()")

    def get_item_profile(self, item_id: int) -> pd.Series:
        """Возвращает TF-IDF профиль одного объекта."""

        self._check_is_fitted()
        if int(item_id) not in self.item_to_index_:
            raise KeyError(f"item_id={item_id} отсутствует в fitted catalog")

        index = self.item_to_index_[int(item_id)]
        vector = self.item_matrix_[index].toarray().ravel()
        profile = pd.Series(vector, index=self.feature_names_, name=int(item_id))
        return profile[profile > 0].sort_values(ascending=False)

    def get_similar_items(self, item_id: int, k: int = 10) -> pd.DataFrame:
        """Находит объекты с похожими TF-IDF признаками."""

        self._check_is_fitted()
        if int(item_id) not in self.item_to_index_:
            raise KeyError(f"item_id={item_id} отсутствует в fitted catalog")

        item_index = self.item_to_index_[int(item_id)]
        similarities = cosine_similarity(self.item_matrix_, self.item_matrix_[item_index]).ravel()
        similar_indices = np.argsort(-similarities)

        rows: list[dict[str, object]] = []
        for index in similar_indices:
            candidate_item_id = int(self.items_.iloc[index][self.item_col])
            if candidate_item_id == int(item_id):
                continue
            rows.append(
                {
                    self.item_col: candidate_item_id,
                    "similarity": float(similarities[index]),
                    self.title_col: self.items_.iloc[index].get(self.title_col),
                    self.text_col: self.items_.iloc[index][self.text_col],
                }
            )
            if len(rows) == k:
                break

        return pd.DataFrame(rows)

    def _build_user_profile(self, history_item_ids: list[int]) -> np.ndarray | None:
        valid_indices = [
            self.item_to_index_[int(item_id)]
            for item_id in history_item_ids
            if int(item_id) in self.item_to_index_
        ]
        if not valid_indices:
            return None

        profile = self.item_matrix_[valid_indices].mean(axis=0)
        return np.asarray(profile).reshape(1, -1)

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
        user_profile = self._build_user_profile(history_item_ids)
        if user_profile is None:
            return []

        scores = cosine_similarity(self.item_matrix_, user_profile).ravel()
        ranked_indices = np.argsort(-scores)

        recommendations: list[int] = []
        for index in ranked_indices:
            item_id = int(self.items_.iloc[index][self.item_col])
            if item_id in seen_items:
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
                    "user_id": int(user_id),
                    self.item_col: int(item_id),
                    "rank": rank,
                }
                for rank, item_id in enumerate(recommendations, start=1)
            )

        return pd.DataFrame(rows)


__all__ = [
    "ContentBasedRecommender",
    "prepare_genre_text_features",
]
