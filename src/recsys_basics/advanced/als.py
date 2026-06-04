"""Тонкая обёртка над библиотекой implicit ALS для advanced-части курса."""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError:  # pragma: no cover - зависит от локального окружения
    _BLAS_THREADPOOL_LIMIT = None
else:  # pragma: no cover - зависит от локального окружения
    _BLAS_THREADPOOL_LIMIT = threadpool_limits(1, "blas")

try:
    from implicit.als import AlternatingLeastSquares
except ModuleNotFoundError as exc:  # pragma: no cover - зависит от локального окружения
    AlternatingLeastSquares = None
    IMPLICIT_IMPORT_ERROR = exc
else:  # pragma: no cover - ветка зависит от локального окружения
    IMPLICIT_IMPORT_ERROR = None


@dataclass
class ImplicitALSRecommender:
    """Минимальная учебная обёртка над `implicit.als.AlternatingLeastSquares`."""

    user_col: str = "user_id"
    item_col: str = "item_id"
    n_factors: int = 16
    n_iterations: int = 8
    regularization: float = 0.1
    alpha: float = 20.0
    random_state: int = 42
    user_to_index_: dict[int, int] = field(init=False, default_factory=dict)
    item_to_index_: dict[int, int] = field(init=False, default_factory=dict)
    index_to_user_: dict[int, int] = field(init=False, default_factory=dict)
    index_to_item_: dict[int, int] = field(init=False, default_factory=dict)
    interaction_matrix_: csr_matrix = field(
        init=False,
        default_factory=lambda: csr_matrix((0, 0), dtype=np.float32),
    )
    user_factors_: np.ndarray = field(init=False, default_factory=lambda: np.empty((0, 0)))
    item_factors_: np.ndarray = field(init=False, default_factory=lambda: np.empty((0, 0)))
    model_: Any = field(init=False, default=None)
    is_fitted_: bool = field(init=False, default=False)

    def fit(self, interactions: pd.DataFrame) -> "ImplicitALSRecommender":
        self._require_implicit()

        required_columns = {self.user_col, self.item_col}
        missing_columns = required_columns.difference(interactions.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Не хватает колонок для fit ALS model: {missing}")

        prepared = interactions[[self.user_col, self.item_col]].drop_duplicates().copy()
        prepared = prepared.sort_values([self.user_col, self.item_col]).reset_index(drop=True)
        if prepared.empty:
            raise ValueError("Для fit ALS model нужны непустые interactions")

        user_ids = sorted(int(user_id) for user_id in prepared[self.user_col].unique().tolist())
        item_ids = sorted(int(item_id) for item_id in prepared[self.item_col].unique().tolist())

        self.user_to_index_ = {user_id: index for index, user_id in enumerate(user_ids)}
        self.item_to_index_ = {item_id: index for index, item_id in enumerate(item_ids)}
        self.index_to_user_ = {index: user_id for user_id, index in self.user_to_index_.items()}
        self.index_to_item_ = {index: item_id for item_id, index in self.item_to_index_.items()}

        row_indices = prepared[self.user_col].map(self.user_to_index_).to_numpy()
        col_indices = prepared[self.item_col].map(self.item_to_index_).to_numpy()
        values = np.full(len(prepared), self.alpha, dtype=np.float32)

        self.interaction_matrix_ = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        )

        model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.n_iterations,
            random_state=self.random_state,
        )
        model.fit(self.interaction_matrix_)

        self.model_ = model
        self.user_factors_ = np.asarray(model.user_factors)
        self.item_factors_ = np.asarray(model.item_factors)
        self.is_fitted_ = True
        return self

    def _require_implicit(self) -> None:
        if AlternatingLeastSquares is None:
            raise ModuleNotFoundError(
                "Библиотека `implicit` не установлена. Установите её в окружение проекта "
                "и перезапустите notebook."
            ) from IMPLICIT_IMPORT_ERROR

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_ or self.model_ is None:
            raise ValueError("Сначала вызовите fit()")

    def score_items(self, user_id: int) -> pd.Series:
        self._check_is_fitted()
        if int(user_id) not in self.user_to_index_:
            raise KeyError(f"user_id={user_id} отсутствует в fitted ALS model")

        user_index = self.user_to_index_[int(user_id)]
        scores = self.item_factors_ @ self.user_factors_[user_index]
        index = pd.Index(
            [self.index_to_item_[index] for index in range(len(self.index_to_item_))],
            name=self.item_col,
        )
        return pd.Series(scores, index=index, name="score")

    def recommend(
        self,
        user_id: int,
        seen_items: set[int] | None = None,
        k: int = 10,
    ) -> list[int]:
        self._check_is_fitted()
        if k <= 0:
            return []

        scores = self.score_items(user_id)
        seen_items = seen_items or set()
        candidate_scores = scores.drop(
            labels=[item_id for item_id in seen_items if item_id in scores.index],
            errors="ignore",
        )
        ranked = candidate_scores.sort_values(ascending=False).head(k)
        return [int(item_id) for item_id in ranked.index.tolist()]

    def recommend_many(
        self,
        user_ids: list[int],
        seen_items_map: dict[int, set[int]] | None = None,
        k: int = 10,
    ) -> pd.DataFrame:
        self._check_is_fitted()
        seen_items_map = seen_items_map or {}
        rows: list[dict[str, int]] = []
        for user_id in user_ids:
            recommendations = self.recommend(
                user_id=int(user_id),
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


ALSImplicitRecommender = ImplicitALSRecommender


__all__ = ["ImplicitALSRecommender", "ALSImplicitRecommender"]
