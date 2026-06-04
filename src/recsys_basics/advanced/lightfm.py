"""Тонкая обёртка над библиотекой LightFM для advanced-части курса."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, hstack, identity

try:
    from lightfm import LightFM
except ModuleNotFoundError as exc:  # pragma: no cover - зависит от локального окружения
    LightFM = None
    LIGHTFM_IMPORT_ERROR = exc
else:  # pragma: no cover - ветка зависит от локального окружения
    LIGHTFM_IMPORT_ERROR = None


def normalize_genre_token(token: str) -> str:
    """Нормализует один genre-token для feature map."""

    cleaned = token.strip().lower().replace(" ", "_")
    if cleaned == "(no_genres_listed)":
        return "no_genres"
    return cleaned


def build_item_genre_feature_map(
    movies: pd.DataFrame,
    item_col: str = "item_id",
    genres_col: str = "genres",
) -> dict[int, list[str]]:
    """Строит отображение `item_id -> genre feature tokens`."""

    required_columns = {item_col, genres_col}
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Не хватает колонок для item genre features: {missing}")

    feature_map: dict[int, list[str]] = {}
    for row in movies[[item_col, genres_col]].itertuples(index=False):
        item_id = int(getattr(row, item_col))
        genres_value = getattr(row, genres_col)
        if pd.isna(genres_value) or genres_value == "":
            feature_map[item_id] = ["no_genres"]
            continue

        tokens = [normalize_genre_token(token) for token in str(genres_value).split("|")]
        tokens = [token for token in tokens if token]
        feature_map[item_id] = tokens or ["no_genres"]

    return feature_map


def normalize_tag_token(tag: str) -> str:
    """Нормализует phrase-level tag в один feature token."""

    cleaned = re.sub(r"[^a-z0-9]+", "_", str(tag).strip().lower())
    cleaned = cleaned.strip("_")
    return f"tag_{cleaned}" if cleaned else ""


def build_item_tag_feature_map(
    tags: pd.DataFrame,
    item_col: str = "item_id",
    tag_col: str = "tag",
    min_tag_frequency: int = 2,
) -> dict[int, list[str]]:
    """Строит `item_id -> normalized tag tokens` c простым frequency filter."""

    required_columns = {item_col, tag_col}
    missing_columns = required_columns.difference(tags.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Не хватает колонок для item tag features: {missing}")

    prepared = tags[[item_col, tag_col]].copy()
    prepared[tag_col] = prepared[tag_col].map(normalize_tag_token)
    prepared = prepared[prepared[tag_col] != ""].drop_duplicates()
    if prepared.empty:
        return {}

    token_counts = prepared[tag_col].value_counts()
    valid_tokens = set(token_counts[token_counts >= min_tag_frequency].index.tolist())
    prepared = prepared[prepared[tag_col].isin(valid_tokens)]
    if prepared.empty:
        return {}

    return (
        prepared.groupby(item_col)[tag_col]
        .agg(lambda values: sorted(set(values.tolist())))
        .astype(object)
        .to_dict()
    )


def build_item_decade_feature_map(
    movies: pd.DataFrame,
    item_col: str = "item_id",
    title_col: str = "title",
) -> dict[int, list[str]]:
    """Строит простую feature map по десятилетию из title year."""

    required_columns = {item_col, title_col}
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Не хватает колонок для decade features: {missing}")

    feature_map: dict[int, list[str]] = {}
    for row in movies[[item_col, title_col]].itertuples(index=False):
        item_id = int(getattr(row, item_col))
        title = str(getattr(row, title_col))
        match = re.search(r"\((\d{4})\)\s*$", title)
        if not match:
            feature_map[item_id] = []
            continue
        year = int(match.group(1))
        decade = (year // 10) * 10
        feature_map[item_id] = [f"decade_{decade}s"]

    return feature_map


def normalize_title_token(token: str) -> str:
    """Нормализует один токен из title в feature token."""

    cleaned = re.sub(r"[^a-z0-9]+", "", str(token).strip().lower())
    return f"title_{cleaned}" if cleaned else ""


def build_item_title_feature_map(
    movies: pd.DataFrame,
    item_col: str = "item_id",
    title_col: str = "title",
    min_token_frequency: int = 2,
    min_token_length: int = 2,
    stopwords: set[str] | None = None,
) -> dict[int, list[str]]:
    """Строит `item_id -> normalized title tokens` с простым frequency filter."""

    required_columns = {item_col, title_col}
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Не хватает колонок для title features: {missing}")

    if stopwords is None:
        stopwords = {
            "a",
            "an",
            "and",
            "at",
            "by",
            "for",
            "from",
            "ii",
            "iii",
            "in",
            "of",
            "on",
            "part",
            "the",
            "to",
        }

    raw_tokens_by_item: dict[int, list[str]] = {}
    token_document_counts: dict[str, int] = {}

    for row in movies[[item_col, title_col]].itertuples(index=False):
        item_id = int(getattr(row, item_col))
        title = str(getattr(row, title_col))
        title_wo_year = re.sub(r"\(\d{4}\)\s*$", "", title).strip()
        base_tokens = re.findall(r"[a-z0-9]+", title_wo_year.lower())
        tokens = {
            normalize_title_token(token)
            for token in base_tokens
            if len(token) >= min_token_length and not token.isdigit() and token not in stopwords
        }
        tokens.discard("")
        raw_tokens_by_item[item_id] = sorted(tokens)
        for token in tokens:
            token_document_counts[token] = token_document_counts.get(token, 0) + 1

    valid_tokens = {
        token for token, count in token_document_counts.items() if count >= min_token_frequency
    }
    return {
        item_id: [token for token in tokens if token in valid_tokens]
        for item_id, tokens in raw_tokens_by_item.items()
    }


def merge_item_feature_maps(*feature_maps: dict[int, list[str]]) -> dict[int, list[str]]:
    """Объединяет несколько `item_id -> features` в одну map без дублей."""

    merged: dict[int, set[str]] = {}
    for feature_map in feature_maps:
        for item_id, tokens in feature_map.items():
            merged.setdefault(int(item_id), set()).update(token for token in tokens if token)
    return {item_id: sorted(tokens) for item_id, tokens in merged.items()}


def prune_item_feature_map(
    item_features: dict[int, list[str]],
    min_item_frequency: int = 2,
    max_item_frequency_ratio: float = 1.0,
    prunable_prefixes: tuple[str, ...] | None = None,
) -> dict[int, list[str]]:
    """Удаляет слишком редкие или слишком частые токены из feature map."""

    if min_item_frequency < 1:
        raise ValueError("min_item_frequency должен быть >= 1")
    if not 0 < max_item_frequency_ratio <= 1.0:
        raise ValueError("max_item_frequency_ratio должен быть в интервале (0, 1]")

    token_document_counts: dict[str, int] = {}
    n_items = len(item_features)
    for tokens in item_features.values():
        for token in set(tokens):
            token_document_counts[token] = token_document_counts.get(token, 0) + 1

    def is_prunable(token: str) -> bool:
        if prunable_prefixes is None:
            return True
        return any(token.startswith(prefix) for prefix in prunable_prefixes)

    kept_tokens = {
        token
        for token, count in token_document_counts.items()
        if (
            not is_prunable(token)
            or (count >= min_item_frequency and count / max(n_items, 1) <= max_item_frequency_ratio)
        )
    }
    return {
        item_id: [token for token in tokens if token in kept_tokens]
        for item_id, tokens in item_features.items()
    }


@dataclass
class LightFMHybridRecommender:
    """Минимальная учебная обёртка над LightFM для hybrid factorization."""

    user_col: str = "user_id"
    item_col: str = "item_id"
    no_components: int = 16
    loss: str = "warp"
    learning_rate: float = 0.05
    item_alpha: float = 1e-6
    user_alpha: float = 1e-6
    epochs: int = 15
    num_threads: int = 1
    random_state: int = 42
    user_to_index_: dict[int, int] = field(init=False, default_factory=dict)
    item_to_index_: dict[int, int] = field(init=False, default_factory=dict)
    index_to_item_: dict[int, int] = field(init=False, default_factory=dict)
    feature_to_index_: dict[str, int] = field(init=False, default_factory=dict)
    seen_items_: dict[int, set[int]] = field(init=False, default_factory=dict)
    interaction_matrix_: csr_matrix = field(
        init=False,
        default_factory=lambda: csr_matrix((0, 0), dtype=np.float32),
    )
    item_feature_matrix_: csr_matrix = field(
        init=False,
        default_factory=lambda: csr_matrix((0, 0), dtype=np.float32),
    )
    model_: Any = field(init=False, default=None)
    is_fitted_: bool = field(init=False, default=False)

    def fit(
        self,
        interactions: pd.DataFrame,
        item_features: dict[int, list[str]],
    ) -> "LightFMHybridRecommender":
        self._require_lightfm()

        required_columns = {self.user_col, self.item_col}
        missing_columns = required_columns.difference(interactions.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Не хватает колонок для fit LightFM model: {missing}")

        prepared = interactions[[self.user_col, self.item_col]].drop_duplicates().copy()
        prepared = prepared.sort_values([self.user_col, self.item_col]).reset_index(drop=True)
        if prepared.empty:
            raise ValueError("Для fit LightFM model нужны непустые interactions")

        user_ids = sorted(int(user_id) for user_id in prepared[self.user_col].unique().tolist())
        item_ids = sorted(int(item_id) for item_id in prepared[self.item_col].unique().tolist())

        self.user_to_index_ = {user_id: index for index, user_id in enumerate(user_ids)}
        self.item_to_index_ = {item_id: index for index, item_id in enumerate(item_ids)}
        self.index_to_item_ = {index: item_id for item_id, index in self.item_to_index_.items()}

        row_indices = prepared[self.user_col].map(self.user_to_index_).to_numpy()
        col_indices = prepared[self.item_col].map(self.item_to_index_).to_numpy()
        values = np.ones(len(prepared), dtype=np.float32)

        self.interaction_matrix_ = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        ).tocsr()

        self.feature_to_index_ = self._build_feature_index(item_ids, item_features)
        self.item_feature_matrix_ = self._build_item_feature_matrix(item_ids, item_features)
        self.seen_items_ = (
            prepared.groupby(self.user_col)[self.item_col]
            .agg(lambda values: {int(item_id) for item_id in values.tolist()})
            .to_dict()
        )

        model = LightFM(
            no_components=self.no_components,
            loss=self.loss,
            learning_rate=self.learning_rate,
            item_alpha=self.item_alpha,
            user_alpha=self.user_alpha,
            random_state=np.random.RandomState(self.random_state),
        )
        model.fit(
            interactions=self.interaction_matrix_,
            item_features=self.item_feature_matrix_,
            epochs=self.epochs,
            num_threads=self.num_threads,
            verbose=False,
        )

        self.model_ = model
        self.is_fitted_ = True
        return self

    def _require_lightfm(self) -> None:
        if LightFM is None:
            raise ModuleNotFoundError(
                "Библиотека `lightfm` не установлена. Установите её в окружение проекта "
                "и перезапустите notebook."
            ) from LIGHTFM_IMPORT_ERROR

    def _build_feature_index(
        self,
        item_ids: list[int],
        item_features: dict[int, list[str]],
    ) -> dict[str, int]:
        feature_tokens = sorted(
            {
                token
                for item_id in item_ids
                for token in item_features.get(item_id, ["no_genres"])
            }
        )
        return {token: index for index, token in enumerate(feature_tokens)}

    def _build_item_feature_matrix(
        self,
        item_ids: list[int],
        item_features: dict[int, list[str]],
    ) -> csr_matrix:
        n_items = len(item_ids)
        n_features = len(self.feature_to_index_)

        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []

        for item_row, item_id in enumerate(item_ids):
            for token in item_features.get(item_id, ["no_genres"]) or ["no_genres"]:
                if token not in self.feature_to_index_:
                    continue
                rows.append(item_row)
                cols.append(self.feature_to_index_[token])
                values.append(1.0)

        genre_matrix = coo_matrix(
            (np.asarray(values, dtype=np.float32), (rows, cols)),
            shape=(n_items, n_features),
            dtype=np.float32,
        ).tocsr()
        identity_matrix = identity(n_items, dtype=np.float32, format="csr")
        return hstack([identity_matrix, genre_matrix], format="csr")

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_ or self.model_ is None:
            raise ValueError("Сначала вызовите fit()")

    def score_items(self, user_id: int) -> pd.Series:
        self._check_is_fitted()
        if int(user_id) not in self.user_to_index_:
            raise KeyError(f"user_id={user_id} отсутствует в fitted LightFM model")

        user_index = self.user_to_index_[int(user_id)]
        item_indices = np.arange(len(self.item_to_index_), dtype=np.int32)
        scores = self.model_.predict(
            user_ids=user_index,
            item_ids=item_indices,
            item_features=self.item_feature_matrix_,
            num_threads=self.num_threads,
        )
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


__all__ = [
    "LightFMHybridRecommender",
    "build_item_decade_feature_map",
    "build_item_genre_feature_map",
    "build_item_tag_feature_map",
    "build_item_title_feature_map",
    "merge_item_feature_maps",
    "prune_item_feature_map",
]
