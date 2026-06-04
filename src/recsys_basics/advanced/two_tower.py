"""Учебная two-tower retrieval модель для advanced-части курса."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover - зависит от локального окружения
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover - ветка зависит от локального окружения
    TORCH_IMPORT_ERROR = None


def build_user_text_profiles(
    interactions: pd.DataFrame,
    item_metadata: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    text_col: str = "item_text",
    timestamp_col: str = "timestamp",
    max_history_items: int = 20,
) -> pd.DataFrame:
    """Строит простые user-text профили из train-истории пользователя."""

    required_interaction_columns = {user_col, item_col}
    if timestamp_col in interactions.columns:
        required_interaction_columns.add(timestamp_col)
    missing_interaction_columns = required_interaction_columns.difference(interactions.columns)
    if missing_interaction_columns:
        missing = ", ".join(sorted(missing_interaction_columns))
        raise ValueError(f"Не хватает колонок для user text profiles: {missing}")

    required_item_columns = {item_col, text_col}
    missing_item_columns = required_item_columns.difference(item_metadata.columns)
    if missing_item_columns:
        missing = ", ".join(sorted(missing_item_columns))
        raise ValueError(f"Не хватает item metadata для user text profiles: {missing}")

    prepared = interactions[[column for column in [user_col, item_col, timestamp_col] if column in interactions.columns]].copy()
    prepared = prepared.merge(
        item_metadata[[item_col, text_col]].drop_duplicates(subset=[item_col]),
        on=item_col,
        how="left",
    )
    prepared[text_col] = prepared[text_col].fillna("").astype(str)
    prepared = prepared[prepared[text_col].str.strip() != ""].copy()
    if prepared.empty:
        return pd.DataFrame(columns=[user_col, "user_text", "history_items"])

    sort_columns = [user_col]
    if timestamp_col in prepared.columns:
        sort_columns.append(timestamp_col)
    sort_columns.append(item_col)
    prepared = prepared.sort_values(sort_columns, ascending=[True] * len(sort_columns))

    rows: list[dict[str, Any]] = []
    for user_id, group in prepared.groupby(user_col, sort=True):
        seen_item_ids: set[Any] = set()
        collected_texts: list[str] = []
        for row in group.sort_values(sort_columns, ascending=[False] * len(sort_columns)).itertuples(index=False):
            item_id = getattr(row, item_col)
            if item_id in seen_item_ids:
                continue
            text_value = str(getattr(row, text_col)).strip()
            if not text_value:
                continue
            seen_item_ids.add(item_id)
            collected_texts.append(text_value)
            if len(collected_texts) >= max_history_items:
                break

        rows.append(
            {
                user_col: user_id,
                "user_text": " ".join(collected_texts),
                "history_items": len(collected_texts),
            }
        )

    return pd.DataFrame(rows)


_TwoTowerModuleBase = nn.Module if nn is not None else object


class _TwoTowerModule(_TwoTowerModuleBase):
    """Минимальная two-tower сеть: отдельные MLP для user/item dense features."""

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_tower = self._build_tower(
            input_dim=user_input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.item_tower = self._build_tower(
            input_dim=item_input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    @staticmethod
    def _build_tower(
        input_dim: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, embedding_dim))
        return nn.Sequential(*layers)

    def encode_users(self, user_features: torch.Tensor) -> torch.Tensor:
        user_embeddings = self.user_tower(user_features)
        return torch.nn.functional.normalize(user_embeddings, dim=-1)

    def encode_items(self, item_features: torch.Tensor) -> torch.Tensor:
        item_embeddings = self.item_tower(item_features)
        return torch.nn.functional.normalize(item_embeddings, dim=-1)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_embeddings = self.encode_users(user_features)
        item_embeddings = self.encode_items(item_features)
        return (user_embeddings * item_embeddings).sum(dim=-1)


@dataclass
class TwoTowerRetrievalRecommender:
    """Учебная two-tower retrieval модель на PyTorch."""

    user_col: str = "user_id"
    item_col: str = "item_id"
    embedding_dim: int = 32
    hidden_dims: tuple[int, ...] = (64,)
    dropout: float = 0.0
    num_epochs: int = 12
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_negatives: int = 4
    random_state: int = 42
    device: str = "cpu"
    user_to_index_: dict[Any, int] = field(init=False, default_factory=dict)
    item_to_index_: dict[str, int] = field(init=False, default_factory=dict)
    index_to_user_: dict[int, Any] = field(init=False, default_factory=dict)
    index_to_item_: dict[int, str] = field(init=False, default_factory=dict)
    seen_items_by_user_: dict[Any, set[str]] = field(init=False, default_factory=dict)
    train_loss_history_: list[float] = field(init=False, default_factory=list)
    model_: Any = field(init=False, default=None)
    user_feature_matrix_: Any = field(init=False, default=None)
    item_feature_matrix_: Any = field(init=False, default=None)
    user_features_df_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    item_features_df_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    is_fitted_: bool = field(init=False, default=False)

    def fit(
        self,
        interactions: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
    ) -> "TwoTowerRetrievalRecommender":
        self._require_torch()

        required_columns = {self.user_col, self.item_col}
        missing_columns = required_columns.difference(interactions.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Не хватает колонок для fit two-tower model: {missing}")

        prepared = interactions[[self.user_col, self.item_col]].drop_duplicates().copy()
        prepared[self.user_col] = prepared[self.user_col].astype(str)
        prepared[self.item_col] = prepared[self.item_col].astype(str)

        prepared_user_features = self._prepare_feature_frame(user_features, entity_col=self.user_col)
        prepared_item_features = self._prepare_feature_frame(item_features, entity_col=self.item_col)

        available_user_ids = set(prepared_user_features.index.tolist())
        available_item_ids = set(prepared_item_features.index.tolist())
        prepared = prepared[
            prepared[self.user_col].isin(available_user_ids)
            & prepared[self.item_col].isin(available_item_ids)
        ].copy()
        if prepared.empty:
            raise ValueError("После пересечения interactions с feature tables не осталось train-пар")

        user_ids = sorted(str(user_id) for user_id in prepared[self.user_col].unique().tolist())
        item_ids = sorted(str(item_id) for item_id in prepared[self.item_col].unique().tolist())

        self.user_to_index_ = {user_id: index for index, user_id in enumerate(user_ids)}
        self.item_to_index_ = {item_id: index for index, item_id in enumerate(item_ids)}
        self.index_to_user_ = {index: user_id for user_id, index in self.user_to_index_.items()}
        self.index_to_item_ = {index: item_id for item_id, index in self.item_to_index_.items()}
        self.seen_items_by_user_ = (
            prepared.groupby(self.user_col)[self.item_col]
            .agg(lambda values: {str(item_id) for item_id in values.tolist()})
            .to_dict()
        )

        self.user_features_df_ = prepared_user_features.loc[user_ids].copy()
        self.item_features_df_ = prepared_item_features.loc[item_ids].copy()
        self.user_feature_matrix_ = torch.tensor(
            self.user_features_df_.to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        self.item_feature_matrix_ = torch.tensor(
            self.item_features_df_.to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        torch.manual_seed(self.random_state)
        model = _TwoTowerModule(
            user_input_dim=self.user_features_df_.shape[1],
            item_input_dim=self.item_features_df_.shape[1],
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        user_indices, positive_item_indices, negative_item_indices = self._build_pairwise_training_samples(
            prepared
        )
        dataset = TensorDataset(
            torch.tensor(user_indices, dtype=torch.long),
            torch.tensor(positive_item_indices, dtype=torch.long),
            torch.tensor(negative_item_indices, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.train_loss_history_ = []
        model.train()
        for _ in range(self.num_epochs):
            total_loss = 0.0
            total_examples = 0
            for batch_user_indices, batch_positive_item_indices, batch_negative_item_indices in loader:
                batch_user_indices = batch_user_indices.to(self.device)
                batch_positive_item_indices = batch_positive_item_indices.to(self.device)
                batch_negative_item_indices = batch_negative_item_indices.to(self.device)

                batch_user_features = self.user_feature_matrix_[batch_user_indices]
                batch_positive_item_features = self.item_feature_matrix_[batch_positive_item_indices]
                batch_negative_item_features = self.item_feature_matrix_[batch_negative_item_indices]

                optimizer.zero_grad()
                positive_scores = model(batch_user_features, batch_positive_item_features)
                negative_scores = model(batch_user_features, batch_negative_item_features)
                loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()
                loss.backward()
                optimizer.step()

                batch_size = int(batch_user_indices.size(0))
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size

            self.train_loss_history_.append(total_loss / max(total_examples, 1))

        self.model_ = model
        self.is_fitted_ = True
        return self

    def _prepare_feature_frame(self, features: pd.DataFrame, entity_col: str) -> pd.DataFrame:
        if entity_col not in features.columns:
            raise ValueError(f"В feature table нет колонки {entity_col}")

        prepared = features.copy()
        if entity_col == self.user_col:
            prepared[entity_col] = prepared[entity_col].astype(str)
        else:
            prepared[entity_col] = prepared[entity_col].astype(str)

        numeric_columns = [column for column in prepared.columns if column != entity_col]
        if not numeric_columns:
            raise ValueError(f"В feature table для {entity_col} нет числовых признаков")

        prepared = prepared[[entity_col] + numeric_columns].drop_duplicates(subset=[entity_col]).set_index(entity_col)
        prepared = prepared.fillna(0.0).astype(np.float32)
        return prepared.sort_index()

    def _build_pairwise_training_samples(
        self,
        prepared: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        all_item_indices = np.arange(len(self.item_to_index_), dtype=np.int64)
        seen_item_indices_by_user = {
            str(user_id): {self.item_to_index_[str(item_id)] for item_id in item_ids}
            for user_id, item_ids in self.seen_items_by_user_.items()
        }

        user_indices: list[int] = []
        positive_item_indices: list[int] = []
        negative_item_indices: list[int] = []

        for row in prepared.itertuples(index=False):
            user_id = str(getattr(row, self.user_col))
            item_id = str(getattr(row, self.item_col))
            user_index = self.user_to_index_[user_id]
            positive_item_index = self.item_to_index_[item_id]
            seen_indices = seen_item_indices_by_user[user_id]

            for _ in range(self.num_negatives):
                sampled_item_index = int(rng.choice(all_item_indices))
                while sampled_item_index in seen_indices:
                    sampled_item_index = int(rng.choice(all_item_indices))
                user_indices.append(user_index)
                positive_item_indices.append(positive_item_index)
                negative_item_indices.append(sampled_item_index)

        return (
            np.asarray(user_indices, dtype=np.int64),
            np.asarray(positive_item_indices, dtype=np.int64),
            np.asarray(negative_item_indices, dtype=np.int64),
        )

    def _require_torch(self) -> None:
        if torch is None:
            raise ModuleNotFoundError(
                "Библиотека `torch` не установлена. Установите её в окружение проекта "
                "и перезапустите notebook."
            ) from TORCH_IMPORT_ERROR

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_ or self.model_ is None:
            raise ValueError("Сначала вызовите fit()")

    def score_items(self, user_id: Any) -> pd.Series:
        self._check_is_fitted()
        if str(user_id) not in self.user_to_index_:
            raise KeyError(f"user_id={user_id} отсутствует в fitted two-tower model")

        self.model_.eval()
        user_index = self.user_to_index_[str(user_id)]
        with torch.no_grad():
            user_features = self.user_feature_matrix_[user_index : user_index + 1]
            user_embeddings = self.model_.encode_users(user_features)
            item_embeddings = self.model_.encode_items(self.item_feature_matrix_)
            scores = (user_embeddings @ item_embeddings.T).squeeze(0).detach().cpu().numpy()

        index = pd.Index(
            [self.index_to_item_[index] for index in range(len(self.index_to_item_))],
            name=self.item_col,
        )
        return pd.Series(scores, index=index, name="score")

    def recommend(
        self,
        user_id: Any,
        seen_items: set[str] | set[int] | None = None,
        k: int = 10,
    ) -> list[str]:
        self._check_is_fitted()
        if k <= 0:
            return []

        scores = self.score_items(user_id)
        normalized_seen_items = {str(item_id) for item_id in (seen_items or set())}
        candidate_scores = scores.drop(
            labels=[item_id for item_id in normalized_seen_items if item_id in scores.index],
            errors="ignore",
        )
        ranked = candidate_scores.sort_values(ascending=False).head(k)
        return [str(item_id) for item_id in ranked.index.tolist()]

    def recommend_many(
        self,
        user_ids: list[Any],
        seen_items_map: dict[Any, set[str] | set[int]] | None = None,
        k: int = 10,
    ) -> pd.DataFrame:
        self._check_is_fitted()
        seen_items_map = seen_items_map or {}
        rows: list[dict[str, Any]] = []
        for user_id in user_ids:
            recommendations = self.recommend(
                user_id=user_id,
                seen_items=seen_items_map.get(user_id, set()),
                k=k,
            )
            rows.extend(
                {
                    self.user_col: user_id,
                    self.item_col: str(item_id),
                    "rank": rank,
                }
                for rank, item_id in enumerate(recommendations, start=1)
            )
        return pd.DataFrame(rows)


__all__ = [
    "TwoTowerRetrievalRecommender",
    "build_user_text_profiles",
]
