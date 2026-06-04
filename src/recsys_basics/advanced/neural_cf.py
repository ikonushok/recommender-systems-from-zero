"""Учебная NeuMF-модель для advanced-части курса."""

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


_NeuralCFModuleBase = nn.Module if nn is not None else object


class _NeuralCFModule(_NeuralCFModuleBase):
    """Упрощённая NeuMF: GMF-ветка + MLP-ветка + общий scorer."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims не должен быть пустым")

        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)

        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(embedding_dim + input_dim, 1)

        nn.init.normal_(self.user_embedding_gmf.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, mean=0.0, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        gmf_user_vectors = self.user_embedding_gmf(user_indices)
        gmf_item_vectors = self.item_embedding_gmf(item_indices)
        gmf_output = gmf_user_vectors * gmf_item_vectors

        mlp_user_vectors = self.user_embedding_mlp(user_indices)
        mlp_item_vectors = self.item_embedding_mlp(item_indices)
        mlp_features = torch.cat([mlp_user_vectors, mlp_item_vectors], dim=-1)
        mlp_output = self.mlp(mlp_features)

        final_features = torch.cat([gmf_output, mlp_output], dim=-1)
        return self.output(final_features).squeeze(-1)


@dataclass
class NeuralCFRecommender:
    """Учебная NeuMF-модель на PyTorch для Neural Collaborative Filtering."""

    user_col: str = "user_id"
    item_col: str = "item_id"
    loss: str = "bpr"
    embedding_dim: int = 32
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.0
    num_epochs: int = 8
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    num_negatives: int = 4
    random_state: int = 42
    device: str = "cpu"
    user_to_index_: dict[int, int] = field(init=False, default_factory=dict)
    item_to_index_: dict[int, int] = field(init=False, default_factory=dict)
    index_to_user_: dict[int, int] = field(init=False, default_factory=dict)
    index_to_item_: dict[int, int] = field(init=False, default_factory=dict)
    seen_items_by_user_: dict[int, set[int]] = field(init=False, default_factory=dict)
    model_: Any = field(init=False, default=None)
    train_loss_history_: list[float] = field(init=False, default_factory=list)
    is_fitted_: bool = field(init=False, default=False)
    architecture_name_: str = field(init=False, default="NeuMF")

    def fit(self, interactions: pd.DataFrame) -> "NeuralCFRecommender":
        self._require_torch()

        required_columns = {self.user_col, self.item_col}
        missing_columns = required_columns.difference(interactions.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Не хватает колонок для fit NeuMF model: {missing}")

        prepared = interactions[[self.user_col, self.item_col]].drop_duplicates().copy()
        prepared = prepared.sort_values([self.user_col, self.item_col]).reset_index(drop=True)
        if prepared.empty:
            raise ValueError("Для fit NeuMF model нужны непустые interactions")

        user_ids = sorted(int(user_id) for user_id in prepared[self.user_col].unique().tolist())
        item_ids = sorted(int(item_id) for item_id in prepared[self.item_col].unique().tolist())

        self.user_to_index_ = {user_id: index for index, user_id in enumerate(user_ids)}
        self.item_to_index_ = {item_id: index for index, item_id in enumerate(item_ids)}
        self.index_to_user_ = {index: user_id for user_id, index in self.user_to_index_.items()}
        self.index_to_item_ = {index: item_id for item_id, index in self.item_to_index_.items()}
        self.seen_items_by_user_ = (
            prepared.groupby(self.user_col)[self.item_col]
            .agg(lambda values: {int(item_id) for item_id in values.tolist()})
            .to_dict()
        )

        torch.manual_seed(self.random_state)
        model = _NeuralCFModule(
            n_users=len(user_ids),
            n_items=len(item_ids),
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.train_loss_history_ = []
        if self.loss == "bce":
            self._fit_with_bce(model=model, optimizer=optimizer, prepared=prepared)
        elif self.loss == "bpr":
            self._fit_with_bpr(model=model, optimizer=optimizer, prepared=prepared)
        else:
            raise ValueError("loss должен быть либо 'bce', либо 'bpr'")

        self.model_ = model
        self.is_fitted_ = True
        return self

    def _fit_with_bce(
        self,
        model: _NeuralCFModule,
        optimizer: Any,
        prepared: pd.DataFrame,
    ) -> None:
        user_indices, item_indices, labels = self._build_pointwise_training_samples(prepared)
        dataset = TensorDataset(
            torch.tensor(user_indices, dtype=torch.long),
            torch.tensor(item_indices, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(self.num_epochs):
            total_loss = 0.0
            total_examples = 0
            for batch_user_indices, batch_item_indices, batch_labels in loader:
                batch_user_indices = batch_user_indices.to(self.device)
                batch_item_indices = batch_item_indices.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = model(batch_user_indices, batch_item_indices)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                batch_size = int(batch_labels.size(0))
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size

            mean_loss = total_loss / max(total_examples, 1)
            self.train_loss_history_.append(mean_loss)

    def _fit_with_bpr(
        self,
        model: _NeuralCFModule,
        optimizer: Any,
        prepared: pd.DataFrame,
    ) -> None:
        user_indices, positive_item_indices, negative_item_indices = (
            self._build_pairwise_training_samples(prepared)
        )
        dataset = TensorDataset(
            torch.tensor(user_indices, dtype=torch.long),
            torch.tensor(positive_item_indices, dtype=torch.long),
            torch.tensor(negative_item_indices, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for _ in range(self.num_epochs):
            total_loss = 0.0
            total_examples = 0
            for batch_user_indices, batch_positive_item_indices, batch_negative_item_indices in loader:
                batch_user_indices = batch_user_indices.to(self.device)
                batch_positive_item_indices = batch_positive_item_indices.to(self.device)
                batch_negative_item_indices = batch_negative_item_indices.to(self.device)

                optimizer.zero_grad()
                positive_scores = model(batch_user_indices, batch_positive_item_indices)
                negative_scores = model(batch_user_indices, batch_negative_item_indices)
                loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()
                loss.backward()
                optimizer.step()

                batch_size = int(batch_user_indices.size(0))
                total_loss += float(loss.item()) * batch_size
                total_examples += batch_size

            mean_loss = total_loss / max(total_examples, 1)
            self.train_loss_history_.append(mean_loss)

    def _build_pointwise_training_samples(
        self,
        prepared: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        n_items = len(self.item_to_index_)
        all_item_indices = np.arange(n_items, dtype=np.int64)

        positive_user_indices = prepared[self.user_col].map(self.user_to_index_).to_numpy(dtype=np.int64)
        positive_item_indices = prepared[self.item_col].map(self.item_to_index_).to_numpy(dtype=np.int64)

        negative_user_indices: list[int] = []
        negative_item_indices: list[int] = []

        seen_item_indices_by_user = {
            int(user_id): {self.item_to_index_[item_id] for item_id in item_ids}
            for user_id, item_ids in self.seen_items_by_user_.items()
        }

        for user_index, item_index in zip(positive_user_indices, positive_item_indices, strict=False):
            user_id = self.index_to_user_[int(user_index)]
            seen_item_indices = seen_item_indices_by_user[int(user_id)]
            for _ in range(self.num_negatives):
                sampled_item_index = int(rng.choice(all_item_indices))
                while sampled_item_index in seen_item_indices:
                    sampled_item_index = int(rng.choice(all_item_indices))
                negative_user_indices.append(int(user_index))
                negative_item_indices.append(sampled_item_index)

        user_indices = np.concatenate(
            [
                positive_user_indices,
                np.asarray(negative_user_indices, dtype=np.int64),
            ]
        )
        item_indices = np.concatenate(
            [
                positive_item_indices,
                np.asarray(negative_item_indices, dtype=np.int64),
            ]
        )
        labels = np.concatenate(
            [
                np.ones(len(positive_user_indices), dtype=np.float32),
                np.zeros(len(negative_user_indices), dtype=np.float32),
            ]
        )
        return user_indices, item_indices, labels

    def _build_pairwise_training_samples(
        self,
        prepared: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        n_items = len(self.item_to_index_)
        all_item_indices = np.arange(n_items, dtype=np.int64)

        positive_user_indices = prepared[self.user_col].map(self.user_to_index_).to_numpy(dtype=np.int64)
        positive_item_indices = prepared[self.item_col].map(self.item_to_index_).to_numpy(dtype=np.int64)

        sampled_user_indices: list[int] = []
        sampled_positive_item_indices: list[int] = []
        sampled_negative_item_indices: list[int] = []

        seen_item_indices_by_user = {
            int(user_id): {self.item_to_index_[item_id] for item_id in item_ids}
            for user_id, item_ids in self.seen_items_by_user_.items()
        }

        for user_index, positive_item_index in zip(
            positive_user_indices,
            positive_item_indices,
            strict=False,
        ):
            user_id = self.index_to_user_[int(user_index)]
            seen_item_indices = seen_item_indices_by_user[int(user_id)]
            for _ in range(self.num_negatives):
                sampled_negative_item_index = int(rng.choice(all_item_indices))
                while sampled_negative_item_index in seen_item_indices:
                    sampled_negative_item_index = int(rng.choice(all_item_indices))
                sampled_user_indices.append(int(user_index))
                sampled_positive_item_indices.append(int(positive_item_index))
                sampled_negative_item_indices.append(sampled_negative_item_index)

        return (
            np.asarray(sampled_user_indices, dtype=np.int64),
            np.asarray(sampled_positive_item_indices, dtype=np.int64),
            np.asarray(sampled_negative_item_indices, dtype=np.int64),
        )

    def _require_torch(self) -> None:
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:
            raise ModuleNotFoundError(
                "Библиотека `torch` не установлена. Установите её в окружение проекта "
                "и перезапустите notebook."
            ) from TORCH_IMPORT_ERROR

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_ or self.model_ is None:
            raise ValueError("Сначала вызовите fit()")

    def score_items(self, user_id: int) -> pd.Series:
        self._check_is_fitted()
        if int(user_id) not in self.user_to_index_:
            raise KeyError(f"user_id={user_id} отсутствует в fitted NeuMF model")

        user_index = self.user_to_index_[int(user_id)]
        item_indices = np.arange(len(self.item_to_index_), dtype=np.int64)
        chunk_size = 4096
        score_chunks: list[np.ndarray] = []

        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(item_indices), chunk_size):
                stop = start + chunk_size
                chunk_item_indices = item_indices[start:stop]
                chunk_user_indices = np.full(len(chunk_item_indices), user_index, dtype=np.int64)
                logits = self.model_(
                    torch.tensor(chunk_user_indices, dtype=torch.long, device=self.device),
                    torch.tensor(chunk_item_indices, dtype=torch.long, device=self.device),
                )
                score_chunks.append(logits.detach().cpu().numpy())

        scores = np.concatenate(score_chunks)
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


NeuMFRecommender = NeuralCFRecommender
NeuralCollaborativeFilteringRecommender = NeuralCFRecommender


__all__ = [
    "NeuralCFRecommender",
    "NeuMFRecommender",
    "NeuralCollaborativeFilteringRecommender",
]
