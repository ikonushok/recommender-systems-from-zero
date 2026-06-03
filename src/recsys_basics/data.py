"""Утилиты для загрузки и базовой подготовки учебных датасетов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOVIELENS_DIR = PROJECT_ROOT / "data" / "raw" / "movielens"
REQUIRED_MOVIELENS_FILES = ("ratings.csv", "movies.csv")


@dataclass(frozen=True)
class MovieLensFrames:
    """Таблицы MovieLens latest small в исходном виде."""

    ratings: pd.DataFrame
    movies: pd.DataFrame
    links: pd.DataFrame | None = None
    tags: pd.DataFrame | None = None


def get_project_root() -> Path:
    """Возвращает корень проекта."""

    return PROJECT_ROOT


def get_movielens_data_dir(data_dir: str | Path | None = None) -> Path:
    """Возвращает директорию с файлами MovieLens."""

    if data_dir is None:
        return DEFAULT_MOVIELENS_DIR
    return Path(data_dir).expanduser().resolve()


def expected_movielens_layout(data_dir: str | Path | None = None) -> str:
    """Текстовая подсказка по ожидаемой раскладке MovieLens."""

    data_path = get_movielens_data_dir(data_dir)
    return "\n".join(
        [
            f"Ожидаемая директория: {data_path}",
            "Внутри должны лежать файлы:",
            "- ratings.csv",
            "- movies.csv",
            "- links.csv",
            "- tags.csv",
        ]
    )


def find_missing_files(data_dir: str | Path, filenames: Iterable[str]) -> list[str]:
    """Проверяет, каких файлов не хватает в директории."""

    data_path = Path(data_dir)
    return [filename for filename in filenames if not (data_path / filename).exists()]


def ensure_movielens_files(
    data_dir: str | Path | None = None,
    required_files: Iterable[str] = REQUIRED_MOVIELENS_FILES,
) -> Path:
    """Проверяет, что в директории есть обязательные файлы MovieLens."""

    data_path = get_movielens_data_dir(data_dir)
    missing_files = find_missing_files(data_path, required_files)

    if missing_files:
        missing = ", ".join(sorted(missing_files))
        raise FileNotFoundError(
            "MovieLens latest small не найден. "
            f"Не хватает файлов: {missing}\n{expected_movielens_layout(data_path)}"
        )

    return data_path


def load_movielens_frames(data_dir: str | Path | None = None) -> MovieLensFrames:
    """Загружает основные таблицы MovieLens latest small."""

    data_path = ensure_movielens_files(data_dir)

    ratings = pd.read_csv(data_path / "ratings.csv")
    movies = pd.read_csv(data_path / "movies.csv")

    links_path = data_path / "links.csv"
    tags_path = data_path / "tags.csv"

    links = pd.read_csv(links_path) if links_path.exists() else None
    tags = pd.read_csv(tags_path) if tags_path.exists() else None

    return MovieLensFrames(
        ratings=ratings,
        movies=movies,
        links=links,
        tags=tags,
    )


def build_explicit_interactions(ratings: pd.DataFrame) -> pd.DataFrame:
    """Приводит `ratings.csv` к канонической схеме interaction table."""

    required_columns = {"userId", "movieId", "rating", "timestamp"}
    missing_columns = required_columns.difference(ratings.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"В ratings не хватает колонок: {missing}")

    interactions = (
        ratings.loc[:, ["userId", "movieId", "rating", "timestamp"]]
        .rename(columns={"userId": "user_id", "movieId": "item_id"})
        .copy()
    )
    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"], unit="s", utc=True
    )
    interactions = interactions.sort_values(["timestamp", "user_id", "item_id"]).reset_index(
        drop=True
    )
    return interactions


def prepare_movielens_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к таблице фильмов простые учебные признаки."""

    required_columns = {"movieId", "title", "genres"}
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"В movies не хватает колонок: {missing}")

    prepared = movies.rename(columns={"movieId": "item_id"}).copy()
    prepared["title_year"] = prepared["title"].str.extract(r"\((\d{4})\)$")
    prepared["title_year"] = pd.to_numeric(prepared["title_year"], errors="coerce")
    prepared["title_year"] = prepared["title_year"].astype("Int64")
    prepared["genre_list"] = prepared["genres"].fillna("").str.split("|")
    prepared["genre_count"] = prepared["genre_list"].map(
        lambda genres: sum(genre != "(no genres listed)" and genre != "" for genre in genres)
    )
    return prepared


__all__ = [
    "DEFAULT_MOVIELENS_DIR",
    "REQUIRED_MOVIELENS_FILES",
    "MovieLensFrames",
    "build_explicit_interactions",
    "ensure_movielens_files",
    "expected_movielens_layout",
    "find_missing_files",
    "get_movielens_data_dir",
    "get_project_root",
    "load_movielens_frames",
    "prepare_movielens_movies",
]
