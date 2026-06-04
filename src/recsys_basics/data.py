"""Утилиты для загрузки и базовой подготовки учебных датасетов."""

from __future__ import annotations

import json
import shutil
import ssl
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

try:
    import certifi
except ModuleNotFoundError:  # pragma: no cover - зависит от локального окружения
    certifi = None

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:  # pragma: no cover - зависит от локального окружения
    load_dataset = None
    DATASETS_IMPORT_ERROR = exc
else:  # pragma: no cover - ветка зависит от локального окружения
    DATASETS_IMPORT_ERROR = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOVIELENS_DIR = PROJECT_ROOT / "data" / "raw" / "movielens"
DEFAULT_AMAZON_REVIEWS_2023_DIR = PROJECT_ROOT / "data" / "raw" / "amazon_reviews_2023"
DEFAULT_RETAILROCKET_DIR = PROJECT_ROOT / "data" / "raw" / "retailrocket"
RETAILROCKET_DOWNLOAD_BASE_URL = (
    "https://huggingface.co/datasets/DanielKiani/RetailRocket-Recommender-Data"
    "/resolve/main/data/RetailRocket-Recommender-Data/data"
)
REQUIRED_MOVIELENS_FILES = ("ratings.csv", "movies.csv")
REQUIRED_RETAILROCKET_FILES = (
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
)


@dataclass(frozen=True)
class MovieLensFrames:
    """Таблицы MovieLens latest small в исходном виде."""

    ratings: pd.DataFrame
    movies: pd.DataFrame
    links: pd.DataFrame | None = None
    tags: pd.DataFrame | None = None


@dataclass(frozen=True)
class AmazonReviews2023Frames:
    """Таблицы Amazon Reviews 2023 для одной категории."""

    reviews: pd.DataFrame
    metadata: pd.DataFrame
    category: str


@dataclass(frozen=True)
class RetailrocketFrames:
    """Таблицы Retailrocket ecommerce dataset."""

    events: pd.DataFrame
    item_properties_part1: pd.DataFrame
    item_properties_part2: pd.DataFrame
    category_tree: pd.DataFrame


def get_project_root() -> Path:
    """Возвращает корень проекта."""

    return PROJECT_ROOT


def get_movielens_data_dir(data_dir: str | Path | None = None) -> Path:
    """Возвращает директорию с файлами MovieLens."""

    if data_dir is None:
        return DEFAULT_MOVIELENS_DIR
    return Path(data_dir).expanduser().resolve()


def get_amazon_reviews_2023_data_dir(data_dir: str | Path | None = None) -> Path:
    """Возвращает директорию с raw-файлами Amazon Reviews 2023."""

    if data_dir is None:
        return DEFAULT_AMAZON_REVIEWS_2023_DIR
    return Path(data_dir).expanduser().resolve()


def get_retailrocket_data_dir(data_dir: str | Path | None = None) -> Path:
    """Возвращает директорию с raw-файлами Retailrocket."""

    if data_dir is None:
        return DEFAULT_RETAILROCKET_DIR
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


def expected_amazon_reviews_2023_layout(
    category: str = "All_Beauty",
    data_dir: str | Path | None = None,
) -> str:
    """Текстовая подсказка по ожидаемой раскладке Amazon Reviews 2023."""

    data_path = get_amazon_reviews_2023_data_dir(data_dir)
    return "\n".join(
        [
            f"Ожидаемая директория: {data_path}",
            "Внутри должны лежать локально сохранённые файлы категории:",
            f"- {category}.jsonl",
            f"- meta_{category}.jsonl",
            "Допустимый альтернативный вариант:",
            f"- {category}.parquet",
            f"- meta_{category}.parquet",
        ]
    )


def expected_retailrocket_layout(data_dir: str | Path | None = None) -> str:
    """Текстовая подсказка по ожидаемой раскладке Retailrocket."""

    data_path = get_retailrocket_data_dir(data_dir)
    return "\n".join(
        [
            f"Ожидаемая директория: {data_path}",
            "Внутри должны лежать файлы Retailrocket:",
            "- events.csv",
            "- item_properties_part1.csv",
            "- item_properties_part2.csv",
            "- category_tree.csv",
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


def ensure_retailrocket_files(data_dir: str | Path | None = None) -> Path:
    """Проверяет, что Retailrocket сохранён локально."""

    data_path = get_retailrocket_data_dir(data_dir)
    missing_files = find_missing_files(data_path, REQUIRED_RETAILROCKET_FILES)

    if missing_files:
        missing = ", ".join(sorted(missing_files))
        raise FileNotFoundError(
            "Retailrocket ecommerce dataset не найден локально. "
            f"Не хватает файлов: {missing}\n{expected_retailrocket_layout(data_path)}"
        )

    return data_path


def get_retailrocket_download_urls() -> dict[str, str]:
    """Возвращает публичные URL raw CSV-файлов Retailrocket."""

    return {
        filename: f"{RETAILROCKET_DOWNLOAD_BASE_URL}/{filename}"
        for filename in REQUIRED_RETAILROCKET_FILES
    }


def download_retailrocket_files(
    data_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Один раз скачивает raw CSV-файлы Retailrocket и сохраняет их в `data/raw`."""

    data_path = get_retailrocket_data_dir(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if not overwrite and not find_missing_files(data_path, REQUIRED_RETAILROCKET_FILES):
        return data_path

    ssl_context = (
        ssl.create_default_context(cafile=certifi.where())
        if certifi is not None
        else ssl.create_default_context()
    )
    for filename, url in get_retailrocket_download_urls().items():
        destination_path = data_path / filename
        if destination_path.exists() and not overwrite:
            continue

        temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
        try:
            with urlopen(url, context=ssl_context) as response, temporary_path.open("wb") as target_file:
                shutil.copyfileobj(response, target_file)
            temporary_path.replace(destination_path)
        except HTTPError as exc:
            if temporary_path.exists():
                temporary_path.unlink()
            raise HTTPError(
                url=exc.url,
                code=exc.code,
                msg=(
                    f"Не удалось скачать Retailrocket: HTTP {exc.code} для URL {url}. "
                    "Проверьте актуальность upstream-ссылки."
                ),
                hdrs=exc.headers,
                fp=exc.fp,
            ) from exc
        except URLError as exc:
            if temporary_path.exists():
                temporary_path.unlink()
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise URLError(
                    "Не удалось проверить SSL-сертификат при скачивании Retailrocket. "
                    "Проверьте системные сертификаты Python или окружение с `certifi`."
                ) from exc
            raise

    return data_path


def load_retailrocket_frames(data_dir: str | Path | None = None) -> RetailrocketFrames:
    """Загружает основные таблицы Retailrocket ecommerce dataset."""

    data_path = ensure_retailrocket_files(data_dir)
    return RetailrocketFrames(
        events=pd.read_csv(data_path / "events.csv"),
        item_properties_part1=pd.read_csv(data_path / "item_properties_part1.csv"),
        item_properties_part2=pd.read_csv(data_path / "item_properties_part2.csv"),
        category_tree=pd.read_csv(data_path / "category_tree.csv"),
    )


def build_retailrocket_interactions(events: pd.DataFrame) -> pd.DataFrame:
    """Приводит `events.csv` Retailrocket к канонической interaction table."""

    required_columns = {"visitorid", "itemid", "event", "timestamp"}
    missing_columns = required_columns.difference(events.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"В Retailrocket events не хватает колонок: {missing}")

    interactions = (
        events.loc[:, ["visitorid", "itemid", "event", "timestamp"]]
        .rename(columns={"visitorid": "user_id", "itemid": "item_id"})
        .copy()
    )
    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"], unit="ms", utc=True, errors="coerce"
    )
    interactions = interactions.dropna(subset=["timestamp"]).reset_index(drop=True)
    interactions = interactions.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    return interactions


def ensure_amazon_reviews_2023_files(
    category: str = "All_Beauty",
    data_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Проверяет наличие локально сохранённых файлов Amazon Reviews 2023."""

    data_path = get_amazon_reviews_2023_data_dir(data_dir)
    parquet_pair = (
        data_path / f"{category}.parquet",
        data_path / f"meta_{category}.parquet",
    )
    jsonl_pair = (
        data_path / f"{category}.jsonl",
        data_path / f"meta_{category}.jsonl",
    )

    for review_path, metadata_path in (jsonl_pair, parquet_pair):
        if review_path.exists() and metadata_path.exists():
            return review_path, metadata_path

    missing = ", ".join(
        sorted(
            {
                jsonl_pair[0].name,
                jsonl_pair[1].name,
            }
        )
    )
    raise FileNotFoundError(
        "Amazon Reviews 2023 не найден локально. "
        f"Не хватает файлов: {missing}\n{expected_amazon_reviews_2023_layout(category, data_path)}"
    )


def get_amazon_reviews_2023_download_urls(category: str = "All_Beauty") -> tuple[str, str]:
    """Возвращает официальные raw `jsonl.gz` URL для одной категории Amazon Reviews 2023."""

    base_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
    review_url = f"{base_url}/review_categories/{category}.jsonl.gz"
    metadata_url = f"{base_url}/meta_categories/meta_{category}.jsonl.gz"
    return review_url, metadata_url


def download_amazon_reviews_2023_files(
    category: str = "All_Beauty",
    data_dir: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Один раз скачивает raw `jsonl.gz` Amazon Reviews 2023 и сохраняет локальные `.jsonl`."""

    data_path = get_amazon_reviews_2023_data_dir(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    review_path = data_path / f"{category}.jsonl"
    metadata_path = data_path / f"meta_{category}.jsonl"
    if not overwrite and review_path.exists() and metadata_path.exists():
        return review_path, metadata_path

    review_url, metadata_url = get_amazon_reviews_2023_download_urls(category=category)
    ssl_context = (
        ssl.create_default_context(cafile=certifi.where())
        if certifi is not None
        else ssl.create_default_context()
    )
    for url, destination_path in (
        (review_url, review_path),
        (metadata_url, metadata_path),
    ):
        if destination_path.exists() and not overwrite:
            continue
        try:
            with urlopen(url, context=ssl_context) as response, gzip.GzipFile(fileobj=response) as gz_file, destination_path.open("wb") as target_file:
                shutil.copyfileobj(gz_file, target_file)
        except HTTPError as exc:
            raise HTTPError(
                url=exc.url,
                code=exc.code,
                msg=(
                    f"Не удалось скачать Amazon Reviews 2023: HTTP {exc.code} для URL {url}. "
                    "Проверьте имя категории или актуальность upstream-ссылки."
                ),
                hdrs=exc.headers,
                fp=exc.fp,
            ) from exc
        except URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise URLError(
                    "Не удалось проверить SSL-сертификат при скачивании Amazon Reviews 2023. "
                    "Проверьте системные сертификаты Python. На macOS обычно помогает запуск "
                    "`Install Certificates.command`, либо использование окружения с `certifi`."
                ) from exc
            raise

    return review_path, metadata_path


def _read_jsonl(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Читает jsonl-файл в DataFrame."""

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file):
            if max_rows is not None and line_index >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_amazon_reviews_2023_frames(
    category: str = "All_Beauty",
    data_dir: str | Path | None = None,
    max_review_rows: int | None = None,
    max_metadata_rows: int | None = None,
) -> AmazonReviews2023Frames:
    """Загружает локально сохранённые reviews и metadata для одной категории Amazon Reviews 2023."""

    review_path, metadata_path = ensure_amazon_reviews_2023_files(category=category, data_dir=data_dir)
    if review_path.suffix == ".parquet":
        reviews = pd.read_parquet(review_path)
    else:
        reviews = _read_jsonl(review_path, max_rows=max_review_rows)

    if metadata_path.suffix == ".parquet":
        metadata = pd.read_parquet(metadata_path)
    else:
        metadata = _read_jsonl(metadata_path, max_rows=max_metadata_rows)

    if max_review_rows is not None:
        reviews = reviews.head(max_review_rows).copy()
    if max_metadata_rows is not None:
        metadata = metadata.head(max_metadata_rows).copy()
    return AmazonReviews2023Frames(reviews=reviews, metadata=metadata, category=category)


def load_amazon_reviews_2023_frames_hf(
    category: str = "All_Beauty",
    review_split: str = "full",
    metadata_split: str = "full",
) -> AmazonReviews2023Frames:
    """Загружает Amazon Reviews 2023 через Hugging Face `datasets`."""

    if load_dataset is None:
        raise ModuleNotFoundError(
            "Библиотека `datasets` не установлена. Установите её (`pip install datasets`) "
            "и перезапустите notebook."
        ) from DATASETS_IMPORT_ERROR

    base_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
    review_url = f"{base_url}/raw_review_{category}/full-00000-of-00001.parquet"
    metadata_url = f"{base_url}/raw_meta_{category}/full-00000-of-00001.parquet"

    review_dataset = load_dataset(
        "parquet",
        data_files={review_split: review_url},
        split=review_split,
    )
    metadata_dataset = load_dataset(
        "parquet",
        data_files={metadata_split: metadata_url},
        split=metadata_split,
    )

    return AmazonReviews2023Frames(
        reviews=review_dataset.to_pandas(),
        metadata=metadata_dataset.to_pandas(),
        category=category,
    )


def build_amazon_reviews_interactions(reviews: pd.DataFrame) -> pd.DataFrame:
    """Приводит raw reviews Amazon Reviews 2023 к канонической interaction table."""

    required_columns = {"user_id", "parent_asin", "rating", "timestamp"}
    missing_columns = required_columns.difference(reviews.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"В reviews не хватает колонок: {missing}")

    interactions = (
        reviews.loc[:, ["user_id", "parent_asin", "rating", "timestamp"]]
        .rename(columns={"parent_asin": "item_id"})
        .copy()
    )
    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"], unit="ms", utc=True, errors="coerce"
    )
    interactions = interactions.dropna(subset=["timestamp"]).reset_index(drop=True)
    interactions = interactions.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    return interactions


def prepare_amazon_item_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Готовит компактную item-таблицу для text-rich retrieval экспериментов."""

    required_columns = {"parent_asin", "title"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"В metadata не хватает колонок: {missing}")

    prepared = metadata.copy()
    prepared = prepared.rename(columns={"parent_asin": "item_id"})
    for column in ("features", "description", "categories"):
        if column not in prepared.columns:
            prepared[column] = [[] for _ in range(len(prepared))]
    if "store" not in prepared.columns:
        prepared["store"] = ""

    def flatten_text(value: object) -> str:
        if isinstance(value, list):
            return " ".join(str(part) for part in value if str(part).strip())
        if pd.isna(value):
            return ""
        return str(value)

    prepared["title_text"] = prepared["title"].fillna("").astype(str)
    prepared["feature_text"] = prepared["features"].map(flatten_text)
    prepared["description_text"] = prepared["description"].map(flatten_text)
    prepared["category_text"] = prepared["categories"].map(flatten_text)
    prepared["store_text"] = prepared["store"].fillna("").astype(str)
    prepared["item_text"] = (
        prepared["title_text"]
        + " "
        + prepared["store_text"]
        + " "
        + prepared["category_text"]
        + " "
        + prepared["feature_text"]
        + " "
        + prepared["description_text"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    columns = [
        "item_id",
        "title",
        "store",
        "categories",
        "features",
        "description",
        "item_text",
    ]
    return prepared.loc[:, columns].drop_duplicates(subset=["item_id"]).reset_index(drop=True)


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
    "AmazonReviews2023Frames",
    "DEFAULT_AMAZON_REVIEWS_2023_DIR",
    "DEFAULT_MOVIELENS_DIR",
    "DEFAULT_RETAILROCKET_DIR",
    "REQUIRED_MOVIELENS_FILES",
    "REQUIRED_RETAILROCKET_FILES",
    "MovieLensFrames",
    "RetailrocketFrames",
    "build_amazon_reviews_interactions",
    "build_explicit_interactions",
    "build_retailrocket_interactions",
    "download_amazon_reviews_2023_files",
    "download_retailrocket_files",
    "ensure_amazon_reviews_2023_files",
    "ensure_movielens_files",
    "ensure_retailrocket_files",
    "expected_amazon_reviews_2023_layout",
    "expected_movielens_layout",
    "expected_retailrocket_layout",
    "find_missing_files",
    "get_amazon_reviews_2023_download_urls",
    "get_amazon_reviews_2023_data_dir",
    "get_movielens_data_dir",
    "get_project_root",
    "get_retailrocket_download_urls",
    "get_retailrocket_data_dir",
    "load_amazon_reviews_2023_frames",
    "load_amazon_reviews_2023_frames_hf",
    "load_movielens_frames",
    "load_retailrocket_frames",
    "prepare_amazon_item_metadata",
    "prepare_movielens_movies",
]
