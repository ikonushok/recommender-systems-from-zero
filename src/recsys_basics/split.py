"""Учебные функции для train/test split без leakage."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    """Результат разбиения interactions на train и test."""

    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class TrainValidationTestSplitResult:
    """Результат разбиения interactions на train, validation и test."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def leave_last_one_out_split(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    min_user_interactions: int = 2,
) -> SplitResult:
    """Для каждого пользователя оставляет последнее взаимодействие в test."""

    required_columns = {user_col, item_col, time_col}
    missing_columns = required_columns.difference(interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Для split не хватает колонок: {missing}")

    if min_user_interactions < 2:
        raise ValueError("min_user_interactions должен быть не меньше 2")

    ordered = interactions.sort_values([user_col, time_col, item_col]).reset_index(drop=True)
    user_activity = ordered.groupby(user_col).size()
    eligible_users = user_activity[user_activity >= min_user_interactions].index

    filtered = ordered[ordered[user_col].isin(eligible_users)].copy()
    test = filtered.groupby(user_col, group_keys=False).tail(1).copy()
    train = filtered.drop(index=test.index).copy()

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return SplitResult(train=train, test=test)


def leave_last_two_out_split(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    min_user_interactions: int = 3,
) -> TrainValidationTestSplitResult:
    """Для каждого пользователя оставляет предпоследнее взаимодействие в validation и последнее в test."""

    required_columns = {user_col, item_col, time_col}
    missing_columns = required_columns.difference(interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Для split не хватает колонок: {missing}")

    if min_user_interactions < 3:
        raise ValueError("min_user_interactions должен быть не меньше 3")

    ordered = interactions.sort_values([user_col, time_col, item_col]).reset_index(drop=True)
    user_activity = ordered.groupby(user_col).size()
    eligible_users = user_activity[user_activity >= min_user_interactions].index

    filtered = ordered[ordered[user_col].isin(eligible_users)].copy()
    test = filtered.groupby(user_col, group_keys=False).tail(1).copy()
    remaining = filtered.drop(index=test.index).copy()
    validation = remaining.groupby(user_col, group_keys=False).tail(1).copy()
    train = remaining.drop(index=validation.index).copy()

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return TrainValidationTestSplitResult(train=train, validation=validation, test=test)


def assert_no_user_time_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "timestamp",
) -> None:
    """Проверяет, что для каждого пользователя test не раньше train."""

    required_train = {user_col, time_col}.difference(train.columns)
    required_test = {user_col, time_col}.difference(test.columns)
    if required_train or required_test:
        missing = ", ".join(sorted(required_train.union(required_test)))
        raise ValueError(f"Для leakage-check не хватает колонок: {missing}")

    train_max = train.groupby(user_col)[time_col].max()
    test_min = test.groupby(user_col)[time_col].min()
    shared_users = train_max.index.intersection(test_min.index)

    if shared_users.empty:
        return

    invalid_users = shared_users[train_max.loc[shared_users] > test_min.loc[shared_users]]
    if len(invalid_users) > 0:
        preview = ", ".join(map(str, invalid_users[:5]))
        raise ValueError(
            "Обнаружено leakage по времени: для некоторых пользователей train позже test. "
            f"Примеры user_id: {preview}"
        )


__all__ = [
    "SplitResult",
    "TrainValidationTestSplitResult",
    "assert_no_user_time_leakage",
    "leave_last_one_out_split",
    "leave_last_two_out_split",
]
