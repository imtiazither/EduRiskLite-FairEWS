"""Helpers for loading and formatting the student success dataset."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATH_CANDIDATES, POSITIVE_CLASS, TARGET_COLUMN


def resolve_data_path(data_path: Path | None = None) -> Path:
    """Return the first available dataset path."""
    if data_path is not None:
        return Path(data_path)

    for candidate in DATA_PATH_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the dataset. Expected one of: "
        + ", ".join(str(path) for path in DATA_PATH_CANDIDATES)
    )


def clean_column_name(column_name: str) -> str:
    """Convert raw dataset headers into predictable snake_case names."""
    cleaned = column_name.replace("\ufeff", "").replace("'", "").strip().lower()
    cleaned = cleaned.replace("/", " ")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def format_feature_name(column_name: str) -> str:
    """Turn a snake_case feature name into a more readable label."""
    words = column_name.replace("_", " ").split()
    formatted_words = []
    for word in words:
        if word.lower() == "gdp":
            formatted_words.append("GDP")
        else:
            formatted_words.append(word.capitalize())

    readable_name = " ".join(formatted_words)
    readable_name = readable_name.replace("1St", "1st").replace("2Nd", "2nd")
    return readable_name


def load_dataset(data_path: Path | None = None) -> pd.DataFrame:
    """Load the dataset and attach a synthetic record identifier."""
    csv_path = resolve_data_path(data_path)
    dataset = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    dataset.columns = [clean_column_name(column_name) for column_name in dataset.columns]
    dataset = dataset.dropna(how="all").reset_index(drop=True)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Expected a '{TARGET_COLUMN}' column in {csv_path}.")

    if "student_id" not in dataset.columns:
        dataset.insert(0, "student_id", np.arange(1, len(dataset) + 1))

    return dataset


def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return the feature matrix, binary target, and feature column names."""
    feature_columns = [
        column_name
        for column_name in dataset.columns
        if column_name not in {"student_id", TARGET_COLUMN}
    ]
    features = dataset[feature_columns].copy()
    target = (dataset[TARGET_COLUMN] == POSITIVE_CLASS).astype(int)
    return features, target, feature_columns

