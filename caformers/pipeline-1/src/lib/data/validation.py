from __future__ import annotations

from pathlib import Path

import pandas as pd
from lib.config.defaults import NUM_CLASSES
from lib.core.logging import data_logger


def validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    data_logger.info(f"train_rows={len(train_df):,} | test_rows={len(test_df):,}")
    required_train_cols = {"ID", "path", "y"}
    required_test_cols = {"ID", "path"}
    missing_train = required_train_cols - set(train_df.columns)
    missing_test = required_test_cols - set(test_df.columns)
    if missing_train:
        raise ValueError(f"Missing train columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Missing test columns: {sorted(missing_test)}")
    if train_df[["ID", "path", "y"]].isnull().any().any():
        raise ValueError("NaNs detected in training metadata")
    if test_df[["ID", "path"]].isnull().any().any():
        raise ValueError("NaNs detected in test metadata")
    train_dup_ids = train_df["ID"].duplicated().sum()
    test_dup_ids = test_df["ID"].duplicated().sum()
    if train_dup_ids:
        data_logger.warning(f"Duplicate train IDs detected: {train_dup_ids}")
    if test_dup_ids:
        data_logger.warning(f"Duplicate test IDs detected: {test_dup_ids}")
    invalid_labels = train_df[~train_df["y"].between(0, NUM_CLASSES - 1)]
    if len(invalid_labels):
        raise ValueError(
            f"Invalid labels detected: {sorted(invalid_labels['y'].unique().tolist())}"
        )
    counts = train_df["y"].value_counts().sort_index()
    data_logger.info("Class distribution:\n" + counts.to_string())
    data_logger.info("Class ratios:\n" + (counts / counts.sum()).round(4).to_string())
    if len(counts) != NUM_CLASSES:
        data_logger.warning(f"Expected {NUM_CLASSES} classes; found {len(counts)}")
    train_exists = train_df["full_path"].map(lambda p: Path(str(p)).exists())
    test_exists = test_df["full_path"].map(lambda p: Path(str(p)).exists())
    missing_train = int((~train_exists).sum())
    missing_test = int((~test_exists).sum())
    data_logger.info(f"Missing train images: {missing_train}")
    data_logger.info(f"Missing test images: {missing_test}")
    if missing_train:
        missing_examples = train_df.loc[~train_exists, "full_path"].head(5).tolist()
        raise FileNotFoundError(
            f"{missing_train} training images missing. Examples: {missing_examples}"
        )
    if missing_test:
        missing_examples = test_df.loc[~test_exists, "full_path"].head(5).tolist()
        raise FileNotFoundError(
            f"{missing_test} test images missing. Examples: {missing_examples}"
        )
    data_logger.info("Data validation passed ✓")
