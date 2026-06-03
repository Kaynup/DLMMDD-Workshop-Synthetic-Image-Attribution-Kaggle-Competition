#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from lib.config.defaults import NUM_FOLDS, RUN_INFERENCE_ONLY, SEED
from lib.config.paths import LOG_DIR, PROCESSED_DIR, RUN_NAME
from lib.core.logging import initialize_logger_safely, pipeline_logger
from lib.core.seed import set_seed
from lib.core.resources import log_resources
from lib.data.loader import load_raw_csvs
from lib.data.validation import validate_data
from lib.data.folds import build_folds
from lib.pipelines.train_pipeline import run_training_pipeline
from lib.pipelines.inference_pipeline import run_inference_pipeline
from lib.pipelines.submission_pipeline import write_submission


def run_pipeline() -> None:
    initialize_logger_safely(LOG_DIR, run_id=RUN_NAME)
    pipeline_logger.info("PIPELINE START")
    set_seed(SEED)
    log_resources("startup")
    train_df, test_df = load_raw_csvs()
    validate_data(train_df, test_df)
    y_train = train_df["y"].to_numpy(dtype=int)
    fold_metadata = build_folds(y_train, NUM_FOLDS, SEED)
    if RUN_INFERENCE_ONLY:
        result = run_inference_pipeline(train_df, test_df, y_train, fold_metadata)
    else:
        result = run_training_pipeline(train_df, test_df, y_train, fold_metadata)
    if result is not None:
        _, preds, conf, test_meta, num_models = result
        write_submission(test_meta=test_meta, preds=preds, conf=conf, num_models=num_models)
    pipeline_logger.info("[pipeline] completed")


if __name__ == "__main__":
    run_pipeline()
