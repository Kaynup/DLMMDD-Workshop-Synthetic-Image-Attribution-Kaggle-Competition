from __future__ import annotations

from lib.core.logging import inference_logger
from lib.inference.ensemble import run_ensemble_inference


def run_inference_pipeline(train_df, test_df, y_train, fold_metadata):
    inference_logger.info("RUN MODE — INFERENCE ONLY")
    return run_ensemble_inference(train_df, test_df, y_train, fold_metadata)
