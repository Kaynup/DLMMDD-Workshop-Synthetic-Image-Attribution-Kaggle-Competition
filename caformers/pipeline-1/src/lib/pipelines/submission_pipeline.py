from __future__ import annotations

import numpy as np
import pandas as pd

from lib.config.defaults import BLEND_MODE, USE_TTA, TTA_N
from lib.config.paths import INFERENCE_DIR, WORKING_ROOT
from lib.core.logging import pipeline_logger
from lib.artifacts.metadata import build_submission_metadata
from lib.core.utils import save_json


def write_submission(test_meta: pd.DataFrame, preds: np.ndarray, conf: np.ndarray, num_models: int) -> pd.DataFrame:
    if len(test_meta) != len(preds):
        raise ValueError(
            f"Prediction length mismatch: len(test_meta)={len(test_meta)} vs len(preds)={len(preds)}"
        )
    if len(conf) != len(preds):
        raise ValueError(
            f"Confidence length mismatch: len(conf)={len(conf)} vs len(preds)={len(preds)}"
        )
    if len(preds) == 0:
        raise RuntimeError("No predictions generated.")
    preds = np.asarray(preds, dtype=np.int64)
    conf = np.asarray(conf, dtype=np.float32)
    submission = test_meta[["ID"]].copy()
    submission["TARGET"] = preds
    submission_root_path = WORKING_ROOT / "submission.csv"
    submission_infer_path = INFERENCE_DIR / "submission.csv"
    confidence_path = INFERENCE_DIR / "prediction_confidence.csv"
    metadata_path = INFERENCE_DIR / "submission_metadata.json"
    submission.to_csv(submission_root_path, index=False)
    submission.to_csv(submission_infer_path, index=False)
    confidence_df = pd.DataFrame(
        {"ID": test_meta["ID"].values, "predicted_class": preds, "confidence": conf}
    )
    confidence_df.to_csv(confidence_path, index=False)
    pred_distribution = pd.Series(preds).value_counts().sort_index()
    metadata = build_submission_metadata(preds, conf, num_models)
    save_json(metadata, metadata_path)
    pipeline_logger.info(f"\n{'=' * 48}\nSubmission preview:\n{submission.head(10).to_string(index=False)}")
    pipeline_logger.info(f"[submission] saved={submission_root_path}")
    pipeline_logger.info(f"[confidence] saved={confidence_path}")
    pipeline_logger.info(f"[metadata] saved={metadata_path}")
    pipeline_logger.info(f"Prediction distribution:\n{pred_distribution.to_string()}")
    pipeline_logger.info(
        f"Confidence stats | mean={conf.mean():.4f} | std={conf.std():.4f} | "
        f"min={conf.min():.4f} | max={conf.max():.4f}"
    )
    return submission
