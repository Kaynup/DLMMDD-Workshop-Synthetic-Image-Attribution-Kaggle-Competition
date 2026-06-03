from __future__ import annotations

import os
import time
from pathlib import Path

from lib.core.utils import ensure_dir

COMPETITION_ROOT = Path(
    os.environ.get(
        "COMPETITION_ROOT",
        "/kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge",
    )
)

DATA_ROOT = COMPETITION_ROOT / "Data" / "Data"
TRAIN_CSV = DATA_ROOT / "training.csv"
TEST_CSV = DATA_ROOT / "test.csv"
TRAIN_DIR = DATA_ROOT / "Training"
TEST_DIR = DATA_ROOT / "Test"

WORKING_ROOT = ensure_dir(
    Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
)

RUN_NAME = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = ensure_dir(WORKING_ROOT / RUN_NAME)

ARTIFACTS_DIR = ensure_dir(RUN_DIR / "artifacts")
PROCESSED_DIR = ensure_dir(ARTIFACTS_DIR / "processed")
CHECKPOINT_DIR = ensure_dir(ARTIFACTS_DIR / "checkpoints")
FINAL_DIR = ensure_dir(ARTIFACTS_DIR / "final_models")
LOG_DIR = ensure_dir(ARTIFACTS_DIR / "logs")
OUTPUT_DIR = ensure_dir(ARTIFACTS_DIR / "outputs")
INFERENCE_DIR = ensure_dir(OUTPUT_DIR / "inference")
VALIDATION_DIR = ensure_dir(OUTPUT_DIR / "validation")
CACHE_DIR = ensure_dir(ARTIFACTS_DIR / "cache")

# Local models directory for predownloaded model weights
MODELS_DIR = ensure_dir(ARTIFACTS_DIR / "models")

os.environ["TORCH_HOME"] = str(CACHE_DIR)
