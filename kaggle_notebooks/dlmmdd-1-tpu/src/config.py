from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PSUTIL_AVAILABLE = True
TF_AVAILABLE = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

COMPETITION_ROOT = Path('/kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge')
DATA_ROOT = COMPETITION_ROOT / 'Data' / 'Data'
TRAIN_CSV = DATA_ROOT / 'training.csv'
TEST_CSV = DATA_ROOT / 'test.csv'
TRAIN_DIR = DATA_ROOT / 'Training'
TEST_DIR = DATA_ROOT / 'Test'
SOURCES = DATA_ROOT / 'sources.txt'
WORKING_ROOT = Path('/kaggle/working')

USE_MANIFEST = False
MANIFEST_DIR_PATH = '/kaggle/input/datasets/punyakdei/dlmmdd-pipe-1-material'

SEED = 42
NUM_CLASSES = 10
NUM_FOLDS = 5

SESSION_START_TIME = time.time()
SESSION_BUDGET_SECS = 8.5 * 3600
DISK_LIMIT_GIB = 17.0
RAM_WARN_GIB = 24.0

AUTO = None
STRATEGY = None
MIXED_PRECISION_POLICY = 'float32'

MODEL_REGISTRY = {
    'maxvit_base_tf_384.in21k_ft_in1k': {
        'image_size': 384, 'batch_size': 32, 'lr': 3e-5,
        'dropout': 0.3, 'drop_path': 0.2, 'weight_decay': 1e-2,
    },
    'convnext_base.fb_in22k_ft_in1k': {
        'image_size': 224, 'batch_size': 64, 'lr': 8e-5,
        'dropout': 0.25, 'drop_path': 0.15, 'weight_decay': 5e-3,
    },
    'efficientnetv2_m.in21k_ft_in1k': {
        'image_size': 384, 'batch_size': 32, 'lr': 6e-5,
        'dropout': 0.3, 'drop_path': 0.2, 'weight_decay': 5e-3,
    },
    'swin_base_patch4_window12_384.ms_in22k_ft_in1k': {
        'image_size': 384, 'batch_size': 32, 'lr': 6e-5,
        'dropout': 0.25, 'drop_path': 0.2, 'weight_decay': 5e-3,
    },
    'caformer_s36.sail_in22k_ft_in1k': {
        'image_size': 224, 'batch_size': 64, 'lr': 3e-5,
        'dropout': 0.2, 'drop_path': 0.15, 'weight_decay': 5e-3,
    },
}

ACTIVE_MODELS = [
    'convnext_base.fb_in22k_ft_in1k',
    'caformer_s36.sail_in22k_ft_in1k',
]

NUM_EPOCHS = 10
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0
WARMUP_EPOCHS = 3
GRAD_CLIP = 1.0
CHECKPOINT_KEEP_TOP_K = 1
CHECKPOINT_SELECTION_METRIC = 'generalization_score'
EARLY_STOP_PATIENCE = 7

USE_ADAPTIVE_DROPOUT = False
ADAPTIVE_DROPOUT_THRESH = 0.08
ADAPTIVE_DROPOUT_STEP = 0.05
ADAPTIVE_DROPOUT_MAX = 0.6
USE_ADAPTIVE_DROP_PATH = False
ADAPTIVE_DROP_PATH_THRESH = 0.08
ADAPTIVE_DROP_PATH_STEP = 0.05
ADAPTIVE_DROP_PATH_MAX = 0.5

USE_LR_PLATEAU = True
PLATEAU_PATIENCE = 4
PLATEAU_FACTOR = 0.5
PLATEAU_MIN_LR = 1e-7

USE_SWA = True
USE_EMA = True
EMA_DECAY = 0.9998

USE_TTA = True
TTA_N = 4

BLEND_MODE = 'stacking'
STACKING_LEARNER = 'logreg'
STACKING_FOLDS = 3

RUN_TRAINING = True
RUN_INFERENCE_ONLY = False
INFERENCE_ONLY_PATH = '/kaggle/input/models/punyakdei/pipe-1-tpu/pytorch/default/1'

NUM_WORKERS = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def initialize_tensorflow_runtime():
    global AUTO, STRATEGY, MIXED_PRECISION_POLICY

    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow is not available in this environment')

    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        STRATEGY = tf.distribute.TPUStrategy(resolver)
        MIXED_PRECISION_POLICY = 'mixed_bfloat16'
        print('Running on TPU')
    except Exception:
        STRATEGY = tf.distribute.get_strategy()
        MIXED_PRECISION_POLICY = 'float32'
        print('Running on CPU')

    AUTO = tf.data.AUTOTUNE
    tf.keras.mixed_precision.set_global_policy(MIXED_PRECISION_POLICY)
    return STRATEGY


STRATEGY = initialize_tensorflow_runtime()
DEVICE_TYPE = 'tpu' if isinstance(STRATEGY, getattr(tf.distribute, 'TPUStrategy', tuple())) else 'cpu'

if DEVICE_TYPE in {'tpu', 'cpu'}:
    NUM_WORKERS = 0
    PIN_MEMORY = False
    PERSISTENT_WORKERS = False

WORKING_ROOT.mkdir(parents=True, exist_ok=True)
if RUN_INFERENCE_ONLY:
    RUN_TRAINING = False

DATA_DIR = DATA_ROOT
INPUT_TRAIN_CSV = TRAIN_CSV
INPUT_TEST_CSV = TEST_CSV
TRAIN_IMAGE_DIR = TRAIN_DIR
TEST_IMAGE_DIR = TEST_DIR
SOURCE_FILE = SOURCES
PROCESSED_DIR = ensure_dir(WORKING_ROOT / 'processed')
CHECKPOINT_DIR = ensure_dir(WORKING_ROOT / 'checkpoints')
FINAL_MODELS_DIR = ensure_dir(WORKING_ROOT / 'final_models')
LOG_DIR = ensure_dir(WORKING_ROOT / 'logs')
OUTPUT_DIR = ensure_dir(WORKING_ROOT / 'outputs')
EDA_DIR = ensure_dir(OUTPUT_DIR / 'eda')
EDA_PLOTS_DIR = ensure_dir(EDA_DIR / 'plots')
VALIDATION_DIR = ensure_dir(OUTPUT_DIR / 'validation')
INFERENCE_DIR = ensure_dir(OUTPUT_DIR / 'inference')


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.keras.utils.set_random_seed(seed)


@dataclass(frozen=True)
class RuntimePaths:
    data_dir: Path = DATA_DIR
    train_csv: Path = INPUT_TRAIN_CSV
    test_csv: Path = INPUT_TEST_CSV
    train_dir: Path = TRAIN_IMAGE_DIR
    test_dir: Path = TEST_IMAGE_DIR
    processed_dir: Path = PROCESSED_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    final_models_dir: Path = FINAL_MODELS_DIR
    log_dir: Path = LOG_DIR
    output_dir: Path = OUTPUT_DIR
    eda_dir: Path = EDA_DIR
    eda_plots_dir: Path = EDA_PLOTS_DIR
    validation_dir: Path = VALIDATION_DIR
    inference_dir: Path = INFERENCE_DIR


def load_manifest_artifact(path: Path):
    import json
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f'Manifest artifact not found: {path}')
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=True)
    if path.suffix == '.json':
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    if path.suffix == '.parquet':
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_path = path.with_suffix('.csv')
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    if path.suffix == '.csv':
        return pd.read_csv(path)
    return path


def load_manifest_bundle() -> dict:
    base = Path(MANIFEST_DIR_PATH)
    if not base.exists():
        raise FileNotFoundError(f'Manifest directory does not exist: {base}')

    bundle = {
        'train_meta': None,
        'test_meta': None,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'fold_metadata': None,
        'source_mapping': None,
    }

    candidates = {
        'train_meta': [base / 'processed' / 'train_metadata.parquet', base / 'processed' / 'train_metadata.csv'],
        'test_meta': [base / 'processed' / 'test_metadata.parquet', base / 'processed' / 'test_metadata.csv'],
        'X_train': [base / 'processed' / 'X_train.npy'],
        'X_test': [base / 'processed' / 'X_test.npy'],
        'y_train': [base / 'processed' / 'y_train.npy'],
        'fold_metadata': [base / 'processed' / 'fold_metadata.json'],
        'source_mapping': [base / 'processed' / 'source_mapping.json'],
    }

    for key, paths in candidates.items():
        for path in paths:
            if path.exists():
                bundle[key] = load_manifest_artifact(path)
                break

    missing = [key for key, value in bundle.items() if value is None and key in {'train_meta', 'test_meta', 'X_train', 'X_test', 'y_train', 'fold_metadata'}]
    if missing:
        raise FileNotFoundError(f'Manifest mode is missing required artifacts: {missing}')

    return bundle
