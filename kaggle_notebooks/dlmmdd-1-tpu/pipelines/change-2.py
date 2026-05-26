from __future__ import annotations

import gc
import json
import os
import logging
import pickle
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm

import psutil
PSUTIL_AVAILABLE = True

import torch
from torch.utils.data import DataLoader, Dataset
TORCH_AVAILABLE = True

import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax.example_libraries import optimizers, stax

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
print(f"MANIFEST USAGE: {USE_MANIFEST}")

SEED = 42
NUM_CLASSES = 10
NUM_FOLDS = 5

NUM_EPOCHS = 2
BATCH_SIZE = 16 if DEVICE_TYPE == "cpu" else 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SESSION_START_TIME = time.time()
SESSION_BUDGET_SECS = 8.5 * 3600
DISK_LIMIT_GIB = 17.0
RAM_WARN_GIB = 24.0

try:
    DEVICE_TYPE = 'tpu' if any(d.platform == 'tpu' for d in jax.devices()) else 'cpu'
except Exception:
    DEVICE_TYPE = 'cpu'
BACKEND = 'jax'
print(f"DEVICE: {DEVICE_TYPE} | Backend: {BACKEND}")

jax.config.update("jax_default_matmul_precision", "bfloat16")
print(f"DEVICE={DEVICE_TYPE} | BACKEND={BACKEND} | MANIFEST={USE_MANIFEST}")

RUN_TRAINING = True
RUN_INFERENCE_ONLY = False
INFERENCE_ONLY_PATH = '/kaggle/input/models/punyakdei/pipe-1-tpu/pytorch/default/1'

NUM_WORKERS = 0 if DEVICE_TYPE in {'tpu', 'cpu'} else 2
PIN_MEMORY = False if DEVICE_TYPE in {'tpu', 'cpu'} else True
PERSISTENT_WORKERS = False if DEVICE_TYPE in {'tpu', 'cpu'} else True

MIXED_PRECISION_POLICY = 'bfloat16' if DEVICE_TYPE == 'tpu' else 'float32'

CHECKPOINT_KEEP_TOP_K = 1
CHECKPOINT_SELECTION_METRIC = 'generalization_score'
EARLY_STOP_PATIENCE = 7

USE_LR_PLATEAU = True
PLATEAU_PATIENCE = 4
PLATEAU_FACTOR = 0.5
PLATEAU_MIN_LR = 1e-7

USE_SWA = False
USE_EMA = False
USE_TTA = False
TTA_N = 4

BLEND_MODE = 'weighted_avg'
STACKING_LEARNER = 'logreg'
STACKING_FOLDS = 3

LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.0
CUTMIX_ALPHA = 0.0

ACTIVE_MODELS = [
    'convnext_base.fb_in22k_ft_in1k',
]

MODEL_REGISTRY = {
    'maxvit_base_tf_384.in21k_ft_in1k': {
        'image_size': 384, 'batch_size': 8 if DEVICE_TYPE == 'cpu' else 16, 'lr': 3e-4,
        'dropout': 0.0, 'width': 96, 'stages': 4,
    },
    'convnext_base.fb_in22k_ft_in1k': {
        'image_size': 224, 'batch_size': 16 if DEVICE_TYPE == 'cpu' else 32, 'lr': 8e-4,
        'dropout': 0.0, 'width': 64, 'stages': 4,
    },
    'efficientnetv2_m.in21k_ft_in1k': {
        'image_size': 384, 'batch_size': 8 if DEVICE_TYPE == 'cpu' else 16, 'lr': 6e-4,
        'dropout': 0.0, 'width': 80, 'stages': 4,
    },
    'swin_base_patch4_window12_384.ms_in22k_ft_in1k': {
        'image_size': 384, 'batch_size': 8 if DEVICE_TYPE == 'cpu' else 16, 'lr': 6e-4,
        'dropout': 0.0, 'width': 96, 'stages': 5,
    },
    'caformer_s36.sail_in22k_ft_in1k': {
        'image_size': 224, 'batch_size': 16 if DEVICE_TYPE == 'cpu' else 32, 'lr': 3e-4,
        'dropout': 0.0, 'width': 72, 'stages': 4,
    },
}

SESSION_CONTEXT = {
    'strategy': DEVICE_TYPE,
    'backend': BACKEND,
}

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def setup_logging(log_dir: Path = LOG_DIR) -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger('hybrid_pipeline')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        log_file = log_dir / f'run_{pd.Timestamp.utcnow():%Y%m%dT%H%M%SZ}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


LOGGER = setup_logging()

try:
    RESAMPLE = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE = Image.BICUBIC


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


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

def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(obj, handle, indent=2, default=str)


def safe_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if path.suffix == '.parquet':
        try:
            df.to_parquet(path, index=False)
            return
        except Exception:
            path = path.with_suffix('.csv')
    df.to_csv(path, index=False)

def manifest_path(*parts: str) -> Path:
    if not MANIFEST_DIR_PATH:
        raise ValueError('Set MANIFEST_DIR_PATH before using manifest mode.')
    return Path(MANIFEST_DIR_PATH).joinpath(*parts)


def load_manifest_artifact(path: Path):
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

def get_disk_used_gib(path=WORKING_ROOT):
    total = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    return total / (1024 ** 3)


def get_ram_used_gib():
    if not PSUTIL_AVAILABLE or psutil is None:
        return 0.0
    return psutil.virtual_memory().used / (1024 ** 3)

def session_elapsed_h():
    return (time.time() - SESSION_START_TIME) / 3600


def session_remaining_h():
    return (SESSION_BUDGET_SECS - (time.time() - SESSION_START_TIME)) / 3600


def log_resources(tag=''):
    disk = get_disk_used_gib()
    ram = get_ram_used_gib()
    elapsed = session_elapsed_h()
    remaining = session_remaining_h()
    print(f'  [resources{" " + tag if tag else ""}] disk={disk:.2f}GiB  ram={ram:.1f}GiB  elapsed={elapsed:.2f}h  remaining={remaining:.2f}h')
    if disk > DISK_LIMIT_GIB:
        raise RuntimeError(f'DISK LIMIT EXCEEDED: {disk:.2f}GiB > {DISK_LIMIT_GIB}GiB')
    if ram > RAM_WARN_GIB:
        print(f'  [WARNING] RAM usage {ram:.1f}GiB is high!')
    return disk, ram, remaining


def time_budget_ok():
    return (time.time() - SESSION_START_TIME) < SESSION_BUDGET_SECS


def cleanup_fold_checkpoints(fold_dir, keep_paths):
    keep = set(str(p) for p in keep_paths)
    for path in Path(fold_dir).glob('epoch_*.pkl'):
        if str(path) not in keep:
            path.unlink()


def restore_full_path_column(df, split_dir, path_col_candidates=('path', 'Path', 'image_path', 'filepath', 'full_path')):
    df = df.copy()
    split_dir = Path(split_dir)

    source_col = next((col for col in path_col_candidates if col in df.columns), None)
    if source_col is None:
        raise KeyError(f'No usable path column found. Tried: {path_col_candidates}')

    def _resolve(raw_path):
        p = Path(str(raw_path))
        if p.is_absolute() and p.exists():
            return str(p)
        return str(split_dir / p.name)

    df['full_path'] = df[source_col].map(_resolve)
    return df


def to_cpu(x):
    return x

def batch_to_numpy(batch):
    if TORCH_AVAILABLE and isinstance(batch, torch.Tensor):
        return batch.detach().cpu().numpy()
    if isinstance(batch, (tuple, list)):
        return type(batch)(batch_to_numpy(x) for x in batch)
    return np.asarray(batch)


def get_selection_metric_value(metrics):
    if CHECKPOINT_SELECTION_METRIC == 'val_accuracy':
        return float(metrics.get('accuracy', 0.0))
    return float(metrics.get('generalization_score', metrics.get('f1_macro', 0.0)))

class MetricsComputer:
    @staticmethod
    def compute_epoch_metrics(y_true, y_pred):
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }

    @staticmethod
    def generalization_score(train_m, val_m, alpha=2.0):
        gap = max(0.0, train_m.get('f1_macro', 0.0) - val_m.get('f1_macro', 0.0))
        return float(val_m.get('f1_macro', 0.0) - alpha * gap**2)


def generalization_score(train_m, val_m, alpha=2.0):
    return MetricsComputer.generalization_score(train_m, val_m, alpha=alpha)


class CheckpointManager:
    @staticmethod
    def prune_to_top_k_checkpoints(saved, keep_top_k):
        while len(saved) > keep_top_k:
            weakest_idx = min(range(len(saved)), key=lambda i: saved[i]['selection_value'])
            weakest = saved.pop(weakest_idx)
            checkpoint_path = Path(weakest['checkpoint_path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            print(f'  [prune] {checkpoint_path.name}  sv={weakest["selection_value"]:.4f}')
        return saved


class CheckpointSelector:
    @staticmethod
    def get_selection_metric_value(metrics):
        return get_selection_metric_value(metrics)

    @staticmethod
    def best_checkpoint(history):
        return max(history, key=lambda row: row['selection_value'])


class DataLoaderHelper:
    @staticmethod
    def load_metadata(data_dir):
        train_df = pd.read_csv(Path(data_dir) / 'training.csv')
        test_df = pd.read_csv(Path(data_dir) / 'test.csv')
        return train_df, test_df

    @staticmethod
    def normalize_image_path(split_dir: Path, raw_path: str) -> str:
        raw_path = Path(str(raw_path))
        if raw_path.is_absolute() and raw_path.exists():
            return str(raw_path)
        candidate = split_dir / raw_path.name
        if candidate.exists():
            return str(candidate)
        return str(split_dir / raw_path.name)

    @staticmethod
    def inspect_image(path: str) -> dict:
        try:
            with Image.open(path) as image:
                width, height = image.size
                fmt = image.format
                mode = image.mode
            return {
                'width': width,
                'height': height,
                'format': fmt,
                'color_mode': mode,
                'file_size_bytes': os.path.getsize(path),
                'is_readable': True,
                'error': None,
            }
        except Exception as exc:
            return {
                'width': None,
                'height': None,
                'format': None,
                'color_mode': None,
                'file_size_bytes': os.path.getsize(path) if os.path.exists(path) else None,
                'is_readable': False,
                'error': str(exc),
            }

    @staticmethod
    def extract_image_metadata(df: pd.DataFrame) -> pd.DataFrame:
        records = [DataLoaderHelper.inspect_image(path) for path in tqdm(df['full_path'], desc='Reading image metadata')]
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)

    @staticmethod
    def load_source_mapping() -> dict:
        return {}

    @staticmethod
    def add_source_names(df: pd.DataFrame, source_mapping: dict) -> pd.DataFrame:
        return df.copy()

    @staticmethod
    def validate_training_data(train_meta: pd.DataFrame) -> dict:
        report = {
            'is_valid': bool(train_meta['is_readable'].all()) if 'is_readable' in train_meta.columns else True,
            'warnings': [],
            'errors': [],
        }
        if 'y' in train_meta.columns:
            counts = train_meta['y'].value_counts().sort_index()
            report['class_counts'] = counts.to_dict()
            if len(counts) != NUM_CLASSES:
                report['warnings'].append(f'Expected {NUM_CLASSES} classes, found {len(counts)}')
        return report

    @staticmethod
    def compute_statistics(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> dict:
        return {
            'train': {
                'total_samples': int(len(train_meta)),
                'classes': int(train_meta['y'].nunique()) if 'y' in train_meta.columns else None,
                'class_distribution': train_meta['y'].value_counts().sort_index().to_dict() if 'y' in train_meta.columns else {},
                'avg_height': float(train_meta['height'].mean()),
                'avg_width': float(train_meta['width'].mean()),
                'avg_file_size_mb': float(train_meta['file_size_bytes'].mean() / 1e6),
                'formats': train_meta['format'].value_counts().to_dict(),
                'color_modes': train_meta['color_mode'].value_counts().to_dict(),
            },
            'test': {
                'total_samples': int(len(test_meta)),
                'avg_height': float(test_meta['height'].mean()),
                'avg_width': float(test_meta['width'].mean()),
                'avg_file_size_mb': float(test_meta['file_size_bytes'].mean() / 1e6),
            },
        }



def _log_metadata_summary(name: str, df: pd.DataFrame) -> None:
    LOGGER.info(f'{name}: rows={len(df)} cols={list(df.columns)[:10]}')
    if 'full_path' in df.columns and len(df):
        sample_path = str(df['full_path'].iloc[0])
        LOGGER.info(f'{name}: sample_path={sample_path} exists={Path(sample_path).exists()}')


def preflight_checks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    LOGGER.info('Running preflight checks...')
    _log_metadata_summary('train', train_df)
    _log_metadata_summary('test', test_df)
    if 'full_path' in train_df.columns:
        missing_train = int((~train_df['full_path'].map(lambda p: Path(p).exists())).sum())
        LOGGER.info(f'train: missing_paths={missing_train}')
    if 'full_path' in test_df.columns:
        missing_test = int((~test_df['full_path'].map(lambda p: Path(p).exists())).sum())
        LOGGER.info(f'test: missing_paths={missing_test}')

def _normalize_image_np(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def load_image_np(path: str, image_size: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image = image.resize((image_size, image_size), RESAMPLE)
            arr = np.asarray(image, dtype=np.float32)
        return _normalize_image_np(arr)
    except Exception as exc:
        raise FileNotFoundError(f'Failed to load image: {path}') from exc


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[str], labels: Sequence[int] | None = None, image_size: int = 224):
        self.paths = [str(p) for p in paths]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = load_image_np(self.paths[idx], self.image_size)
        if self.labels is None:
            return image
        return image, int(self.labels[idx])


def build_torch_dataloader(paths, labels=None, image_size: int = 224, batch_size: int = 32, training: bool = False, cache: bool | str = False):
    if not TORCH_AVAILABLE:
        raise RuntimeError('PyTorch is required for the CPU data pipeline.')
    ds = ImagePathDataset(paths, labels=labels, image_size=image_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=bool(training),
        drop_last=bool(training),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
    )


def build_tf_dataset(paths, labels=None, image_size: int = 224, batch_size: int = 32, training: bool = False, cache: bool | str = False):
    return build_torch_dataloader(paths, labels=labels, image_size=image_size, batch_size=batch_size, training=training, cache=cache)


def build_train_dataset(paths, labels, image_size, batch_size, shuffle=True):
    return build_torch_dataloader(paths, labels, image_size=image_size, batch_size=batch_size, training=shuffle)


def build_eval_dataset(paths, labels, image_size, batch_size):
    return build_torch_dataloader(paths, labels, image_size=image_size, batch_size=batch_size, training=False)


def build_predict_dataset(paths, image_size, batch_size):
    return build_torch_dataloader(paths, labels=None, image_size=image_size, batch_size=batch_size, training=False)


class TrainValSplitter:
    @staticmethod
    def build_folds(y_train: np.ndarray, num_folds: int, seed: int) -> dict:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        fold_metadata = {}
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
            fold_metadata[fold_idx] = {
                'fold_idx': int(fold_idx),
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'train_count': int(len(train_idx)),
                'val_count': int(len(val_idx)),
                'train_class_counts': np.bincount(y_train[train_idx], minlength=NUM_CLASSES).tolist(),
                'val_class_counts': np.bincount(y_train[val_idx], minlength=NUM_CLASSES).tolist(),
            }
        return fold_metadata


def one_hot(labels):
    return jax.nn.one_hot(jnp.asarray(labels, dtype=jnp.int32), NUM_CLASSES)


def mixup_batch(images, labels, alpha=0.4, key=None):
    if alpha <= 0:
        return images, labels
    if key is None:
        key = jrandom.PRNGKey(np.random.randint(0, 1_000_000))
    batch_size = images.shape[0]
    key, k1, k2 = jrandom.split(key, 3)
    lam = jrandom.beta(k1, alpha, alpha)
    lam = jnp.maximum(lam, 1.0 - lam)
    perm = jrandom.permutation(k2, batch_size)
    mixed_images = lam * images + (1.0 - lam) * images[perm]
    mixed_labels = lam * one_hot(labels) + (1.0 - lam) * one_hot(labels[perm])
    return mixed_images, mixed_labels


def cutmix_batch(images, labels, alpha=1.0, key=None):
    if alpha <= 0:
        return images, labels
    if key is None:
        key = jrandom.PRNGKey(np.random.randint(0, 1_000_000))
    b, h, w, c = images.shape
    key, k1, k2, k3, k4 = jrandom.split(key, 5)
    lam = jrandom.beta(k1, alpha, alpha)
    perm = jrandom.permutation(k2, b)
    cut_ratio = jnp.sqrt(1.0 - lam)
    cut_w = jnp.asarray(w * cut_ratio, dtype=jnp.int32)
    cut_h = jnp.asarray(h * cut_ratio, dtype=jnp.int32)
    cx = jrandom.randint(k3, (), 0, w)
    cy = jrandom.randint(k4, (), 0, h)
    x1 = jnp.clip(cx - cut_w // 2, 0, w)
    x2 = jnp.clip(cx + cut_w // 2, 0, w)
    y1 = jnp.clip(cy - cut_h // 2, 0, h)
    y2 = jnp.clip(cy + cut_h // 2, 0, h)

    mixed = images.copy()
    patch = images[perm, y1:y2, x1:x2, :]
    mixed = mixed.at[:, y1:y2, x1:x2, :].set(patch)

    lam_adjusted = 1.0 - (jnp.asarray(x2 - x1, dtype=jnp.float32) * jnp.asarray(y2 - y1, dtype=jnp.float32)) / jnp.asarray(w * h, dtype=jnp.float32)
    mixed_labels = lam_adjusted * one_hot(labels) + (1.0 - lam_adjusted) * one_hot(labels[perm])
    return mixed, mixed_labels


def _make_cnn(image_size: int, num_classes: int, width: int = 64, stages: int = 4):
    layers = []
    channels = width
    for stage in range(stages):
        layers.extend([
            stax.Conv(channels, (3, 3), padding='SAME'),
            stax.Relu,
            stax.Conv(channels, (3, 3), padding='SAME'),
            stax.Relu,
        ])
        if stage < stages - 1:
            layers.append(stax.MaxPool((2, 2), (2, 2), padding='VALID'))
            channels = min(channels * 2, 512)

    downsample = 2 ** max(0, stages - 1)
    pool_size = max(1, image_size // downsample)
    layers.extend([
        stax.AvgPool((pool_size, pool_size), (pool_size, pool_size), padding='VALID'),
        stax.Flatten,
        stax.Dense(max(128, channels)),
        stax.Relu,
        stax.Dense(num_classes),
    ])
    return stax.serial(*layers)


def build_model(model_name, num_classes, dropout=0.0, image_size=224):
    registry = MODEL_REGISTRY.get(model_name)
    if registry is None:
        raise ValueError(f'Unknown model: {model_name}')
    init_fn, apply_fn = _make_cnn(
        image_size=image_size,
        num_classes=num_classes,
        width=int(registry.get('width', 64)),
        stages=int(registry.get('stages', 4)),
    )
    return init_fn, apply_fn


def _cross_entropy_with_logits(logits, labels_onehot):
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(labels_onehot * log_probs, axis=-1))


def _accuracy_from_logits(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == labels).astype(jnp.float32))


def _to_jax_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        images, labels = batch
        images = np.asarray(images)
        labels = np.asarray(labels, dtype=np.int32)
        return jnp.asarray(images), jnp.asarray(labels)
    images = np.asarray(batch)
    return jnp.asarray(images)


def _loader_to_arrays(loader, with_labels: bool):
    images_list = []
    labels_list = []
    for batch in loader:
        batch = batch_to_numpy(batch)
        if with_labels:
            images, labels = batch
            images_list.append(np.asarray(images))
            labels_list.append(np.asarray(labels))
        else:
            images_list.append(np.asarray(batch))
    if with_labels:
        return np.concatenate(images_list, axis=0), np.concatenate(labels_list, axis=0)
    return np.concatenate(images_list, axis=0)


def _predict_proba(apply_fn, params, loader):
    probs = []
    for batch in loader:
        batch = batch_to_numpy(batch)
        if isinstance(batch, (tuple, list)):
            images = np.asarray(batch[0])
        else:
            images = np.asarray(batch)
        logits = apply_fn(params, jnp.asarray(images))
        probs.append(np.asarray(jax.nn.softmax(logits)))
    return np.concatenate(probs, axis=0)


def _make_jax_train_step(apply_fn, opt_update, get_params):
    @jax.jit
    def train_step(step, opt_state, images, labels, mixup_alpha=0.0, cutmix_alpha=0.0):
        params = get_params(opt_state)

        def loss_fn(p):
            logits = apply_fn(p, images)
            targets = one_hot(labels)
            loss = _cross_entropy_with_logits(logits, targets)
            acc = _accuracy_from_logits(logits, labels)
            return loss, (logits, acc)

        (loss, (logits, acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        opt_state = opt_update(step, grads, opt_state)
        return opt_state, loss, acc

    return train_step


def _make_jax_eval_step(apply_fn):
    @jax.jit
    def eval_step(params, images, labels):
        logits = apply_fn(params, images)
        loss = _cross_entropy_with_logits(logits, one_hot(labels))
        acc = _accuracy_from_logits(logits, labels)
        return loss, acc, logits

    return eval_step


def _collect_predictions(apply_fn, params, loader):
    y_true = []
    y_pred = []
    probs = []
    for batch in loader:
        batch = batch_to_numpy(batch)
        images, labels = batch
        logits = apply_fn(params, jnp.asarray(np.asarray(images)))
        batch_probs = np.asarray(jax.nn.softmax(logits))
        probs.append(batch_probs)
        y_true.append(np.asarray(labels))
        y_pred.append(batch_probs.argmax(axis=1))
    if probs:
        return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(probs)
    return np.array([]), np.array([]), np.array([])


def _save_checkpoint(path: Path, params, metadata: dict):
    ensure_dir(path.parent)
    payload = {
        'params': jax.tree_util.tree_map(lambda x: np.asarray(x), params),
        'metadata': metadata,
    }
    with path.open('wb') as handle:
        pickle.dump(payload, handle)
    save_json(metadata, path.with_suffix('.json'))


def _load_checkpoint(path: Path):
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), payload['params'])
    metadata = payload.get('metadata', {})
    return params, metadata


class EpochCheckpointCallback:
    def __init__(self, fold_dir, train_eval_loader, val_loader, y_train, y_val, model_name, model_cfg):
        self.fold_dir = Path(fold_dir)
        self.train_eval_loader = train_eval_loader
        self.val_loader = val_loader
        self.y_train = np.asarray(y_train, dtype=np.int64)
        self.y_val = np.asarray(y_val, dtype=np.int64)
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.history = []
        self.saved = []
        self.best_val_f1 = 0.0
        self.no_improve = 0
        self.fold_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, params, apply_fn, opt_state, logs=None):
        logs = logs or {}
        train_true, train_pred, train_probs = _collect_predictions(apply_fn, params, self.train_eval_loader)
        val_true, val_pred, val_probs = _collect_predictions(apply_fn, params, self.val_loader)

        train_m = MetricsComputer.compute_epoch_metrics(train_true, train_pred)
        val_m = MetricsComputer.compute_epoch_metrics(val_true, val_pred)
        train_m['loss'] = float(logs.get('loss')) if logs.get('loss') is not None else None
        val_m['loss'] = float(logs.get('val_loss')) if logs.get('val_loss') is not None else None
        val_m['generalization_score'] = MetricsComputer.generalization_score(train_m, val_m)
        selection_value = CheckpointSelector.get_selection_metric_value(val_m)

        checkpoint_path = self.fold_dir / f'epoch_{epoch:03d}.pkl'
        _save_checkpoint(
            checkpoint_path,
            params,
            {
                'model_name': self.model_name,
                'fold_idx': int(self.model_cfg['fold_idx']),
                'epoch': int(epoch),
                'image_size': int(self.model_cfg['image_size']),
                'batch_size': int(self.model_cfg['batch_size']),
                'train_metrics': train_m,
                'val_metrics': val_m,
                'selection_value': float(selection_value),
            },
        )

        self.saved.append({'epoch': epoch, 'checkpoint_path': str(checkpoint_path), 'selection_value': selection_value})
        self.saved = CheckpointManager.prune_to_top_k_checkpoints(self.saved, CHECKPOINT_KEEP_TOP_K)
        self.history.append({
            'epoch': int(epoch),
            'checkpoint_path': str(checkpoint_path),
            'train': train_m,
            'val': val_m,
            'selection_value': float(selection_value),
        })

        if val_m['f1_macro'] > self.best_val_f1 + 1e-4:
            self.best_val_f1 = val_m['f1_macro']
            self.no_improve = 0
        else:
            self.no_improve += 1


def train_fold(fold_idx, fold_info, train_df, model_name, mcfg, arch_dir):
    LOGGER.info(f"{'=' * 68}")
    LOGGER.info(f"MODEL={model_name} | FOLD={fold_idx} | train={fold_info['train_count']} | val={fold_info['val_count']}")
    LOGGER.info(f"{'=' * 68}")

    if 'full_path' not in train_df.columns:
        raise ValueError('train_df must include a full_path column before training')

    fold_dir = Path(arch_dir) / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(mcfg['image_size'])
    batch_size = int(mcfg['batch_size'])
    lr = float(mcfg['lr'])
    width = int(mcfg.get('width', 64))
    stages = int(mcfg.get('stages', 4))

    train_rows = train_df.iloc[fold_info['train_indices']].reset_index(drop=True)
    val_rows = train_df.iloc[fold_info['val_indices']].reset_index(drop=True)
    x_train = train_rows['full_path'].astype(str).to_numpy()
    y_train = train_rows['y'].astype(np.int64).to_numpy()
    x_val = val_rows['full_path'].astype(str).to_numpy()
    y_val = val_rows['y'].astype(np.int64).to_numpy()

    LOGGER.info(f'train sample path: {x_train[0] if len(x_train) else '<empty>'}')
    LOGGER.info(f'val sample path: {x_val[0] if len(x_val) else '<empty>'}')
    LOGGER.info(f'image_size={image_size} batch_size={batch_size} lr={lr}')

    train_loader = build_torch_dataloader(x_train, y_train, image_size=image_size, batch_size=batch_size, training=True)
    train_eval_loader = build_torch_dataloader(x_train, y_train, image_size=image_size, batch_size=batch_size, training=False)
    val_loader = build_torch_dataloader(x_val, y_val, image_size=image_size, batch_size=batch_size, training=False)

    init_fn, apply_fn = build_model(model_name, NUM_CLASSES, image_size=image_size)
    rng = jrandom.PRNGKey(SEED + fold_idx)
    _, params = init_fn(rng, (-1, image_size, image_size, 3))
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    train_step = _make_jax_train_step(apply_fn, opt_update, get_params)
    eval_step = _make_jax_eval_step(apply_fn)

    callback = EpochCheckpointCallback(
        fold_dir=fold_dir,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        y_train=y_train,
        y_val=y_val,
        model_name=model_name,
        model_cfg={**mcfg, 'fold_idx': fold_idx, 'image_size': image_size, 'batch_size': batch_size},
    )

    if not time_budget_ok():
        print('  [TIME BUDGET] Skipping fold before training starts.')
        return []

    best_val_f1 = 0.0
    no_improve = 0
    step = 0
    history = []

    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        epoch_accs = []
        for batch in train_loader:
            images, labels = batch_to_numpy(batch)
            images = jnp.asarray(images)
            labels = jnp.asarray(labels, dtype=jnp.int32)
            opt_state, loss, acc = train_step(step, opt_state, images, labels, MIXUP_ALPHA, CUTMIX_ALPHA)
            step += 1
            epoch_losses.append(float(loss))
            epoch_accs.append(float(acc))

        params = get_params(opt_state)
        train_true, train_pred, _ = _collect_predictions(apply_fn, params, train_eval_loader)
        val_true, val_pred, _ = _collect_predictions(apply_fn, params, val_loader)

        train_m = MetricsComputer.compute_epoch_metrics(train_true, train_pred)
        val_m = MetricsComputer.compute_epoch_metrics(val_true, val_pred)
        train_m['loss'] = float(np.mean(epoch_losses)) if epoch_losses else None
        val_m['loss'] = None
        val_m['generalization_score'] = MetricsComputer.generalization_score(train_m, val_m)
        selection_value = CheckpointSelector.get_selection_metric_value(val_m)

        checkpoint_path = fold_dir / f'epoch_{epoch:03d}.pkl'
        _save_checkpoint(
            checkpoint_path,
            params,
            {
                'model_name': model_name,
                'fold_idx': int(fold_idx),
                'epoch': int(epoch),
                'image_size': int(image_size),
                'batch_size': int(batch_size),
                'train_metrics': train_m,
                'val_metrics': val_m,
                'selection_value': float(selection_value),
            },
        )

        history.append({
            'epoch': int(epoch),
            'checkpoint_path': str(checkpoint_path),
            'train': train_m,
            'val': val_m,
            'selection_value': float(selection_value),
            'learning_rate': float(lr),
        })

        print(
            f"  epoch {epoch:02d} | loss={np.mean(epoch_losses):.4f} "
            f"| train_f1={train_m['f1_macro']:.4f} | val_f1={val_m['f1_macro']:.4f} | gen={val_m['generalization_score']:.4f}"
        )

        if val_m['f1_macro'] > best_val_f1 + 1e-4:
            best_val_f1 = val_m['f1_macro']
            no_improve = 0
            best_dst_dir = FINAL_MODELS_DIR / model_name
            best_dst_dir.mkdir(parents=True, exist_ok=True)
            best_dst = best_dst_dir / f'fold_{fold_idx}_best.pkl'
            shutil.copy2(checkpoint_path, best_dst)
            shutil.copy2(checkpoint_path.with_suffix('.json'), best_dst.with_suffix('.json'))
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f'  early stopping at epoch {epoch}')
                break

    save_json(history, LOG_DIR / f'{model_name[:30]}_history_fold_{fold_idx}.json')
    log_resources(f'after fold {fold_idx}')
    jax.clear_caches()
    gc.collect()
    return history


def load_model_from_ckpt(ckpt_path):
    ckpt_path = Path(ckpt_path)
    params, metadata = _load_checkpoint(ckpt_path)
    model_name = metadata.get('model_name', ACTIVE_MODELS[0])
    val_metrics = metadata.get('val_metrics', {})
    print(
        f'  [load] {ckpt_path.name}  arch={model_name}  epoch={metadata.get("epoch", "?")}  '
        f'val_f1={val_metrics.get("f1_macro", float("nan")):.4f}  gen={val_metrics.get("generalization_score", float("nan")):.4f}'
    )
    return params, metadata, model_name


def infer_single_pass(apply_fn, params, dataset):
    return _predict_proba(apply_fn, params, dataset)


def infer_with_tta(apply_fn, params, paths, img_sz, batch_size, n_tta=4):
    clean_ds = build_torch_dataloader(paths, labels=None, image_size=img_sz, batch_size=batch_size, training=False)
    probs = _predict_proba(apply_fn, params, clean_ds)

    for idx in range(n_tta):
        aug_images = []
        for batch in clean_ds:
            batch = batch_to_numpy(batch)
            images = np.asarray(batch)
            if idx % 2 == 0:
                images = np.flip(images, axis=2)
            aug_images.append(images)
        if aug_images:
            aug_images = np.concatenate(aug_images, axis=0)
            logits = apply_fn(params, jnp.asarray(aug_images))
            probs += np.asarray(jax.nn.softmax(logits))
        print(f'    TTA {idx + 1}/{n_tta}')

    return probs / float(n_tta + 1)


def _predict_class_probabilities(apply_fn, params, paths, img_sz, batch_size):
    dataset = build_torch_dataloader(paths, labels=None, image_size=img_sz, batch_size=batch_size, training=False)
    return infer_single_pass(apply_fn, params, dataset)


def run_ensemble_inference(train_meta, test_meta, y_train, fold_metadata):
    if test_meta.empty:
        print('test_meta empty -- skipped.')
        return None

    print(f'\nRunning inference on {len(test_meta)} test images ...')
    log_resources('inference start')

    if RUN_INFERENCE_ONLY:
        ckpt_paths = sorted(Path(INFERENCE_ONLY_PATH).glob('**/*best.pkl'))
    else:
        ckpt_paths = sorted(FINAL_MODELS_DIR.rglob('fold_*_best.pkl'))

    print(f'Found {len(ckpt_paths)} checkpoint(s).')

    all_probs = []
    all_weights = []
    all_oof = []

    test_paths = test_meta['full_path'].astype(str).to_numpy()
    fold_metadata = {int(k): v for k, v in fold_metadata.items()}

    for checkpoint_path in ckpt_paths:
        params, metadata, model_name = load_model_from_ckpt(checkpoint_path)
        img_sz = int(metadata.get('image_size', MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[ACTIVE_MODELS[0]])['image_size']))
        batch_size = int(metadata.get('batch_size', MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[ACTIVE_MODELS[0]])['batch_size']))
        init_fn, apply_fn = build_model(model_name, NUM_CLASSES, image_size=img_sz)

        if USE_TTA:
            probs = infer_with_tta(apply_fn, params, test_paths, img_sz, batch_size, TTA_N)
        else:
            probs = _predict_class_probabilities(apply_fn, params, test_paths, img_sz, batch_size)

        all_probs.append(probs)
        weight = metadata.get('val_metrics', {}).get('f1_macro', metadata.get('val_metrics', {}).get('accuracy', 1.0))
        all_weights.append(weight)
        print(f'  [weight] {checkpoint_path.name}  f1={weight:.4f}')

        if BLEND_MODE == 'stacking':
            match = re.search(r'fold_(\d+)', checkpoint_path.stem)
            fold_idx = int(match.group(1)) if match else -1
            if fold_idx >= 0 and fold_idx in fold_metadata:
                val_idx = fold_metadata[fold_idx]['val_indices']
                val_df = train_meta.iloc[val_idx].reset_index(drop=True)
                val_paths = val_df['full_path'].astype(str).to_numpy()
                val_ds = build_torch_dataloader(val_paths, labels=None, image_size=img_sz, batch_size=batch_size, training=False)
                oof_probs = _predict_proba(apply_fn, params, val_ds)
                all_oof.append((fold_idx, val_idx, oof_probs))

        del params
        gc.collect()

    if not all_probs:
        print('No checkpoints found.')
        return None

    stacked_probs = np.stack(all_probs, 0)
    weights = np.asarray(all_weights, np.float32)
    weights = np.nan_to_num(weights, nan=1e-6)
    weights = np.maximum(weights, 1e-6)

    if BLEND_MODE == 'simple_avg':
        print('\n[ensemble] simple average')
        ensemble = stacked_probs.mean(0)
    elif BLEND_MODE == 'weighted_avg':
        weights = weights / weights.sum()
        print(f'\n[ensemble] weighted average  w={weights.round(3).tolist()}')
        ensemble = np.average(stacked_probs, 0, weights=weights)
    elif BLEND_MODE == 'rank_avg':
        print('\n[ensemble] rank average')
        from scipy.stats import rankdata
        ranked = np.stack([np.apply_along_axis(rankdata, 1, stacked_probs[idx]) for idx in range(stacked_probs.shape[0])], 0)
        ensemble = ranked.mean(0)
    elif BLEND_MODE == 'stacking':
        print(f'\n[ensemble] stacking meta-learner={STACKING_LEARNER}')
        if not all_oof:
            print('  [stacking] no OOF data -- falling back to weighted_avg')
            weights = weights / weights.sum()
            ensemble = np.average(stacked_probs, 0, weights=weights)
        else:
            n_train = len(train_meta)
            n_feat = len(all_oof) * NUM_CLASSES
            X_oof = np.zeros((n_train, n_feat), np.float32)
            X_test_stack = np.zeros((stacked_probs.shape[1], n_feat), np.float32)
            for idx, (fold_idx, val_idx, oof_probs) in enumerate(all_oof):
                sl = slice(idx * NUM_CLASSES, (idx + 1) * NUM_CLASSES)
                X_oof[val_idx, sl] = oof_probs
                X_test_stack[:, sl] = stacked_probs[idx]
            y_oof = y_train

            if STACKING_LEARNER == 'logreg':
                meta = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial', random_state=SEED)
            elif STACKING_LEARNER == 'ridge':
                meta = RidgeClassifier(alpha=1.0)
            elif STACKING_LEARNER == 'mlp':
                meta = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500, random_state=SEED, early_stopping=True)
            else:
                raise ValueError(f'Unknown STACKING_LEARNER: {STACKING_LEARNER}')

            print(f'  Fitting {STACKING_LEARNER} on {X_oof.shape} OOF features...')
            meta.fit(X_oof, y_oof)
            meta_f1 = f1_score(y_oof, meta.predict(X_oof), average='macro', zero_division=0)
            print(f'  OOF train f1_macro={meta_f1:.4f}')

            if hasattr(meta, 'predict_proba'):
                ensemble = meta.predict_proba(X_test_stack)
            else:
                decision = meta.decision_function(X_test_stack)
                ensemble = np.exp(decision) / np.exp(decision).sum(1, keepdims=True)

            save_json({'learner': STACKING_LEARNER, 'oof_f1_macro': float(meta_f1)}, INFERENCE_DIR / 'stacking_meta.json')
    else:
        raise ValueError(f'Unknown BLEND_MODE: {BLEND_MODE}')

    preds = ensemble.argmax(1).astype(int)
    conf = ensemble.max(1)

    submission_df = test_meta[['ID']].copy()
    submission_df['TARGET'] = preds
    submission_df.to_csv(WORKING_ROOT / 'submission.csv', index=False)
    submission_df.to_csv(INFERENCE_DIR / 'submission.csv', index=False)

    pd.DataFrame({'ID': test_meta['ID'].values, 'predicted_class': preds, 'confidence': conf}).to_csv(INFERENCE_DIR / 'prediction_confidence.csv', index=False)

    save_json({
        'timestamp': pd.Timestamp.utcnow().isoformat(),
        'num_models': len(ckpt_paths),
        'blend_mode': BLEND_MODE,
        'tta_passes': TTA_N if USE_TTA else 0,
        'mean_confidence': float(conf.mean()),
        'min_confidence': float(conf.min()),
        'max_confidence': float(conf.max()),
        'num_predictions': len(preds),
    }, INFERENCE_DIR / 'submission_metadata.json')

    print('\nSubmission preview:')
    print(submission_df.head(10).to_string(index=False))
    print(f'Avg confidence: {conf.mean():.4f}  Min: {conf.min():.4f}  Max: {conf.max():.4f}')
    print(pd.Series(preds).value_counts().sort_index().to_string())
    log_resources('inference done')
    return submission_df


if __name__ == "__main__":
    set_seed(SEED)

    if RUN_INFERENCE_ONLY:
        bundle = load_manifest_bundle()
        train_meta = bundle["train_meta"]
        test_meta = bundle["test_meta"]
        y_train = bundle["y_train"]
        fold_metadata = bundle["fold_metadata"]
        run_ensemble_inference(train_meta, test_meta, y_train, fold_metadata)

    elif RUN_TRAINING:
        # You need a prepared DataFrame with full_path and y columns
        train_df = pd.read_csv(INPUT_TRAIN_CSV)
        train_df = restore_full_path_column(train_df, TRAIN_IMAGE_DIR)

        splitter = TrainValSplitter()
        fold_metadata = splitter.build_folds(train_df["y"].to_numpy(), NUM_FOLDS, SEED)

        for model_name in ACTIVE_MODELS:
            mcfg = MODEL_REGISTRY[model_name]
            for fold_idx, fold_info in fold_metadata.items():
                train_fold(fold_idx, fold_info, train_df, model_name, mcfg, CHECKPOINT_DIR / model_name)

    else:
        raise RuntimeError("Nothing to do: set RUN_TRAINING or RUN_INFERENCE_ONLY")