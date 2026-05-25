from __future__ import annotations

import gc
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False

import tensorflow as tf
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_convnext_backbone(include_top=False, weights='imagenet', input_shape=None, **kwargs):
    return tf.keras.applications.ConvNeXtBase(
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        **kwargs,
    )


def _build_efficientnetv2_backbone(include_top=False, weights='imagenet', input_shape=None, **kwargs):
    return tf.keras.applications.EfficientNetV2M(
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        **kwargs,
    )


def _build_maxvit_backbone(include_top=False, weights='imagenet', input_shape=None, **kwargs):
    for candidate_name in ('MaxViTBase', 'MaxViTSmall', 'MaxViTTiny'):
        builder = getattr(tf.keras.applications, candidate_name, None)
        if builder is not None:
            return builder(include_top=include_top, weights=weights, input_shape=input_shape, **kwargs)
    print('  [warn] MaxViT backbone unavailable; falling back to ConvNeXtBase')
    return _build_convnext_backbone(include_top=include_top, weights=weights, input_shape=input_shape, **kwargs)


def _build_swin_backbone(include_top=False, weights='imagenet', input_shape=None, **kwargs):
    print('  [warn] Swin backbone unavailable in this runtime; falling back to ConvNeXtBase')
    return _build_convnext_backbone(include_top=include_top, weights=weights, input_shape=input_shape, **kwargs)


def _build_caformer_backbone(include_top=False, weights='imagenet', input_shape=None, **kwargs):
    print('  [warn] CaFormer backbone unavailable in this runtime; falling back to EfficientNetV2M')
    return _build_efficientnetv2_backbone(include_top=include_top, weights=weights, input_shape=input_shape, **kwargs)


MODEL_REGISTRY = {
    'maxvit_base_tf_384.in21k_ft_in1k': {
        'builder': _build_maxvit_backbone,
        'image_size': 384, 'batch_size': 32, 'lr': 3e-5,
        'dropout': 0.3, 'drop_path': 0.2, 'weight_decay': 1e-2,
    },
    'convnext_base.fb_in22k_ft_in1k': {
        'builder': _build_convnext_backbone,
        'image_size': 224, 'batch_size': 64, 'lr': 8e-5,
        'dropout': 0.25, 'drop_path': 0.15, 'weight_decay': 5e-3,
    },
    'efficientnetv2_m.in21k_ft_in1k': {
        'builder': _build_efficientnetv2_backbone,
        'image_size': 384, 'batch_size': 32, 'lr': 6e-5,
        'dropout': 0.3, 'drop_path': 0.2, 'weight_decay': 5e-3,
    },
    'swin_base_patch4_window12_384.ms_in22k_ft_in1k': {
        'builder': _build_swin_backbone,
        'image_size': 384, 'batch_size': 32, 'lr': 6e-5,
        'dropout': 0.25, 'drop_path': 0.2, 'weight_decay': 5e-3,
    },
    'caformer_s36.sail_in22k_ft_in1k': {
        'builder': _build_caformer_backbone,
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


def initialize_tensorflow_runtime():
    global AUTO, STRATEGY, MIXED_PRECISION_POLICY

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
DEVICE_TYPE = 'tpu' if hasattr(tf.distribute, 'TPUStrategy') and isinstance(STRATEGY, tf.distribute.TPUStrategy) else 'cpu'

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
    for path in Path(fold_dir).glob('epoch_*.keras'):
        if str(path) not in keep:
            path.unlink()


def restore_full_path_column(df, split_dir):
    df = df.copy()
    if 'full_path' not in df.columns:
        df['full_path'] = df['path'].apply(lambda p: str(split_dir / Path(str(p))))
    return df


def to_cpu(tensor):
    return tensor


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

    @staticmethod
    def cleanup_fold_checkpoints(fold_dir, keep_paths):
        keep = set(str(path) for path in keep_paths)
        for path in Path(fold_dir).glob('epoch_*.keras'):
            if str(path) not in keep:
                path.unlink()


class CheckpointSelector:
    @staticmethod
    def get_selection_metric_value(metrics):
        return get_selection_metric_value(metrics)

    @staticmethod
    def best_checkpoint(history):
        return max(history, key=lambda row: row['selection_value'])


class DataLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load_metadata(self):
        train_df = pd.read_csv(self.data_dir / 'training.csv')
        test_df = pd.read_csv(self.data_dir / 'test.csv')
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

    def extract_image_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        records = [self.inspect_image(path) for path in tqdm(df['full_path'], desc='Reading image metadata')]
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


class ImagePreprocessor:
    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def __call__(self, image_path: str) -> np.ndarray:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [self.target_size, self.target_size], antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
        std = tf.constant(IMAGENET_STD, dtype=tf.float32)
        return ((image - mean) / std).numpy().astype(np.float32)

    def preprocess_paths(self, paths: pd.Series, cache_path: Path) -> np.ndarray:
        if cache_path.exists():
            return np.load(cache_path, mmap_mode='r')
        n = len(paths)
        sample = self(paths.iloc[0])
        shape = (n,) + sample.shape
        arr = np.lib.format.open_memmap(cache_path, mode='w+', dtype=np.float32, shape=shape)
        arr[0] = sample
        for i, image_path in enumerate(tqdm(paths.iloc[1:], desc=f'Preprocessing {cache_path.stem}', initial=1, total=n), start=1):
            arr[i] = self(image_path)
        arr.flush()
        return arr


def load_image_tensor(path: tf.Tensor, image_size: int) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size], antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std = tf.constant(IMAGENET_STD, dtype=tf.float32)
    return (image - mean) / std


def build_tf_dataset(paths, labels=None, image_size: int = 224, batch_size: int = 32, training: bool = False, cache: bool | str = False):
    path_tensor = tf.convert_to_tensor(paths)
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(path_tensor)
        dataset = dataset.map(lambda path: load_image_tensor(path, image_size), num_parallel_calls=AUTO)
    else:
        label_tensor = tf.convert_to_tensor(labels)
        dataset = tf.data.Dataset.from_tensor_slices((path_tensor, label_tensor))
        dataset = dataset.map(lambda path, label: (load_image_tensor(path, image_size), tf.cast(label, tf.int32)), num_parallel_calls=AUTO)
        if training:
            dataset = dataset.shuffle(min(len(paths), 2048), reshuffle_each_iteration=True)
    if cache:
        dataset = dataset.cache(cache) if isinstance(cache, str) else dataset.cache()
    dataset = dataset.batch(batch_size, drop_remainder=training)
    return dataset.prefetch(AUTO)


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


def _normalize_image(image):
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std = tf.constant(IMAGENET_STD, dtype=tf.float32)
    return (image - mean) / std


def build_augmentation_layer():
    return keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name='augmentation')


def load_image(path, label=None, image_size=224, training=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size], antialias=True)
    image = _normalize_image(image)
    if label is None:
        return image
    return image, tf.cast(label, tf.int32)


def build_train_dataset(paths, labels, image_size, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(min(len(paths), 2048), reshuffle_each_iteration=True)
    dataset = dataset.map(lambda path, label: load_image(path, label, image_size=image_size, training=True), num_parallel_calls=AUTO)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset


def build_eval_dataset(paths, labels, image_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda path, label: load_image(path, label, image_size=image_size, training=False), num_parallel_calls=AUTO)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(AUTO)
    return dataset


def build_predict_dataset(paths, image_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(lambda path: load_image(path, image_size=image_size, training=False), num_parallel_calls=AUTO)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(AUTO)
    return dataset


def one_hot(labels):
    return tf.one_hot(tf.cast(labels, tf.int32), NUM_CLASSES)


def mixup_batch(images, labels, alpha=0.4):
    if alpha <= 0:
        return images, labels
    batch_size = tf.shape(images)[0]
    lam = tf.random.gamma([], alpha, beta=alpha)
    lam = tf.maximum(lam, 1.0 - lam)
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, index)
    mixed_labels = lam * one_hot(labels) + (1.0 - lam) * one_hot(tf.gather(labels, index))
    return mixed_images, mixed_labels


def cutmix_batch(images, labels, alpha=1.0):
    if alpha <= 0:
        return images, labels
    batch_size = tf.shape(images)[0]
    lam = tf.random.gamma([], alpha, beta=alpha)
    index = tf.random.shuffle(tf.range(batch_size))
    image_shape = tf.shape(images)
    height = image_shape[1]
    width = image_shape[2]
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_w = tf.cast(tf.cast(width, tf.float32) * cut_ratio, tf.int32)
    cut_h = tf.cast(tf.cast(height, tf.float32) * cut_ratio, tf.int32)
    cx = tf.random.uniform([], 0, width, dtype=tf.int32)
    cy = tf.random.uniform([], 0, height, dtype=tf.int32)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, width)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, height)

    mixed_images = tf.identity(images)
    patch = tf.gather(images, index)[:, y1:y2, x1:x2, :]
    mixed_images = tf.tensor_scatter_nd_update(
        mixed_images,
        indices=tf.reshape(tf.where(tf.ones_like(mixed_images[:, y1:y2, x1:x2, :], dtype=tf.bool)), [-1, 4]),
        updates=tf.reshape(patch, [-1]),
    )
    lam_adjusted = 1.0 - (tf.cast(x2 - x1, tf.float32) * tf.cast(y2 - y1, tf.float32)) / tf.cast(width * height, tf.float32)
    mixed_labels = lam_adjusted * one_hot(labels) + (1.0 - lam_adjusted) * one_hot(tf.gather(labels, index))
    return mixed_images, mixed_labels


def build_model(model_name, num_classes, dropout=0.2, image_size=224):
    registry = MODEL_REGISTRY.get(model_name)
    if registry is None:
        raise ValueError(f'Unknown model: {model_name}')

    builder = registry['builder']
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name='image')
    backbone = builder(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs, name=model_name)


class EpochCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, fold_dir, train_eval_ds, val_ds, y_train, y_val, model_name, model_cfg):
        super().__init__()
        self.fold_dir = Path(fold_dir)
        self.train_eval_ds = train_eval_ds
        self.val_ds = val_ds
        self.y_train = np.asarray(y_train, dtype=np.int64)
        self.y_val = np.asarray(y_val, dtype=np.int64)
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.history = []
        self.saved = []
        self.best_val_f1 = 0.0
        self.no_improve = 0
        self.fold_dir.mkdir(parents=True, exist_ok=True)

    def _predict(self, ds):
        probs = self.model.predict(ds, verbose=0)
        return np.asarray(probs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_probs = self._predict(self.train_eval_ds)
        val_probs = self._predict(self.val_ds)
        train_pred = train_probs.argmax(axis=1)
        val_pred = val_probs.argmax(axis=1)

        train_m = MetricsComputer.compute_epoch_metrics(self.y_train, train_pred)
        val_m = MetricsComputer.compute_epoch_metrics(self.y_val, val_pred)
        train_m['loss'] = float(logs.get('loss')) if logs.get('loss') is not None else None
        val_m['loss'] = float(logs.get('val_loss')) if logs.get('val_loss') is not None else None
        val_m['generalization_score'] = MetricsComputer.generalization_score(train_m, val_m)
        selection_value = CheckpointSelector.get_selection_metric_value(val_m)

        checkpoint_path = self.fold_dir / f'epoch_{epoch:03d}.keras'
        self.model.save(checkpoint_path)
        save_json(
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
            checkpoint_path.with_suffix('.json'),
        )

        self.saved.append({'epoch': epoch, 'checkpoint_path': str(checkpoint_path), 'selection_value': selection_value})
        self.saved = CheckpointManager.prune_to_top_k_checkpoints(self.saved, CHECKPOINT_KEEP_TOP_K)

        self.history.append({
            'epoch': int(epoch),
            'checkpoint_path': str(checkpoint_path),
            'train': train_m,
            'val': val_m,
            'learning_rate': float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),
            'selection_value': float(selection_value),
        })

        if val_m['f1_macro'] > self.best_val_f1 + 1e-4:
            self.best_val_f1 = val_m['f1_macro']
            self.no_improve = 0
        else:
            self.no_improve += 1
            if self.no_improve >= EARLY_STOP_PATIENCE:
                self.model.stop_training = True


def _make_optimizer(lr):
    return tf.keras.optimizers.Adam(learning_rate=lr)


def train_fold(fold_idx, fold_info, train_df, model_name, mcfg, arch_dir):
    print(f"\n{'=' * 68}")
    print(f"  MODEL={model_name}  FOLD={fold_idx}  train={fold_info['train_count']}  val={fold_info['val_count']}")
    print(f"{'=' * 68}")

    if 'full_path' not in train_df.columns:
        raise ValueError('train_df must include a full_path column before training')

    fold_dir = Path(arch_dir) / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(mcfg['image_size'])
    batch_size = int(mcfg['batch_size'])
    lr = float(mcfg['lr'])
    dropout = float(mcfg['dropout'])

    train_rows = train_df.iloc[fold_info['train_indices']].reset_index(drop=True)
    val_rows = train_df.iloc[fold_info['val_indices']].reset_index(drop=True)
    x_train = train_rows['full_path'].astype(str).to_numpy()
    y_train = train_rows['y'].astype(np.int64).to_numpy()
    x_val = val_rows['full_path'].astype(str).to_numpy()
    y_val = val_rows['y'].astype(np.int64).to_numpy()

    train_ds = build_tf_dataset(x_train, y_train, image_size=image_size, batch_size=batch_size, training=True)
    train_eval_ds = build_tf_dataset(x_train, y_train, image_size=image_size, batch_size=batch_size, training=False, cache=str(fold_dir / 'train_eval.cache'))
    val_ds = build_tf_dataset(x_val, y_val, image_size=image_size, batch_size=batch_size, training=False, cache=str(fold_dir / 'val.cache'))

    with STRATEGY.scope():
        model = build_model(model_name, NUM_CLASSES, dropout=dropout, image_size=image_size)
        model.compile(
            optimizer=_make_optimizer(lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
            jit_compile=(DEVICE_TYPE == 'tpu'),
            steps_per_execution=16 if DEVICE_TYPE == 'tpu' else 1,
        )

    callback = EpochCheckpointCallback(
        fold_dir=fold_dir,
        train_eval_ds=train_eval_ds,
        val_ds=val_ds,
        y_train=y_train,
        y_val=y_val,
        model_name=model_name,
        model_cfg={**mcfg, 'fold_idx': fold_idx, 'image_size': image_size, 'batch_size': batch_size},
    )

    callbacks = [callback]
    if USE_LR_PLATEAU:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                mode='max',
                patience=PLATEAU_PATIENCE,
                factor=PLATEAU_FACTOR,
                min_lr=PLATEAU_MIN_LR,
                verbose=1,
            )
        )

    if not time_budget_ok():
        print('  [TIME BUDGET] Skipping fold before training starts.')
        return []

    model.fit(train_ds, validation_data=val_ds, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=2)

    history = callback.history
    save_json(history, LOG_DIR / f'{model_name[:30]}_history_fold_{fold_idx}.json')

    if history:
        best = max(history, key=lambda row: row['selection_value'])
        best_src = Path(best['checkpoint_path'])
        best_dst_dir = FINAL_MODELS_DIR / model_name
        best_dst_dir.mkdir(parents=True, exist_ok=True)
        best_dst = best_dst_dir / f'fold_{fold_idx}_best.keras'
        shutil.copy2(best_src, best_dst)
        shutil.copy2(best_src.with_suffix('.json'), best_dst.with_suffix('.json'))
        print(f"\n  Done. Best E={best['epoch']}  val_f1={best['val']['f1_macro']:.4f}  gen={best['val']['generalization_score']:.4f}")

    log_resources(f'after fold {fold_idx}')
    gc.collect()
    return history


def load_model_from_ckpt(ckpt_path):
    ckpt_path = Path(ckpt_path)
    model = tf.keras.models.load_model(ckpt_path)
    meta_path = ckpt_path.with_suffix('.json')
    metadata = {}
    if meta_path.exists():
        with meta_path.open('r', encoding='utf-8') as handle:
            metadata = json.load(handle)
    model_name = metadata.get('model_name', ACTIVE_MODELS[0])
    val_metrics = metadata.get('val_metrics', {})
    print(
        f'  [load] {ckpt_path.name}  arch={model_name}  epoch={metadata.get("epoch", "?")}  '
        f'val_f1={val_metrics.get("f1_macro", float("nan")):.4f}  gen={val_metrics.get("generalization_score", float("nan")):.4f}'
    )
    return model, metadata, model_name


def infer_single_pass(model, dataset):
    probs = model.predict(dataset, verbose=0)
    return np.asarray(probs)


def infer_with_tta(model, paths, img_sz, batch_size, n_tta=4):
    cache_path = INFERENCE_DIR / f'test_{img_sz}.cache'
    clean_ds = build_tf_dataset(paths, labels=None, image_size=img_sz, batch_size=batch_size, cache=str(cache_path))
    probs = infer_single_pass(model, clean_ds)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    def _augment_batch(images):
        return augmentation(images, training=True)

    for idx in range(n_tta):
        tta_ds = build_tf_dataset(paths, labels=None, image_size=img_sz, batch_size=batch_size, cache=str(cache_path))
        tta_ds = tta_ds.map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        probs += infer_single_pass(model, tta_ds)
        print(f'    TTA {idx + 1}/{n_tta}')

    return probs / float(n_tta + 1)


def _predict_class_probabilities(model, paths, img_sz, batch_size):
    dataset = build_tf_dataset(paths, labels=None, image_size=img_sz, batch_size=batch_size, cache=str(INFERENCE_DIR / f'test_{img_sz}.cache'))
    return infer_single_pass(model, dataset)


def run_ensemble_inference(train_meta, test_meta, y_train, fold_metadata):
    if test_meta.empty:
        print('test_meta empty -- skipped.')
        return None

    print(f'\nRunning inference on {len(test_meta)} test images ...')
    log_resources('inference start')

    if RUN_INFERENCE_ONLY:
        ckpt_paths = sorted(Path(INFERENCE_ONLY_PATH).glob('**/*best.keras'))
    else:
        ckpt_paths = sorted(FINAL_MODELS_DIR.rglob('fold_*_best.keras'))

    print(f'Found {len(ckpt_paths)} checkpoint(s).')

    all_probs = []
    all_weights = []
    all_oof = []

    test_paths = test_meta['full_path'].astype(str).to_numpy()
    fold_metadata = {int(k): v for k, v in fold_metadata.items()}

    for checkpoint_path in ckpt_paths:
        model, metadata, model_name = load_model_from_ckpt(checkpoint_path)
        img_sz = int(metadata.get('image_size', MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[ACTIVE_MODELS[0]])['image_size']))
        batch_size = int(metadata.get('batch_size', MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[ACTIVE_MODELS[0]])['batch_size']))

        if USE_TTA:
            probs = infer_with_tta(model, test_paths, img_sz, batch_size, TTA_N)
        else:
            probs = _predict_class_probabilities(model, test_paths, img_sz, batch_size)

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
                val_ds = build_tf_dataset(val_paths, labels=None, image_size=img_sz, batch_size=batch_size, cache=str(INFERENCE_DIR / f'fold_{fold_idx}_{img_sz}.cache'))
                oof_probs = infer_single_pass(model, val_ds)
                all_oof.append((fold_idx, val_idx, oof_probs))

        del model
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