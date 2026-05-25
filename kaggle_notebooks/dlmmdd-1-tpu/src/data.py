from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from .config import AUTO, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def build_tf_dataset(
    paths,
    labels=None,
    image_size: int = 224,
    batch_size: int = 32,
    training: bool = False,
    cache: bool | str = False,
):
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


def restore_full_path_column(df, split_dir):
    df = df.copy()
    if 'full_path' not in df.columns:
        df['full_path'] = df['path'].apply(lambda p: str(split_dir / Path(str(p))))
    return df
