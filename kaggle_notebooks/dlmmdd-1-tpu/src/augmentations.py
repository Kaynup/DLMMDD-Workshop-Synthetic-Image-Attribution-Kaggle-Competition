from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import AUTO, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


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
