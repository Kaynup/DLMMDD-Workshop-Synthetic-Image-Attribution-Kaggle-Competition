from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps

from lib.core.logging import data_logger
from lib.config.constants import IMAGENET_MEAN, IMAGENET_STD, RESAMPLE

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _normalize_np(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def load_image(path: str, image_size: int) -> np.ndarray:
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img = img.resize((image_size, image_size), RESAMPLE)
            arr = np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        raise FileNotFoundError(f"Cannot load image: {path}") from exc
    arr = _normalize_np(arr)
    arr = arr.transpose(2, 0, 1)
    arr = np.ascontiguousarray(arr)
    data_logger.info(
        f"[image_check] shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min():.3f} max={arr.max():.3f}"
    )
    return arr


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths: list[str],
        labels: list[int] | None = None,
        image_size: int = 224,
    ) -> None:
        self.paths = [str(p) for p in paths]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = load_image(self.paths[idx], self.image_size)
        tensor = torch.from_numpy(img)
        if self.labels is None:
            return tensor
        return tensor, int(self.labels[idx])
