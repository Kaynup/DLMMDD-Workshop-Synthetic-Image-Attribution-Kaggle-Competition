from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lib.config.paths import TEST_CSV, TEST_DIR, TRAIN_CSV, TRAIN_DIR
from lib.core.seed import seed_worker
from lib.core.device import NUM_WORKERS, PIN_MEMORY
from lib.data.dataset import ImageDataset


def load_raw_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    def _full_path(split_dir: Path, raw: str) -> str:
        p = Path(str(raw))
        if p.is_absolute() and p.exists():
            return str(p)
        return str(split_dir / p.name)

    train_df["full_path"] = train_df["path"].map(lambda r: _full_path(TRAIN_DIR, r))
    test_df["full_path"] = test_df["path"].map(lambda r: _full_path(TEST_DIR, r))
    return train_df, test_df


def make_loader(
    paths: Sequence[str],
    labels: Sequence[int] | None = None,
    *,
    image_size: int = 224,
    batch_size: int = 32,
    shuffle: bool = False,
    sampler=None,
    drop_last: bool = False,
    fold_seed: int = 0,
) -> DataLoader:
    ds = ImageDataset(paths, labels, image_size=image_size)
    g = torch.Generator()
    g.manual_seed(fold_seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
