from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from lib.config.defaults import NUM_CLASSES
from lib.core.logging import data_logger


def build_folds(y: np.ndarray, n_folds: int, seed: int) -> dict[int, dict]:
    meta: dict[int, dict] = {}
    if n_folds == 1:
        indices = np.arange(len(y))
        tr, va = train_test_split(
            indices,
            test_size=0.10,
            stratify=y,
            random_state=seed,
        )
        meta[0] = {
            "fold_idx": 0,
            "train_indices": tr.tolist(),
            "val_indices": va.tolist(),
            "train_count": int(len(tr)),
            "val_count": int(len(va)),
            "train_class_counts": np.bincount(y[tr], minlength=NUM_CLASSES).tolist(),
            "val_class_counts": np.bincount(y[va], minlength=NUM_CLASSES).tolist(),
        }
        data_logger.info(
            f"[single split] train={len(tr)}  val={len(va)}  "
            f"val_cls={meta[0]['val_class_counts']}"
        )
        return meta
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        meta[fold_idx] = {
            "fold_idx": int(fold_idx),
            "train_indices": tr.tolist(),
            "val_indices": va.tolist(),
            "train_count": int(len(tr)),
            "val_count": int(len(va)),
            "train_class_counts": np.bincount(y[tr], minlength=NUM_CLASSES).tolist(),
            "val_class_counts": np.bincount(y[va], minlength=NUM_CLASSES).tolist(),
        }
        data_logger.info(
            f"[fold {fold_idx}] train={len(tr)}  val={len(va)}  "
            f"val_cls={meta[fold_idx]['val_class_counts']}"
        )
    return meta
