from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .config import (
    DEVICE_TYPE,
    DISK_LIMIT_GIB,
    RAM_WARN_GIB,
    SESSION_BUDGET_SECS,
    SESSION_START_TIME,
    WORKING_ROOT,
    PSUTIL_AVAILABLE,
    ensure_dir,
    psutil,
)


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
    from .config import MANIFEST_DIR_PATH

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
    for path in Path(fold_dir).glob('epoch_*.pth'):
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
    from .config import CHECKPOINT_SELECTION_METRIC

    if CHECKPOINT_SELECTION_METRIC == 'val_accuracy':
        return float(metrics.get('accuracy', 0.0))
    return float(metrics.get('generalization_score', metrics.get('f1_macro', 0.0)))
