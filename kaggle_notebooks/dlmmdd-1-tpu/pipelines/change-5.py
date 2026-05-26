"""
Synthetic Source Attribution — End-to-End PyTorch / PyTorch-XLA Pipeline
=========================================================================
Pipeline stages (in order):
  1. System config, globals, imports, seeding
  2. Data loading & validation
  3. Data preprocessing (normalisation, fold splitting)
  4. Model training with per-epoch train + val metrics (CV, no leakage)
  5. Inference / test-set prediction (ensemble)
  6. Submission & outputs
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — SYSTEM CONFIG, GLOBALS, IMPORTS, SEEDING
# ─────────────────────────────────────────────────────────────────────────────

import gc
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from tqdm.auto import tqdm
import psutil

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    raise RuntimeError("timm is required: pip install timm")

# ── XLA execution mode toggles ───────────────────────────────────────────────
# Default = single-process TPU (safe). Set True to spawn one process per TPU core.
USE_XLA_MULTIPROCESSING = False

# ── PyTorch-XLA (optional) ────────────────────────────────────────────────────
try:
    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl

    XLA_AVAILABLE  = True
    DEVICE_TYPE    = "tpu"
    BACKEND        = "pytorch_xla"
    NUM_DEVICES    = max(1, xr.global_device_count())

    if not USE_XLA_MULTIPROCESSING:
        device = torch_xla.device()
        print(f"[XLA] Device            : {device}")
    else:
        device = None
        print("[XLA] Multiprocessing enabled; device will be bound per worker")
    print(f"[XLA] Hardware type     : {xr.device_type()}")
    print(f"[XLA] Global cores      : {xr.global_device_count()}")
    print(f"[XLA] World size (procs): {xr.world_size()}")

except Exception as _xla_err:
    print(f"[XLA] Not available ({_xla_err}); falling back to CPU/CUDA")
    XLA_AVAILABLE  = False
    DEVICE_TYPE    = "cuda" if torch.cuda.is_available() else "cpu"
    BACKEND        = "pytorch"
    device         = torch.device(DEVICE_TYPE)
    NUM_DEVICES    = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1

print(f"DEVICE={DEVICE_TYPE} | BACKEND={BACKEND} | NUM_DEVICES={NUM_DEVICES}")

# ── Paths ─────────────────────────────────────────────────────────────────────
COMPETITION_ROOT = Path(
    "/kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge"
)
DATA_ROOT     = COMPETITION_ROOT / "Data" / "Data"
TRAIN_CSV     = DATA_ROOT / "training.csv"
TEST_CSV      = DATA_ROOT / "test.csv"
TRAIN_DIR     = DATA_ROOT / "Training"
TEST_DIR      = DATA_ROOT / "Test"
WORKING_ROOT  = Path("/kaggle/working")

for _d in [WORKING_ROOT]:
    _d.mkdir(parents=True, exist_ok=True)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

PROCESSED_DIR  = ensure_dir(WORKING_ROOT / "processed")
CHECKPOINT_DIR = ensure_dir(WORKING_ROOT / "checkpoints")
FINAL_DIR      = ensure_dir(WORKING_ROOT / "final_models")
LOG_DIR        = ensure_dir(WORKING_ROOT / "logs")
OUTPUT_DIR     = ensure_dir(WORKING_ROOT / "outputs")
INFERENCE_DIR  = ensure_dir(OUTPUT_DIR   / "inference")
VALIDATION_DIR = ensure_dir(OUTPUT_DIR   / "validation")

# ── Control panel ─────────────────────────────────────────────────────────────
CONTROL_PANEL: dict[str, dict] = {
    "training": {
        "seed": 42,
        "num_classes": 10,
        "num_folds": 5,
        "num_epochs": 10,
        "checkpoint_selection_metric": "generalization_score",  # val_accuracy | val_f1_macro | generalization_score
        "early_stop_patience": 5,
        "checkpoint_keep_top_k": 1,
        "use_lr_plateau": True,
        "plateau_patience": 3,
        "plateau_factor": 0.5,
        "plateau_min_lr": 1e-7,
        "use_bf16": DEVICE_TYPE == "tpu",
    },
    "regularization_defaults": {
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.0,
        "mixup_prob": 0.0,
        "cutmix_alpha": 0.0,
        "cutmix_prob": 0.0,
        "dropout": 0.0,
        "grad_clip_norm": 1.0,
    },
    "blend": {
        "mode": "stacking",          # simple_avg | weighted_avg | stacking
        "weight_metric": "selection_value",  # selection_value | val_f1_macro | accuracy
        "stacking_learner": "logreg", # logreg | ridge | mlp
        "use_tta": False,
        "tta_n": 4,
    },
    "monitoring": {
        "verbose_training": True,
        "train_eval_each_epoch": True,
        "print_xla_memory_every": 20,
    },
    "generalization": {
        "val_weight": 1.0,
        "low_train_reward": 0.15,
        "overfit_penalty": 2.0,
        "balance_penalty": 0.5,
    },
    "resources": {
        "session_budget_secs": 8.5 * 3600,
        "disk_limit_gib": 17.0,
        "ram_warn_gib": 26.0,
    },
    "runtime": {
        "run_training": True,
        "run_inference_only": False,
        "inference_only_path": "/kaggle/input/models/punyakdei/pipe-1-tpu/pytorch/default/1",
    },
}

SEED            = int(CONTROL_PANEL["training"]["seed"])
NUM_CLASSES     = int(CONTROL_PANEL["training"]["num_classes"])
NUM_FOLDS       = int(CONTROL_PANEL["training"]["num_folds"])
NUM_EPOCHS      = int(CONTROL_PANEL["training"]["num_epochs"])
CHECKPOINT_SELECTION_METRIC = str(CONTROL_PANEL["training"]["checkpoint_selection_metric"])
EARLY_STOP_PATIENCE = int(CONTROL_PANEL["training"]["early_stop_patience"])
CHECKPOINT_KEEP_TOP_K = int(CONTROL_PANEL["training"]["checkpoint_keep_top_k"])
USE_LR_PLATEAU  = bool(CONTROL_PANEL["training"]["use_lr_plateau"])
PLATEAU_PATIENCE = int(CONTROL_PANEL["training"]["plateau_patience"])
PLATEAU_FACTOR   = float(CONTROL_PANEL["training"]["plateau_factor"])
PLATEAU_MIN_LR   = float(CONTROL_PANEL["training"]["plateau_min_lr"])
USE_BF16        = bool(CONTROL_PANEL["training"]["use_bf16"])

USE_TTA    = bool(CONTROL_PANEL["blend"]["use_tta"])
TTA_N      = int(CONTROL_PANEL["blend"]["tta_n"])
BLEND_MODE = str(CONTROL_PANEL["blend"]["mode"])
STACKING_LEARNER = str(CONTROL_PANEL["blend"]["stacking_learner"])
BLEND_WEIGHT_METRIC = str(CONTROL_PANEL["blend"]["weight_metric"])

VERBOSE_TRAINING = bool(CONTROL_PANEL["monitoring"]["verbose_training"])
TRAIN_EVAL_EACH_EPOCH = bool(CONTROL_PANEL["monitoring"]["train_eval_each_epoch"])
PRINT_XLA_MEMORY_EVERY = int(CONTROL_PANEL["monitoring"]["print_xla_memory_every"])

SESSION_START_TIME  = time.time()
SESSION_BUDGET_SECS = float(CONTROL_PANEL["resources"]["session_budget_secs"])
DISK_LIMIT_GIB      = float(CONTROL_PANEL["resources"]["disk_limit_gib"])
RAM_WARN_GIB        = float(CONTROL_PANEL["resources"]["ram_warn_gib"])

NUM_WORKERS = 0 if DEVICE_TYPE in {"tpu", "cpu"} else 4
PIN_MEMORY  = False if DEVICE_TYPE in {"tpu", "cpu"} else True

RUN_TRAINING       = bool(CONTROL_PANEL["runtime"]["run_training"])
RUN_INFERENCE_ONLY = bool(CONTROL_PANEL["runtime"]["run_inference_only"])
INFERENCE_ONLY_PATH = str(CONTROL_PANEL["runtime"]["inference_only_path"])

# ── Active models ─────────────────────────────────────────────────────────────
ACTIVE_MODELS = [
    "convnext_base.fb_in22k_ft_in1k",
    "caformer_s36.sail_in22k_ft_in1k",
]

MODEL_REGISTRY: dict[str, dict] = {
    "maxvit_base_tf_384.in21k_ft_in1k": {
        "image_size": 384,
        "batch_size": 8 if DEVICE_TYPE == "cpu" else 64,
        "lr": 3e-4,
        "regularization": {},
    },
    "convnext_base.fb_in22k_ft_in1k": {
        "image_size": 224,
        "batch_size": 16 if DEVICE_TYPE == "cpu" else 32,
        "lr": 8e-4,
        "regularization": {
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "mixup_alpha": 0.2,
            "mixup_prob": 0.5,
            "cutmix_alpha": 0.0,
            "cutmix_prob": 0.0,
            "dropout": 0.0,
            "grad_clip_norm": 1.0,
        },
    },
    "efficientnetv2_m.in21k_ft_in1k": {
        "image_size": 384,
        "batch_size": 8 if DEVICE_TYPE == "cpu" else 64,
        "lr": 6e-4,
        "regularization": {},
    },
    "swin_base_patch4_window12_384.ms_in22k_ft_in1k": {
        "image_size": 384,
        "batch_size": 8 if DEVICE_TYPE == "cpu" else 64,
        "lr": 6e-4,
        "regularization": {},
    },
    "caformer_s36.sail_in22k_ft_in1k": {
        "image_size": 224,
        "batch_size": 16 if DEVICE_TYPE == "cpu" else 48,
        "lr": 3e-4,
        "regularization": {
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "mixup_alpha": 0.2,
            "mixup_prob": 0.5,
            "cutmix_alpha": 0.0,
            "cutmix_prob": 0.0,
            "dropout": 0.0,
            "grad_clip_norm": 1.0,
        },
    },
}

# ── ImageNet stats ────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESAMPLE      = Image.Resampling.BICUBIC

# ── Logging ───────────────────────────────────────────────────────────────────
def _setup_logging(log_dir: Path = LOG_DIR) -> logging.Logger:
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh  = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
        fh  = logging.FileHandler(
            log_dir / f"run_{pd.Timestamp.now('UTC'):%Y%m%dT%H%M%SZ}.log",
            encoding="utf-8"
        )
        fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

LOGGER = _setup_logging()
LOGGER.info(f"DEVICE={DEVICE_TYPE} | BACKEND={BACKEND} | NUM_DEVICES={NUM_DEVICES}")
LOGGER.info(f"epochs={NUM_EPOCHS} seed={SEED} | control_panel=enabled")

# ── Seeding ───────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ── Utility helpers ───────────────────────────────────────────────────────────
def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)

def xla_mark_step() -> None:
    """Call xm.mark_step() only when XLA is active."""
    if XLA_AVAILABLE:
        torch_xla.sync()

def xla_rendezvous(tag: str = "sync") -> None:
    if XLA_AVAILABLE:
        xm.rendezvous(tag)


def resolve_model_settings(model_name: str, mcfg: dict | None = None) -> dict:
    """Merge per-model settings with the control-panel defaults."""
    base = dict(MODEL_REGISTRY[model_name] if mcfg is None else mcfg)
    reg = dict(CONTROL_PANEL["regularization_defaults"])
    reg.update(base.get("regularization", {}))
    base["regularization"] = reg
    return base


def build_weighting_value(meta: dict) -> float:
    """Value used by weighted averaging."""
    if BLEND_WEIGHT_METRIC == "val_f1_macro":
        return float(meta.get("val_metrics", {}).get("f1_macro", meta.get("selection_value", 0.0)))
    if BLEND_WEIGHT_METRIC == "accuracy":
        return float(meta.get("val_metrics", {}).get("accuracy", meta.get("selection_value", 0.0)))
    return float(meta.get("selection_value", meta.get("val_metrics", {}).get("f1_macro", 0.0)))


def make_stacking_learner(name: str):
    name = name.lower().strip()
    if name == "logreg":
        return LogisticRegression(max_iter=3000, multi_class="multinomial", solver="lbfgs")
    if name == "ridge":
        return RidgeClassifier()
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256,), activation="relu", max_iter=500, random_state=SEED)
    raise ValueError(f"Unknown STACKING_LEARNER={name}")


def predict_proba_from_learner(learner, x: np.ndarray) -> np.ndarray:
    if hasattr(learner, "predict_proba"):
        return np.asarray(learner.predict_proba(x), dtype=np.float32)
    scores = np.asarray(learner.decision_function(x), dtype=np.float32)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)

def is_master_process() -> bool:
    if not XLA_AVAILABLE:
        return True
    if not USE_XLA_MULTIPROCESSING:
        return True
    return xm.is_master_ordinal(local=False)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_csvs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read training.csv and test.csv; resolve full image paths."""
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    def _full_path(split_dir: Path, raw: str) -> str:
        p = Path(str(raw))
        if p.is_absolute() and p.exists():
            return str(p)
        return str(split_dir / p.name)

    train_df["full_path"] = train_df["path"].map(lambda r: _full_path(TRAIN_DIR, r))
    test_df["full_path"]  = test_df["path"].map(lambda r: _full_path(TEST_DIR,  r))
    return train_df, test_df


def validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Hard checks before any GPU/TPU work starts."""
    LOGGER.info("=" * 60)
    LOGGER.info("STAGE 2 — DATA LOADING & VALIDATION")
    LOGGER.info("=" * 60)
    LOGGER.info(f"train rows={len(train_df)}  test rows={len(test_df)}")

    # Label column
    assert "y" in train_df.columns, "training.csv must contain a 'y' column"
    counts = train_df["y"].value_counts().sort_index()
    LOGGER.info(f"Class distribution:\n{counts.to_string()}")
    if len(counts) != NUM_CLASSES:
        LOGGER.warning(f"Expected {NUM_CLASSES} classes; found {len(counts)}")

    # Missing files
    missing_train = (~train_df["full_path"].map(lambda p: Path(p).exists())).sum()
    missing_test  = (~test_df["full_path"].map(lambda p: Path(p).exists())).sum()
    LOGGER.info(f"Missing train images : {missing_train}")
    LOGGER.info(f"Missing test  images : {missing_test}")
    if missing_train:
        raise FileNotFoundError(
            f"{missing_train} training images not found. "
            f"Sample: {train_df.loc[~train_df['full_path'].map(lambda p: Path(p).exists()), 'full_path'].head(3).tolist()}"
        )
    if missing_test:
        raise FileNotFoundError(
            f"{missing_test} test images not found."
        )

    LOGGER.info("Data validation passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_np(arr: np.ndarray) -> np.ndarray:
    """uint8 HWC → float32 HWC, ImageNet-normalised."""
    arr = arr.astype(np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


_image_stats_logged = False

def load_image(path: str, image_size: int) -> np.ndarray:
    """
    Load, resize, normalise one image.
    Returns float32 array of shape (C, H, W) — channels-first for PyTorch.
    """
    global _image_stats_logged
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img = img.resize((image_size, image_size), RESAMPLE)
            arr = np.asarray(img, dtype=np.uint8)          # (H, W, 3)  uint8
    except Exception as exc:
        raise FileNotFoundError(f"Cannot load image: {path}") from exc

    arr = _normalize_np(arr)                                # (H, W, 3)  float32
    arr = arr.transpose(2, 0, 1)                            # (3, H, W)  ← PyTorch expects channels-first
    arr = np.ascontiguousarray(arr)

    if not _image_stats_logged:
        LOGGER.info(
            f"[image_check] shape={arr.shape} dtype={arr.dtype} "
            f"min={arr.min():.3f} max={arr.max():.3f}"
        )
        _image_stats_logged = True
    return arr


class ImageDataset(Dataset):
    """Minimal dataset: loads images on the fly (no caching to save RAM)."""

    def __init__(
        self,
        paths: Sequence[str],
        labels: Sequence[int] | None = None,
        image_size: int = 224,
    ):
        self.paths      = [str(p) for p in paths]
        self.labels     = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = load_image(self.paths[idx], self.image_size)  # (3, H, W) float32
        t   = torch.from_numpy(img)
        if self.labels is None:
            return t
        return t, int(self.labels[idx])


def make_loader(
    paths,
    labels=None,
    *,
    image_size: int = 224,
    batch_size: int = 32,
    shuffle: bool = False,
    sampler=None,
    drop_last: bool = False,
) -> DataLoader:
    ds = ImageDataset(paths, labels, image_size=image_size)
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
    )


def build_folds(y: np.ndarray, n_folds: int, seed: int) -> dict[int, dict]:
    """Stratified K-Fold — returns dict[fold_idx → {train_indices, val_indices, …}]."""
    skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    meta = {}
    for fold_idx, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        meta[fold_idx] = {
            "fold_idx"          : int(fold_idx),
            "train_indices"     : tr.tolist(),
            "val_indices"       : va.tolist(),
            "train_count"       : int(len(tr)),
            "val_count"         : int(len(va)),
            "train_class_counts": np.bincount(y[tr], minlength=NUM_CLASSES).tolist(),
            "val_class_counts"  : np.bincount(y[va], minlength=NUM_CLASSES).tolist(),
        }
        LOGGER.info(
            f"[fold {fold_idx}] train={len(tr)}  val={len(va)}  "
            f"val_cls={meta[fold_idx]['val_class_counts']}"
        )
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class TimmModel(nn.Module):
    """timm backbone with a linear head for NUM_CLASSES."""

    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        LOGGER.info(f"[model] Loaded {model_name} (num_classes={num_classes}, dropout={dropout})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (N, C, H, W)  float32
        return self.backbone(x)


def build_model(model_name: str, num_classes: int = NUM_CLASSES) -> TimmModel:
    cfg = resolve_model_settings(model_name)
    dropout = float(cfg["regularization"].get("dropout", 0.0))
    return TimmModel(model_name, num_classes, dropout)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — TRAINING  (with per-epoch train + val metrics, no data leakage)
# ─────────────────────────────────────────────────────────────────────────────

# ── Label-smoothed cross-entropy (works with one-hot soft targets) ─────────────
def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    confidence = 1.0 - smoothing
    with torch.no_grad():
        oh = torch.zeros(labels.size(0), num_classes, device=labels.device)
        oh.fill_(smoothing / (num_classes - 1))
        oh.scatter_(1, labels.unsqueeze(1), confidence)
    return oh


def cross_entropy_soft(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    return -(soft_labels * log_p).sum(dim=-1).mean()


# ── Regularized batch mixing ──────────────────────────────────────────────────
def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float, p: float = 1.0):
    """Standard mixup: one Beta sample, one permutation, convex label mix."""
    if alpha <= 0.0 or p <= 0.0 or images.size(0) < 2 or random.random() > p:
        return images, labels
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    mixed_x = lam * images + (1.0 - lam) * images[perm]
    mixed_y = lam * labels + (1.0 - lam) * labels[perm]
    return mixed_x, mixed_y


def cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float, p: float = 1.0):
    """Standard CutMix for NCHW tensors and soft targets."""
    if alpha <= 0.0 or p <= 0.0 or images.size(0) < 2 or random.random() > p:
        return images, labels

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)

    b, c, h, w = images.shape
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = random.randint(0, w - 1)
    cy = random.randint(0, h - 1)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)

    mixed_x = images.clone()
    mixed_x[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

    area = max(1, (x2 - x1) * (y2 - y1))
    lam_adjusted = 1.0 - area / float(h * w)
    mixed_y = lam_adjusted * labels + (1.0 - lam_adjusted) * labels[perm]
    return mixed_x, mixed_y


def apply_batch_regularization(
    images: torch.Tensor,
    soft_labels: torch.Tensor,
    reg: dict,
):
    """Apply CutMix or MixUp according to the control-panel settings."""
    cutmix_alpha = float(reg.get("cutmix_alpha", 0.0))
    cutmix_prob = float(reg.get("cutmix_prob", 0.0))
    mixup_alpha = float(reg.get("mixup_alpha", 0.0))
    mixup_prob = float(reg.get("mixup_prob", 0.0))

    if cutmix_alpha > 0.0 and random.random() < cutmix_prob:
        return cutmix(images, soft_labels, cutmix_alpha, p=1.0)
    if mixup_alpha > 0.0 and random.random() < mixup_prob:
        return mixup(images, soft_labels, mixup_alpha, p=1.0)
    return images, soft_labels


# ── Metric helpers ────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy"   : float(accuracy_score(y_true, y_pred)),
        "f1_macro"   : float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def generalization_score(train_m: dict, val_m: dict, *, cfg: dict | None = None, return_parts: bool = False):
    """Reward higher val F1, reward val > train, and penalize train > val overfit."""
    gcfg = CONTROL_PANEL["generalization"] if cfg is None else cfg
    train_f1 = float(train_m.get("f1_macro", 0.0))
    val_f1 = float(val_m.get("f1_macro", 0.0))

    val_weight = float(gcfg["val_weight"])
    low_train_reward = float(gcfg["low_train_reward"])
    overfit_penalty = float(gcfg["overfit_penalty"])
    balance_penalty = float(gcfg["balance_penalty"])

    positive_gap = max(0.0, val_f1 - train_f1)
    negative_gap = max(0.0, train_f1 - val_f1)
    balance_gap = abs(train_f1 - val_f1)

    score = (
        val_weight * val_f1
        + low_train_reward * positive_gap
        - overfit_penalty * (negative_gap ** 2)
        - balance_penalty * (balance_gap ** 2)
    )

    parts = {
        "train_f1": train_f1,
        "val_f1": val_f1,
        "positive_gap": positive_gap,
        "negative_gap": negative_gap,
        "balance_gap": balance_gap,
        "val_term": val_weight * val_f1,
        "low_train_reward": low_train_reward * positive_gap,
        "overfit_penalty_term": overfit_penalty * (negative_gap ** 2),
        "balance_penalty_term": balance_penalty * (balance_gap ** 2),
        "score": float(score),
    }
    return (float(score), parts) if return_parts else float(score)


def get_selection_value(train_m: dict, val_m: dict) -> float:
    metric = CHECKPOINT_SELECTION_METRIC.lower().strip()
    if metric == "val_accuracy":
        return float(val_m.get("accuracy", 0.0))
    if metric == "val_f1_macro":
        return float(val_m.get("f1_macro", 0.0))
    if metric == "generalization_score":
        return float(val_m.get("generalization_score", val_m.get("f1_macro", 0.0)))
    return float(val_m.get(metric, val_m.get("f1_macro", 0.0)))


# ── Resource helpers ──────────────────────────────────────────────────────────
def _disk_gib() -> float:
    return sum(f.stat().st_size for f in WORKING_ROOT.rglob("*") if f.is_file()) / 1024 ** 3

def _ram_gib() -> float:
    return psutil.virtual_memory().used / 1024 ** 3

def _elapsed_h() -> float:
    return (time.time() - SESSION_START_TIME) / 3600

def _remaining_h() -> float:
    return (SESSION_BUDGET_SECS - (time.time() - SESSION_START_TIME)) / 3600

def log_resources(tag: str = "") -> None:
    disk, ram = _disk_gib(), _ram_gib()
    LOGGER.info(
        f"[resources{' ' + tag if tag else ''}] "
        f"disk={disk:.2f}GiB  ram={ram:.1f}GiB  "
        f"elapsed={_elapsed_h():.2f}h  remaining={_remaining_h():.2f}h"
    )
    if disk > DISK_LIMIT_GIB:
        raise RuntimeError(f"DISK LIMIT EXCEEDED: {disk:.2f} > {DISK_LIMIT_GIB} GiB")
    if ram > RAM_WARN_GIB:
        LOGGER.warning(f"[ram] High RAM usage: {ram:.1f} GiB")

def budget_ok() -> bool:
    return (time.time() - SESSION_START_TIME) < SESSION_BUDGET_SECS


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(path: Path, model: nn.Module, optimizer, meta: dict) -> None:
    ensure_dir(path.parent)
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metadata": meta}, path)
    save_json(meta, path.with_suffix(".json"))


def load_checkpoint(path: Path, model: nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt.get("metadata", {})


def prune_checkpoints(saved: list, keep: int) -> list:
    while len(saved) > keep:
        weakest = min(saved, key=lambda r: r["sv"])
        p = Path(weakest["path"])
        if p.exists(): p.unlink()
        jf = p.with_suffix(".json")
        if jf.exists(): jf.unlink()
        LOGGER.info(f"[prune] removed {p.name}  sv={weakest['sv']:.4f}")
        saved.remove(weakest)
    return saved


# ── Per-split predictions ─────────────────────────────────────────────────────
@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_true = []

    for batch in loader:
        has_labels = isinstance(batch, (list, tuple)) and len(batch) == 2
        if has_labels:
            imgs, lbls = batch
            all_true.append(lbls.cpu().numpy().copy())
        else:
            imgs = batch

        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=-1)
        probs_np = probs.detach().cpu().numpy().copy()
        all_probs.append(probs_np)

        del imgs, logits, probs
        if has_labels:
            del lbls
        xla_mark_step()

    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, NUM_CLASSES), dtype=np.float32)
    preds_arr = probs_arr.argmax(axis=1) if len(probs_arr) else np.array([], dtype=np.int64)
    true_arr = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int64)
    return true_arr, preds_arr, probs_arr


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion=None,
    label_smoothing: float = 0.0,
) -> dict:
    """Return loss, metrics, and probabilities for a full evaluation pass."""
    model.eval()
    all_probs = []
    all_true = []
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        has_labels = isinstance(batch, (list, tuple)) and len(batch) == 2
        if has_labels:
            imgs, lbls = batch
            all_true.append(lbls.cpu().numpy().copy())
            lbls_dev = lbls.to(device)
        else:
            imgs = batch
            lbls_dev = None

        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=-1)
        probs_np = probs.detach().cpu().numpy().copy()
        all_probs.append(probs_np)

        if has_labels and criterion is not None:
            soft = smooth_one_hot(lbls_dev, NUM_CLASSES, label_smoothing)
            loss = criterion(logits, soft)
            bs = int(imgs.size(0))
            total_loss += float(loss.item()) * bs
            total_count += bs

        del imgs, logits, probs
        if has_labels:
            del lbls, lbls_dev
        xla_mark_step()

    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, NUM_CLASSES), dtype=np.float32)
    preds_arr = probs_arr.argmax(axis=1) if len(probs_arr) else np.array([], dtype=np.int64)
    true_arr = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int64)

    metrics = compute_metrics(true_arr, preds_arr) if len(true_arr) else {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}
    metrics["loss"] = float(total_loss / max(total_count, 1)) if total_count else None
    metrics["probs"] = probs_arr
    metrics["y_true"] = true_arr
    metrics["y_pred"] = preds_arr
    return metrics


# ── Core training loop for one fold ───────────────────────────────────────────
def train_fold(
    fold_idx: int,
    fold_info: dict,
    train_df: pd.DataFrame,
    model_name: str,
    mcfg: dict,
    arch_ckpt_dir: Path,
    rank: int = 0,
    world_size: int = 1,
) -> list[dict]:

    cfg = resolve_model_settings(model_name, mcfg)
    reg = cfg["regularization"]

    LOGGER.info("=" * 68)
    LOGGER.info(
        f"STAGE 4 — TRAINING | model={model_name} | fold={fold_idx} | "
        f"train={fold_info['train_count']}  val={fold_info['val_count']}"
    )
    LOGGER.info("=" * 68)
    LOGGER.info(f"[fold {fold_idx}] model_cfg = {json.dumps(cfg, indent=2, default=str)}")
    LOGGER.info(f"[fold {fold_idx}] regularization = {json.dumps(reg, indent=2, default=str)}")

    img_sz     = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    lr         = float(cfg["lr"])
    weight_decay = float(reg.get("weight_decay", 0.0))
    label_smoothing = float(reg.get("label_smoothing", 0.0))
    grad_clip_norm = float(reg.get("grad_clip_norm", 1.0))
    fold_dir   = ensure_dir(arch_ckpt_dir / f"fold_{fold_idx}")

    tr_rows = train_df.iloc[fold_info["train_indices"]].reset_index(drop=True)
    va_rows = train_df.iloc[fold_info["val_indices"]].reset_index(drop=True)

    x_tr = tr_rows["full_path"].astype(str).to_numpy()
    y_tr = tr_rows["y"].astype(np.int64).to_numpy()
    x_va = va_rows["full_path"].astype(str).to_numpy()
    y_va = va_rows["y"].astype(np.int64).to_numpy()

    LOGGER.info(f"[fold {fold_idx}] train_sample: {x_tr[0]}")
    LOGGER.info(f"[fold {fold_idx}] val_sample  : {x_va[0]}")

    train_ds = ImageDataset(x_tr, y_tr, image_size=img_sz)
    val_ds   = ImageDataset(x_va, y_va, image_size=img_sz)

    if XLA_AVAILABLE and USE_XLA_MULTIPROCESSING:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        train_loader = make_loader(x_tr, y_tr, image_size=img_sz, batch_size=batch_size, shuffle=False, sampler=train_sampler, drop_last=True)
        train_loader = pl.MpDeviceLoader(train_loader, device)
        train_eval_loader = make_loader(x_tr, y_tr, image_size=img_sz, batch_size=batch_size, shuffle=False, drop_last=False)
        val_loader = make_loader(x_va, y_va, image_size=img_sz, batch_size=batch_size, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        train_loader = make_loader(x_tr, y_tr, image_size=img_sz, batch_size=batch_size, shuffle=True, drop_last=True)
        train_eval_loader = make_loader(x_tr, y_tr, image_size=img_sz, batch_size=batch_size, shuffle=False, drop_last=False)
        val_loader = make_loader(x_va, y_va, image_size=img_sz, batch_size=batch_size, shuffle=False, drop_last=False)

    model     = build_model(model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR
        ) if USE_LR_PLATEAU else None
    )

    criterion = cross_entropy_soft

    history        : list[dict] = []
    saved_ckpts    : list[dict] = []
    best_sel_value = -float("inf")
    no_improve_cnt = 0

    for epoch in range(NUM_EPOCHS):
        if not budget_ok():
            LOGGER.info("[budget] Time limit reached — stopping training early.")
            break

        epoch_start  = time.time()
        epoch_losses : list[float] = []
        epoch_accs   : list[float] = []
        epoch_grad_norms : list[float] = []

        model.train()
        for step, (imgs, lbls) in enumerate(train_loader):
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            soft = smooth_one_hot(lbls, NUM_CLASSES, label_smoothing)
            imgs, soft = apply_batch_regularization(imgs, soft, reg)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("xla", dtype=torch.bfloat16, enabled=USE_BF16):
                logits = model(imgs)
                loss = criterion(logits, soft)

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm).item())
            optimizer.step()
            xla_mark_step()

            with torch.no_grad():
                preds = logits.argmax(1)
                acc_val = float((preds == lbls).float().mean().detach().cpu().item())

            loss_val = float(loss.detach().cpu().item())
            epoch_losses.append(loss_val)
            epoch_accs.append(acc_val)
            epoch_grad_norms.append(grad_norm)

            if np.isnan(loss_val):
                LOGGER.warning(f"[train] NaN loss at epoch={epoch} step={step}")

            if VERBOSE_TRAINING and (step % 20 == 0):
                LOGGER.info(
                    f"[train] epoch={epoch:02d} step={step:04d} "
                    f"loss={loss_val:.4f}  acc={acc_val:.4f}  grad_norm={grad_norm:.4f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                if XLA_AVAILABLE and PRINT_XLA_MEMORY_EVERY > 0:
                    if step % PRINT_XLA_MEMORY_EVERY == 0:
                        try:
                            LOGGER.info(f"[xla_memory] {xm.get_memory_info()}")
                        except Exception as mem_exc:
                            LOGGER.info(f"[xla_memory] unavailable: {mem_exc}")

            del imgs, lbls, soft, logits, loss, preds

        train_eval = None
        val_eval = None
        if TRAIN_EVAL_EACH_EPOCH:
            train_eval = evaluate_loader(model, train_eval_loader, criterion=criterion, label_smoothing=label_smoothing)
        val_eval = evaluate_loader(model, val_loader, criterion=criterion, label_smoothing=label_smoothing)

        train_m = train_eval if train_eval is not None else {
            "accuracy": float(np.mean(epoch_accs)) if epoch_accs else 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "probs": np.empty((0, NUM_CLASSES), dtype=np.float32),
            "y_true": np.array([], dtype=np.int64),
            "y_pred": np.array([], dtype=np.int64),
        }
        val_m = val_eval
        train_m.pop("probs", None); train_m.pop("y_true", None); train_m.pop("y_pred", None)
        val_m.pop("probs", None); val_m.pop("y_true", None); val_m.pop("y_pred", None)

        train_m["loss"] = float(np.mean(epoch_losses)) if epoch_losses else None
        val_m["generalization_score"], gen_parts = generalization_score(train_m, val_m, return_parts=True)
        sv = get_selection_value(train_m, val_m)

        per_class_f1 = f1_score(val_eval["y_true"], val_eval["y_pred"], average=None, zero_division=0) if len(val_eval["y_true"]) else np.zeros(NUM_CLASSES, dtype=np.float32)
        worst3 = sorted(enumerate(per_class_f1), key=lambda t: t[1])[:3]

        epoch_time    = time.time() - epoch_start
        imgs_per_sec  = len(x_tr) / epoch_time if epoch_time > 0 else 0.0

        train_gap = train_m["f1_macro"] - val_m["f1_macro"]
        LOGGER.info(
            f"\n{'─'*96}\n"
            f"  Epoch {epoch:02d}/{NUM_EPOCHS-1}  |  fold={fold_idx}  |  model={model_name}\n"
            f"  TRAIN loss={train_m['loss']:.4f}  acc={train_m['accuracy']:.4f}  f1={train_m['f1_macro']:.4f}  "
            f"f1w={train_m['f1_weighted']:.4f}\n"
            f"  VAL   loss={val_m['loss']:.4f}  acc={val_m['accuracy']:.4f}  f1={val_m['f1_macro']:.4f}  "
            f"f1w={val_m['f1_weighted']:.4f}\n"
            f"  gap(train-val)={train_gap:+.4f}  gen_score={val_m['generalization_score']:.4f}  "
            f"selection_metric={CHECKPOINT_SELECTION_METRIC}  selection_value={sv:.4f}\n"
            f"  gen_parts: val_term={gen_parts['val_term']:.4f}  low_train_reward={gen_parts['low_train_reward']:.4f}  "
            f"overfit_penalty={gen_parts['overfit_penalty_term']:.4f}  balance_penalty={gen_parts['balance_penalty_term']:.4f}\n"
            f"  throughput={imgs_per_sec:.0f} img/s  lr={optimizer.param_groups[0]['lr']:.2e}  grad_clip={grad_clip_norm:.2f}\n"
            f"  Worst-3 val classes: {worst3}\n"
            f"{'─'*96}"
        )

        ckpt_path = fold_dir / f"epoch_{epoch:03d}.pt"
        meta = {
            "model_name": model_name, "fold_idx": fold_idx, "epoch": epoch,
            "image_size": img_sz, "batch_size": batch_size,
            "train_metrics": train_m, "val_metrics": val_m,
            "selection_metric": CHECKPOINT_SELECTION_METRIC,
            "selection_value": float(sv),
            "generalization_parts": gen_parts,
            "regularization": reg,
        }
        save_checkpoint(ckpt_path, model, optimizer, meta)
        saved_ckpts.append({"path": str(ckpt_path), "sv": float(sv)})
        saved_ckpts = prune_checkpoints(saved_ckpts, CHECKPOINT_KEEP_TOP_K)
        history.append({"epoch": epoch, "train": train_m, "val": val_m, "sv": float(sv), "generalization_parts": gen_parts})

        if scheduler:
            scheduler.step(val_m["f1_macro"])

        if sv > best_sel_value + 1e-4:
            best_sel_value = sv
            no_improve_cnt = 0
            best_dst = ensure_dir(FINAL_DIR / model_name) / f"fold_{fold_idx}_best.pt"
            shutil.copy2(ckpt_path, best_dst)
            shutil.copy2(ckpt_path.with_suffix(".json"), best_dst.with_suffix(".json"))
            LOGGER.info(
                f"[best] fold={fold_idx} epoch={epoch} selection_value={best_sel_value:.4f} "
                f"(metric={CHECKPOINT_SELECTION_METRIC}) → {best_dst.name}"
            )
        else:
            no_improve_cnt += 1
            LOGGER.info(f"[early_stop] no improvement {no_improve_cnt}/{EARLY_STOP_PATIENCE} | best_selection={best_sel_value:.4f}")
            if no_improve_cnt >= EARLY_STOP_PATIENCE:
                LOGGER.info(f"[early_stop] Triggered at epoch {epoch}. Stopping fold {fold_idx}.")
                break

        LOGGER.info(f"[resources] epoch={epoch}")
        log_resources(f"fold={fold_idx} epoch={epoch}")

        del train_eval, val_eval, train_m, val_m
        gc.collect()

    save_json(history, LOG_DIR / f"{model_name[:30]}_fold{fold_idx}_history.json")

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    log_resources(f"after fold {fold_idx}")
    return history
# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — INFERENCE / TEST-SET PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_ckpt(ckpt_path: Path) -> tuple[nn.Module, dict, str]:
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    meta      = ckpt.get("metadata", {})
    model_name = meta.get("model_name", ACTIVE_MODELS[0])
    val_m      = meta.get("val_metrics", {})
    LOGGER.info(
        f"[load_ckpt] {ckpt_path.name}  arch={model_name}  "
        f"epoch={meta.get('epoch','?')}  "
        f"val_f1={val_m.get('f1_macro', float('nan')):.4f}  "
        f"gen={val_m.get('generalization_score', float('nan')):.4f}"
    )
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, meta, model_name


@torch.no_grad()
def infer_single(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(device)
        p    = F.softmax(model(imgs), dim=-1).cpu().numpy()
        probs.append(p)
        xla_mark_step()
    return np.concatenate(probs, axis=0) if probs else np.array([])


@torch.no_grad()
def infer_tta(model: nn.Module, paths, img_sz: int, batch_size: int, n: int) -> np.ndarray:
    base_loader = make_loader(paths, image_size=img_sz, batch_size=batch_size)
    acc = infer_single(model, base_loader)
    for t in range(n):
        aug_probs = []
        for batch in base_loader:
            imgs = (batch[0] if isinstance(batch, (list, tuple)) else batch).numpy()
            if t % 2 == 0:
                imgs = np.flip(imgs, axis=3).copy()   # horizontal flip
            p = F.softmax(model(torch.from_numpy(imgs).float().to(device)), dim=-1).cpu().numpy()
            aug_probs.append(p)
            xla_mark_step()
        if aug_probs:
            acc = acc + np.concatenate(aug_probs, axis=0)
        LOGGER.info(f"[TTA] pass {t+1}/{n}")
    return acc / (n + 1)


def run_ensemble_inference(
    train_meta: pd.DataFrame,
    test_meta:  pd.DataFrame,
    y_train:    np.ndarray,
    fold_metadata: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int] | None:

    LOGGER.info("=" * 60)
    LOGGER.info("STAGE 6 — INFERENCE / TEST-SET PREDICTION")
    LOGGER.info("=" * 60)
    LOGGER.info(f"[blend] mode={BLEND_MODE} weight_metric={BLEND_WEIGHT_METRIC} stacking_learner={STACKING_LEARNER}")

    if test_meta.empty:
        LOGGER.warning("test_meta is empty — skipping inference.")
        return None

    if RUN_INFERENCE_ONLY:
        ckpt_paths = sorted(Path(INFERENCE_ONLY_PATH).glob("**/*best.pt"))
    else:
        ckpt_paths = sorted(FINAL_DIR.rglob("fold_*_best.pt"))

    LOGGER.info(f"[inference] Found {len(ckpt_paths)} checkpoint(s).")
    if not ckpt_paths:
        LOGGER.error("No checkpoints found — cannot produce submission.")
        return None

    test_paths  = test_meta["full_path"].astype(str).to_numpy()
    all_probs   : list[np.ndarray] = []
    all_weights : list[float]      = []

    if BLEND_MODE in {"simple_avg", "weighted_avg"}:
        for idx, ckpt_path in enumerate(ckpt_paths):
            model, meta, model_name = load_model_from_ckpt(ckpt_path)
            img_sz     = int(meta.get("image_size", MODEL_REGISTRY[model_name]["image_size"]))
            batch_size = int(meta.get("batch_size", MODEL_REGISTRY[model_name]["batch_size"]))

            t0 = time.time()
            if USE_TTA:
                probs = infer_tta(model, test_paths, img_sz, batch_size, TTA_N)
            else:
                loader = make_loader(test_paths, image_size=img_sz, batch_size=batch_size)
                probs  = infer_single(model, loader)
            elapsed = time.time() - t0

            LOGGER.info(
                f"[predict] ckpt={idx}  model={model_name}  shape={probs.shape}  "
                f"speed={len(test_paths)/elapsed:.1f} img/s"
            )

            all_probs.append(probs)
            weight = build_weighting_value(meta)
            all_weights.append(max(float(weight), 1e-6))
            LOGGER.info(f"  weight={weight:.4f}  selection_value={meta.get('selection_value', float('nan')):.4f}")

            del model; gc.collect()

        stacked = np.stack(all_probs, axis=0)
        weights = np.asarray(all_weights, dtype=np.float32)
        weights = np.nan_to_num(weights, nan=1e-6)

        if BLEND_MODE == "simple_avg":
            ensemble = stacked.mean(axis=0)
            LOGGER.info("[ensemble] simple_avg applied.")
        elif BLEND_MODE == "weighted_avg":
            w = weights / weights.sum()
            LOGGER.info(f"[ensemble] weighted_avg weights={w.round(4).tolist()}")
            ensemble = np.average(stacked, axis=0, weights=w)
        else:
            w = weights / weights.sum()
            LOGGER.info(f"[ensemble] fallback weighted_avg weights={w.round(4).tolist()}")
            ensemble = np.average(stacked, axis=0, weights=w)

        preds = ensemble.argmax(axis=1).astype(int)
        conf  = ensemble.max(axis=1)
        LOGGER.info(
            f"[ensemble] models={len(all_probs)}  "
            f"mean_conf={conf.mean():.4f}  min={conf.min():.4f}  max={conf.max():.4f}"
        )
        return ensemble, preds, conf, test_meta, len(ckpt_paths)

    model_ckpts: dict[str, list[Path]] = {}
    for model_name in ACTIVE_MODELS:
        model_ckpts[model_name] = sorted((FINAL_DIR / model_name).glob("fold_*_best.pt"))
        LOGGER.info(f"[stacking] model={model_name} ckpts={len(model_ckpts[model_name])}")

    n_train = len(train_meta)
    n_classes = NUM_CLASSES
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []

    for model_name, ckpts in model_ckpts.items():
        if not ckpts:
            raise RuntimeError(f"No checkpoints for stacking: {model_name}")

        model_oof = np.zeros((n_train, n_classes), dtype=np.float32)
        model_test_fold_probs = []

        for ckpt_path in ckpts:
            model, meta, loaded_name = load_model_from_ckpt(ckpt_path)
            fold_idx = int(meta.get("fold_idx", -1))
            if fold_idx not in fold_metadata:
                raise RuntimeError(f"Fold metadata missing for fold_idx={fold_idx}")

            img_sz     = int(meta.get("image_size", MODEL_REGISTRY[loaded_name]["image_size"]))
            batch_size = int(meta.get("batch_size", MODEL_REGISTRY[loaded_name]["batch_size"]))

            va_idx = fold_metadata[fold_idx]["val_indices"]
            va_rows = train_meta.iloc[va_idx].reset_index(drop=True)
            va_loader = make_loader(
                va_rows["full_path"].astype(str).to_numpy(),
                va_rows["y"].astype(np.int64).to_numpy(),
                image_size=img_sz,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
            _, _, va_probs = predict_loader(model, va_loader)
            model_oof[va_idx] = va_probs
            LOGGER.info(f"[stacking] oof filled model={model_name} fold={fold_idx} block={va_probs.shape}")

            test_loader = make_loader(test_paths, image_size=img_sz, batch_size=batch_size, shuffle=False, drop_last=False)
            _, _, te_probs = predict_loader(model, test_loader)
            model_test_fold_probs.append(te_probs)
            del model; gc.collect()

        model_test_avg = np.mean(np.stack(model_test_fold_probs, axis=0), axis=0).astype(np.float32)
        train_blocks.append(model_oof)
        test_blocks.append(model_test_avg)
        LOGGER.info(
            f"[stacking] model={model_name} train_block={model_oof.shape} test_block={model_test_avg.shape}"
        )

    X_train = np.concatenate(train_blocks, axis=1)
    X_test = np.concatenate(test_blocks, axis=1)
    LOGGER.info(f"[stacking] X_train={X_train.shape} X_test={X_test.shape}")

    stacker = make_stacking_learner(STACKING_LEARNER)
    stacker.fit(X_train, y_train)
    ensemble = predict_proba_from_learner(stacker, X_test)
    preds = ensemble.argmax(axis=1).astype(int)
    conf = ensemble.max(axis=1)
    LOGGER.info(
        f"[stacking] learner={STACKING_LEARNER} models={len(model_ckpts)}  "
        f"mean_conf={conf.mean():.4f}  min={conf.min():.4f}  max={conf.max():.4f}"
    )
    return ensemble, preds, conf, test_meta, len(ckpt_paths)
# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — SUBMISSION & OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def write_submission(
    test_meta: pd.DataFrame,
    preds: np.ndarray,
    conf: np.ndarray,
    num_models: int,
) -> pd.DataFrame:

    LOGGER.info("=" * 60)
    LOGGER.info("STAGE 7 — SUBMISSION & OUTPUTS")
    LOGGER.info("=" * 60)

    sub = test_meta[["ID"]].copy()
    sub["TARGET"] = preds

    sub.to_csv(WORKING_ROOT / "submission.csv",  index=False)
    sub.to_csv(INFERENCE_DIR / "submission.csv", index=False)

    pd.DataFrame({
        "ID": test_meta["ID"].values,
        "predicted_class": preds,
        "confidence": conf,
    }).to_csv(INFERENCE_DIR / "prediction_confidence.csv", index=False)

    save_json({
        "timestamp"       : pd.Timestamp.now("UTC").isoformat(),
        "num_models"      : int(num_models),
        "blend_mode"      : BLEND_MODE,
        "tta_passes"      : TTA_N if USE_TTA else 0,
        "mean_confidence" : float(conf.mean()),
        "min_confidence"  : float(conf.min()),
        "max_confidence"  : float(conf.max()),
        "num_predictions" : int(len(preds)),
    }, INFERENCE_DIR / "submission_metadata.json")

    LOGGER.info(f"\n{'='*40}\nSubmission preview:\n{sub.head(10).to_string(index=False)}")
    LOGGER.info(f"Prediction distribution:\n{pd.Series(preds).value_counts().sort_index().to_string()}")
    LOGGER.info(f"Avg confidence: {conf.mean():.4f}  Min: {conf.min():.4f}  Max: {conf.max():.4f}")
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(rank: int = 0, world_size: int = 1) -> None:
    global device

    if XLA_AVAILABLE and USE_XLA_MULTIPROCESSING:
        device = xm.xla_device()
        if not xm.is_master_ordinal(local=False):
            LOGGER.setLevel(logging.ERROR)
    elif XLA_AVAILABLE:
        device = torch_xla.device()

    set_seed(SEED + rank)
    if is_master_process():
        log_resources("startup")

    # ── STAGE 2: Load & validate ──────────────────────────────────────────────
    train_df, test_df = load_raw_csvs()
    validate_data(train_df, test_df)

    # ── STAGE 3: Preprocessing / fold metadata ────────────────────────────────
    LOGGER.info("=" * 60)
    LOGGER.info("STAGE 3 — DATA PREPROCESSING")
    LOGGER.info("=" * 60)
    y_train       = train_df["y"].to_numpy(dtype=np.int64)
    fold_metadata = build_folds(y_train, NUM_FOLDS, SEED)
    if is_master_process():
        save_json(fold_metadata, PROCESSED_DIR / "fold_metadata.json")

    if RUN_INFERENCE_ONLY:
        # ── STAGE 6 + 7 only ──────────────────────────────────────────────────
        result = run_ensemble_inference(train_df, test_df, y_train, fold_metadata)
        if result is not None:
            _, preds, conf, test_meta, num_models = result
            if is_master_process():
                write_submission(test_meta, preds, conf, num_models=num_models)

    elif RUN_TRAINING:
        # ── STAGES 4-7 ────────────────────────────────────────────────────────
        all_history: dict = {}
        proc_rank = 0
        proc_world = 1
        if XLA_AVAILABLE and USE_XLA_MULTIPROCESSING:
            proc_rank = xm.get_ordinal()
            proc_world = NUM_DEVICES

        for model_name in ACTIVE_MODELS:
            mcfg        = MODEL_REGISTRY[model_name]
            arch_dir    = CHECKPOINT_DIR / model_name
            model_hist  = {}

            for fold_idx, fold_info in fold_metadata.items():
                if not budget_ok():
                    LOGGER.info("[budget] Session budget exhausted — stopping.")
                    break
                h = train_fold(
                    fold_idx,
                    fold_info,
                    train_df,
                    model_name,
                    mcfg,
                    arch_dir,
                    rank=proc_rank,
                    world_size=proc_world,
                )
                model_hist[fold_idx] = h

            all_history[model_name] = model_hist
            if is_master_process():
                save_json(model_hist, LOG_DIR / f"{model_name[:30]}_all_folds.json")

        if is_master_process():
            save_json(all_history, LOG_DIR / "full_training_history.json")

        # ── Inference & submission ─────────────────────────────────────────────
        result = run_ensemble_inference(train_df, test_df, y_train, fold_metadata)
        if result is not None:
            _, preds, conf, test_meta, num_models = result
            if is_master_process():
                write_submission(test_meta, preds, conf, num_models=num_models)
        if is_master_process():
            log_resources("final")

    else:
        raise RuntimeError("Set RUN_TRAINING=True or RUN_INFERENCE_ONLY=True")

if __name__ == "__main__":
    if XLA_AVAILABLE and USE_XLA_MULTIPROCESSING:
        xmp.spawn(run_pipeline, nprocs=NUM_DEVICES, start_method="fork")
    else:
        run_pipeline()
