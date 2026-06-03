from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.config.defaults import NUM_CLASSES
from lib.core.device import DEVICE, AMP_DTYPE, USE_AMP


def predict_loader(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    for batch in loader:
        has_labels = isinstance(batch, (list, tuple)) and len(batch) == 2
        if has_labels:
            imgs, lbls = batch
            all_true.append(lbls.numpy())
        else:
            imgs = batch
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(imgs)
            probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.float().cpu().numpy())
        del imgs, logits, probs
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, NUM_CLASSES), dtype=np.float32)
    preds_arr = probs_arr.argmax(axis=1) if len(probs_arr) else np.array([], dtype=np.int64)
    true_arr = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int64)
    return true_arr, preds_arr, probs_arr


@torch.inference_mode()
def evaluate_loader(model: torch.nn.Module, loader: DataLoader, criterion=None) -> dict:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        has_labels = isinstance(batch, (list, tuple)) and len(batch) == 2
        if has_labels:
            imgs, lbls = batch
            all_true.append(lbls.numpy())
            lbls_dev = lbls.to(DEVICE, non_blocking=True)
        else:
            imgs = batch
            lbls_dev = None
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(imgs)
            probs = F.softmax(logits, dim=-1)
            if has_labels and criterion is not None:
                loss = F.cross_entropy(logits, lbls_dev)
        all_probs.append(probs.float().cpu().numpy())
        if has_labels and criterion is not None:
            bs = int(imgs.size(0))
            total_loss += float(loss.item()) * bs
            total_count += bs
        del imgs, logits, probs
        if has_labels:
            del lbls, lbls_dev
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, NUM_CLASSES), dtype=np.float32)
    preds_arr = probs_arr.argmax(axis=1) if len(probs_arr) else np.array([], dtype=np.int64)
    true_arr = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int64)
    from lib.training.metrics import compute_metrics
    metrics = compute_metrics(true_arr, preds_arr)
    metrics["loss"] = float(total_loss / max(total_count, 1)) if total_count else None
    metrics["probs"] = probs_arr
    metrics["y_true"] = true_arr
    metrics["y_pred"] = preds_arr
    return metrics
