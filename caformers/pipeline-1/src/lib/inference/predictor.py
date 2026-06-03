from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from lib.config.defaults import NUM_CLASSES
from lib.core.device import DEVICE, AMP_DTYPE, USE_AMP
from lib.core.logging import inference_logger
from lib.models.factory import build_model


def load_model_from_ckpt(ckpt_path: Path) -> tuple[torch.nn.Module, dict, str]:
    inference_logger.info(f"[load_ckpt] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("metadata", {})
    model_name = meta.get("model_name", "caformer_s18.sail_in22k_ft_in1k")
    val_m = meta.get("val_metrics", {})
    inference_logger.info(
        f"[load_ckpt] {ckpt_path.name} | arch={model_name} | epoch={meta.get('epoch', '?')} | "
        f"val_f1={val_m.get('f1_macro', float('nan')):.4f} | "
        f"gen={val_m.get('generalization_score', float('nan')):.4f}"
    )
    model = build_model(model_name)
    state_dict = ckpt["model_state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        inference_logger.warning(f"[load_ckpt] missing_keys={missing_keys}")
    if unexpected_keys:
        inference_logger.warning(f"[load_ckpt] unexpected_keys={unexpected_keys}")
    model.eval()
    return model, meta, model_name


@torch.inference_mode()
def infer_single(model: torch.nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(imgs)
            p = F.softmax(logits, dim=-1)
        probs.append(p.float().cpu().numpy())
        del imgs, logits, p
    return np.concatenate(probs, axis=0) if probs else np.empty((0, __import__("lib.config.defaults", fromlist=["NUM_CLASSES"]).NUM_CLASSES), dtype=np.float32)
