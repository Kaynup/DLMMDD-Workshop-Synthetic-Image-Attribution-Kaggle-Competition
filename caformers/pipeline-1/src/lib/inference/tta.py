from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lib.config.defaults import TTA_N, USE_TTA
from lib.core.device import AMP_DTYPE, DEVICE, DEVICE_TYPE, USE_AMP
from lib.core.logging import inference_logger
from lib.data.loader import make_loader


@torch.inference_mode()
def infer_tta(model: torch.nn.Module, paths: list[str], img_sz: int, batch_size: int, n: int) -> np.ndarray:
    loader = make_loader(paths, image_size=img_sz, batch_size=batch_size, shuffle=False)
    accum_probs = None
    total_passes = n + 1
    for t in range(total_passes):
        pass_probs: list[np.ndarray] = []
        for batch in loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(DEVICE, non_blocking=True)
            if DEVICE_TYPE == "cuda":
                imgs = imgs.contiguous(memory_format=torch.channels_last)
            if t > 0 and t % 2 == 1:
                imgs = torch.flip(imgs, dims=[3])
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                logits = model(imgs)
                probs = F.softmax(logits, dim=-1)
            pass_probs.append(probs.float().cpu().numpy())
            del imgs, logits, probs
        pass_probs = np.concatenate(pass_probs, axis=0)
        accum_probs = pass_probs if accum_probs is None else accum_probs + pass_probs
        inference_logger.info(f"[TTA] pass {t + 1}/{total_passes}")
    return accum_probs / float(total_passes)
