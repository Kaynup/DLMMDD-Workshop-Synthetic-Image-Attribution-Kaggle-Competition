from __future__ import annotations

import torch

from lib.config.defaults import NUM_CLASSES
from lib.config.model_registry import resolve_model_settings
from lib.core.device import DEVICE, USE_COMPILE
from lib.core.logging import model_logger
from lib.models.timm_model import TimmModel


def build_model(model_name: str, num_classes: int = NUM_CLASSES, mcfg: dict | None = None) -> torch.nn.Module:
    cfg = resolve_model_settings(model_name, mcfg)
    reg_cfg = cfg["regularization"]
    dropout = float(reg_cfg.get("dropout", 0.0))
    freeze_stages = int(cfg.get("freeze_stages", 0))
    model = TimmModel(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_stages=freeze_stages,
    )
    model = model.to(DEVICE)
    if USE_COMPILE:
        model_logger.info(f"[compile] compiling {model_name}")
        model = torch.compile(model)
    return model
