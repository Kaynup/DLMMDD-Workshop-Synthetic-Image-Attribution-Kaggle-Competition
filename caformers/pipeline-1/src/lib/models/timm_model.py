from __future__ import annotations

import timm
import torch.nn as nn

from lib.core.logging import model_logger
from lib.config.paths import MODELS_DIR
from pathlib import Path
import torch
from lib.models.freezing import freeze_backbone_children


class TimmModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.0, freeze_stages: int = 0) -> None:
        super().__init__()
        self.model_name = model_name
        local_ckpt = Path(MODELS_DIR) / f"{model_name}_{num_classes}.pt"
        if local_ckpt.exists():
            model_logger.info(f"[models] loading local pretrained weights: {local_ckpt.name}")
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
                drop_rate=dropout,
            )
            try:
                state = torch.load(local_ckpt, map_location="cpu")
                self.backbone.load_state_dict(state)
            except Exception as exc:  # fallback to remote if load fails
                model_logger.warning(f"[models] failed loading local weights ({exc}), falling back to remote pretrained")
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=num_classes,
                    drop_rate=dropout,
                )
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes,
                drop_rate=dropout,
            )
        frozen = 0
        if freeze_stages > 0:
            frozen = freeze_backbone_children(self.backbone, freeze_stages)
            model_logger.info(f"[freeze] froze {frozen} backbone stages")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_logger.info(
            f"[model] {model_name} | params={total_params:,} | trainable={trainable_params:,} | "
            f"dropout={dropout:.3f} | freeze_stages={freeze_stages}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
