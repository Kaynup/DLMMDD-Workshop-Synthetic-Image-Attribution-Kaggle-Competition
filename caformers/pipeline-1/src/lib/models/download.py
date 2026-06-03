from __future__ import annotations

from pathlib import Path
import timm
import torch

from lib.core.logging import model_logger
from lib.config.paths import MODELS_DIR


def predownload_model(model_name: str, num_classes: int = 10, dest: Path | None = None) -> Path:
    """Download a pretrained model via timm and save its state_dict locally.

    Note: num_classes must match the target head size you will use when loading the model.
    """
    dest = Path(dest or MODELS_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"{model_name}_{num_classes}.pt"
    if out_path.exists():
        model_logger.info(f"[models] already downloaded: {out_path.name}")
        return out_path
    model_logger.info(f"[models] downloading model weights for {model_name} (num_classes={num_classes})")
    # create model with pretrained weights (this will use timm/torch cache)
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    state = model.state_dict()
    torch.save(state, out_path)
    model_logger.info(f"[models] saved pretrained weights -> {out_path}")
    return out_path
