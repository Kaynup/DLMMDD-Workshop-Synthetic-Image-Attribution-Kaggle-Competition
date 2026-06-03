from __future__ import annotations
from copy import deepcopy

from lib.config.defaults import CONTROL_PANEL

ACTIVE_MODELS = [
    "caformer_s18.sail_in22k_ft_in1k",
]

MODEL_REGISTRY: dict[str, dict] = {
    "caformer_s18.sail_in22k_ft_in1k": {
        "image_size": 224,
        "batch_size": 48,
        "lr": 5e-4,
        "freeze_stages": 1,
        "regularization": {
            "weight_decay": 2e-4,
            "label_smoothing": 0.10,
            "mixup_alpha": 0.4,
            "mixup_prob": 0.5,
            "cutmix_alpha": 1.0,
            "cutmix_prob": 0.2,
            "dropout": 0.22,
            "grad_clip_norm": 0.8,
        },
    },
}


def resolve_model_settings(model_name: str, mcfg: dict | None = None) -> dict:
    base = deepcopy(MODEL_REGISTRY[model_name] if mcfg is None else mcfg)
    reg = deepcopy(CONTROL_PANEL["regularization_defaults"])
    reg.update(base.get("regularization", {}))
    base["regularization"] = reg
    return base
