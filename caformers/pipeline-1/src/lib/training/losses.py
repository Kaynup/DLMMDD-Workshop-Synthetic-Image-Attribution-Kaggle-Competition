from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    if not (0.0 <= smoothing < 1.0):
        raise ValueError(f"Invalid label smoothing value: {smoothing}")
    confidence = 1.0 - smoothing
    with torch.no_grad():
        soft_targets = torch.full(
            (labels.size(0), num_classes),
            fill_value=smoothing / max(num_classes - 1, 1),
            device=labels.device,
            dtype=torch.float32,
        )
        soft_targets.scatter_(1, labels.unsqueeze(1), confidence)
    return soft_targets


def cross_entropy_soft(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    soft_targets = soft_targets.to(dtype=logits.dtype)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    return loss.mean()
