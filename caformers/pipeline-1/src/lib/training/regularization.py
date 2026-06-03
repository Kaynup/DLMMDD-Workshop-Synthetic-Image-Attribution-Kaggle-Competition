from __future__ import annotations

import math
import random
import torch


def _randperm(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randperm(batch_size, device=device)


def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    if alpha <= 0.0 or images.size(0) < 2:
        return images, labels
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = _randperm(images.size(0), images.device)
    lam_t = torch.tensor(lam, device=images.device, dtype=images.dtype)
    mixed_images = lam_t * images + (1.0 - lam_t) * images[perm]
    mixed_labels = lam * labels + (1.0 - lam) * labels[perm]
    return mixed_images, mixed_labels


def cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    if alpha <= 0.0 or images.size(0) < 2:
        return images, labels
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = _randperm(images.size(0), images.device)
    b, c, h, w = images.shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)
    cx = torch.randint(0, w, (1,), device=images.device).item()
    cy = torch.randint(0, h, (1,), device=images.device).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    area = max(1, (x2 - x1) * (y2 - y1))
    lam_adjusted = 1.0 - (area / float(h * w))
    mixed_labels = lam_adjusted * labels + (1.0 - lam_adjusted) * labels[perm]
    return mixed_images, mixed_labels


def apply_batch_regularization(images: torch.Tensor, soft_labels: torch.Tensor, reg: dict):
    cutmix_alpha = float(reg.get("cutmix_alpha", 0.0))
    cutmix_prob = float(reg.get("cutmix_prob", 0.0))
    mixup_alpha = float(reg.get("mixup_alpha", 0.0))
    mixup_prob = float(reg.get("mixup_prob", 0.0))
    r = random.random()
    if cutmix_alpha > 0.0 and r < cutmix_prob:
        return cutmix(images, soft_labels, cutmix_alpha)
    if mixup_alpha > 0.0 and r < mixup_prob:
        return mixup(images, soft_labels, mixup_alpha)
    return images, soft_labels
