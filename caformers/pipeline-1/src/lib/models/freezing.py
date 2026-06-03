from __future__ import annotations


def freeze_backbone_children(backbone, freeze_stages: int) -> int:
    children = list(backbone.children())
    frozen = 0
    for idx, module in enumerate(children):
        if idx >= freeze_stages:
            break
        for param in module.parameters():
            param.requires_grad = False
        frozen += 1
    return frozen
