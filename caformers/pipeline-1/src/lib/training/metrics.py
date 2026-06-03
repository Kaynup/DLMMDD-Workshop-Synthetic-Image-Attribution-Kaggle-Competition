from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from lib.config.defaults import CONTROL_PANEL, NUM_CLASSES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "per_class_f1": [0.0] * NUM_CLASSES,
        }
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": per_class_f1.tolist(),
    }


def generalization_score(train_m: dict, val_m: dict, *, cfg: dict | None = None, return_parts: bool = False):
    gcfg = CONTROL_PANEL["generalization"] if cfg is None else cfg
    train_f1 = float(train_m.get("f1_macro", 0.0))
    val_f1 = float(val_m.get("f1_macro", 0.0))
    val_weight = float(gcfg.get("val_weight", 1.0))
    low_train_reward = float(gcfg.get("low_train_reward", 0.0))
    overfit_penalty = float(gcfg.get("overfit_penalty", 1.0))
    balance_penalty = float(gcfg.get("balance_penalty", 0.0))
    positive_gap = max(0.0, val_f1 - train_f1)
    negative_gap = max(0.0, train_f1 - val_f1)
    balance_gap = abs(train_f1 - val_f1)
    score = (
        val_weight * val_f1
        + low_train_reward * positive_gap
        - overfit_penalty * (negative_gap ** 2)
        - balance_penalty * (balance_gap ** 2)
    )
    parts = {
        "train_f1": float(train_f1),
        "val_f1": float(val_f1),
        "positive_gap": float(positive_gap),
        "negative_gap": float(negative_gap),
        "balance_gap": float(balance_gap),
        "val_term": float(val_weight * val_f1),
        "low_train_reward": float(low_train_reward * positive_gap),
        "overfit_penalty_term": float(overfit_penalty * (negative_gap ** 2)),
        "balance_penalty_term": float(balance_penalty * (balance_gap ** 2)),
        "score": float(score),
    }
    if return_parts:
        return float(score), parts
    return float(score)


def get_selection_value(train_m: dict, val_m: dict, checkpoint_metric: str = "generalization_score") -> float:
    metric = checkpoint_metric.strip().lower()
    if metric == "val_accuracy":
        return float(val_m.get("accuracy", 0.0))
    if metric == "val_f1_macro":
        return float(val_m.get("f1_macro", 0.0))
    if metric == "generalization_score":
        return float(val_m.get("generalization_score", val_m.get("f1_macro", 0.0)))
    return float(val_m.get(metric, val_m.get("f1_macro", 0.0)))
