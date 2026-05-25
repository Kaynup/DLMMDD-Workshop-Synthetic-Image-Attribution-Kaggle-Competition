from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class MetricsComputer:
    @staticmethod
    def compute_epoch_metrics(y_true, y_pred):
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }

    @staticmethod
    def generalization_score(train_m, val_m, alpha=2.0):
        gap = max(0.0, train_m.get('f1_macro', 0.0) - val_m.get('f1_macro', 0.0))
        return float(val_m.get('f1_macro', 0.0) - alpha * gap**2)


def generalization_score(train_m, val_m, alpha=2.0):
    return MetricsComputer.generalization_score(train_m, val_m, alpha=alpha)
