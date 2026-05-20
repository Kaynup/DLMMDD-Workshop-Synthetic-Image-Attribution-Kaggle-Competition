"""Metrics computation and tracking."""
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from .logger_setup import setup_logger

logger = setup_logger(__name__)


class MetricsComputer:
    """Compute and track metrics during training."""
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize metrics computer.
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        logger.info(f"Initialized MetricsComputer for {num_classes} classes")
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics_list: list = None,
        compute_per_class: bool = True,
    ) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            metrics_list: List of metrics to compute. Defaults to ['accuracy', 'f1_macro']
            compute_per_class: Whether to compute per-class metrics
            
        Returns:
            Dictionary with computed metrics
        """
        if metrics_list is None:
            metrics_list = ['accuracy', 'f1_macro']
        
        results = {}
        
        # Accuracy (main metric for this competition)
        if 'accuracy' in metrics_list:
            results['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # F1 scores
        if 'f1_macro' in metrics_list:
            results['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        if 'f1_weighted' in metrics_list:
            results['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Precision and Recall
        if 'precision_macro' in metrics_list:
            results['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        if 'recall_macro' in metrics_list:
            results['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Per-class metrics
        if compute_per_class:
            results['per_class'] = {}
            for class_id in range(self.num_classes):
                y_true_binary = (y_true == class_id).astype(int)
                y_pred_binary = (y_pred == class_id).astype(int)
                
                results['per_class'][class_id] = {
                    'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
                    'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
                    'f1': float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
                }
        
        return results
    
    def compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
    
    def compute_generalization_score(self, train_metrics: Dict, val_metrics: Dict) -> float:
        """
        Compute generalization score for checkpoint selection.
        
        Formula:
            Gen_Score = val_accuracy - |train_accuracy - val_accuracy|
        
        Higher score = better generalization (less overfitting)
        
        Args:
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            
        Returns:
            Generalization score (float)
        """
        train_acc = train_metrics.get('accuracy', 0)
        val_acc = val_metrics.get('accuracy', 0)
        
        overfitting_magnitude = abs(train_acc - val_acc)
        gen_score = val_acc - overfitting_magnitude
        
        return float(gen_score)


class MetricsTracker:
    """Track metrics throughout training."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize metrics tracker.
        
        Args:
            log_dir: Directory to save metrics logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {}
        logger.info(f"Initialized MetricsTracker with log_dir={log_dir}")
    
    def log_epoch(self, fold_idx: int, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """
        Log metrics for an epoch.
        
        Args:
            fold_idx: Fold index
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        fold_key = f"fold_{fold_idx}"
        
        if fold_key not in self.history:
            self.history[fold_key] = {
                'epochs': [],
                'train': [],
                'val': [],
            }
        
        self.history[fold_key]['epochs'].append(epoch)
        self.history[fold_key]['train'].append(train_metrics)
        self.history[fold_key]['val'].append(val_metrics)
        
        # Log summary
        train_acc = train_metrics.get('accuracy', 0)
        val_acc = val_metrics.get('accuracy', 0)
        logger.debug(
            f"Fold {fold_idx}, Epoch {epoch}: "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )
    
    def save_history(self, fold_idx: int, output_path: str):
        """
        Save metrics history to JSON.
        
        Args:
            fold_idx: Fold index
            output_path: Path to save JSON
        """
        fold_key = f"fold_{fold_idx}"
        
        if fold_key not in self.history:
            logger.warning(f"No history for fold {fold_idx}")
            return
        
        with open(output_path, 'w') as f:
            json.dump(self.history[fold_key], f, indent=2)
        
        logger.info(f"Saved metrics history to {output_path}")
    
    def save_all_history(self, output_dir: str = "logs"):
        """
        Save all metrics history to separate JSON files.
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fold_key, fold_history in self.history.items():
            file_path = output_path / f"{fold_key}_history.json"
            with open(file_path, 'w') as f:
                json.dump(fold_history, f, indent=2)
            logger.info(f"Saved {fold_key} history")
