"""Checkpoint selection and model management."""
import json
import torch
from pathlib import Path
from typing import Dict, Optional, List
from .logger_setup import setup_logger
from .metrics import MetricsComputer

logger = setup_logger(__name__)


class CheckpointManager:
    """Manage checkpoint saving and selection."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", save_all_epochs: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_all_epochs: Whether to save all epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_all_epochs = save_all_epochs
        logger.info(f"Initialized CheckpointManager: {checkpoint_dir}")
    
    def save_checkpoint(
        self,
        fold_idx: int,
        epoch: int,
        model,
        optimizer,
        metrics: Dict,
        config: Dict,
    ):
        """
        Save model checkpoint.
        
        Args:
            fold_idx: Fold index
            epoch: Epoch number
            model: PyTorch model
            optimizer: Optimizer state
            metrics: Metrics dictionary
            config: Config dictionary
        """
        fold_dir = self.checkpoint_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = fold_dir / f"epoch_{epoch:03d}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'fold_idx': fold_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, fold_idx: int, epoch: int) -> Optional[Dict]:
        """
        Load model checkpoint.
        
        Args:
            fold_idx: Fold index
            epoch: Epoch number
            
        Returns:
            Checkpoint dictionary or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"fold_{fold_idx}" / f"epoch_{epoch:03d}.pth"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            return None
    
    def get_fold_checkpoints(self, fold_idx: int) -> List[int]:
        """
        Get list of available epochs for a fold.
        
        Args:
            fold_idx: Fold index
            
        Returns:
            List of epoch numbers
        """
        fold_dir = self.checkpoint_dir / f"fold_{fold_idx}"
        
        if not fold_dir.exists():
            return []
        
        epochs = []
        for checkpoint_file in sorted(fold_dir.glob("epoch_*.pth")):
            epoch = int(checkpoint_file.stem.split('_')[1])
            epochs.append(epoch)
        
        return sorted(epochs)


class CheckpointSelector:
    """Select best checkpoint based on metrics."""
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize checkpoint selector.
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.metrics_computer = MetricsComputer(num_classes)
        logger.info("Initialized CheckpointSelector")
    
    def select_best_checkpoint(
        self,
        fold_idx: int,
        fold_metrics_history: List[Dict],
        selection_metric: str = "generalization_score",
    ) -> Dict:
        """
        Select best checkpoint for a fold.
        
        Args:
            fold_idx: Fold index
            fold_metrics_history: List of epoch metrics
            selection_metric: Metric to use for selection
                - "val_accuracy": Best validation accuracy
                - "generalization_score": Best generalization (less overfitting)
                
        Returns:
            Dictionary with selection info:
                {
                    'fold_idx': int,
                    'selected_epoch': int,
                    'metrics': dict,
                    'selection_metric': str,
                    'selection_value': float,
                }
        """
        best_epoch = None
        best_value = -float('inf')
        best_metrics = None
        
        for epoch_idx, epoch_metrics in enumerate(fold_metrics_history):
            train_metrics = epoch_metrics.get('train', {})
            val_metrics = epoch_metrics.get('val', {})
            
            if selection_metric == "val_accuracy":
                value = val_metrics.get('accuracy', 0)
            elif selection_metric == "generalization_score":
                value = self.metrics_computer.compute_generalization_score(
                    train_metrics, val_metrics
                )
            else:
                value = val_metrics.get('accuracy', 0)
            
            if value > best_value:
                best_value = value
                best_epoch = epoch_idx
                best_metrics = {
                    'train': train_metrics,
                    'val': val_metrics,
                }
        
        if best_epoch is None:
            logger.warning(f"No valid checkpoints found for fold {fold_idx}")
            return {
                'fold_idx': fold_idx,
                'selected_epoch': None,
                'metrics': None,
                'selection_metric': selection_metric,
                'selection_value': None,
            }
        
        result = {
            'fold_idx': fold_idx,
            'selected_epoch': best_epoch,
            'metrics': best_metrics,
            'selection_metric': selection_metric,
            'selection_value': best_value,
        }
        
        logger.info(
            f"Fold {fold_idx}: Selected epoch {best_epoch} "
            f"({selection_metric}={best_value:.4f})"
        )
        
        return result
    
    def select_best_checkpoints_all_folds(
        self,
        num_folds: int,
        metrics_histories: Dict[int, List[Dict]],
        selection_metric: str = "generalization_score",
    ) -> Dict:
        """
        Select best checkpoint for all folds.
        
        Args:
            num_folds: Number of folds
            metrics_histories: Dictionary mapping fold_idx to metrics history
            selection_metric: Metric for selection
            
        Returns:
            Dictionary with selection results for all folds
        """
        selections = {}
        
        for fold_idx in range(num_folds):
            if fold_idx not in metrics_histories:
                logger.warning(f"No metrics history for fold {fold_idx}")
                continue
            
            selection = self.select_best_checkpoint(
                fold_idx,
                metrics_histories[fold_idx],
                selection_metric
            )
            selections[fold_idx] = selection
        
        return selections
    
    def save_selections(self, selections: Dict, output_path: str):
        """
        Save checkpoint selections to JSON.
        
        Args:
            selections: Selections dictionary
            output_path: Path to save JSON
        """
        with open(output_path, 'w') as f:
            json.dump(selections, f, indent=2)
        logger.info(f"Saved checkpoint selections to {output_path}")
