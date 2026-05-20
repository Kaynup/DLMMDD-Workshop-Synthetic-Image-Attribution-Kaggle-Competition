"""Train/validation split and cross-validation utilities."""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold
from .logger_setup import setup_logger

logger = setup_logger(__name__)


class TrainValSplitter:
    """Create stratified train/validation splits."""
    
    def __init__(self, num_splits: int = 5, random_state: int = 42, shuffle: bool = True):
        """
        Initialize splitter.
        
        Args:
            num_splits: Number of CV folds
            random_state: Random seed
            shuffle: Whether to shuffle data
        """
        self.num_splits = num_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.skf = StratifiedKFold(
            n_splits=num_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        logger.info(f"Initialized TrainValSplitter: {num_splits} folds, seed={random_state}")
    
    def create_folds(self, X: np.ndarray, y: np.ndarray) -> Dict[int, Dict]:
        """
        Create stratified K-fold splits.
        
        Args:
            X: Feature array (ignored, used for shape only)
            y: Label array
            
        Returns:
            Dictionary mapping fold_idx to {'train_indices': [...], 'val_indices': [...]}
        """
        fold_metadata = {}
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            fold_metadata[fold_idx] = {
                'fold_idx': fold_idx,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'train_count': len(train_idx),
                'val_count': len(val_idx),
                'train_class_counts': np.bincount(y[train_idx]).tolist(),
                'val_class_counts': np.bincount(y[val_idx]).tolist(),
            }
            
            logger.info(f"Fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val")
        
        return fold_metadata
    
    def save_fold_metadata(self, fold_metadata: Dict, output_path: str):
        """
        Save fold metadata to JSON.
        
        Args:
            fold_metadata: Fold metadata dictionary
            output_path: Path to save JSON
        """
        with open(output_path, 'w') as f:
            json.dump(fold_metadata, f, indent=2)
        logger.info(f"Saved fold metadata to {output_path}")
    
    @staticmethod
    def load_fold_metadata(input_path: str) -> Dict:
        """
        Load fold metadata from JSON.
        
        Args:
            input_path: Path to load JSON
            
        Returns:
            Fold metadata dictionary
        """
        with open(input_path, 'r') as f:
            fold_metadata = json.load(f)
        logger.info(f"Loaded fold metadata from {input_path}")
        return fold_metadata


def get_fold_data(
    fold_metadata: Dict,
    fold_idx: int,
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get train/val data for a specific fold.
    
    Args:
        fold_metadata: Fold metadata dictionary
        fold_idx: Fold index
        X: Feature array
        y: Label array
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    fold = fold_metadata[fold_idx]
    train_idx = np.array(fold['train_indices'])
    val_idx = np.array(fold['val_indices'])
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    return X_train, X_val, y_train, y_val


def create_data_loaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 4,
    augmenter=None,
    shuffle_train: bool = True,
):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        batch_size: Batch size
        num_workers: Number of data loading workers
        augmenter: Optional ImageAugmenter instance
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Convert to torch tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        X_val_tensor = torch.from_numpy(X_val).float()
        y_train_tensor = torch.from_numpy(y_train).long()
        y_val_tensor = torch.from_numpy(y_val).long()
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        return train_loader, val_loader
    
    except ImportError:
        logger.error("PyTorch not available. Install torch to use create_data_loaders.")
        return None, None
