"""Utility functions for the pipeline."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """
    Get the best available device (CUDA or CPU).
    
    Returns:
        Device name ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")
    return device


def count_parameters(model) -> Tuple[int, int]:
    """
    Count trainable and total parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def get_model_summary(model, trainable_params: int, total_params: int) -> str:
    """
    Generate model summary string.
    
    Args:
        model: PyTorch model
        trainable_params: Number of trainable parameters
        total_params: Total number of parameters
        
    Returns:
        Summary string
    """
    return f"""
    Model Architecture: {model.__class__.__name__}
    Total Parameters: {total_params:,}
    Trainable Parameters: {trainable_params:,}
    Non-trainable Parameters: {total_params - trainable_params:,}
    """


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def ensure_dir_exists(path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_checkpoint_model(model, checkpoint_path: str, device: str = 'cuda'):
    """
    Load model weights from checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def save_submission_csv(predictions: np.ndarray, test_ids: np.ndarray, output_path: str):
    """
    Save submission in Kaggle format.
    
    Args:
        predictions: Predicted class labels (N,)
        test_ids: Test image IDs (N,)
        output_path: Path to save CSV
    """
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'TARGET': predictions
    })
    submission_df.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y: Labels array
        num_classes: Number of classes
        
    Returns:
        Tensor of class weights
    """
    counts = np.bincount(y, minlength=num_classes)
    weights = num_classes / (counts + 1e-6)  # Avoid division by zero
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights).float()


def get_train_transforms(augmenter=None):
    """Get training transforms with optional augmentation."""
    return augmenter


def get_eval_transforms():
    """Get evaluation transforms (no augmentation)."""
    return None
