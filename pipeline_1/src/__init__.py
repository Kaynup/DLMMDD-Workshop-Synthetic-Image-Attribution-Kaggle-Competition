"""
Synthetic Image Attribution Challenge - Production Pipeline
=====================================================

A powerful, modular pipeline for the DLMMDD Synthetic Image Attribution competition.

Modules:
    - config_loader: Configuration management with TOML support
    - logger_setup: Structured logging setup
    - data_loader: Data loading and validation
    - preprocessor: Image preprocessing and augmentation
    - train_val_split: Stratified K-fold cross-validation
    - metrics: Metrics computation and tracking
    - checkpoint_selector: Checkpoint management and selection
"""

from .config_loader import AppConfig, load_config
from .logger_setup import setup_logger
from .data_loader import DataLoader
from .preprocessor import ImagePreprocessor, ImageAugmenter
from .train_val_split import TrainValSplitter, get_fold_data, create_data_loaders
from .metrics import MetricsComputer, MetricsTracker
from .checkpoint_selector import CheckpointManager, CheckpointSelector

__all__ = [
    'AppConfig',
    'load_config',
    'setup_logger',
    'DataLoader',
    'ImagePreprocessor',
    'ImageAugmenter',
    'TrainValSplitter',
    'get_fold_data',
    'create_data_loaders',
    'MetricsComputer',
    'MetricsTracker',
    'CheckpointManager',
    'CheckpointSelector',
]

__version__ = '1.0.0'
