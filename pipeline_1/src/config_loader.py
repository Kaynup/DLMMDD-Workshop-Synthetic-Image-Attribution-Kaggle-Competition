"""Configuration loader with environment variable override support."""
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass
from .logger_setup import setup_logger

logger = setup_logger(__name__)


@dataclass
class DataConfig:
    """Data loading configuration."""
    target_height: int = 224
    target_width: int = 224
    color_mode: str = "RGB"
    image_format: str = "numpy"


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    normalization: str = "imagenet"
    resize_method: str = "bilinear"
    handle_missing: str = "skip"


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enabled: bool = True
    train_augmentations: list = None
    horizontal_flip_p: float = 0.5
    rotation_degrees: int = 15
    brightness_factor: float = 0.2
    contrast_factor: float = 0.2


@dataclass
class TrainValSplitConfig:
    """Train/validation split configuration."""
    strategy: str = "stratified_kfold"
    num_splits: int = 5
    random_state: int = 42
    shuffle: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = "efficientnet_b4"
    pretrained: bool = True
    num_classes: int = 10
    dropout_rate: float = 0.3


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    early_stopping_patience: int = 10


@dataclass
class ValidationConfig:
    """Validation configuration."""
    metrics: list = None
    compute_per_class_metrics: bool = True
    save_confusion_matrix: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint saving and selection configuration."""
    selection_metric: str = "generalization_score"
    save_all_epochs: bool = True
    save_frequency: int = 1


@dataclass
class InferenceConfig:
    """Inference configuration."""
    tta_enabled: bool = True
    tta_transforms: int = 5
    batch_size_inference: int = 32


@dataclass
class AppConfig:
    """Root application configuration."""
    competition_name: str = "DLMMDD SIA Challenge"
    num_classes: int = 10
    data_dir: str = "../Data"
    seed: int = 42
    
    data: DataConfig = None
    preprocessing: PreprocessingConfig = None
    augmentation: AugmentationConfig = None
    train_val_split: TrainValSplitConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    validation: ValidationConfig = None
    checkpoint: CheckpointConfig = None
    inference: InferenceConfig = None


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from TOML file with environment variable overrides.
    
    Priority order:
    1. Environment variables (e.g., PIPELINE_SEED=123)
    2. config.toml file
    3. Hardcoded defaults
    
    Args:
        config_path: Path to config.toml. If None, searches for it in:
                    - Current directory
                    - Parent directory
                    - ./pipeline_1/config.toml
                    
    Returns:
        AppConfig dataclass instance
    """
    # Locate config file
    if config_path is None:
        candidates = [
            Path("config.toml"),
            Path("../config.toml"),
            Path("./pipeline_1/config.toml"),
        ]
        config_path = None
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break
    
    # Load TOML if found
    config_dict = {}
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'rb') as f:
                config_dict = tomllib.load(f)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.warning("Using default configuration")
    else:
        logger.warning(f"Config file not found at {config_path}. Using defaults.")
    
    # Environment variable overrides (e.g., PIPELINE_SEED=123)
    env_overrides = {}
    for key, value in os.environ.items():
        if key.startswith("PIPELINE_"):
            env_key = key.replace("PIPELINE_", "").lower()
            try:
                # Try to parse as int, float, or bool
                if value.lower() in ('true', 'false'):
                    env_overrides[env_key] = value.lower() == 'true'
                elif '.' in value:
                    env_overrides[env_key] = float(value)
                else:
                    env_overrides[env_key] = int(value)
            except ValueError:
                env_overrides[env_key] = value
    
    if env_overrides:
        logger.info(f"Environment overrides: {env_overrides}")
    
    # Build config object with defaults, then TOML, then env overrides
    config = AppConfig(
        competition_name=env_overrides.get(
            'competition_name',
            config_dict.get('competition', {}).get('name', 'DLMMDD SIA Challenge')
        ),
        num_classes=env_overrides.get(
            'num_classes',
            config_dict.get('competition', {}).get('num_classes', 10)
        ),
        data_dir=env_overrides.get(
            'data_dir',
            config_dict.get('competition', {}).get('data_dir', '../Data')
        ),
        seed=env_overrides.get(
            'seed',
            config_dict.get('reproducibility', {}).get('seed', 42)
        ),
        data=DataConfig(
            target_height=env_overrides.get(
                'target_height',
                config_dict.get('data', {}).get('target_height', 224)
            ),
            target_width=env_overrides.get(
                'target_width',
                config_dict.get('data', {}).get('target_width', 224)
            ),
            color_mode=config_dict.get('data', {}).get('color_mode', 'RGB'),
            image_format=config_dict.get('data', {}).get('image_format', 'numpy'),
        ),
        preprocessing=PreprocessingConfig(
            normalization=config_dict.get('preprocessing', {}).get('normalization', 'imagenet'),
            resize_method=config_dict.get('preprocessing', {}).get('resize_method', 'bilinear'),
            handle_missing=config_dict.get('preprocessing', {}).get('handle_missing', 'skip'),
        ),
        augmentation=AugmentationConfig(
            enabled=config_dict.get('augmentation', {}).get('enabled', True),
            train_augmentations=config_dict.get('augmentation', {}).get('train_augmentations', []),
            horizontal_flip_p=config_dict.get('augmentation', {}).get('horizontal_flip_p', 0.5),
            rotation_degrees=config_dict.get('augmentation', {}).get('rotation_degrees', 15),
            brightness_factor=config_dict.get('augmentation', {}).get('brightness_factor', 0.2),
            contrast_factor=config_dict.get('augmentation', {}).get('contrast_factor', 0.2),
        ),
        train_val_split=TrainValSplitConfig(
            strategy=config_dict.get('train_val_split', {}).get('strategy', 'stratified_kfold'),
            num_splits=env_overrides.get(
                'num_splits',
                config_dict.get('train_val_split', {}).get('num_splits', 5)
            ),
            random_state=config_dict.get('train_val_split', {}).get('random_state', 42),
            shuffle=config_dict.get('train_val_split', {}).get('shuffle', True),
        ),
        model=ModelConfig(
            architecture=config_dict.get('model', {}).get('architecture', 'efficientnet_b4'),
            pretrained=config_dict.get('model', {}).get('pretrained', True),
            num_classes=config_dict.get('model', {}).get('num_classes', 10),
            dropout_rate=config_dict.get('model', {}).get('dropout_rate', 0.3),
        ),
        training=TrainingConfig(
            num_epochs=env_overrides.get(
                'num_epochs',
                config_dict.get('training', {}).get('num_epochs', 100)
            ),
            batch_size=env_overrides.get(
                'batch_size',
                config_dict.get('training', {}).get('batch_size', 32)
            ),
            num_workers=config_dict.get('training', {}).get('num_workers', 4),
            learning_rate=env_overrides.get(
                'learning_rate',
                config_dict.get('training', {}).get('learning_rate', 1e-4)
            ),
            optimizer=config_dict.get('training', {}).get('optimizer', 'adam'),
            weight_decay=config_dict.get('training', {}).get('weight_decay', 1e-5),
            scheduler=config_dict.get('training', {}).get('scheduler', 'cosine'),
            warmup_epochs=config_dict.get('training', {}).get('warmup_epochs', 5),
            early_stopping_patience=config_dict.get('training', {}).get('early_stopping_patience', 10),
        ),
        validation=ValidationConfig(
            metrics=config_dict.get('validation', {}).get('metrics', ['accuracy', 'f1_macro']),
            compute_per_class_metrics=config_dict.get('validation', {}).get('compute_per_class_metrics', True),
            save_confusion_matrix=config_dict.get('validation', {}).get('save_confusion_matrix', True),
        ),
        checkpoint=CheckpointConfig(
            selection_metric=config_dict.get('checkpoint', {}).get('selection_metric', 'generalization_score'),
            save_all_epochs=config_dict.get('checkpoint', {}).get('save_all_epochs', True),
            save_frequency=config_dict.get('checkpoint', {}).get('save_frequency', 1),
        ),
        inference=InferenceConfig(
            tta_enabled=config_dict.get('inference', {}).get('tta_enabled', True),
            tta_transforms=config_dict.get('inference', {}).get('tta_transforms', 5),
            batch_size_inference=config_dict.get('inference', {}).get('batch_size_inference', 32),
        ),
    )
    
    logger.info(f"Configuration loaded: {config.competition_name} ({config.num_classes} classes)")
    return config
