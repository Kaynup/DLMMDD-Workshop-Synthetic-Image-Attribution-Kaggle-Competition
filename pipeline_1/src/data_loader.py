"""Data loading and validation pipeline."""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple, Optional, List
from .logger_setup import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Load and validate training and test datasets."""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to Data directory containing training.csv, test.csv, etc.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        logger.info(f"Initialized DataLoader with data_dir={self.data_dir}")
    
    def load_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test metadata from CSV files.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        train_csv = self.data_dir / "training.csv"
        test_csv = self.data_dir / "test.csv"
        
        if not train_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {train_csv}")
        if not test_csv.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv}")
        
        # Load with explicit dtypes
        train_df = pd.read_csv(train_csv, dtype={'ID': 'int64', 'path': 'str', 'y': 'int64'})
        test_df = pd.read_csv(test_csv, dtype={'ID': 'int64', 'path': 'str'})
        
        logger.info(f"Loaded training data: {len(train_df)} samples")
        logger.info(f"Loaded test data: {len(test_df)} samples")
        
        return train_df, test_df
    
    def resolve_paths(self, df: pd.DataFrame, image_dir: str = 'Training') -> pd.DataFrame:
        """
        Convert relative image paths to absolute paths.
        
        Args:
            df: DataFrame with 'path' column
            image_dir: Subdirectory name ('Training' or 'Test')
            
        Returns:
            DataFrame with added 'full_path' column
        """
        base_path = self.data_dir / image_dir
        df = df.copy()
        df['full_path'] = df['path'].apply(lambda p: str(base_path / p))
        
        logger.info(f"Resolved paths for {len(df)} samples in {image_dir}")
        return df
    
    def inspect_image(self, img_path: str) -> Dict:
        """
        Extract image metadata without loading full image.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Dictionary with image metadata
        """
        try:
            img = Image.open(img_path)
            # Don't verify here as it's slow for many images
            width, height = img.size
            
            # Get file size
            file_size = os.path.getsize(img_path) if os.path.exists(img_path) else None
            
            return {
                'width': width,
                'height': height,
                'format': img.format,
                'color_mode': img.mode,
                'file_size_bytes': file_size,
                'is_readable': True,
                'error': None
            }
        except Exception as e:
            return {
                'width': None,
                'height': None,
                'format': None,
                'color_mode': None,
                'file_size_bytes': os.path.getsize(img_path) if os.path.exists(img_path) else None,
                'is_readable': False,
                'error': str(e)
            }
    
    def extract_image_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract metadata for all images in dataframe.
        
        Args:
            df: DataFrame with 'full_path' column
            
        Returns:
            DataFrame with added image metadata columns
        """
        df = df.copy()
        
        metadata = []
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                logger.debug(f"Processing image {idx}/{len(df)}")
            
            meta = self.inspect_image(row['full_path'])
            metadata.append(meta)
        
        metadata_df = pd.DataFrame(metadata)
        result = pd.concat([df, metadata_df], axis=1)
        
        logger.info(f"Extracted metadata for {len(result)} images")
        return result
    
    def validate_training_data(self, train_df: pd.DataFrame) -> Dict:
        """
        Validate training data integrity.
        
        Args:
            train_df: Training dataframe
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # Check class distribution
        class_counts = train_df['y'].value_counts().sort_index()
        results['stats']['class_counts'] = class_counts.to_dict()
        
        # Verify all 10 classes present
        if len(class_counts) != 10:
            results['warnings'].append(f"Expected 10 classes, found {len(class_counts)}")
        
        # Check balance
        expected_count = 1000
        for class_id, count in class_counts.items():
            if count != expected_count:
                results['warnings'].append(f"Class {class_id}: expected {expected_count}, got {count}")
        
        # Check for missing files
        missing_count = (train_df['is_readable'] == False).sum()
        if missing_count > 0:
            results['errors'].append(f"{missing_count} files are not readable")
            results['is_valid'] = False
        
        # Log results
        if results['errors']:
            logger.error(f"Validation errors: {results['errors']}")
        if results['warnings']:
            logger.warning(f"Validation warnings: {results['warnings']}")
        if not results['errors']:
            logger.info("Training data validation passed")
        
        return results
    
    def load_source_mapping(self) -> Dict[int, str]:
        """
        Load mapping of class IDs to generator names.
        
        Returns:
            Dictionary mapping class ID to generator name
        """
        sources_file = self.data_dir / "sources.txt"
        
        if not sources_file.exists():
            logger.warning(f"Sources file not found: {sources_file}")
            # Return default mapping if not found
            return {i: f"Generator_{i}" for i in range(10)}
        
        mapping = {}
        try:
            with open(sources_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('|')
                    if len(parts) == 2:
                        mapping[int(parts[0])] = parts[1]
            logger.info(f"Loaded source mapping: {mapping}")
        except Exception as e:
            logger.error(f"Error loading source mapping: {e}")
            mapping = {i: f"Generator_{i}" for i in range(10)}
        
        return mapping
    
    def add_source_names(self, df: pd.DataFrame, source_mapping: Dict[int, str]) -> pd.DataFrame:
        """
        Add human-readable generator names to dataframe.
        
        Args:
            df: DataFrame with 'y' column (for training data)
            source_mapping: Mapping of class ID to generator name
            
        Returns:
            DataFrame with 'source_name' column added (if 'y' exists)
        """
        df = df.copy()
        
        if 'y' in df.columns:
            df['source_name'] = df['y'].map(source_mapping)
            logger.info(f"Added source names to {len(df)} samples")
        
        return df
    
    def compute_statistics(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Compute summary statistics for the dataset.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'train': {
                'total_samples': len(train_df),
                'classes': train_df['y'].nunique() if 'y' in train_df.columns else None,
                'class_distribution': train_df['y'].value_counts().to_dict() if 'y' in train_df.columns else {},
                'avg_height': train_df['height'].mean() if 'height' in train_df.columns else None,
                'avg_width': train_df['width'].mean() if 'width' in train_df.columns else None,
                'avg_file_size_mb': (train_df['file_size_bytes'].mean() / 1e6) if 'file_size_bytes' in train_df.columns else None,
                'formats': train_df['format'].value_counts().to_dict() if 'format' in train_df.columns else {},
                'color_modes': train_df['color_mode'].value_counts().to_dict() if 'color_mode' in train_df.columns else {},
            },
            'test': {
                'total_samples': len(test_df),
                'avg_height': test_df['height'].mean() if 'height' in test_df.columns else None,
                'avg_width': test_df['width'].mean() if 'width' in test_df.columns else None,
                'avg_file_size_mb': (test_df['file_size_bytes'].mean() / 1e6) if 'file_size_bytes' in test_df.columns else None,
            }
        }
        
        logger.info(f"Computed statistics: {len(train_df)} train, {len(test_df)} test")
        return stats
