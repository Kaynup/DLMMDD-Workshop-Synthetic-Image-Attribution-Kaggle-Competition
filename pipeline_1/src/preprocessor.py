"""Image preprocessing pipeline."""
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import torchvision.transforms as transforms
from .logger_setup import setup_logger

logger = setup_logger(__name__)

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImagePreprocessor:
    """Preprocess images for model training."""
    
    def __init__(
        self,
        target_height: int = 224,
        target_width: int = 224,
        normalization: str = "imagenet",
        color_mode: str = "RGB",
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_height: Target image height
            target_width: Target image width
            normalization: "imagenet" or "minmax"
            color_mode: "RGB" or "Grayscale"
        """
        self.target_height = target_height
        self.target_width = target_width
        self.normalization = normalization
        self.color_mode = color_mode
        
        # Create transform pipeline
        self.transforms = self._build_transforms()
        logger.info(f"Initialized ImagePreprocessor: {target_height}x{target_width}, {normalization}")
    
    def _build_transforms(self):
        """Build transform pipeline."""
        if self.normalization == "imagenet":
            normalize = transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        else:  # minmax
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        
        transform_list = [
            transforms.Resize((self.target_height, self.target_width), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            normalize,
        ]
        
        return transforms.Compose(transform_list)
    
    def preprocess_image(self, img_path: str) -> Optional[np.ndarray]:
        """
        Preprocess a single image.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array or None if error
        """
        try:
            # Load image
            img = Image.open(img_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply transforms (returns tensor)
            tensor = self.transforms(img)
            
            # Convert to numpy
            img_array = tensor.numpy()
            
            return img_array
        
        except Exception as e:
            logger.warning(f"Error preprocessing {img_path}: {e}")
            return None
    
    def preprocess_batch(self, image_paths: list) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Array of preprocessed images (N, C, H, W)
        """
        batch = []
        failed_count = 0
        
        for idx, img_path in enumerate(image_paths):
            if idx % 500 == 0:
                logger.debug(f"Preprocessing {idx}/{len(image_paths)}")
            
            img_array = self.preprocess_image(img_path)
            if img_array is not None:
                batch.append(img_array)
            else:
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(f"Failed to preprocess {failed_count}/{len(image_paths)} images")
        
        logger.info(f"Preprocessed batch: {len(batch)} images")
        return np.array(batch)


class ImageAugmenter:
    """Data augmentation for training."""
    
    def __init__(
        self,
        enabled: bool = True,
        horizontal_flip_p: float = 0.5,
        rotation_degrees: int = 15,
        brightness_factor: float = 0.2,
        contrast_factor: float = 0.2,
    ):
        """
        Initialize augmenter.
        
        Args:
            enabled: Whether augmentation is enabled
            horizontal_flip_p: Probability of horizontal flip
            rotation_degrees: Max rotation angle
            brightness_factor: Brightness adjustment range
            contrast_factor: Contrast adjustment range
        """
        self.enabled = enabled
        
        if enabled:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=horizontal_flip_p),
                transforms.RandomRotation(degrees=rotation_degrees),
                transforms.ColorJitter(
                    brightness=brightness_factor,
                    contrast=contrast_factor,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ])
        else:
            self.transforms = transforms.Compose([])
        
        logger.info(f"Initialized ImageAugmenter: enabled={enabled}")
    
    def augment_tensor(self, tensor) -> None:
        """
        Apply augmentation to tensor in-place.
        This is typically called within a DataLoader collate function.
        
        Args:
            tensor: Image tensor
            
        Returns:
            Augmented tensor
        """
        if self.enabled:
            return self.transforms(tensor)
        return tensor
