# Pipeline 04: Augmentation (Training-Time)

## Flow Diagram

```
Input: Preprocessed image batch (224, 224, 3)
       y_batch (labels)
       ↓
┌──────────────────────────────────────┐
│ Initialize Augmentation Pipeline     │
│ - Random rotation ([-5°, +5°])       │
│ - Random horizontal flip (p=0.5)     │
│ - Random vertical flip (p=0.2)       │
│ - Random crop (crop_size=224)        │
│ - Random brightness (factor: 0.8-1.2)│
│ - Random contrast (factor: 0.8-1.2)  │
│ - Random hue shift (small)           │
│ - Random saturation (factor: 0.8-1.2)│
│ - Gaussian blur (σ=0.5-1.0)          │
└────────────┬─────────────────────────┘
             │
      ┌──────▼─────────────────────┐
      │ For Each Batch:            │
      │ Apply Random Augmentations │
      │ - Each image gets unique   │
      │   random combination       │
      │ - Deterministic with seed  │
      │   per epoch                │
      └──────┬─────────────────────┘
             │
      ┌──────▼─────────────────────┐
      │ Output Augmented Batch:    │
      │ - X_augmented: modified   │
      │ - y_batch: unchanged      │
      │ - Ready for model input   │
      └────────────────────────────┘
```

## Key Augmentations

| Augmentation | Probability/Range | Justification |
|---|---|---|
| **Rotation** | [-5°, +5°] | Handle slight rotation in post-processing |
| **H-Flip** | 50% | Standard augmentation, doesn't break semantics |
| **V-Flip** | 20% | Less common in faces, apply rarely |
| **Brightness** | [0.8, 1.2] | Simulate lighting variations |
| **Contrast** | [0.8, 1.2] | Simulate different camera/compression |
| **Saturation** | [0.8, 1.2] | Handle color shifts from compression |
| **Gaussian Blur** | σ ∈ [0.5, 1.0] | Simulate blur from post-processing |
| **Random Crop** | 224→224 | Generate spatial variation |

## Code Pattern (albumentations)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline
train_augmentations = A.Compose([
    A.Rotate(limit=5, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.RandomRain(p=0.1),  # Simulate rain/artifacts
    A.GaussNoise(p=0.2),  # Simulate noise
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))

# Apply during training
augmented = train_augmentations(image=X_train_batch)
X_train_augmented = augmented['image']
```

## Dependencies
- `albumentations` - Fast augmentation library
- `torch` - If using PyTorch models
- `numpy` - Array operations

## Integration
**Inputs:** X_train (from Preprocessing)
**Outputs:** Augmented batches (on-the-fly during training)
**Consumed by:** Model Training (DataLoader)
