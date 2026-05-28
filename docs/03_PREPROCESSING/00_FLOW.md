# Pipeline 03: Preprocessing

## Flow Diagram

```
Input: train_df, test_df
       ↓
┌─────────────────────────────────┐
│ Load Images from Disk (batch)   │
│ - Use full_path column          │
│ - Parallel loading (8 workers)  │
│ - Handle corrupted gracefully   │
└────────────┬────────────────────┘
             │
      ┌──────▼──────────────────┐
      │ Resize to (H, W, 3)     │
      │ - Standardize all to    │
      │   224×224 or 256×256    │
      │ - Use PIL.Image.resize  │
      └──────┬──────────────────┘
             │
      ┌──────▼──────────────────┐
      │ Convert to RGB/Tensor   │
      │ - Ensure 3-channel RGB  │
      │ - Convert to numpy array│
      │ - Convert to float32    │
      └──────┬──────────────────┘
             │
      ┌──────▼──────────────────┐
      │ Normalize Pixel Values  │
      │ - Subtract ImageNet mean│
      │ - Divide by ImageNet std│
      │ - Scale to [-1, 1] or   │
      │   [0, 1] (config)       │
      └──────┬──────────────────┘
             │
      ┌──────▼──────────────────┐
      │ Cache Preprocessed Data │
      │ (Optional: pickle/npz)  │
      │ - Save to cache dir     │
      │ - Track version/checksum│
      └──────┬──────────────────┘
             │
      ┌──────▼──────────────────┐
      │ Output:                 │
      │ - Preprocessed train arr│
      │ - Preprocessed test arr │
      │ - Normalization params  │
      └──────────────────────────┘
```

## Key Design Decisions

### 1. **Standardization Strategy**
- **ImageNet normalization:** mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- **Why:** Pretrained models expect ImageNet-normalized inputs
- **Format:** Normalize AFTER conversion to [0, 1] float

### 2. **Resize Strategy**
- **Target size:** 224×224 (standard for ImageNet)
- **Why:** Most pretrained models (ResNet, ViT, etc.) require this
- **Interpolation:** PIL.Image.LANCZOS (high-quality, slow)

### 3. **Memory Strategy**
- **Don't load all 10K images at once**
- **Instead:** Create generator/dataloader class
- **Batch loading:** Load 32 images at a time during training

### 4. **Caching Strategy**
- **Cache preprocessed train data** (reuse across folds)
- **Don't cache test data** (processed once at inference)
- **Store as:** .npy files with metadata

## Code Pattern

```python
class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), normalization='imagenet'):
        self.target_size = target_size
        self.normalization = normalization
        self.norm_params = self._get_normalization_params()
    
    def preprocess(self, img_path):
        """Load and preprocess single image."""
        # Load
        img = PIL.Image.open(img_path).convert('RGB')
        
        # Resize
        img = img.resize(self.target_size, PIL.Image.LANCZOS)
        
        # Convert to numpy
        img_array = np.array(img, dtype='float32') / 255.0
        
        # Normalize
        mean, std = self.norm_params
        img_array = (img_array - mean) / std
        
        return img_array  # shape: (224, 224, 3)
    
    def _get_normalization_params(self):
        if self.normalization == 'imagenet':
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
        elif self.normalization == 'standard':
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
        return mean, std

# Usage
preprocessor = ImagePreprocessor()
train_arrays = []
for path in train_df['full_path']:
    img_array = preprocessor.preprocess(path)
    train_arrays.append(img_array)
X_train = np.stack(train_arrays)  # shape: (7000, 224, 224, 3)
```

## Output Format

```python
X_train: np.ndarray
  Shape: (7000, 224, 224, 3)
  Dtype: float32
  Values: [-2 to 2] (after ImageNet normalization)

X_test: np.ndarray
  Shape: (3000, 224, 224, 3)
  Dtype: float32

normalization_metadata: dict
  {
    'target_size': (224, 224),
    'normalization_type': 'imagenet',
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'dtype': 'float32'
  }
```

## Dependencies
- `PIL` (Pillow) - Image loading/resizing
- `numpy` - Array operations
- `joblib` - Parallel processing

## Integration
**Inputs:** train_df, test_df (from Data Loading)
**Outputs:** X_train, X_test, normalization_metadata
**Consumed by:** Train/Val Split, Model Training
