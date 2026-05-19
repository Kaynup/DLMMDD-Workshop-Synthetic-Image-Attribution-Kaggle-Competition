# 🚀 Pipe-1: Complete End-to-End ML Pipeline
## Synthetic Image Attribution Challenge

**Complete workflow:** Data Loading → EDA → Preprocessing → CV Split → Model Training → Validation → Checkpoint Selection → Inference & Submission

---

## 📋 Pipeline Overview

This **single integrated pipeline** implements the complete solution from raw data to Kaggle submission. It combines multiple stages into one cohesive notebook.

### What This Pipeline Does

```
Raw Data (training.csv, test.csv, /Data/)
    ↓
STAGE 1: Load & Validate Data
    ↓
STAGE 2: Exploratory Data Analysis (EDA)
    ↓
STAGE 3: Preprocess Images (224×224, normalize)
    ↓
STAGE 4: Create 5-Fold Stratified CV Splits
    ↓
STAGE 5: Train Models (Loop through 5 folds)
    ↓
STAGE 6: Validate & Compute Metrics
    ↓
STAGE 7: Select Best Checkpoint per Fold
    ↓
STAGE 8: Inference on Test Set
    ↓
submission.csv (Ready for Kaggle!)
```

---

## 🎯 Competition Context

| Aspect | Details |
|--------|---------|
| **Task** | 10-class image classification (synthetic attribution) |
| **Data** | 7,000 training images (700 per class), 3,000 test images |
| **Classes** | AuraFlow, Freepik, Lumina, Photon, Pixart-sigma, Playground v2.5, StableDiffusion3, SD3.5, SDXL-Turbo, Tencent Hunyuan |
| **Challenge** | Test images have 1-3 unknown post-processing operations |
| **Metric** | Classification Accuracy |
| **Expected** | 65-70% after first run, 72-80% optimized |

---

## 📊 Pipeline Architecture

### Stage 1: Data Loading & Validation

**Reference:** `docs/01_DATA_LOADING_VALIDATION/`

**Exact File Paths (Kaggle Environment):**
```
NOTE: Double Data directory structure - /Data/Data/
Full paths (from root):
  - training.csv: /kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge/Data/Data/training.csv
  - test.csv: /kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge/Data/Data/test.csv
  - Training/: /kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge/Data/Data/Training/ (7000 .png files)
  - Test/: /kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge/Data/Data/Test/ (3000 .png files)

Output Files (Created in /kaggle/working/):
  - logs/training_log.csv
  - logs/validation_metrics.csv
  - preprocessed/X_train_preprocessed.npy
  - preprocessed/X_test_preprocessed.npy
  - checkpoints/fold_*/epoch_*.pth (500 total)
  - final_models/fold_*_best.pth (5 best)
  - submission/submission.csv ← UPLOAD THIS
```

**Exact Base Paths:**
```python
BASE_INPUT = '/kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge/Data/'

# CSV paths
train_csv_path = f'{BASE_INPUT}Data/training.csv'
test_csv_path = f'{BASE_INPUT}Data/test.csv'
```

**CSV Structure:**
```
training.csv columns: ID, path, y
  ID: integer image identifier
  path: relative path like "Data/Training/123.png"
  y: numeric class index (0-9)

test.csv columns: ID, path
  ID: integer image identifier  
  path: relative path like "Data/Test/456.png"
```

**Exact Execution Steps:**

**Step 1.1: Load CSVs**
```python
# EXACT CODE PATTERN for Kaggle
BASE_INPUT = '/kaggle/input/competitions/dlmmdd-workshop-synthetic-source-attribution-challenge/Data/'

train_df = pd.read_csv(f'{BASE_INPUT}Data/training.csv')  # shape: (7000, 3)
test_df = pd.read_csv(f'{BASE_INPUT}Data/test.csv')       # shape: (3000, 2)

logger.info(f"Loaded training.csv: {train_df.shape}")
logger.info(f"  Columns: {list(train_df.columns)}")
logger.info(f"  Sample path: {train_df['path'].iloc[0]}")
logger.info(f"Loaded test.csv: {test_df.shape}")

# VERIFY COLUMN NAMES
assert list(train_df.columns) == ['ID', 'path', 'y'], f"Got columns: {list(train_df.columns)}"
assert list(test_df.columns) == ['ID', 'path'], f"Got columns: {list(test_df.columns)}"
```

**Step 1.2: Verify File Existence**
```python
# EXACT: Check every single file exists
train_missing = []
for idx, row in train_df.iterrows():
    path = f"{BASE_INPUT}{row['path']}"
    if not os.path.exists(path):
        train_missing.append(path)
        
if train_missing:
    raise FileNotFoundError(f"Missing {len(train_missing)} training files: {train_missing[:5]}")
else:
    logger.info(f"✓ All 7000 training files exist")

test_missing = []
for idx, row in test_df.iterrows():
    path = f"{BASE_INPUT}{row['path']}"
    if not os.path.exists(path):
        test_missing.append(path)
        
if test_missing:
    raise FileNotFoundError(f"Missing {len(test_missing)} test files: {test_missing[:5]}")
else:
    logger.info(f"✓ All 3000 test files exist")
```

**Step 1.3: Extract Image Metadata**
```python
# EXACT: Create class mapping (numeric → name)
class_idx_to_name = {
    0: 'AuraFlow',
    1: 'Freepik',
    2: 'Lumina',
    3: 'Photon',
    4: 'Pixart-sigma',
    5: 'Playground v2.5',
    6: 'StableDiffusion3',
    7: 'StableDiffusion3.5',
    8: 'StableDiffusionXL-Turbo',
    9: 'Tencent Hunyuan'
}

# EXACT: For each image, get dimensions and file size
train_metadata = []
for idx, row in train_df.iterrows():
    path = f"{BASE_INPUT}{row['path']}"
    img = Image.open(path)
    size_bytes = os.path.getsize(path)
    train_metadata.append({
        'image_id': row['ID'],
        'source': class_idx_to_name[row['y']],  # Convert numeric y to class name
        'width': img.width,
        'height': img.height,
        'format': img.format,
        'size_kb': size_bytes / 1024,
        'mode': img.mode  # RGB, RGBA, L, etc.
    })
    if (idx + 1) % 1000 == 0:
        logger.info(f"Processed {idx + 1} training images")

train_metadata_df = pd.DataFrame(train_metadata)
```

**Step 1.4: Validate Class Distribution**
```python
# EXACT: Every class must have EXACTLY 700 images
class_counts = train_metadata_df['source'].value_counts()
logger.info(f"\nClass distribution:\n{class_counts}")

# VERIFY: Each class has exactly 700
expected_per_class = 700
for class_name, count in class_counts.items():
    assert count == expected_per_class, f"Class {class_name} has {count} images, expected {expected_per_class}"
logger.info(f"✓ All {len(class_counts)} classes have exactly {expected_per_class} images ({len(class_counts) * expected_per_class} total)")

# VERIFY: No duplicate IDs
assert len(train_df) == len(train_df['ID'].unique()), "Duplicate image IDs found"
logger.info("✓ No duplicate image IDs")
```

**Outputs After Stage 1:**
- `train_df`: 7000 rows, columns=['ID', 'path', 'y']
- `test_df`: 3000 rows, columns=['ID', 'path']
- `train_metadata_df`: 7000 rows with source (mapped), width, height, format, size_kb, mode
- Validation report logged to console

**Key Checks (MUST All Pass):**
```
✓ train_df.shape == (7000, 3)
✓ test_df.shape == (3000, 2)
✓ train_df['y'].min() == 0, train_df['y'].max() == 9
✓ All image paths resolve correctly
✓ class_counts: {class: 700 for each of 10 classes}
✓ No duplicate IDs
✓ All images readable (no corrupt files)
✓ Image modes: RGB (primary) or RGBA (handle separately)
```

**If ANY Check Fails:**
- Stop execution immediately
- Log which check failed
- Investigate file paths or data integrity before proceeding

---

### Stage 2: Exploratory Data Analysis (EDA)

**Reference:** `docs/02_EDA/`

**Exact Imports Required:**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import json

# For generating HTML report
from pathlib import Path
```

**Exact Execution Steps:**

**Step 2.1: Class Distribution Analysis**
```python
# EXACT: Count images per class
class_counts = train_metadata_df['source'].value_counts().sort_values(ascending=False)

# Create bar chart and save
fig, ax = plt.subplots(figsize=(12, 6))
class_counts.plot(kind='bar', ax=ax, color='steelblue')
ax.set_xlabel('Generator Class')
ax.set_ylabel('Number of Images')
ax.set_title('Training Data: Class Distribution')
ax.set_ylim([900, 1100])
for i, v in enumerate(class_counts):
    ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/eda_outputs/class_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

logger.info("✓ Saved class_distribution.png")

# VERIFY: All should be 700
assert all(v == 700 for v in class_counts.values), "Classes not balanced!"
logger.info("✓ Confirmed: All 10 classes have exactly 700 images")
```

**Step 2.2: Image Dimension Analysis**
```python
# EXACT: Analyze width, height distributions
logger.info(f"\nImage Dimensions Statistics:")
logger.info(f"Width:  mean={train_metadata_df['width'].mean():.1f}, std={train_metadata_df['width'].std():.1f}")
logger.info(f"        min={train_metadata_df['width'].min()}, max={train_metadata_df['width'].max()}")
logger.info(f"Height: mean={train_metadata_df['height'].mean():.1f}, std={train_metadata_df['height'].std():.1f}")
logger.info(f"        min={train_metadata_df['height'].min()}, max={train_metadata_df['height'].max()}")

# Histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train_metadata_df['width'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Width (pixels)')
axes[0].set_ylabel('Count')
axes[0].set_title('Width Distribution')
axes[1].hist(train_metadata_df['height'], bins=50, color='lightcoral', edgecolor='black')
axes[1].set_xlabel('Height (pixels)')
axes[1].set_ylabel('Count')
axes[1].set_title('Height Distribution')
plt.tight_layout()
plt.savefig('/kaggle/working/eda_outputs/plots/dimension_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

logger.info("✓ Saved dimension_distribution.png")
```

**Step 2.3: File Format & Size Analysis**
```python
# EXACT: Analyze PNG vs JPEG, sizes
logger.info(f"\nImage Format Statistics:")
format_counts = train_metadata_df['format'].value_counts()
logger.info(format_counts)

logger.info(f"\nImage Size (KB) Statistics:")
logger.info(f"Mean: {train_metadata_df['size_kb'].mean():.2f} KB")
logger.info(f"Std:  {train_metadata_df['size_kb'].std():.2f} KB")
logger.info(f"Min:  {train_metadata_df['size_kb'].min():.2f} KB")
logger.info(f"Max:  {train_metadata_df['size_kb'].max():.2f} KB")

# Size distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot([train_metadata_df[train_metadata_df['source'] == src]['size_kb'].values 
            for src in sorted(train_metadata_df['source'].unique())],
           labels=sorted(train_metadata_df['source'].unique()))
ax.set_ylabel('File Size (KB)')
ax.set_title('File Size Distribution by Generator')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('/kaggle/working/eda_outputs/plots/filesize_boxplot.png', dpi=100, bbox_inches='tight')
plt.close()

logger.info("✓ Saved filesize_boxplot.png")
```

**Step 2.4: Generator Sample Grid (MOST IMPORTANT FOR EDA)**
```python
# EXACT: Show sample images per class to understand generator signatures
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 10 classes
axes = axes.flatten()

classes = sorted(train_metadata_df['source'].unique())
for col_idx, class_name in enumerate(classes):
    # Get one random image from this class
    class_images = train_metadata_df[train_metadata_df['source'] == class_name]
    sample_row = class_images.sample(1).iloc[0]
    
    # Load and display (use BASE_INPUT prefix)
    img_id = sample_row['image_id']
    # Find matching row in train_df to get path
    train_row = train_df[train_df['ID'] == img_id].iloc[0]
    img_path = f"{BASE_INPUT}{train_row['path']}"
    img = Image.open(img_path)
    
    axes[col_idx].imshow(img)
    axes[col_idx].set_title(class_name, fontsize=10, fontweight='bold')
    axes[col_idx].axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/eda_outputs/generator_samples.png', dpi=100, bbox_inches='tight')
plt.close()

logger.info("✓ Saved generator_samples.png - Visual inspection of each class")
logger.info("   → Look for distinctive patterns (lighting, style, artifacts)")
```

**Step 2.5: Save EDA Insights to JSON**
```python
# EXACT: Save computed statistics for later reference
eda_insights = {
    'class_distribution': class_counts.to_dict(),
    'image_dimensions': {
        'width': {
            'mean': float(train_metadata_df['width'].mean()),
            'std': float(train_metadata_df['width'].std()),
            'min': int(train_metadata_df['width'].min()),
            'max': int(train_metadata_df['width'].max())
        },
        'height': {
            'mean': float(train_metadata_df['height'].mean()),
            'std': float(train_metadata_df['height'].std()),
            'min': int(train_metadata_df['height'].min()),
            'max': int(train_metadata_df['height'].max())
        }
    },
    'file_format': format_counts.to_dict(),
    'file_size_kb': {
        'mean': float(train_metadata_df['size_kb'].mean()),
        'std': float(train_metadata_df['size_kb'].std()),
        'min': float(train_metadata_df['size_kb'].min()),
        'max': float(train_metadata_df['size_kb'].max())
    },
    'image_modes': train_metadata_df['mode'].value_counts().to_dict(),
    'num_training_images': len(train_df),
    'num_test_images': len(test_df),
    'num_classes': len(class_counts)
}

with open('/kaggle/working/eda_outputs/eda_insights.json', 'w') as f:
    json.dump(eda_insights, f, indent=2)

logger.info("✓ Saved eda_insights.json")
```

**Outputs After Stage 2:**
- `/kaggle/working/eda_outputs/class_distribution.png`
- `/kaggle/working/eda_outputs/dimension_distribution.png`
- `/kaggle/working/eda_outputs/filesize_boxplot.png`
- `/kaggle/working/eda_outputs/generator_samples.png`
- `/kaggle/working/eda_outputs/eda_insights.json`

**Key Insights to Observe:**
```
LOOK FOR THESE PATTERNS:
1. Are all classes visually similar or distinct?
   → If identical, model will memorize generator "signatures"
   
2. Do certain generators have different image sizes?
   → May indicate different post-processing or upscaling

3. Is file size similar across generators?
   → Large variation suggests different compression levels

4. Do all images have RGB mode?
   → Some might be RGBA (handle in preprocessing)

5. Are dimensions consistent within each class?
   → Variation suggests random crops or resizing in test set
```

**Critical Finding:**
Look at the `generator_samples.png` - if you can visually distinguish classes, the model can learn genuine differences. If all look identical, the model will learn artificial signatures (risky for leaderboard generalization).

---

### Stage 3: Preprocessing

**Reference:** `docs/03_PREPROCESSING/`

**Exact Imports Required:**
```python
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import json
```

**Exact Preprocessing Pipeline:**

**Step 3.1: Define Normalization Constants**
```python
# EXACT: ImageNet statistics (MUST use these exact values)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_SIZE = 224  # EXACT: All pretrained models expect 224x224

# Log these for reference
logger.info(f"Preprocessing Configuration:")
logger.info(f"  Target size: {TARGET_SIZE}x{TARGET_SIZE}")
logger.info(f"  Normalization: ImageNet mean={IMAGENET_MEAN}, std={IMAGENET_STD}")
logger.info(f"  This is REQUIRED for pretrained models to work correctly")
```

**Step 3.2: Create Preprocessing Function**
```python
# EXACT: This is the function you'll use for EVERY image
def preprocess_image(image_path, target_size=224):
    """
    Load, resize, and normalize a single image.
    
    EXACT STEPS:
    1. Load image from disk (PIL)
    2. Resize to 224x224 (LANCZOS for quality)
    3. Convert to RGB (handle RGBA → RGB, grayscale → RGB)
    4. Convert to numpy array
    5. Normalize with ImageNet stats
    6. Return as numpy array (height, width, 3)
    
    Args:
        image_path (str): Path to image file
        target_size (int): 224 (hardcoded, don't change)
    
    Returns:
        np.array: Shape (224, 224, 3), values in [-2, 2] range
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert RGBA → RGB (discard alpha channel)
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        
        # Convert grayscale → RGB (replicate channel 3x)
        elif img.mode == 'L':
            img = img.convert('RGB')
        
        # Ensure RGB (already RGB, no change needed)
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224 using high-quality LANCZOS
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        
        # Convert to numpy array (range [0, 255])
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Apply ImageNet normalization: (x - mean) / std
        # This converts to approximately [-2, 2] range
        img_array = (img_array - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        raise
```

**Step 3.3: Preprocess All Training Images**
```python
# EXACT: Load all 7000 training images into memory
logger.info(f"Preprocessing {len(train_df)} training images...")
X_train = np.zeros((len(train_df), 224, 224, 3), dtype=np.float32)

for idx, row in train_df.iterrows():
    image_path = f"Data/Training/{row['image_id']}.png"
    X_train[idx] = preprocess_image(image_path, target_size=224)
    
    if (idx + 1) % 1000 == 0:
        logger.info(f"  Processed {idx + 1} training images")

logger.info(f"✓ Loaded all training images: X_train.shape = {X_train.shape}")

# Verify values are in expected range [-2, 2]
logger.info(f"  Value range: min={X_train.min():.3f}, max={X_train.max():.3f}")
assert X_train.min() < -1.0, "Preprocessing may have failed (values too high)"
assert X_train.max() > 1.0, "Preprocessing may have failed (values too low)"
logger.info(f"✓ Value range looks correct for ImageNet normalization")

# Check for NaN or Inf
assert not np.isnan(X_train).any(), "Found NaN values in X_train!"
assert not np.isinf(X_train).any(), "Found Inf values in X_train!"
logger.info(f"✓ No NaN or Inf values")
```

**Step 3.4: Preprocess All Test Images**
```python
# EXACT: Load all 3000 test images into memory
logger.info(f"Preprocessing {len(test_df)} test images...")
X_test = np.zeros((len(test_df), 224, 224, 3), dtype=np.float32)

for idx, row in test_df.iterrows():
    image_path = f"Data/Test/{row['image_id']}.png"
    X_test[idx] = preprocess_image(image_path, target_size=224)
    
    if (idx + 1) % 500 == 0:
        logger.info(f"  Processed {idx + 1} test images")

logger.info(f"✓ Loaded all test images: X_test.shape = {X_test.shape}")
logger.info(f"  Value range: min={X_test.min():.3f}, max={X_test.max():.3f}")

assert not np.isnan(X_test).any(), "Found NaN values in X_test!"
assert not np.isinf(X_test).any(), "Found Inf values in X_test!"
logger.info(f"✓ No NaN or Inf values")
```

**Step 3.5: Save Preprocessed Data**
```python
# EXACT: Save to disk for future use (avoid reprocessing)
os.makedirs('preprocessed', exist_ok=True)

# Save numpy arrays
np.save('preprocessed/X_train_preprocessed.npy', X_train)
np.save('preprocessed/X_test_preprocessed.npy', X_test)

logger.info(f"✓ Saved X_train_preprocessed.npy ({X_train.nbytes / 1e9:.2f} GB)")
logger.info(f"✓ Saved X_test_preprocessed.npy ({X_test.nbytes / 1e9:.2f} GB)")

# Save metadata
preprocessing_metadata = {
    'target_size': 224,
    'imagenet_mean': IMAGENET_MEAN,
    'imagenet_std': IMAGENET_STD,
    'X_train_shape': list(X_train.shape),
    'X_test_shape': list(X_test.shape),
    'X_train_value_range': [float(X_train.min()), float(X_train.max())],
    'X_test_value_range': [float(X_test.min()), float(X_test.max())],
    'timestamp': str(pd.Timestamp.now())
}

with open('preprocessed/normalization_metadata.json', 'w') as f:
    json.dump(preprocessing_metadata, f, indent=2)

logger.info(f"✓ Saved normalization_metadata.json")
```

**Outputs After Stage 3:**
- `preprocessed/X_train_preprocessed.npy`: (7000, 224, 224, 3) float32 array (~1.68 GB)
- `preprocessed/X_test_preprocessed.npy`: (3000, 224, 224, 3) float32 array (~0.72 GB)
- `preprocessed/normalization_metadata.json`: Configuration and statistics

**Verification Checks (MUST Pass):**
```python
✓ X_train.shape == (7000, 224, 224, 3)
✓ X_test.shape == (3000, 224, 224, 3)
✓ X_train.dtype == np.float32
✓ X_test.dtype == np.float32
✓ -2.5 < X_train.min() < -1.0  (ImageNet norm puts ~95% in [-2, 2])
✓ 1.0 < X_train.max() < 2.5
✓ No NaN or Inf values
✓ Files saved successfully
```

**Memory Requirements:**
- X_train: ~1.7 GB RAM
- X_test: ~0.7 GB RAM
- Total: ~2.4 GB RAM
- **If you run out of memory:** Process in batches and save to disk, don't load all at once

**WHY These Exact Values:**
- **224×224:** Standard size for ResNet/EfficientNet/ViT (pretrained weights expect this)
- **LANCZOS:** Highest quality resampling, preserves details
- **ImageNet normalization:** REQUIRED for transfer learning to work
- **float32:** Standard for neural networks, saves memory vs float64
- **Save to disk:** Avoid reprocessing on every epoch (I/O expensive)

---

### Stage 4: Train/Validation Stratified Split

**Reference:** `docs/05_TRAIN_VAL_SPLIT/`

**What happens:**
- Create **5-fold StratifiedKFold** cross-validation
- Ensures each fold has balanced class distribution
- Fixed random seed for reproducibility

**Exact Imports Required:**
```python
from sklearn.model_selection import StratifiedKFold
import json
import numpy as np
```

**Exact Code:**

**Step 4.1: Create 5-Fold Stratified K-Fold**
```python
# EXACT: These parameters are hardcoded, don't change them
kf = StratifiedKFold(
    n_splits=5,           # Create 5 folds (change to 10 later if time permits)
    shuffle=True,         # Shuffle before splitting
    random_state=42       # CRITICAL: Fixed seed for reproducibility
)

logger.info("Creating 5-Fold Stratified Cross-Validation...")
logger.info(f"  n_splits: 5")
logger.info(f"  shuffle: True")
logger.info(f"  random_state: 42 (for reproducibility)")

# Get class labels for stratification
# EXACT: Extract 'source' column from train_df as y
y_train = train_df['source'].values  # Shape: (7000,)

# Create fold indices
fold_metadata = {}
fold_idx = 0

for train_indices, val_indices in kf.split(np.arange(len(train_df)), y_train):
    logger.info(f"\nFold {fold_idx}:")
    logger.info(f"  Train indices: {len(train_indices)} images")
    logger.info(f"  Val indices: {len(val_indices)} images")
    
    fold_metadata[f'fold_{fold_idx}'] = {
        'train_indices': train_indices.tolist(),  # Convert to list for JSON
        'val_indices': val_indices.tolist(),
        'num_train': len(train_indices),
        'num_val': len(val_indices)
    }
    
    fold_idx += 1

logger.info(f"\n✓ Created {fold_idx} folds")
```

**Step 4.2: Verify Fold Balance**
```python
# EXACT: For each fold, verify all 10 classes present and balanced

for fold_num in range(5):
    fold_key = f'fold_{fold_num}'
    val_indices = np.array(fold_metadata[fold_key]['val_indices'])
    
    # Get validation labels
    val_labels = y_train[val_indices]
    val_classes = pd.Categorical(val_labels, categories=train_df['source'].unique())
    class_counts = np.bincount(val_classes.codes)
    
    # Check: all 10 classes present
    assert len(class_counts) == 10, f"Fold {fold_num} missing classes!"
    
    logger.info(f"Fold {fold_num}: ~{class_counts.mean():.0f} images per class")
    
    # Check: not too imbalanced
    for class_idx, count in enumerate(class_counts):
        ratio = count / class_counts.mean()
        assert 0.8 < ratio < 1.2, f"Fold {fold_num} class {class_idx} imbalanced ({ratio:.2f}x)"

logger.info("✓ All folds have balanced class distribution")
```

**Step 4.3: Save Fold Metadata**
```python
# EXACT: Save fold information to JSON
os.makedirs('preprocessed', exist_ok=True)

with open('preprocessed/fold_metadata.json', 'w') as f:
    json.dump(fold_metadata, f, indent=2)

logger.info("✓ Saved fold_metadata.json")
```

**Outputs After Stage 4:**
- `preprocessed/fold_metadata.json`: Train/val indices for each fold

**Verification Checks (MUST Pass):**
```python
✓ len(fold_metadata) == 5
✓ For each fold: train_indices + val_indices = all 7000 unique indices
✓ No overlap between train and val per fold
✓ All 10 classes in each val_indices
✓ Class distribution ±20% of average
```

---

### Stage 5: Model Training

**Reference:** `docs/07_MODEL_TRAINING/`

**Exact Imports Required:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm  # pytorch-image-models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import csv
from sklearn.metrics import accuracy_score, f1_score
```

**Exact Configuration (Hardcoded):**
```python
# DEVICE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# MODEL
MODEL_NAME = 'efficientnet_b4'       # Exact model architecture
PRETRAINED = True                      # Use ImageNet pretrained weights
NUM_CLASSES = 10                        # 10 generators

# TRAINING
EPOCHS = 100                           # Total epochs per fold
BATCH_SIZE_TRAIN = 32                  # Training batch size
BATCH_SIZE_VAL = 32                    # Validation batch size
LEARNING_RATE = 1e-4                  # Initial learning rate
WEIGHT_DECAY = 1e-5                    # L2 regularization

# OPTIMIZER
OPTIMIZER_NAME = 'AdamW'
BETAS = (0.9, 0.999)
EPS = 1e-8

# SCHEDULER
SCHEDULER_NAME = 'CosineAnnealingLR'
T_MAX = EPOCHS

# LOSS
LABEL_SMOOTHING = 0.1

logger.info(f"""
Training Configuration:
  Model: {MODEL_NAME} (pretrained={PRETRAINED})
  Epochs: {EPOCHS}
  Batch size: train={BATCH_SIZE_TRAIN}, val={BATCH_SIZE_VAL}
  Optimizer: {OPTIMIZER_NAME} (lr={LEARNING_RATE})
  Scheduler: {SCHEDULER_NAME} (T_max={T_MAX})
  Loss: CrossEntropyLoss (label_smoothing={LABEL_SMOOTHING})
  Device: {DEVICE}
""")
```

**Step 5.1: Define Augmentation Pipeline**
```python
# EXACT: Train-time augmentation ONLY (not for validation)
train_augmentation = A.Compose([
    A.Rotate(limit=5, p=0.7),              # ±5° rotation
    A.HorizontalFlip(p=0.5),               # 50% horizontal flip
    A.VerticalFlip(p=0.2),                 # 20% vertical flip
    A.RandomBrightnessContrast(
        brightness_limit=(-0.2, 0.2),      # 0.8-1.2x
        contrast_limit=(-0.2, 0.2),
        p=0.5
    ),
    A.GaussBlur(blur_limit=3, p=0.3),      # Gaussian blur
    A.RandomCrop(224, 224, p=0.1),         # Occasionally crop
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=1.0  # Our images already in [0, 1]
    ),
    ToTensorV2()
], p=1.0)

# EXACT: Validation augmentation (normalization ONLY, no random transforms)
val_augmentation = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=1.0
    ),
    ToTensorV2()
], p=1.0)

logger.info("✓ Augmentation pipelines defined")
```

**Step 5.2: Create Dataset Class**
```python
# EXACT: Custom Dataset for loading from preprocessed numpy arrays
class ImageDataset(Dataset):
    def __init__(self, images, labels, augmentation=None):
        """
        Args:
            images: numpy array (N, 224, 224, 3) with values in [-2, 2]
            labels: numpy array (N,) with class indices 0-9
            augmentation: albumentations transform or None
        """
        self.images = images
        self.labels = labels
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image (already preprocessed, in [-2, 2])
        image = self.images[idx]  # Shape: (224, 224, 3)
        label = self.labels[idx]   # Shape: ()
        
        # Apply augmentation if provided
        if self.augmentation is not None:
            augmented = self.augmentation(image=image)
            image = augmented['image']  # Already tensor after ToTensorV2
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

logger.info("✓ ImageDataset class defined")
```

**Step 5.3: Training Loop (Loop Through Each Fold)**
```python
# EXACT: Main training loop for all 5 folds

# Create training log CSV
os.makedirs('logs', exist_ok=True)
csv_file = 'logs/training_log.csv'
csv_header = ['fold', 'epoch', 'train_loss', 'train_acc', 'train_f1', 
              'val_loss', 'val_acc', 'val_f1', 'learning_rate', 'time_sec']

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

logger.info(f"Training log will be saved to {csv_file}")

# Loop through folds
for fold_num in range(5):
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_num}")
    logger.info(f"{'='*60}")
    
    # Load fold indices
    fold_key = f'fold_{fold_num}'
    train_indices = np.array(fold_metadata[fold_key]['train_indices'])
    val_indices = np.array(fold_metadata[fold_key]['val_indices'])
    
    logger.info(f"Train: {len(train_indices)} | Val: {len(val_indices)}")
    
    # Get train/val labels
    y_train_fold = train_df.iloc[train_indices]['source'].values
    y_val_fold = train_df.iloc[val_indices]['source'].values
    
    # Get train/val images from preprocessed arrays
    X_train_fold = X_train[train_indices]  # (5600, 224, 224, 3)
    X_val_fold = X_train[val_indices]      # (1400, 224, 224, 3)
    
    # Convert class names to indices if needed
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(train_df['source'].unique()))}
    y_train_fold_idx = np.array([class_to_idx[cls] for cls in y_train_fold])
    y_val_fold_idx = np.array([class_to_idx[cls] for cls in y_val_fold])
    
    # Create datasets
    train_dataset = ImageDataset(X_train_fold, y_train_fold_idx, augmentation=train_augmentation)
    val_dataset = ImageDataset(X_val_fold, y_val_fold_idx, augmentation=val_augmentation)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"DataLoaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Initialize model
    model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-6)
    
    logger.info(f"Model initialized: {MODEL_NAME}")
    logger.info(f"Optimizer: AdamW (lr={LEARNING_RATE})")
    logger.info(f"Scheduler: CosineAnnealingLR")
    
    # Create checkpoint directory
    checkpoint_dir = f'checkpoints/fold_{fold_num}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # TRAIN PHASE
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            # Accumulate metrics
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        
        # VAL PHASE
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                # Forward
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Accumulate metrics
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Log epoch
        if (epoch + 1) % 10 == 0:
            logger.info(f"Fold {fold_num} | Epoch {epoch+1:3d}/{EPOCHS} | "
                       f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | train_f1: {train_f1:.4f} | "
                       f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | "
                       f"lr: {current_lr:.2e} | time: {epoch_time:.1f}s")
        
        # Save checkpoint (ALL epochs)
        checkpoint_path = f'{checkpoint_dir}/epoch_{epoch:03d}.pth'
        torch.save({
            'epoch': epoch,
            'fold': fold_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'learning_rate': current_lr,
            'model_name': MODEL_NAME
        }, checkpoint_path)
        
        # Write to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([fold_num, epoch, train_loss, train_acc, train_f1,
                           val_loss, val_acc, val_f1, current_lr, epoch_time])
    
    logger.info(f"✓ Fold {fold_num} training complete. Saved 100 checkpoints to {checkpoint_dir}/")

logger.info(f"\n✓ Training complete for all 5 folds")
logger.info(f"✓ Total checkpoints saved: 500 (5 folds × 100 epochs)")
logger.info(f"✓ Training log: {csv_file}")
```

**Outputs After Stage 5:**
- `checkpoints/fold_0/epoch_000.pth` through `epoch_099.pth`
- `checkpoints/fold_1/epoch_000.pth` through `epoch_099.pth`
- ... (5 folds total)
- `logs/training_log.csv`: Per-epoch metrics

**Each Checkpoint Contains:**
```python
{
    'epoch': int (0-99),
    'fold': int (0-4),
    'model_state_dict': OrderedDict(...),
    'optimizer_state_dict': {...},
    'train_loss': float,
    'train_acc': float,
    'train_f1': float,
    'val_loss': float,
    'val_acc': float,
    'val_f1': float,
    'learning_rate': float,
    'model_name': 'efficientnet_b4'
}
```

**Verification Checks (MUST Pass):**
```python
✓ 500 checkpoint files created (5 folds × 100 epochs)
✓ Each checkpoint ~100 MB (4 GB total)
✓ training_log.csv has 500 rows
✓ Validation accuracy increases over first 10-20 epochs
✓ Final val_acc > 15% (random = 10%)
✓ No NaN or Inf in metrics
```

**Expected Time:**
- ~3-4 hours on GPU (V100/A100)
- ~15-20 hours on CPU (not recommended)

### Stage 6: Validation

**Reference:** `docs/08_VALIDATION/`

**Exact Imports Required:**
```python
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import json
```

**Exact Code:**

**Step 6.1: Load All Checkpoints and Evaluate**
```python
# EXACT: Evaluate all 500 checkpoints on their validation sets

val_augmentation = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=1.0
    ),
    ToTensorV2()
], p=1.0)

# Create directory for validation metrics
os.makedirs('logs', exist_ok=True)

# Loop through all folds and all checkpoints
validation_results = []

for fold_num in range(5):
    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATING FOLD {fold_num}")
    logger.info(f"{'='*60}")
    
    # Load fold indices and data
    fold_key = f'fold_{fold_num}'
    val_indices = np.array(fold_metadata[fold_key]['val_indices'])
    X_val_fold = X_train[val_indices]
    y_val_fold = train_df.iloc[val_indices]['source'].values
    
    # Convert class names to indices
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(train_df['source'].unique()))}
    y_val_fold_idx = np.array([class_to_idx[cls] for cls in y_val_fold])
    
    # Create dataset and loader
    val_dataset = ImageDataset(X_val_fold, y_val_fold_idx, augmentation=val_augmentation)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False, num_workers=4)
    
    checkpoint_dir = f'checkpoints/fold_{fold_num}'
    
    # Evaluate all 100 checkpoints for this fold
    for epoch in range(100):
        checkpoint_path = f'{checkpoint_dir}/epoch_{epoch:03d}.pth'
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Load model
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        model.eval()
        
        # Evaluate
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                outputs = model(images)
                val_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(val_labels, val_preds)
        f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        # Get train metrics from checkpoint
        train_acc = checkpoint['train_acc']
        train_f1 = checkpoint['train_f1']
        
        validation_results.append({
            'fold': fold_num,
            'epoch': epoch,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_acc': accuracy,
            'val_f1': f1_macro,
            'checkpoint_path': checkpoint_path
        })
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch:3d} | train_acc: {train_acc:.4f} | val_acc: {accuracy:.4f} | "
                       f"train_f1: {train_f1:.4f} | val_f1: {f1_macro:.4f}")
    
    logger.info(f"✓ Validated all 100 checkpoints for fold {fold_num}")

logger.info(f"\n✓ Validation complete for all 500 checkpoints")

# Convert to DataFrame
val_df = pd.DataFrame(validation_results)
val_df.to_csv('logs/validation_metrics.csv', index=False)

logger.info(f"✓ Saved validation_metrics.csv ({len(val_df)} rows)")
```

**Step 6.2: Compute Per-Fold Metrics**
```python
# EXACT: Aggregate metrics per fold

for fold_num in range(5):
    fold_data = val_df[val_df['fold'] == fold_num]
    
    logger.info(f"\nFold {fold_num} Summary:")
    logger.info(f"  Best val_acc epoch: {fold_data['val_acc'].idxmax()}, "
               f"val_acc: {fold_data['val_acc'].max():.4f}")
    logger.info(f"  Best val_f1 epoch: {fold_data['val_f1'].idxmax()}, "
               f"val_f1: {fold_data['val_f1'].max():.4f}")
    logger.info(f"  Final epoch (99) val_acc: {fold_data.iloc[-1]['val_acc']:.4f}")
    logger.info(f"  Final epoch (99) val_f1: {fold_data.iloc[-1]['val_f1']:.4f}")
```

**Outputs After Stage 6:**
- `logs/validation_metrics.csv`: 500 rows (5 folds × 100 epochs), columns=[fold, epoch, train_acc, train_f1, val_acc, val_f1, checkpoint_path]

**Verification Checks (MUST Pass):**
```python
✓ validation_metrics.csv has 500 rows
✓ All val_acc values in [0, 1]
✓ All f1_macro values in [0, 1]
✓ No NaN or Inf values
✓ Final epoch accuracies > 15% (random = 10%)
```

---

### Stage 7: Checkpoint Selection

**Reference:** `docs/09_CHECKPOINT_SELECTION/`

**The Key Formula:**
$$\text{Generalization Score} = \text{val\_acc} - |\text{train\_acc} - \text{val\_acc}|$$

**Why This Formula:**
- Penalizes overfitting (large gap between train/val)
- Prefers models that generalize well
- Example: train=0.99, val=0.80 → Gen_Score=0.61 (overfitting)
- Example: train=0.92, val=0.90 → Gen_Score=0.88 (good generalization)

**Exact Code:**

**Step 7.1: Compute Generalization Score for All Checkpoints**
```python
# EXACT: Apply formula to all checkpoints

val_df['gen_score'] = val_df['val_acc'] - np.abs(val_df['train_acc'] - val_df['val_acc'])

logger.info(f"Generalization Score computed for all checkpoints")
logger.info(f"  Gen_Score range: [{val_df['gen_score'].min():.4f}, {val_df['gen_score'].max():.4f}]")
```

**Step 7.2: Select Best Checkpoint Per Fold**
```python
# EXACT: For each fold, find epoch with max Gen_Score

best_checkpoints = {}

for fold_num in range(5):
    fold_data = val_df[val_df['fold'] == fold_num].copy()
    
    # Find row with max gen_score
    best_row = fold_data.loc[fold_data['gen_score'].idxmax()]
    
    best_epoch = int(best_row['epoch'])
    best_gen_score = best_row['gen_score']
    best_train_acc = best_row['train_acc']
    best_val_acc = best_row['val_acc']
    best_checkpoint_path = best_row['checkpoint_path']
    
    logger.info(f"Fold {fold_num}: Selected epoch {best_epoch:3d} "
               f"(Gen_Score={best_gen_score:.4f}, train_acc={best_train_acc:.4f}, "
               f"val_acc={best_val_acc:.4f})")
    
    best_checkpoints[fold_num] = {
        'epoch': best_epoch,
        'gen_score': float(best_gen_score),
        'train_acc': float(best_train_acc),
        'val_acc': float(best_val_acc),
        'checkpoint_path': best_checkpoint_path
    }
```

**Step 7.3: Copy Best Checkpoints to Final Models Directory**
```python
# EXACT: Copy selected checkpoints to final_models/

os.makedirs('final_models', exist_ok=True)

for fold_num in range(5):
    best_info = best_checkpoints[fold_num]
    source_path = best_info['checkpoint_path']
    dest_path = f'final_models/fold_{fold_num}_best.pth'
    
    # Copy checkpoint
    import shutil
    shutil.copy(source_path, dest_path)
    
    logger.info(f"✓ Copied {source_path} → {dest_path}")
```

**Step 7.4: Save Selection Rationale**
```python
# EXACT: Document why each checkpoint was selected

selection_rationale = {
    'formula': 'Gen_Score = val_acc - |train_acc - val_acc|',
    'description': 'Selects checkpoints that generalize well (penalizes overfitting)',
    'selected_folds': {}
}

for fold_num in range(5):
    best_info = best_checkpoints[fold_num]
    selection_rationale['selected_folds'][f'fold_{fold_num}'] = best_info

with open('final_models/checkpoint_selection_rationale.json', 'w') as f:
    json.dump(selection_rationale, f, indent=2)

logger.info("✓ Saved checkpoint_selection_rationale.json")

# Also save as CSV
selection_df = pd.DataFrame([
    {
        'fold': f,
        'epoch': best_checkpoints[f]['epoch'],
        'gen_score': best_checkpoints[f]['gen_score'],
        'train_acc': best_checkpoints[f]['train_acc'],
        'val_acc': best_checkpoints[f]['val_acc']
    }
    for f in range(5)
])
selection_df.to_csv('final_models/checkpoint_selection_summary.csv', index=False)

logger.info("✓ Saved checkpoint_selection_summary.csv")
```

**Step 7.5: Verify Selected Checkpoints**
```python
# EXACT: Verify all 5 best checkpoints exist and are valid

logger.info(f"\nVerifying 5 selected checkpoints...")

for fold_num in range(5):
    dest_path = f'final_models/fold_{fold_num}_best.pth'
    
    # Check file exists
    assert os.path.exists(dest_path), f"Missing {dest_path}"
    
    # Check file size
    file_size_mb = os.path.getsize(dest_path) / 1e6
    assert 90 < file_size_mb < 150, f"Checkpoint size unexpected: {file_size_mb:.0f} MB"
    
    # Load and verify checkpoint structure
    checkpoint = torch.load(dest_path, map_location='cpu')
    assert 'model_state_dict' in checkpoint, "Missing model_state_dict"
    assert 'train_acc' in checkpoint, "Missing train_acc"
    assert 'val_acc' in checkpoint, "Missing val_acc"
    
    logger.info(f"✓ fold_{fold_num}_best.pth verified")

logger.info(f"✓ All 5 checkpoints validated successfully")
```

**Outputs After Stage 7:**
- `final_models/fold_0_best.pth` through `fold_4_best.pth`: 5 best checkpoints
- `final_models/checkpoint_selection_rationale.json`: Why each was selected
- `final_models/checkpoint_selection_summary.csv`: Summary table

**Verification Checks (MUST Pass):**
```python
✓ 5 checkpoints in final_models/
✓ Each checkpoint ~100 MB
✓ Each checkpoint valid (loadable)
✓ Gen_Score values in 0.5-0.9 range
✓ Selected epochs distributed across 0-100 range
```

---

### Stage 8: Inference & Submission

**Reference:** `docs/10_INFERENCE_SUBMISSION/`

**Exact Imports Required:**
```python
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
```

**Exact Code:**

**Step 8.1: Load Test Data**
```python
# EXACT: Load preprocessed test images and create dataset/loader

class ImageDatasetTest(Dataset):
    def __init__(self, images, augmentation=None):
        self.images = images
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.augmentation is not None:
            augmented = self.augmentation(image=image)
            image = augmented['image']
        
        return image  # Only return image, no label

# Define validation augmentation (no random transforms for test)
val_augmentation = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=1.0
    ),
    ToTensorV2()
], p=1.0)

# Load test images
logger.info("Loading test images...")
test_dataset = ImageDatasetTest(X_test, augmentation=val_augmentation)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE_VAL,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

logger.info(f"✓ Test loader created: {len(test_loader)} batches")
```

**Step 8.2: Generate Predictions from All 5 Models**
```python
# EXACT: Load each best checkpoint and generate predictions on test set

test_predictions_all_folds = []

for fold_num in range(5):
    logger.info(f"\nGenerating predictions from fold {fold_num}...")
    
    # Load checkpoint
    checkpoint_path = f'final_models/fold_{fold_num}_best.pth'
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load model
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # Generate predictions
    fold_predictions = []
    
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            images = images.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            
            # Get probabilities (softmax already applied in outputs)
            probs = torch.softmax(outputs, dim=1)
            
            # Store probabilities (NOT argmax yet)
            fold_predictions.append(probs.detach().cpu().numpy())  # Shape: (batch_size, 10)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{len(test_loader)}")
    
    # Concatenate all batches for this fold
    fold_probs = np.vstack(fold_predictions)  # Shape: (3000, 10)
    test_predictions_all_folds.append(fold_probs)
    
    logger.info(f"✓ Fold {fold_num} predictions: shape={fold_probs.shape}")

logger.info(f"\n✓ Generated predictions from all 5 folds")
```

**Step 8.3: Ensemble Predictions (Average Probabilities)**
```python
# EXACT: Average probabilities across 5 folds

# Stack all fold predictions
all_folds_probs = np.stack(test_predictions_all_folds)  # Shape: (5, 3000, 10)

# Average across folds (axis 0)
ensemble_probs = all_folds_probs.mean(axis=0)  # Shape: (3000, 10)

logger.info(f"Ensemble probabilities computed:")
logger.info(f"  Shape: {ensemble_probs.shape}")
logger.info(f"  Mean per sample sums to 1.0? {np.allclose(ensemble_probs.sum(axis=1), 1.0)}")

# Get final predictions (argmax of ensemble probabilities)
final_predictions = ensemble_probs.argmax(axis=1)  # Shape: (3000,)

logger.info(f"Final predictions computed:")
logger.info(f"  Shape: {final_predictions.shape}")
logger.info(f"  Class distribution: {np.bincount(final_predictions)}")
```

**Step 8.4: Map Predictions to Class Names**
```python
# EXACT: Convert class indices back to generator names

# Create mapping from index to class name
idx_to_class = {idx: cls for idx, cls in enumerate(sorted(train_df['source'].unique()))}

# Convert predictions to class names
final_predictions_names = np.array([idx_to_class[idx] for idx in final_predictions])

logger.info(f"Predictions mapped to class names:")
for class_name, count in pd.Series(final_predictions_names).value_counts().items():
    logger.info(f"  {class_name}: {count}")
```

**Step 8.5: Create Submission CSV**
```python
# EXACT: Format submission CSV exactly as Kaggle requires

os.makedirs('submission', exist_ok=True)

# Get test image IDs (from test.csv)
test_image_ids = test_df['image_id'].values  # Shape: (3000,)

# Create submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_image_ids,
    'TARGET': final_predictions_names
})

# Save to CSV (EXACT format required by Kaggle)
submission_path = 'submission/submission.csv'
submission_df.to_csv(submission_path, index=False)

logger.info(f"\n✓ Submission CSV created: {submission_path}")
logger.info(f"  Shape: {submission_df.shape}")
logger.info(f"  Columns: {list(submission_df.columns)}")
logger.info(f"\n  First 5 rows:")
logger.info(submission_df.head())
```

**Step 8.6: Verify Submission Format**
```python
# EXACT: Critical checks before uploading

logger.info(f"\nVerifying submission format...")

# Check 1: Correct number of rows
assert len(submission_df) == 3000, f"Wrong number of rows: {len(submission_df)}"
logger.info(f"✓ 3000 rows")

# Check 2: Correct columns
assert list(submission_df.columns) == ['ID', 'TARGET'], f"Wrong columns: {list(submission_df.columns)}"
logger.info(f"✓ Columns: ID, TARGET")

# Check 3: No missing values
assert not submission_df['ID'].isna().any(), "Missing values in ID column"
assert not submission_df['TARGET'].isna().any(), "Missing values in TARGET column"
logger.info(f"✓ No missing values")

# Check 4: All test IDs present
assert len(submission_df['ID'].unique()) == 3000, "Duplicate IDs in submission"
assert set(submission_df['ID'].values) == set(test_df['image_id'].values), "ID mismatch"
logger.info(f"✓ All test IDs present, no duplicates")

# Check 5: All predictions are valid class names
valid_classes = set(train_df['source'].unique())
invalid_preds = ~submission_df['TARGET'].isin(valid_classes)
assert not invalid_preds.any(), f"Invalid class predictions found"
logger.info(f"✓ All predictions are valid class names")

# Check 6: File exists and is readable
assert os.path.exists(submission_path), f"Submission file not found"
file_size_mb = os.path.getsize(submission_path) / 1e3
logger.info(f"✓ File saved: {file_size_mb:.1f} KB")

logger.info(f"\n{'='*60}")
logger.info(f"✓✓✓ SUBMISSION VERIFIED AND READY FOR KAGGLE ✓✓✓")
logger.info(f"{'='*60}")
logger.info(f"\nNext step: Upload {submission_path} to Kaggle competition")
```

**Step 8.7: Save Metadata and Confidence Scores**
```python
# EXACT: Save metadata and prediction confidence for analysis

# Save metadata
metadata = {
    'timestamp': str(pd.Timestamp.now()),
    'num_test_samples': len(submission_df),
    'num_classes': 10,
    'ensemble_strategy': '5-fold average',
    'model_architecture': MODEL_NAME,
    'model_source': 'pytorch-image-models (timm)',
    'fold_checkpoints': [f'fold_{f}_best.pth' for f in range(5)]
}

with open('submission/submission_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

logger.info(f"✓ Saved submission_metadata.json")

# Save confidence scores
max_probs = ensemble_probs.max(axis=1)
min_probs = ensemble_probs.min(axis=1)
std_probs = ensemble_probs.std(axis=1)

confidence_df = pd.DataFrame({
    'ID': test_image_ids,
    'TARGET': final_predictions_names,
    'max_probability': max_probs,
    'min_probability': min_probs,
    'std_probability': std_probs,
    'confidence_range': max_probs - min_probs
})

confidence_df.to_csv('submission/prediction_confidence.csv', index=False)

logger.info(f"✓ Saved prediction_confidence.csv")
logger.info(f"\nConfidence Statistics:")
logger.info(f"  Max prob - mean: {max_probs.mean():.4f}, std: {max_probs.std():.4f}")
logger.info(f"  Confidence range - mean: {(max_probs - min_probs).mean():.4f}")
```

**Outputs After Stage 8:**
- `submission/submission.csv`: Final submission (ready for Kaggle!)
- `submission/submission_metadata.json`: Metadata about submission
- `submission/prediction_confidence.csv`: Confidence scores per prediction

**Exact Submission CSV Format:**
```
ID,TARGET
test_0000.png,StableDiffusion3
test_0001.png,SDXL-Turbo
test_0002.png,AuraFlow
...
test_2999.png,Pixart-sigma
```

**Verification Checks (MUST ALL Pass):**
```python
✓ submission.csv has exactly 3000 rows
✓ Columns: ['ID', 'TARGET'] (exact order)
✓ All 3000 test_image_ids present in ID column
✓ No duplicate IDs
✓ No missing values
✓ All TARGET values are valid class names
✓ File readable and correct format
```

**Expected Accuracy:**
- First submission: 65-70%
- After optimization: 72-80%

---

## 📁 Directory Structure Created

```
kaggle_notebooks/pipe-1/
├── README.md                          [This file]
├── notebook.ipynb                     [To be created]
│
└── (After running, creates:)
    ├── checkpoints/
    │   ├── fold_0/
    │   │   ├── epoch_000.pth
    │   │   ├── epoch_001.pth
    │   │   └── ... (100 epochs)
    │   ├── fold_1/
    │   └── ... (5 folds total)
    │
    ├── final_models/
    │   ├── fold_0_best.pth
    │   ├── fold_1_best.pth
    │   ├── fold_2_best.pth
    │   ├── fold_3_best.pth
    │   └── fold_4_best.pth
    │
    ├── logs/
    │   ├── training_log.csv
    │   ├── validation_metrics.csv
    │   └── fold_0_metrics.json through fold_4_metrics.json
    │
    ├── eda_outputs/
    │   ├── eda_report.html
    │   ├── eda_insights.json
    │   ├── plots/
    │   │   ├── class_distribution.png
    │   │   ├── dimension_distribution.png
    │   │   └── generator_samples.png
    │
    ├── preprocessed/
    │   ├── X_train_preprocessed.npy
    │   ├── X_test_preprocessed.npy
    │   └── normalization_metadata.json
    │
    ├── fold_metadata.json
    ├── data_validation_report.txt
    │
    └── submission/
        ├── submission.csv                 [FINAL: Upload to Kaggle!]
        ├── submission_metadata.json
        └── prediction_confidence.csv
```

---

## ⏱️ Execution Timeline

| Stage | Time | Notes |
|-------|------|-------|
| Stage 1: Data Loading | 5 min | Quick CSV read + file verify |
| Stage 2: EDA | 10 min | Analyze distributions, create plots |
| Stage 3: Preprocessing | 10 min | Load, resize, normalize all 10K images |
| Stage 4: CV Splits | 2 min | Create fold indices |
| Stage 5: Training | 3-4 hrs | Loop 5 folds × 100 epochs (GPU-intensive!) |
| Stage 6: Validation | 10 min | Evaluate all checkpoints |
| Stage 7: Selection | 5 min | Apply Gen_Score formula |
| Stage 8: Inference | 10 min | Ensemble predictions, create submission |
| **Total** | **~3.5-4.5 hrs** | Plus compute time for training |

---

## 🔑 Key Design Decisions

### 1. Why Save ALL Epochs?
- Flexibility in checkpoint selection
- Allows post-hoc analysis of training dynamics
- Different selection criteria can be applied later
- Without this, you'd be stuck with just the "best" epoch

### 2. Why Use Generalization Score?
- Prevents selecting overfitted models
- A model with train=0.99, val=0.80 is memorizing, not generalizing
- Better predicted test performance than max(val_acc)
- Correlates with leaderboard performance

### 3. Why 5-Fold Ensemble?
- Each fold sees different validation set
- 5 diverse views of training data
- Better than single model: +1-2% accuracy typical
- Natural byproduct of CV training

### 4. Why Stratified K-Fold?
- Maintains class balance per fold
- Prevents one fold from being "harder" (e.g., 70% class 0, 30% others)
- Ensures stable CV estimates
- Essential for imbalanced scenarios (though we have balanced data)

### 5. Why ImageNet Normalization?
- Pretrained models expect this specific normalization
- Standard for transfer learning
- Makes features more interpretable
- Required for convergence

### 6. Why Apply Augmentation Only in Training?
- Validation/test should be on original images (unbiased evaluation)
- Augmentation is a regularization technique
- Helps model become robust to variations

---

## ✅ Success Criteria

**After running pipe-1, you should have:**

- ✅ `submission.csv` with 3000 rows
- ✅ All validation checks passed
- ✅ Training logs showing learning progress
- ✅ 5 best checkpoints (one per fold)
- ✅ Expected accuracy: **65-70%** (first attempt)
- ✅ No errors or missing predictions
- ✅ Readable EDA visualizations

**If you don't have these, something went wrong!**

---

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Out of memory during training | Reduce batch size (32 → 16), use smaller model (B2 instead of B4) |
| Images not found | Verify data path, check file permissions, ensure `/Data/` exists |
| Very low accuracy (<15%) | Check data loading (labels might be swapped), verify preprocessing |
| Training loss increasing | Learning rate too high, check for data issues |
| NaN in loss | Reduce learning rate, check for corrupt images |
| Submission format error | Verify ID column matches exactly, no extra/missing rows |

---

## 📊 Expected Performance

| Metric | Expected | Notes |
|--------|----------|-------|
| **First submission accuracy** | 65-70% | Baseline EfficientNet-B4 |
| **F1-Macro** | 0.65-0.70 | Similar to accuracy (balanced classes) |
| **Variance across folds** | ±2-3% | Should be relatively stable |
| **Generalization gap** | 5-8% | train_acc - val_acc (acceptable) |

---

## 🎓 References

This pipeline integrates these documentation sources:

- [`docs/00_PIPELINE_OVERVIEW.md`](../../docs/00_PIPELINE_OVERVIEW.md) - System architecture
- [`docs/01_DATA_LOADING_VALIDATION/`](../../docs/01_DATA_LOADING_VALIDATION/) - Stage 1
- [`docs/02_EDA/`](../../docs/02_EDA/) - Stage 2
- [`docs/03_PREPROCESSING/`](../../docs/03_PREPROCESSING/) - Stage 3
- [`docs/05_TRAIN_VAL_SPLIT/`](../../docs/05_TRAIN_VAL_SPLIT/) - Stage 4
- [`docs/07_MODEL_TRAINING/`](../../docs/07_MODEL_TRAINING/) - Stage 5
- [`docs/08_VALIDATION/`](../../docs/08_VALIDATION/) - Stage 6
- [`docs/09_CHECKPOINT_SELECTION/`](../../docs/09_CHECKPOINT_SELECTION/) - Stage 7
- [`docs/10_INFERENCE_SUBMISSION/`](../../docs/10_INFERENCE_SUBMISSION/) - Stage 8

---

## 🚀 Ready?

Once you review this README and give the ✅, I'll create the corresponding **notebook.ipynb** with:
- Complete, working Python code
- Proper error handling
- Logging at each stage
- Progress indicators
- Ready to run cell by cell!

---

**Questions about this pipeline?** Ask before we create the notebook! 🤔
