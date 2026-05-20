# DLMMDD Synthetic Image Attribution Challenge - Pipeline 1
## Architecture & Design Document

---

## Executive Summary

**Pipeline 1** is a production-ready, end-to-end machine learning system for the DLMMDD Synthetic Image Attribution Challenge. It provides a modular, configurable framework for training multi-class image classification models with built-in:

- ✅ Data loading and validation
- ✅ Image preprocessing and augmentation
- ✅ Stratified K-fold cross-validation
- ✅ Model training with checkpointing
- ✅ Metric tracking and validation
- ✅ Checkpoint selection with generalization score
- ✅ Inference and submission generation
- ✅ Optional ensemble support

**Key Metrics:**
- **10 Classes:** AuraFlow, Freepik, Lumina, Photon, Pixart(sigma), Playground v2.5, StableDiffusion3, StableDiffusion3.5, StableDiffusionXL-Turbo, Tencent Hunyuan
- **7,000 Training Images** + **3,000 Test Images**
- **Accuracy Metric** (Leaderboard Split: 50% public, 50% private)

---

## Architecture Overview

### Service-Oriented Pipeline (SOA)

The pipeline is organized as a series of independent **stages**, each with clear inputs and outputs:

```
Stage 0           Stage 1              Stage 2            Stage 3
Data Loading  →  Preprocessing   →   Training      →   Validation
   CSV            Images          Models              Metrics
   Images         Arrays          Checkpoints        Selection
   Metadata       Folds           Logs               Best Model

                                      ↓
                                  
Stage 4           Stage 5
Inference      →  Ensemble (Optional)
Submission       Robust Predictions
```

Each stage is:
- **Independent:** Can run in isolation (with proper inputs)
- **Trackable:** Produces logs and metrics
- **Reproducible:** Uses fixed seeds and saved configs
- **Extensible:** Easy to add new stages or modify existing ones

---

## Core Components

### 1. Configuration System (`config_loader.py`)

**Purpose:** Centralized management of all hyperparameters and settings

**Features:**
- TOML-based configuration with sensible defaults
- Environment variable overrides (e.g., `PIPELINE_BATCH_SIZE=64`)
- Type-safe dataclass-based config objects
- Validation of required fields

**Example Usage:**
```python
from config_loader import load_config

config = load_config()  # Loads config.toml
print(config.training.batch_size)  # 32
print(config.model.architecture)   # efficientnet_b4
```

### 2. Logging System (`logger_setup.py`)

**Purpose:** Structured, trackable logging throughout pipeline

**Features:**
- Console + file output
- Rotating file handlers (10MB max, 5 backups)
- Consistent format: timestamp, level, module, line number
- Per-module loggers for fine-grained control

**Example Usage:**
```python
from logger_setup import setup_logger

logger = setup_logger(__name__)
logger.info("Processing started")
logger.warning("Potential issue detected")
logger.error("Critical failure")
```

### 3. Data Loading (`data_loader.py`)

**Purpose:** Load, validate, and enrich dataset metadata

**Components:**
- **CSV Loader:** Reads training.csv and test.csv with explicit dtypes
- **Path Resolver:** Converts relative paths to absolute paths
- **Image Inspector:** Extracts metadata without loading full images
- **Distribution Validator:** Checks class balance (1,000 per class)
- **Source Mapper:** Links class IDs to generator names
- **Statistics Aggregator:** Computes dataset-wide statistics

**Key Decision:** Store both relative and absolute paths for reproducibility

### 4. Image Preprocessing (`preprocessor.py`)

**Purpose:** Standardize image format and normalize pixel values

**ImagePreprocessor:**
- Resize to 224×224 (configurable)
- Convert to RGB (handles various formats)
- Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Convert to PyTorch tensors
- Output shape: (N, 3, 224, 224)

**ImageAugmenter:**
- Applied during training only
- Random augmentations: flip, rotation, brightness, contrast, Gaussian blur
- Configurable probability for each augmentation
- Prevents overfitting by introducing training-time variations

### 5. Train/Validation Splitting (`train_val_split.py`)

**Purpose:** Create stratified K-fold splits for robust evaluation

**Strategy:**
- Stratified K-Fold (preserves class distribution)
- Default: 5 folds
- Random seed for reproducibility
- Prevents data leakage

**Outputs:**
- `fold_metadata.json`: For each fold: train indices, val indices, class counts

### 6. Metrics Computation (`metrics.py`)

**Purpose:** Track and compute evaluation metrics

**MetricsComputer:**
- **Accuracy** (primary metric)
- **F1-Macro** (robust to class imbalance)
- **Precision/Recall** (per-class metrics)
- **Confusion Matrix** (error analysis)
- **Generalization Score** (for checkpoint selection)

**Generalization Score Formula:**
```
Gen_Score = val_accuracy - |train_accuracy - val_accuracy|
```
Penalizes overfitting while rewarding validation performance.

**MetricsTracker:**
- Logs metrics per epoch
- Saves history to JSON
- Tracks per-fold history

### 7. Checkpoint Management (`checkpoint_selector.py`)

**Purpose:** Save and select best models

**CheckpointManager:**
- Saves model state + optimizer state + metrics + config per epoch
- Saves all epochs (configurable)
- Organizes checkpoints by fold

**CheckpointSelector:**
- Evaluates all epochs on validation set
- Selects best by generalization score (or val accuracy)
- Handles per-fold selection
- Aggregates results across folds

---

## Execution Flow

### Stage 00: Data Loading & EDA (Notebook 00)

**Input:** `Data/training.csv`, `Data/test.csv`, image files

**Process:**
1. Load metadata CSVs
2. Resolve image file paths
3. Extract metadata for all images:
   - Dimensions (width, height)
   - Format (PNG, JPEG, WebP)
   - Color mode (RGB, RGBA, Grayscale)
   - File size
   - Readability check
4. Validate training data:
   - Check class balance (1,000 per class)
   - Verify all 10 classes present
   - Check for missing/corrupted files
5. Compute statistics:
   - Class distribution
   - Image dimension statistics
   - Format distribution
6. Generate EDA visualizations:
   - Class distribution bar charts
   - Dimension histograms
   - File size distributions

**Output:** 
- `train_metadata.parquet` (7000 rows × 10 cols)
- `test_metadata.parquet` (3000 rows × 8 cols)
- `data_stats.json` (summary statistics)
- Visualization PNGs

**Duration:** ~5 minutes (metadata extraction + visualization)

---

### Stage 01: Preprocessing & Augmentation (Notebook 01)

**Input:** `train_metadata.parquet`, `test_metadata.parquet`

**Process:**
1. Initialize ImagePreprocessor
   - Target size: 224×224
   - Normalization: ImageNet
2. Preprocess training images:
   - Batch process 7,000 images
   - Output shape: (7000, 3, 224, 224)
3. Preprocess test images:
   - Batch process 3,000 images
   - Output shape: (3000, 3, 224, 224)
4. Create stratified K-fold splits:
   - 5 folds with ~1,400 train + ~600 val per fold
   - Preserve class distribution in each fold
5. Verify stratification:
   - Ensure no overlap between train/val
   - Confirm class counts match expected

**Output:**
- `X_train.npy` (7000, 3, 224, 224) - preprocessed training images
- `X_test.npy` (3000, 3, 224, 224) - preprocessed test images
- `y_train.npy` (7000,) - training labels
- `fold_metadata.json` - fold assignments

**Duration:** ~15-30 minutes (image preprocessing is bottleneck)

---

### Stage 02: Model Training (Notebook 02)

**Input:** `X_train.npy`, `X_test.npy`, `y_train.npy`, `fold_metadata.json`, `config.toml`

**Process:**
For each fold (0-4):
1. Initialize model (default: EfficientNet-B4)
   - Load pretrained ImageNet weights
   - Add classification head for 10 classes
   - Count parameters (typically ~17M)
2. Create data loaders:
   - Training: batch_size=32, shuffle=True, augmentation enabled
   - Validation: batch_size=32, shuffle=False, no augmentation
3. Training loop (per epoch):
   - Forward pass on training batch
   - Compute cross-entropy loss
   - Backward pass + optimizer step
   - Compute training metrics (accuracy, F1)
   - Forward pass on validation set
   - Compute validation metrics
   - Save checkpoint if epoch % save_frequency == 0
   - Log all metrics
   - Check early stopping (patience=10 epochs)
4. Save final checkpoint with metadata

**Training Configuration:**
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
- Scheduler: Cosine annealing (T_max=95)
- Loss: CrossEntropyLoss
- Epochs: 100
- Early stopping: 10 epochs patience

**Output:**
- `checkpoints/fold_0/epoch_*.pth` - all saved checkpoints
- `checkpoints/fold_1/epoch_*.pth`
- ... (fold 2-4)
- `logs/training_fold_*.log` - training logs

**Expected Duration:**
- Per fold: ~30-60 minutes (GPU dependent)
- Total (5 folds): ~2.5-5 hours

**Expected Metrics:**
- Training accuracy: ~95%+ (may overfit)
- Validation accuracy: ~80-90%

---

### Stage 03: Validation & Checkpoint Selection (Notebook 03)

**Input:** All checkpoints, training logs, fold metadata

**Process:**
1. Load metrics history from logs
2. For each fold:
   - Evaluate each checkpoint on validation set
   - Compute generalization score: `val_acc - |train_acc - val_acc|`
   - Select epoch with highest generalization score
3. Aggregate metrics across all folds:
   - Mean accuracy across folds
   - Standard deviation
   - Per-class accuracy
4. Generate confusion matrices
5. Save selection results

**Output:**
- `checkpoint_selections.json` - selected epochs per fold
- `validation_metrics.csv` - aggregated metrics
- Confusion matrices (PNG)

**Duration:** ~5 minutes

**Decision Logic:**
```python
def select_best_checkpoint(metrics_history):
    best_epoch = None
    best_score = -inf
    
    for epoch, metrics in enumerate(metrics_history):
        train_acc = metrics['train']['accuracy']
        val_acc = metrics['val']['accuracy']
        
        # Generalization score penalizes overfitting
        gen_score = val_acc - abs(train_acc - val_acc)
        
        if gen_score > best_score:
            best_score = gen_score
            best_epoch = epoch
    
    return best_epoch, best_score
```

---

### Stage 04: Inference & Submission (Notebook 04)

**Input:** Best checkpoints, `X_test.npy`, test IDs

**Process:**
1. Load best checkpoint for each fold
2. Move to evaluation mode (disable dropout, batch norm updates)
3. For each test image:
   - Pass through all 5 models (one per fold)
   - Collect predictions (shape: 5×10 probabilities per image)
4. Ensemble predictions:
   - Average probabilities across 5 folds
   - Select argmax class
5. Optional: Test-Time Augmentation (TTA)
   - Apply 5 augmented versions per image
   - Average predictions across augmentations
   - Further improves robustness
6. Format submission CSV:
   - Column 1: Image ID
   - Column 2: Predicted class (0-9)
7. Verify format and save

**Output:**
- `submissions/submission_final.csv` - final predictions

**Expected Accuracy:**
- Without TTA: ~85-92%
- With TTA: ~86-93%
- With ensemble: ~88-94%

**Duration:** ~10 minutes

---

### Stage 05: Ensemble (Optional, Notebook 05)

**Input:** Multiple trained models/checkpoints

**Process:**
1. Train different model architectures:
   - EfficientNet-B4 (already trained)
   - ResNet50 (initialize and train)
   - ViT Base (initialize and train)
2. Use checkpoint selections from Stage 03
3. Ensemble predictions:
   - Weighted average: w_0×pred_0 + w_1×pred_1 + w_2×pred_2
   - Or voting: majority class
4. Save ensemble submission

**Output:**
- `submissions/submission_ensemble.csv` - ensemble predictions

**Expected Improvement:** +1-2% accuracy

---

## Design Patterns Used

### 1. Configuration as Code (CaC)
- All parameters in `config.toml`
- Environment variable overrides
- Type-safe dataclass validation
- No hard-coded magic numbers

### 2. Dependency Injection
- Components receive configuration, not global state
- Easy to test and reuse
- Clear dependencies between modules

### 3. Structured Logging
- Logger per module
- Consistent format
- Both console and file output
- Historical record of all decisions

### 4. Separation of Concerns
- Data loading ≠ preprocessing ≠ training
- Each stage produces clear outputs
- Easy to debug and modify individual stages

### 5. Reproducibility
- Fixed random seeds
- Saved configuration per run
- Checkpoints with full metadata
- Logged hyperparameters

---

## Performance Characteristics

### Memory Usage
```
Training Set Preprocessing:
  7,000 images × (3×224×224 × 4 bytes) = 7,056 MB ≈ 7 GB

Test Set Preprocessing:
  3,000 images × (3×224×224 × 4 bytes) = 3,024 MB ≈ 3 GB

Total: ~10 GB (fits in typical GPU VRAM with batch processing)
```

### Compute Time
```
Stage 00: ~5 min      (metadata extraction + visualization)
Stage 01: ~20 min     (image preprocessing + fold creation)
Stage 02: ~150 min    (5 folds × ~30 min per fold)
Stage 03: ~5 min      (checkpoint evaluation)
Stage 04: ~10 min     (inference + submission)
─────────────────────
Total:   ~190 minutes (≈3.2 hours)

With ensemble (Stage 05): +150 minutes (total ≈5 hours)
```

---

## Extension Points

### 1. Model Architecture
Edit `notebooks/02_model_training.ipynb`:
```python
if config.model.architecture == "resnet50":
    model = torchvision.models.resnet50(pretrained=True)
elif config.model.architecture == "vit_base":
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
```

### 2. Custom Augmentations
Edit `src/preprocessor.py`:
```python
self.transforms = transforms.Compose([
    ...,
    RandomPerspective(distortion_scale=0.2),  # Custom augmentation
    ...
])
```

### 3. Loss Functions
Add to config:
```toml
[training]
loss_function = "focal_loss"  # Instead of cross_entropy
```

### 4. Checkpoint Selection Metrics
Edit `config.toml`:
```toml
[checkpoint]
selection_metric = "val_f1_macro"  # Instead of generalization_score
```

---

## Error Handling

### Data Validation
- Check file existence before processing
- Verify CSV dtypes
- Log warnings for missing/corrupted files
- Graceful degradation (don't crash on individual errors)

### Training Robustness
- Early stopping to prevent excessive overfitting
- Learning rate scheduling for convergence
- Gradient clipping (if needed)
- Save checkpoints even if validation fails

### Checkpoint Loading
- Verify checkpoint integrity
- Check model state dict compatibility
- Handle missing files gracefully
- Log detailed error messages

---

## Testing & Validation

### Data Validation (Stage 00)
```python
assert len(train_df) == 7000, "Expected 7000 training samples"
assert train_df['y'].value_counts().min() == 1000, "Imbalanced classes"
assert train_df['is_readable'].all(), "Missing/corrupted files"
```

### Preprocessing Validation (Stage 01)
```python
assert X_train.shape == (7000, 3, 224, 224), "Incorrect shape"
assert X_train.dtype == np.float32, "Incorrect dtype"
assert len(train_idx) + len(val_idx) == 7000, "Split mismatch"
```

### Training Validation (Stage 02)
```python
assert loss.isfinite(), "Diverged loss"
assert train_acc > 0.1, "Below random baseline"
assert val_acc > 0.1, "Below random baseline"
```

---

## Future Improvements

1. **Hyperparameter Optimization**
   - Integrate Optuna for automated tuning
   - Search over learning rate, batch size, augmentation strength

2. **Advanced Augmentation**
   - Mixup / CutMix
   - AutoAugment
   - RandAugment

3. **Model Architectures**
   - Vision Transformer (ViT)
   - DenseNet
   - EfficientNetV2

4. **Ensemble Strategies**
   - Stacking
   - Blending
   - Weighted ensemble with learned weights

5. **Monitoring & Logging**
   - TensorBoard integration
   - Weights & Biases logging
   - Gradient visualization

---

## Competition Context

### Challenge Details
- **Participants:** ~50-100 (workshop setting)
- **Duration:** ~1 month
- **Dataset:** 10K images, 10 classes, balanced
- **Metric:** Accuracy (public/private split: 50/50)

### Ranking Strategy
1. **Baseline:** Random guessing (10%)
2. **Tier 1:** Simple CNN (60-70%)
3. **Tier 2:** Pretrained ResNet50 (75-85%)
4. **Tier 3:** Fine-tuned EfficientNet (85-92%)
5. **Tier 4:** Ensemble + TTA (92-96%)
6. **Top:** Advanced techniques (96%+)

This pipeline aims for **Tier 3-4** performance with standard configuration.

---

## References

1. **EfficientNet:** https://arxiv.org/abs/1905.11946
2. **ImageNet Normalization:** https://pytorch.org/vision/stable/models.html
3. **Stratified K-Fold:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
4. **PyTorch Training Loop:** https://pytorch.org/tutorials/beginner/nn_tutorial.html

---

**Document Version:** 1.0.0  
**Last Updated:** 2026-05-20  
**Status:** Complete & Production Ready ✅
