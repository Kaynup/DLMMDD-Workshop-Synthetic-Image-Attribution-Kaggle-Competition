# PIPE-3: Custom Vision Transformer Variants for Cotton Leaf Disease Classification

## Overview

**Pipe-3** is a specialized framework for **exploring custom Vision Transformer (ViT) architectures** on the **SAR-CLD-2024 Cotton Leaf Disease Detection** task. Unlike Pipe-1 (ensemble of CNNs) and Pipe-2 (comparative CNNs), this pipeline emphasizes **custom ViT design** with **interpretability through attention visualization**.

### Key Distinction from Other Pipes
- **Pipe-1**: Pre-trained single architecture, 5-fold CV
- **Pipe-2**: Pre-trained multi-model comparison, single split
- **Pipe-3**: Custom ViT from scratch, 16 hyperparameter variants, attention analysis

---

## Competition & Dataset

| Aspect | Details |
|--------|---------|
| **Dataset** | SAR-CLD-2024 Cotton Leaf Disease Detection |
| **Source** | Original + Augmented subsets (merged) |
| **Task Type** | Multi-class image classification |
| **Number of Classes** | Multiple cotton leaf disease categories (~5-10) |
| **Total Images** | Original + Augmented combined (varies) |
| **Data Organization** | Separate directories for Original and Augmented subsets |
| **Train/Val Split** | Stratified 90/10 |
| **Image Format** | PNG/JPG, 224×224 standard |
| **Special Feature** | Augmented subset provided (doubles training data) |

---

## Technical Stack

### Framework & Design
- **Deep Learning Framework**: PyTorch (custom implementation)
- **Precision**: Standard float32 (interpretability priority)
- **Architecture**: Fully custom Vision Transformer (not timm pre-built)
- **GPU**: CUDA/CuDNN optimized

### Custom Vision Transformer Architecture

#### Core Components

1. **Patch Embedding**:
   ```
   Input: 224×224 RGB image
   ↓
   Conv2d(kernel_size=P, stride=P) → (C, H', W')
   ↓
   Flatten to sequence: (num_patches, dim)
   ↓
   Add learnable CLS token at position 0
   ↓
   Add position embeddings (learnable)
   ↓
   Output: (1 + num_patches, dim)
   ```

2. **Transformer Encoder** (stacked layers):
   ```
   For each layer:
   ├─ LayerNorm
   ├─ MultiHeadAttention (self-attention)
   ├─ Residual connection
   ├─ LayerNorm
   ├─ MLP(dim → 4×dim → dim, GELU)
   └─ Residual connection
   ```

3. **Classification Head**:
   ```
   CLS token output (after all layers)
   ↓
   Linear(dim → 256, GELU)
   ↓
   Dropout(p)
   ↓
   Linear(256 → num_classes)
   ↓
   Softmax
   ```

#### 16 ViT Variants (Hyperparameter Grid)

| Variant | Patch Size | Depth | Embedding Dim | Dropout |
|---------|------------|-------|---------------|---------|
| ViT-P14-D6-256-D15 | 14×14 | 6 | 256 | 0.15 |
| ViT-P14-D6-256-D25 | 14×14 | 6 | 256 | 0.25 |
| ViT-P14-D6-384-D15 | 14×14 | 6 | 384 | 0.15 |
| ViT-P14-D6-384-D25 | 14×14 | 6 | 384 | 0.25 |
| ViT-P14-D8-256-D15 | 14×14 | 8 | 256 | 0.15 |
| ViT-P14-D8-256-D25 | 14×14 | 8 | 256 | 0.25 |
| ViT-P14-D8-384-D15 | 14×14 | 8 | 384 | 0.15 |
| ViT-P14-D8-384-D25 | 14×14 | 8 | 384 | 0.25 |
| ViT-P16-D8-192-D15 | 16×16 | 8 | 192 | 0.15 |
| ViT-P16-D8-192-D25 | 16×16 | 8 | 192 | 0.25 |
| ViT-P16-D10-192-D15 | 16×16 | 10 | 192 | 0.15 |
| ViT-P16-D10-192-D25 | 16×16 | 10 | 192 | 0.25 |
| ViT-P16-D10-384-D15 | 16×16 | 10 | 384 | 0.15 |
| ViT-P16-D10-384-D25 | 16×16 | 10 | 384 | 0.25 |
| ViT-P16-D12-384-D15 | 16×16 | 12 | 384 | 0.15 |
| ViT-P16-D12-384-D25 | 16×16 | 12 | 384 | 0.25 |

**Parameter Space**:
- Patch sizes: 14×14, 16×16
- Depths: 6, 8, 10, 12 transformer layers
- Embedding dims: 192, 256, 384
- Dropout rates: 0.15, 0.25

---

## Data Pipeline

### Stage 1: Dataset Structure & Merging

**Input Structure**:
```
/kaggle/input/cotton-leaf-disease-dataset/
├── Original/
│   ├── disease_class_1/
│   │   ├── image1.png
│   │   └── ...
│   ├── disease_class_2/
│   └── ...
│
└── Augmented/
    ├── disease_class_1/
    │   ├── aug_image1.png
    │   └── ...
    └── ...
```

**Merging Strategy**:
- Combine Original and Augmented subsets into single training pool
- Increases training data size (~2× coverage)
- Metadata tracked per source (Original vs Augmented)

### Stage 2: Comprehensive Metadata Extraction

**Per-Image Metadata**:
- Filepath, label, split (train/val)
- Dimensions: width, height, aspect ratio, area (px²)
- File mode: RGB/RGBA/Grayscale
- File size (bytes)
- MD5 hash (for duplicate detection)

**Output**: CSV with all metadata for reproducibility

### Stage 3: Duplicate Detection (Unique Feature!)

**Duplicate Categories**:

1. **Global Duplicates**:
   - Exact duplicates across entire dataset
   - Using MD5 hash comparison

2. **Within-Split Duplicates**:
   - Duplicates within Original subset only
   - Duplicates within Augmented subset only
   - Suggests data quality issues

3. **Cross-Duplicates**:
   - Overlaps between Original and Augmented
   - Same image in both subsets (potential data leakage)

**Output**: Grouped lists of matching file hashes with counts

### Stage 4: Stratified Train/Val Split (90/10)

**StratifiedShuffleSplit**:
- Maintains class distribution in both sets
- 90% training, 10% validation
- Seed: 123 (reproducibility)

**Class Imbalance Handling**:
1. **Oversampling** (enabled):
   - Find max class count in training set
   - Resample minority classes to match max
   - Random resampling with replacement
   - Prevents information loss while balancing

2. **Class Weights** (in loss):
   - Balanced cross-entropy weights
   - Weight = total_samples / (num_classes × class_samples)
   - Higher weight for minority classes

### Stage 5: Image Preprocessing & Augmentation

**Train Augmentation** (Albumentations):
```python
train_transforms = [
    Resize(224, 224),           # Standardize size
    Rotate(limit=15, p=1.0),    # ±15° rotation
    HorizontalFlip(p=0.5),      # 50% horizontal flip
    VerticalFlip(p=0.5),        # 50% vertical flip
    ShiftScaleRotate(
        shift_limit=0.08,       # ±8% shift
        scale_limit=0.1,        # ±10% scale
        rotate_limit=10,        # ±10° rotation
        p=0.5
    ),
    Normalize(mean=0.5, std=0.5),  # [0,1] → [-1,1]
    ToTensorV2()
]
```

**Validation Augmentation**:
```python
val_transforms = [
    Resize(224, 224),
    Normalize(mean=0.5, std=0.5),
    ToTensorV2()
]
```

**Key Choices**:
- Normalization: mean=0.5, std=0.5 (centers to [-1, 1] from [0, 1])
- Augmentation applied only to training set
- Moderate augmentation (preserves disease features)

### Stage 6: DataLoader Setup

- **Batch size**: 64
- **Workers**: 2 parallel data loading processes
- **Pin memory**: True (faster host-to-device transfer)
- **Shuffle**: Train=True, Val=False

---

## Training Pipeline

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Image Size** | 224×224 | Standard ViT input |
| **Batch Size** | 64 | Balanced for GPU |
| **Epochs** | 50 | Per variant |
| **Optimizer** | AdamW | Deep learning standard |
| **Learning Rate** | 3e-5 | Small (vision transformers are sensitive) |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Loss** | Weighted CrossEntropyLoss | Balanced for class imbalance |
| **Warmup** | max(3, EPOCHS/10) | Linear LR warmup phase |

### Learning Rate Schedule

**Two-Phase Strategy**:

1. **Warmup Phase** (~5 epochs):
   - Linear increase from 0 → learning_rate
   - Stabilizes training initialization
   - Formula: `lr = base_lr × (epoch / warmup_epochs)`

2. **Decay Phase** (~45 epochs):
   - Cosine annealing from learning_rate → eta_min (1e-7)
   - Smooth convergence
   - Formula: `lr = eta_min + 0.5 × (base_lr - eta_min) × (1 + cos(π × epoch / total_epochs))`

### Training Loop Structure

```
For each of 16 ViT variants:
  1. Initialize custom ViT architecture
  2. Setup optimizer (AdamW) + scheduler (warmup + cosine)
  3. Train for 50 epochs:
     a. Forward pass on train batch
     b. Compute weighted loss
     c. Backward + optimizer step
     d. Log metrics (loss, acc, F1, etc.)
     e. Validate on val set
     f. Compute GnP score
     g. Save checkpoint if best GnP
     h. Update learning rate
  4. Save final best checkpoint and logs
```

### Checkpointing Strategy (GnP-based)

**GnP Metric (Generalization & Performance)**:
```
GnP = val_f1 - |train_f1 - val_f1|
```

- **Rewards**: High validation F1
- **Penalizes**: Large gap between train and val F1 (overfitting)
- **Best checkpoint**: Highest GnP across all 50 epochs
- **Kept**: Only best checkpoint per variant (space efficient)

**Why GnP over val_loss?**
- F1 is classification-specific (more meaningful than loss)
- Gap term forces generalization (prevents overfitting)
- Custom metric suits multi-class problem

---

## Evaluation & Metrics

### Per-Epoch Tracking

| Metric | Computed | Formula |
|--------|----------|---------|
| **Loss** | Train/Val | CrossEntropyLoss (weighted) |
| **Accuracy** | Train/Val | Correct / Total (%) |
| **Precision** | Train/Val | mean(TP_i / (TP_i + FP_i)) macro |
| **Recall** | Train/Val | mean(TP_i / (TP_i + FN_i)) macro |
| **F1-Score** | Train/Val | 2 × (P × R) / (P + R) macro |
| **ROC-AUC** | Train/Val | mean(AUC_i) per class, macro |
| **GnP Score** | Val | F1 - \|train_F1 - val_F1\| |
| **Learning Rate** | Current | Updated per epoch |
| **Epoch Time** | Wall-clock | Seconds per epoch |

### Final Evaluation (per variant)

1. **Classification Report**:
   - Per-class: precision, recall, F1
   - Weighted averages

2. **Confusion Matrix**:
   - NxN matrix (N = num_classes)
   - Shows misclassification patterns

3. **Attention Analysis**:
   - Attention rollout (aggregated across layers)
   - Per-head raw attention maps
   - Shows which image patches ViT attends to

---

## Attention Visualization (Unique Feature!)

### Attention Rollout

**Purpose**: Understand which image regions ViT considers important

**Computation**:
```
1. For each transformer layer, extract attention weights
2. Multiply attention matrices across layers (attention rollout)
3. Normalize to [0, 1]
4. Resize to image dimensions (224×224)
5. Overlay on original image as heatmap
```

**Interpretation**:
- Red/hot regions: ViT focuses here
- Blue/cool regions: Less important
- Helps identify disease-relevant features

### Per-Head Attention Maps

**Purpose**: Detect what each attention head specializes in

**Output** (per validation sample):
- 8-12 grids (number of attention heads)
- Each shows CLS token's attention to patches
- Pattern analysis: global vs local features

### Validation Samples Visualization

- 16 validation images per variant
- True label vs predicted label
- Color-coded: Green (correct), Red (incorrect)
- Saved as grid visualization

---

## EDA (Exploratory Data Analysis)

### EDA Components

1. **Class Distribution**:
   - Bar chart (overall counts)
   - Heatmap (Original vs Augmented split)
   - Per-split statistics

2. **Image Characteristics**:
   - Histogram: width, height, area, aspect ratio, file size
   - Scatter: width vs height (colored by aspect ratio)
   - Correlation heatmap: dimensions, area, file size relationships
   - Summary stats: mean, std, min, max per dimension

3. **Split-wise Statistics**:
   - Boxplots: area by split, area by class
   - Per-class size analysis (mean/std/min/max)

4. **Sample Gallery**:
   - 3 samples per class from Original
   - 3 samples per class from Augmented
   - Visual inspection of augmentation effectiveness

5. **Duplicate Analysis** (unique!):
   - Lists of duplicate groups by MD5
   - Cross-subset duplicates identified
   - Counts and percentages

6. **Output**: Metadata CSV + visualizations

---

## Output Artifacts

### Per-Variant Directory Structure

```
/kaggle/working/
├── vit_variants/
│   ├── ViT-P14-D6-256-D15/
│   │   ├── checkpoints/
│   │   │   └── best_gnp_0.92345.pt (only best, to save space)
│   │   │
│   │   ├── plots/
│   │   │   ├── training_loss.png
│   │   │   ├── training_accuracy.png
│   │   │   ├── training_precision.png
│   │   │   ├── training_recall.png
│   │   │   ├── training_f1.png
│   │   │   ├── training_auc.png
│   │   │   ├── gnp_score.png
│   │   │   ├── lr_schedule.png
│   │   │   └── epoch_timing.png
│   │   │
│   │   ├── logs/
│   │   │   ├── training.log (full text output)
│   │   │   ├── training.csv (epoch-by-epoch metrics)
│   │   │   ├── classification_report.txt
│   │   │   ├── confusion_matrix.npy
│   │   │   └── training_history.npz (all curves, numpy format)
│   │   │
│   │   ├── samples/
│   │   │   └── validation_predictions.png (16 samples grid)
│   │   │
│   │   ├── attention/
│   │   │   ├── sample_1_attention_rollout.png
│   │   │   ├── sample_1_head_attention.png
│   │   │   ├── sample_2_attention_rollout.png
│   │   │   ├── sample_2_head_attention.png
│   │   │   └── ... (2 samples per variant)
│   │   │
│   │   ├── config.json
│   │   │   {patch_size, depth, embedding_dim, dropout, num_classes}
│   │   │
│   │   └── best_result.json
│   │       {best_gnp, best_epoch, metrics}
│   │
│   ├── ViT-P14-D6-256-D25/
│   │   └── [same structure]
│   │
│   └── ... (14 more variants)
│
├── eda_outputs/
│   ├── class_distribution.png
│   ├── original_vs_augmented_heatmap.png
│   ├── resolution_scatter.png
│   ├── dimension_histograms.png
│   ├── correlation_heatmap.png
│   ├── area_by_split_boxplot.png
│   ├── samples_grid.png
│   ├── duplicate_report.txt
│   └── metadata.csv (all images with file info)
│
├── global_best_variant.json
│   {variant_name, best_gnp, metrics, epoch}
│
└── pipeline.log
    [Text log of all console output]
```

### Final Outputs

1. **16 variant checkpoints** (best GnP per variant)
2. **16 variant results** (per-variant logs, plots, attention viz)
3. **Comparative summary**: Global best variant identified
4. **EDA artifacts**: Metadata CSV, distribution plots, samples
5. **Attention visualizations**: 32 PNG files (2 samples × 16 variants)
6. **Training curves**: 9 PNG files per variant (loss, acc, F1, GnP, etc.)
7. **Metadata CSV**: All images with dimensions, hashes, labels

---

## Training Expectations

| Metric | Range | Notes |
|--------|-------|-------|
| **Val Accuracy** | 75-92% | Disease classification complexity varies |
| **Val F1-Score** | 0.72-0.90 | Class-weighted average |
| **Best GnP Score** | 0.60-0.85 | Custom metric balancing F1 and generalization |
| **Training Time** | 15-45 min/variant | Depends on depth and embedding dim |
| **Total Pipeline Time** | 6-15 hours | 16 variants × 50 epochs each |
| **Overfitting (gap)** | 2-10% | Small gaps = good generalization |

**Performance Expectations**:
- Larger patch sizes (16×16) and depths (10+): Better accuracy, slower training
- Smaller patch sizes (14×14) and depths (6): Faster training, similar accuracy
- Higher embedding dims (384): Better representation, larger model
- Dropout (0.25): Slightly more regularization vs (0.15)

---

## Key Pipeline Stages

### Stage 1: Setup & Configuration
- Install/import all packages
- Set random seeds (reproducibility)
- Configure logging (tee to console + per-variant log file)
- Create output directories

### Stage 2: Dataset Loading & Metadata
- Load Original and Augmented subsets
- Extract comprehensive metadata (MD5 hashes, dimensions, etc.)
- Export metadata to CSV

### Stage 3: Duplicate Detection
- Compute MD5 hashes
- Identify global, within-split, and cross-split duplicates
- Report findings

### Stage 4: EDA & Visualization
- Class distribution analysis
- Image dimension analysis with statistics
- Sample gallery visualization
- Export metadata CSV and EDA plots

### Stage 5: Data Preparation
- Stratified train/val split (90/10)
- Oversample minority classes
- Compute class weights
- Create PyTorch DataLoaders

### Stage 6: ViT Architecture Definition
- Define PatchEmbedding class
- Define TransformerLayer class
- Define CustomViT class (combining all components)
- Configure 16 variants

### Stage 7: Training (16 times)
- For each variant:
  - Initialize custom ViT
  - Setup optimizer (AdamW) + warmup+cosine scheduler
  - Train for 50 epochs
  - Track all metrics, compute GnP score
  - Save best checkpoint by GnP
  - Generate training curves
  - Save logs and config

### Stage 8: Evaluation & Analysis
- Generate classification reports per variant
- Compute confusion matrices
- Extract attention weights
- Create attention visualizations
- Generate validation prediction grid

### Stage 9: Global Best Selection
- Compare GnP scores across all 16 variants
- Identify overall best variant
- Output summary to global_best_variant.json

---

## Notebook Cell Breakdown

**Expected ~80 cells** organized as:

| Cell Range | Purpose | Count |
|------------|---------|-------|
| 1-5 | Headers, Cleanup, Imports | 5 |
| 6-10 | Configuration & Logging Setup | 5 |
| 11-15 | Dataset Loading & Metadata | 5 |
| 16-20 | Duplicate Detection | 5 |
| 21-30 | EDA & Visualization | 10 |
| 31-35 | Data Preparation (split, augment, dataloaders) | 5 |
| 36-40 | ViT Architecture Components | 5 |
| 41-45 | Custom ViT Class Definition | 5 |
| 46-50 | ViT Variant Configuration (16 variants) | 5 |
| 51-60 | Training Loop & Main Execution | 10 |
| 61-70 | Per-Variant Evaluation & Visualization | 10 |
| 71-75 | Attention Analysis & Visualization | 5 |
| 76-80 | Global Summary & Best Variant Selection | 5 |

---

## Implementation Notes

1. **Custom ViT**: Fully implemented from scratch (not using timm)
2. **PyTorch over TensorFlow**: Better for custom architectures
3. **GnP Metric**: Custom to this pipeline, balances validation performance and generalization
4. **Attention Visualization**: Unique interpretability feature
5. **Warm-up + Cosine Scheduling**: Standard practice for vision transformers
6. **Metadata + Duplicate Detection**: Quality assurance for dataset
7. **Per-Variant Logging**: Tee output to console AND per-variant log file

---

## Success Criteria

✅ All 16 ViT variants trained successfully  
✅ Metrics tracked across all variants  
✅ Best variant identified by GnP score  
✅ Attention visualizations generated  
✅ Per-variant logs, plots, and results saved  
✅ Global best variant summary exported  
✅ EDA complete with duplicate analysis  
✅ Metadata CSV exported for reproducibility  
✅ Pipeline runs without errors  

---

## Key Differences from Pipe-1 and Pipe-2

| Aspect | Pipe-1 | Pipe-2 | Pipe-3 |
|--------|--------|--------|--------|
| **Framework** | PyTorch | TensorFlow | PyTorch |
| **Architecture** | Pre-trained EfficientNet | Pre-trained CNNs (24) | Custom ViT (16 variants) |
| **Models** | 1 | 24 | 16 |
| **CV Strategy** | 5-fold | Single split | Single split |
| **Metric** | Accuracy | Composite (acc+efficiency) | GnP (generalization+performance) |
| **Output** | Submission CSV | Model comparison | Attention visualizations |
| **Focus** | Ensemble prediction | Model selection | Architecture exploration |

