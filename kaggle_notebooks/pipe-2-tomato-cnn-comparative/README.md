# PIPE-2: Comparative Analysis of 24 Pre-Trained CNN Models for Tomato Leaf Disease Classification

## Overview

**Pipe-2** is a comprehensive framework for **systematic comparison of 24 pre-trained CNN models** on the **PlantVillage Tomato Leaf Disease Classification** task. Unlike Pipe-1 (focused on synthetic image attribution), this pipeline emphasizes **model selection through comparative analysis** using both accuracy and efficiency metrics.

### Key Distinction from Pipe-1
- **Pipe-1**: Single ensemble approach (5-fold CV, one architecture)
- **Pipe-2**: Multi-model comparison (24 models, single split, composite scoring)

---

## Competition & Dataset

| Aspect | Details |
|--------|---------|
| **Dataset** | PlantVillage Tomato Leaf Disease Classification |
| **Source** | `/kaggle/input/plantvillage-tomato-leaf-dataset/` |
| **Task Type** | Multi-class image classification |
| **Number of Classes** | ~9-10 disease categories for tomato leaves |
| **Total Images** | ~9,000-15,000 (varies by dataset size) |
| **Train/Val Split** | Stratified 80/20 |
| **Image Format** | JPEG/PNG, variable resolution |

---

## Technical Stack

### Framework & Optimization
- **Deep Learning Framework**: TensorFlow 2.x + Keras
- **Precision**: Mixed precision (float16 + float32) with LossScaleOptimizer
- **Optimization Level**: Reduced memory footprint, increased throughput
- **GPU**: CUDA/CuDNN optimized

### Models Compared (24 Total)

#### VGG Family (2)
- VGG16
- VGG19

#### ResNet Family (5)
- ResNet50, ResNet101, ResNet152
- ResNet50V2, ResNet101V2

#### Inception Family (2)
- InceptionV3
- InceptionResNetV2

#### Mobile & Efficient (9)
- MobileNetV2
- MobileNetV3 (Small, Large)
- EfficientNetB0, EfficientNetB3
- EfficientNetV2B0, EfficientNetV2B3, EfficientNetV2S

#### DenseNet Family (3)
- DenseNet121, DenseNet169, DenseNet201

#### Others (3)
- Xception
- NASNetMobile, NASNetLarge

---

## Architecture Pattern

All 24 models follow the **Transfer Learning** pattern:

```
┌─────────────────────────────────────┐
│ Pre-Trained Base (ImageNet weights) │
│   [Frozen: no gradient updates]     │
└────────────────┬────────────────────┘
                 │
         ┌───────▼──────────┐
         │ GlobalAvgPool2D  │
         └────────┬─────────┘
                  │
         ┌────────▼──────────┐
         │ Dense(128, relu)  │
         │ + Dropout(0.5)    │
         └────────┬──────────┘
                  │
    ┌─────────────▼────────────────┐
    │ Dense(num_classes, softmax)  │
    │ [Trainable: float32 dtype]   │
    └──────────────────────────────┘
```

### Head Architecture
- **GlobalAveragePooling2D**: Compress spatial dimensions
- **Dense(128, relu) + Dropout(0.5)**: Adaptation layer
- **Dense(num_classes, softmax)**: Classification head (float32 for stability)

---

## Data Pipeline

### Stage 1: Dataset Detection & Loading

**Input Path Structure**:
```
/kaggle/input/plantvillage-tomato-leaf-dataset/
└── [subdirectories organized by disease class]
    ├── class_1/
    │   ├── image1.jpg
    │   ├── image2.png
    │   └── ...
    ├── class_2/
    │   └── ...
    └── ...
```

**Dataset Detection Logic**:
- Handles both flat and nested directory structures
- Detects single level of nesting (e.g., Tomato → Tomato_Early_blight)
- Validates at least one class directory with images

**Output**: DataFrame with columns `[filepath, label]`

### Stage 2: Data Splitting & Class Handling

**Stratified Train/Val Split (80/20)**:
- Preserves class distribution in both splits
- Special handling for classes with <2 images:
  - Placed in training set only (insufficient for validation)
  - Logged as "small classes"

**Class Imbalance Mitigation**:
1. **Oversampling (during training)**:
   - Identify max class count
   - Resample minority classes to match max count
   - Random resampling with replacement
   
2. **Class Weights (in loss function)**:
   - Compute balanced class weights: `weight = N / (n_classes * n_class_samples)`
   - Applied to loss function during training
   - Higher weight for minority classes

### Stage 3: tf.data Pipeline (Optimized)

```python
# Pseudo-code
train_ds = create_dataset(train_df, BATCH_SIZE=64)
  .map(load_image, num_parallel_calls=AUTOTUNE)
  .map(augment_image, num_parallel_calls=AUTOTUNE)
  .shuffle(buffer_size=1000)
  .batch(BATCH_SIZE)
  .prefetch(AUTOTUNE)
```

**Optimizations**:
- `num_parallel_calls=AUTOTUNE`: Adaptive parallelization
- `prefetch(AUTOTUNE)`: Overlap data loading with GPU training
- `shuffle()`: Randomize batch order for better generalization
- Separate train/val augmentation strategies

### Stage 4: Image Preprocessing

**Input Image Specs**:
- Size: Variable (224×224 target)
- Format: JPEG/PNG
- Channels: RGB (grayscale converted to RGB if needed)

**Preprocessing Steps**:
1. **Load**: PIL Image or tf.io.decode_image
2. **Resize**: Bilinear interpolation to 224×224
3. **Normalize**: Model-specific (e.g., `vgg16.preprocess_input`)
   - Values centered around 0
   - Different for each model family

### Stage 5: Augmentation (Train-Only)

**Training Augmentation** (applied via tf.data):
- **Horizontal Flip**: p=0.5 (prevent left-right bias)
- **Brightness**: ±10% (handle lighting variation)
- **Contrast**: ±10% (handle exposure variation)
- **Saturation**: ±10% (handle color balance)
- **Rotation**: Stochastic 90° (handle arbitrary orientations)

**Validation Augmentation**: None (only normalization)

---

## Training Pipeline

### Training Loop Structure

```
For each of 24 models:
  1. Load pre-trained base (ImageNet weights)
  2. Attach custom head
  3. Build model with mixed precision
  4. Train for 10 epochs with callbacks
  5. Evaluate and save metrics
  6. Save best checkpoint
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Image Size** | 224×224 | ImageNet standard |
| **Batch Size** | 64 | Balanced for GPU memory |
| **Epochs** | 10 | Per model (short for comparison) |
| **Optimizer** | Adam | Learning rate: 0.001 (default) |
| **Loss** | Categorical Cross-Entropy | Weighted by class weights |
| **Mixed Precision** | float16 base, float32 output | LossScaleOptimizer |
| **Early Stopping** | patience=5 | Stop if no improvement for 5 epochs |
| **LR Reduce** | factor=0.5, patience=3 | Reduce LR when val_loss plateaus |

### Callbacks

1. **ModelCheckpoint**:
   - Monitor: `val_loss`
   - Save best only
   - Output: `models/model_name_best.h5`

2. **ReduceLROnPlateau**:
   - Factor: 0.5 (multiply LR by 0.5)
   - Patience: 3 epochs
   - Prevents overfitting on plateau

3. **EarlyStopping**:
   - Patience: 5 epochs
   - Stop if no improvement in val_loss
   - Prevents unnecessary training

---

## Evaluation & Metrics

### Per-Model Metrics (tracked for all 24)

| Metric | Formula | Notes |
|--------|---------|-------|
| **Accuracy** | `correct / total` | Overall classification accuracy (%) |
| **Precision (macro)** | `mean(TP_i / (TP_i + FP_i))` | Unweighted average across classes |
| **Recall (macro)** | `mean(TP_i / (TP_i + FN_i))` | Sensitivity per class, averaged |
| **F1-Score (macro)** | `2 * (P * R) / (P + R)` | Harmonic mean of precision & recall |
| **Params (M)** | From model summary | Number of trainable parameters (millions) |
| **Training Time (min)** | Wall-clock time | Per model, 10 epochs |
| **Overfitting Gap** | `max(train_acc - val_acc)` | Generalization measure (lower = better) |

### Composite Scoring (for model selection)

**Best Overall Score = 0.50 × F1 + 0.30 × anti_overfit + 0.20 × efficiency**

Where:
- **F1 Score**: Raw F1-macro (normalized 0-1)
- **Anti-overfit**: `1 - (overfitting_gap / max_gap)` (penalizes large train-val gaps)
- **Efficiency**: `F1 / (params_M × time_min)` (performance per resource)

### Selection Criteria (5 categories)

1. **Best F1-Score**: Highest F1-macro across all models
2. **Best Accuracy**: Highest accuracy (%)
3. **Least Overfitting**: Lowest train-val gap, restricted to F1 > 0.85
4. **Most Efficient**: Best F1 per parameter (F1 / params_M)
5. **Overall Best**: Composite score combining accuracy, generalization, and efficiency

---

## EDA (Exploratory Data Analysis)

### EDA Outputs

1. **Class Distribution Analysis**:
   - Bar chart (class counts)
   - Pie chart (proportions)
   - Normalized heatmap (imbalance visualization)
   - Box plot (quartiles, outliers)
   - **Statistical test**: Chi-square goodness-of-fit
     - Null hypothesis: Classes follow uniform distribution
     - Result indicates significance of imbalance

2. **Image Dimensions Analysis**:
   - Scatter plot: width vs height (colored by aspect ratio)
   - Histograms: width, height, area (px²), aspect ratio, file size (KB)
   - Summary stats: mean, std, min, max for each dimension
   - **Insight**: Identifies if images have consistent aspect ratios

3. **Sample Gallery**:
   - 3 random samples per class (100×100 px thumbnails)
   - Visual inspection of disease presentation

4. **Output**: `eda_summary.json`
   ```json
   {
     "class_distribution": {class: count},
     "image_dimensions": {
       "width": {mean, std, min, max},
       "height": {mean, std, min, max},
       "area": {mean, std, min, max},
       "aspect_ratio": {mean, std, min, max}
     },
     "file_statistics": {...},
     "class_imbalance_ratio": max_count / min_count,
     "chi_square_test": {statistic, p_value, interpretation}
   }
   ```

---

## Output Artifacts

### Directory Structure

```
/kaggle/working/
├── output/
│   ├── models/
│   │   ├── VGG16_best.h5
│   │   ├── ResNet50_best.h5
│   │   ├── EfficientNetB0_best.h5
│   │   └── ... (24 model checkpoints)
│   │
│   ├── plots/
│   │   ├── training_history/
│   │   │   ├── VGG16_history.png (train/val accuracy & loss)
│   │   │   └── ... (1 per model)
│   │   │
│   │   ├── class_distribution.png
│   │   ├── class_distribution_pie.png
│   │   ├── resolution_scatter.png
│   │   ├── resolution_histograms.png
│   │   │
│   │   ├── comparative_metrics/
│   │   │   ├── accuracy_comparison.png (bar chart, all 24 models)
│   │   │   ├── precision_comparison.png
│   │   │   ├── recall_comparison.png
│   │   │   ├── f1_comparison.png
│   │   │   ├── parameters_comparison.png (model size)
│   │   │   ├── training_time_comparison.png
│   │   │   ├── accuracy_vs_params.png (scatter: size vs performance)
│   │   │   ├── accuracy_vs_time.png (scatter: speed vs performance)
│   │   │   ├── overfitting_analysis.png (train-val gap per model)
│   │   │   └── efficiency_scatter.png (F1/params × F1/time)
│   │   │
│   │   └── confusion_matrices/
│   │       ├── VGG16_cm.png (heatmap)
│   │       └── ... (best model only, space efficient)
│   │
│   ├── eda/
│   │   └── eda_summary.json
│   │
│   └── results_summary.csv
│       Columns: Model | Accuracy | Precision | Recall | F1Score |
│                Params | TrainTime | OverfitGap | CompositeScore
│
├── best_models.json
│   {
│     "best_f1_model": {model, f1, accuracy},
│     "best_accuracy_model": {model, accuracy, f1},
│     "least_overfitting_model": {model, overfit_gap},
│     "most_efficient_model": {model, f1_per_param},
│     "overall_best_model": {model, composite_score, breakdown}
│   }
│
└── pipeline.log
    [Text log of all console output]
```

### Final Outputs

1. **results_summary.csv**: All 24 models with metrics
2. **best_models.json**: Top performers in 5 categories
3. **Model checkpoints**: 24 `.h5` files (best per model)
4. **Visualizations**: 40+ PNG files (training curves, comparisons, analysis)
5. **EDA summary**: JSON with dataset statistics

---

## Training Expectations

| Metric | Range | Notes |
|--------|-------|-------|
| **Accuracy** | 85-95% | Tomato disease classification is relatively clean |
| **F1-Score** | 0.84-0.94 | Macro-averaged across disease classes |
| **Best Model** | EfficientNet or ResNet50 | Typically balances accuracy & efficiency |
| **Training Time** | 2-8 min/model | Depends on model size (10 epochs each) |
| **Total Pipeline Time** | 3-6 hours | ~24 models × 3-15 min each |
| **Overfitting Gap** | 2-8% | Small gaps indicate good generalization |

---

## Key Pipeline Stages

### Stage 1: Setup & Configuration
- Install/import all packages
- Set mixed precision policy
- Configure logging
- Create output directories

### Stage 2: Data Loading & Analysis
- Detect dataset structure
- Gather image paths and labels
- Create metadata DataFrame
- Perform stratified train/val split
- Oversample minority classes
- Compute class weights

### Stage 3: EDA
- Class distribution analysis (with statistical testing)
- Image dimension analysis
- Sample visualization gallery
- Export EDA summary to JSON

### Stage 4: tf.data Pipeline
- Load images from disk
- Apply augmentation/normalization
- Create train and validation datasets
- Prefetch for performance

### Stage 5: Model Building (24 times)
- Load pre-trained base (ImageNet weights)
- Add custom head (GlobalAvgPool → Dense → Softmax)
- Compile with mixed precision
- Build model configuration

### Stage 6: Training (24 times)
- Fit model on train_ds with val_ds
- Track metrics per epoch
- Apply callbacks (checkpoint, LR reduce, early stop)
- Save best checkpoint

### Stage 7: Evaluation
- Generate confusion matrices per model
- Classification reports
- Log all metrics

### Stage 8: Comparative Analysis
- Compute composite scores
- Create comparison visualizations (6 bar charts + 2 scatter plots)
- Identify top performers in 5 categories
- Export results summary CSV

### Stage 9: Model Selection
- Determine best model by composite score
- Output best_models.json
- Summarize findings

---

## Notebook Cell Breakdown

**Expected ~50 cells** organized as:

| Cell Range | Purpose | Count |
|------------|---------|-------|
| 1-3 | Headers & Config | 3 |
| 4-8 | Imports & Setup | 5 |
| 9-12 | Helper Functions | 4 |
| 13-16 | Dataset Loading | 4 |
| 17-25 | EDA & Visualization | 9 |
| 26-28 | tf.data Pipeline | 3 |
| 29-35 | Model Architecture & Training Loop | 7 |
| 36-40 | Individual Model Training | 5 |
| 41-45 | Evaluation & Analysis | 5 |
| 46-50 | Summary & Export | 5 |

---

## Implementation Notes

1. **Framework Choice**: TensorFlow/Keras (matches original)
2. **Mixed Precision**: Reduces memory by ~50%, increases training speed ~20-30%
3. **Model Registry**: Dictionary mapping model names to builders and preprocessors
4. **Stratified Split**: Handles class imbalance without excluding small classes
5. **Composite Scoring**: Balances accuracy, generalization, and efficiency
6. **Statistical Rigor**: Chi-square test validates significance of class imbalance
7. **Comparative Framework**: Systematically logs all models for side-by-side analysis

---

## Success Criteria

✅ All 24 models trained successfully  
✅ Metrics tracked for all models  
✅ Top 5 performers identified by different criteria  
✅ Comparative visualizations generated  
✅ results_summary.csv complete  
✅ best_models.json informative  
✅ Pipeline runs without errors  

