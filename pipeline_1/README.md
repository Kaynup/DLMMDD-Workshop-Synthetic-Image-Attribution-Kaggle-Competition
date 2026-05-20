# Synthetic Image Attribution Challenge - Pipeline 1

A powerful, production-ready end-to-end machine learning pipeline for the **DLMMDD Synthetic Image Attribution Challenge**.

**Competition:** Synthetic Image Source Attribution (10-class classification)  
**Task:** Classify face images by their generation source (10 text-to-image generators)  
**Dataset:** 7,000 training images + 3,000 test images  
**Metric:** Classification Accuracy  

---

## 📋 Pipeline Overview

```
Pipeline 1 - Modular ML System
├── Stage 00: Data Loading, Validation & EDA
│   └── Load CSVs, verify integrity, extract metadata, analyze distributions
├── Stage 01: Preprocessing, Augmentation & CV Splits
│   └── Normalize images, create augmentation pipelines, 5-fold stratification
├── Stage 02: Model Training
│   └── Train models with logging, checkpointing, early stopping per fold
├── Stage 03: Validation & Checkpoint Selection
│   └── Evaluate all epochs, select best by generalization score
├── Stage 04: Inference & Submission
│   └── Predict on test set, format submission CSV
└── Stage 05: Ensemble (Optional)
    └── Combine multiple models for robust predictions
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Data Structure
Ensure your `Data/` directory has:
```
Data/
├── Training/           # 7,000 images (labeled)
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── Test/              # 3,000 images (unlabeled)
│   ├── 6.png
│   ├── 12.png
│   └── ...
├── training.csv       # ID, path, y (labels)
├── test.csv          # ID, path
├── sources.txt       # Class ID → Generator name mapping
└── sample_submission.csv
```

### 3. Run Notebooks in Order

```bash
# Navigate to pipeline_1/
cd pipeline_1

# Run notebooks sequentially
jupyter notebook notebooks/00_data_loading_validation_eda.ipynb
jupyter notebook notebooks/01_preprocessing_augmentation_splits.ipynb
jupyter notebook notebooks/02_model_training.ipynb
jupyter notebook notebooks/03_validation_checkpoint_selection.ipynb
jupyter notebook notebooks/04_inference_submission.ipynb
jupyter notebook notebooks/05_ensemble.ipynb  # Optional
```

---

## 📁 Project Structure

```
pipeline_1/
├── config.toml                    # Configuration file (TOML)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── src/                          # Core modules (reusable)
│   ├── __init__.py               # Public API
│   ├── config_loader.py          # Configuration management
│   ├── logger_setup.py           # Structured logging
│   ├── data_loader.py            # Data loading & validation
│   ├── preprocessor.py           # Image preprocessing & augmentation
│   ├── train_val_split.py        # Stratified K-fold splitting
│   ├── metrics.py                # Metrics computation
│   └── checkpoint_selector.py    # Checkpoint management
│
├── notebooks/                    # Jupyter notebooks
│   ├── 00_data_loading_validation_eda.ipynb
│   ├── 01_preprocessing_augmentation_splits.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_validation_checkpoint_selection.ipynb
│   ├── 04_inference_submission.ipynb
│   └── 05_ensemble.ipynb
│
├── checkpoints/                  # Model checkpoints (auto-generated)
│   ├── fold_0/
│   │   ├── epoch_000.pth
│   │   └── ...
│   └── ...
│
├── processed/                    # Preprocessed data (auto-generated)
│   ├── train_metadata.parquet
│   ├── test_metadata.parquet
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── fold_metadata.json
│   └── source_mapping.json
│
├── logs/                         # Training & pipeline logs
│   ├── pipeline.log
│   ├── pipeline_00.log
│   ├── training_fold_0.log
│   └── ...
│
└── submissions/                  # Final submission CSVs
    ├── submission_final.csv
    └── submission_ensemble.csv
```

---

## 🔧 Configuration

Edit `config.toml` to customize:

```toml
[competition]
num_classes = 10
data_dir = "../Data"

[data]
target_height = 224
target_width = 224
color_mode = "RGB"

[preprocessing]
normalization = "imagenet"   # ImageNet or minmax

[augmentation]
enabled = true
train_augmentations = ["random_horizontal_flip", "random_rotation", ...]

[training]
num_epochs = 100
batch_size = 32
learning_rate = 1e-4
scheduler = "cosine"
early_stopping_patience = 10

[model]
architecture = "efficientnet_b4"   # ResNet50, ViT, etc.
pretrained = true

[train_val_split]
strategy = "stratified_kfold"
num_splits = 5

[checkpoint]
selection_metric = "generalization_score"   # or "val_accuracy"

[inference]
tta_enabled = true
tta_transforms = 5
```

**Environment Variable Overrides:**
```bash
# Override config values via environment variables
export PIPELINE_NUM_EPOCHS=50
export PIPELINE_BATCH_SIZE=64
export PIPELINE_LEARNING_RATE=0.0005
```

---

## 📊 Pipeline Details

### Stage 00: Data Loading & Validation
- Load training.csv (7,000 labeled images)
- Load test.csv (3,000 unlabeled images)
- Extract image metadata: dimensions, format, color mode, file size
- Validate class balance (should be 1,000 per class)
- Load source mapping (class ID → generator name)
- Compute dataset statistics
- Generate EDA visualizations (class distribution, dimensions, formats)

**Outputs:** `train_metadata.parquet`, `test_metadata.parquet`, `data_stats.json`

### Stage 01: Preprocessing & Augmentation
- Resize images to 224×224 (configurable)
- Apply ImageNet normalization (or minmax)
- Convert to PyTorch tensors
- Create augmentation pipelines for training:
  - Horizontal flip (50%)
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation)
  - Gaussian blur
  - Random affine transforms

- Create stratified K-fold splits (5-fold by default)
  - Preserves class distribution in each fold
  - Ensures no data leakage

**Outputs:** `X_train.npy`, `X_test.npy`, `y_train.npy`, `fold_metadata.json`

### Stage 02: Model Training
- Load preprocessed data and fold metadata
- For each fold (0-4):
  - Initialize model (EfficientNet-B4 by default)
  - Create data loaders with augmentation
  - Training loop:
    - Forward pass
    - Compute loss (CrossEntropy)
    - Backward pass + optimizer step
    - Track metrics (accuracy, F1-macro)
    - Save checkpoints every epoch
    - Early stopping if validation metric plateaus
  - Log all events (epochs, losses, metrics, LR)

**Outputs:** `checkpoints/fold_0/epoch_*.pth`, `logs/training_fold_*.log`

### Stage 03: Validation & Checkpoint Selection
- Load all saved checkpoints
- Evaluate each epoch on validation set
- Compute generalization score: `val_acc - |train_acc - val_acc|`
  - Penalizes overfitting
  - Promotes generalizable models
- Select best checkpoint per fold
- Aggregate metrics across all folds (mean ± std)
- Save selection results

**Outputs:** `checkpoint_selections.json`, `validation_metrics.csv`

### Stage 04: Inference & Submission
- Load best checkpoint per fold
- Ensemble predictions across 5 folds (averaging)
- Optional: Test-time augmentation (TTA)
  - Apply 5 augmented versions per image
  - Average predictions
- Format submission CSV: `ID, TARGET` (class predictions)
- Include prediction confidence/probabilities

**Outputs:** `submission_final.csv`

### Stage 05: Ensemble (Optional)
- Train multiple model architectures (ResNet50, ViT, DenseNet)
- Use weighted ensemble to combine predictions
- Improve robustness by leveraging model diversity

**Outputs:** `submission_ensemble.csv`

---

## 🎯 Key Features

✅ **Modular Design**
- Reusable components: preprocessor, augmenter, metrics tracker
- Easy to swap model architectures or training strategies
- Clean separation of concerns

✅ **Reproducibility**
- Fixed random seeds
- Config-driven execution (all parameters in TOML)
- Comprehensive logging of all decisions
- Saved checkpoints with full metadata

✅ **Robustness**
- Stratified K-fold cross-validation (no data leakage)
- Generalization score for checkpoint selection (reduces overfitting)
- Per-class metrics to identify weak sources
- Confusion matrices for error analysis

✅ **Production-Ready**
- Structured logging with file output
- Metrics tracking throughout training
- Early stopping to prevent overfitting
- TTA for improved inference
- Ensemble support

✅ **Flexibility**
- Easy configuration via TOML or environment variables
- Multiple architectures supported (EfficientNet, ResNet, ViT)
- Customizable augmentation pipelines
- Support for different optimizers and schedulers

---

## 📈 Expected Performance

Based on competition structure:
- **Baseline:** Random guessing = 10% accuracy
- **Reasonable Model:** ~70-80% accuracy (single fold CV)
- **Strong Model:** ~85-92% accuracy (5-fold ensemble)
- **Competition Winner:** ~95%+ accuracy (advanced techniques + ensemble)

Your actual performance depends on:
1. Model architecture choice
2. Hyperparameter tuning
3. Ensemble strategy
4. Training duration

---

## 🔍 Debugging & Monitoring

### Logs
```bash
# View pipeline logs
tail -f logs/pipeline.log

# View training logs for specific fold
tail -f logs/training_fold_0.log
```

### Checkpoint Management
```bash
# List available checkpoints for fold 0
ls checkpoints/fold_0/

# Inspect a checkpoint
python -c "import torch; ckpt = torch.load('checkpoints/fold_0/epoch_050.pth'); print(ckpt.keys())"
```

### Metrics History
```bash
# View training history
python -c "import json; print(json.load(open('logs/fold_0_history.json')))"
```

---

## 🐛 Troubleshooting

**Issue:** "Data directory not found"
```bash
# Ensure Data/ is in the parent directory
ls ../Data/training.csv ../Data/test.csv
```

**Issue:** "No module named 'src'"
```bash
# Run notebooks from pipeline_1/ directory
cd pipeline_1
jupyter notebook notebooks/00_...ipynb
```

**Issue:** "CUDA out of memory"
```bash
# Reduce batch size in config.toml
[training]
batch_size = 16  # Instead of 32
```

**Issue:** "Class imbalance in folds"
```bash
# Verify stratification in notebook output
# Should see ~700 per class in train, ~300 in val
```

---

## 📝 Sample Submission Format

Your final `submission.csv` should look like:

```csv
ID,TARGET
0,5
1,2
2,5
3,0
...
2999,7
```

Where `TARGET` is the predicted class (0-9).

---

## 🤝 Contributing & Extending

### Add a New Model Architecture
1. Edit `notebooks/02_model_training.ipynb`:
```python
# Load pretrained model
if config.model.architecture == "resnet50":
    model = torchvision.models.resnet50(pretrained=True)
elif config.model.architecture == "vit_base":
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
```

### Add Custom Augmentations
1. Edit `src/preprocessor.py`:
```python
# In ImageAugmenter._build_transforms()
self.transforms = transforms.Compose([
    ...,
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
    CustomAugmentation(),  # Your custom transform
])
```

### Use Different Checkpoint Selection Metric
1. Edit `config.toml`:
```toml
[checkpoint]
selection_metric = "val_f1_macro"  # Instead of "generalization_score"
```

---

## 📚 References

- **Competition:** [DLMMDD Workshop - Synthetic Image Attribution Challenge](https://icann2026.org/)
- **Pretrained Models:** [Hugging Face Model Hub](https://huggingface.co/models)
- **EfficientNet:** [Paper](https://arxiv.org/abs/1905.11946)
- **ImageNet Normalization:** RGB (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)

---

## 📄 License

This pipeline is provided as-is for the DLMMDD 2026 competition.

---

## ❓ Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review configuration in `config.toml`
3. Inspect notebook outputs for error messages
4. Verify data integrity in Stage 00 notebook

---

**Last Updated:** 2026-05-20  
**Version:** 1.0.0  
**Status:** Production Ready ✅
