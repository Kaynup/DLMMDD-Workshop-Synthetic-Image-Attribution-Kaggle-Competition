# Synthetic Image Attribution Challenge - Pipeline Architecture

## Competition Summary
- **Task:** 10-class closed-set classification (source attribution)
- **Training Set:** 7,000 labeled face images (1,000 per generator)
- **Test Set:** 3,000 unlabeled images with 1-3 post-processing operations
- **Metric:** Classification Accuracy (assume F1-macro for robustness validation)
- **Sources:** 10 text-to-image generators (AuraFlow, Freepik, Lumina, Photon, Pixart(sigma), Playground v2.5, StableDiffusion3, StableDiffusion3.5, StableDiffusionXL-Turbo, Tencent Hunyuan)

---

## Service-Oriented Architecture (SOA) Pipelines

### Pipeline Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│  Data Loading & Validation Pipeline                         │
│  (Verify splits, check distributions, handle paths)         │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┬──────────────────┐
        │                         │                  │
        ▼                         ▼                  ▼
┌──────────────┐    ┌────────────────────┐  ┌──────────────┐
│ EDA Pipeline │    │ Augmentation       │  │ Preprocessing│
│ (Analysis)   │    │ Pipeline (Train)   │  │ Pipeline     │
└──────────────┘    └────────────────────┘  └──────────────┘
        │                   │                      │
        └───────────────────┼──────────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Train/Val Split│
                    │ (Stratified CV) │
                    └───────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────────┐  ┌───────────────┐
│ Feature      │  │ Model Training   │  │ Validation    │
│ Extraction   │  │ Pipeline         │  │ Pipeline      │
│ Pipeline     │  │ (Logging, CKPTs) │  │ (CV, Metrics) │
└──────────────┘  └──────────────────┘  └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼────────┐
                    │ Checkpoint     │
                    │ Selection      │
                    │ (Gen. Score)   │
                    └───────┬────────┘
                            │
        ┌───────────────────┴──────────────────┐
        │                                      │
        ▼                                      ▼
┌──────────────────┐              ┌────────────────────┐
│ Inference &      │              │ Ensemble           │
│ Submission       │              │ Prediction         │
│ Pipeline         │              │ Pipeline (Optional)│
└──────────────────┘              └────────────────────┘
```

---

## SOA Pipeline Specifications

| # | Pipeline | Purpose | Key Outputs | Dependencies |
|---|----------|---------|------------|--------------|
| 1 | **Data Loading & Validation** | Load CSVs/images, verify integrity, check class distribution | `train_df`, `test_df`, data statistics | None |
| 2 | **EDA Pipeline** | Statistical analysis, visualization, post-processing sensitivity analysis | Plots, distributions, insights report | Data Loading |
| 3 | **Preprocessing Pipeline** | Normalization, resizing, format conversion, color space handling | Preprocessed image arrays/tensors | Data Loading |
| 4 | **Augmentation Pipeline** | Train-time augmentation (rotation, flip, brightness, etc.) | Augmented batch generator | Preprocessing |
| 5 | **Train/Val Stratified Split** | Create balanced CV folds (5-fold or 10-fold) with stratification | `fold_assignments`, `train_indices`, `val_indices` | Preprocessing |
| 6 | **Feature Extraction Pipeline** | Extract features from pretrained models (ResNet, Vision Transformer, etc.) | Feature matrices `[N, feature_dim]` | Preprocessing |
| 7 | **Model Training Pipeline** | Training loop with logging, checkpoint saving, early stopping | Model checkpoints, training logs, metrics history | Augmentation + Train Split |
| 8 | **Validation Pipeline** | Compute metrics per fold, aggregate statistics, track generalization | Per-fold metrics, aggregated CV results, generalization scores | Model Training |
| 9 | **Checkpoint Selection Pipeline** | Select best checkpoint using generalization score formula | Best checkpoint path, selection metrics | Validation |
| 10 | **Inference & Submission Pipeline** | Load best model, predict on test set, format submission CSV | `submission.csv`, prediction probabilities | Checkpoint Selection |
| 11 | **Ensemble Pipeline (Optional)** | Combine multiple models/checkpoints for robust predictions | Ensemble predictions, weighted probabilities | Multiple Checkpoint Selection |

---

## Key Formulas & Metrics

### Generalization Score (Checkpoint Selection)
```
Gen_Score = val_metric - |train_metric - val_metric|

Where:
  - val_metric: Validation accuracy/F1 on held-out fold
  - train_metric: Training accuracy/F1 on training fold
  - |train_metric - val_metric|: Absolute overfitting magnitude

Higher score = better generalization (less overfitting)
```

### CV Evaluation
- **5-Fold or 10-Fold Stratified K-Fold**
- Per-fold metrics: Accuracy, F1-Macro, Precision, Recall
- Aggregate: Mean ± Std across folds
- Final validation: Use fold scores to select best checkpoint per fold

### Metrics Suite
- **Accuracy:** Multi-class accuracy (matches leaderboard)
- **F1-Macro:** Average F1 across 10 classes (robustness to class imbalance)
- **Per-Class Metrics:** Precision, Recall, F1 per generator source
- **Confusion Matrix:** Identify which sources are confused
- **Generalization Gap:** train_acc - val_acc (should be small)

---

## File Structure

```
docs/
├── 00_PIPELINE_OVERVIEW.md (this file)
├── 01_DATA_LOADING_VALIDATION/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
├── 02_EDA/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
├── 03_PREPROCESSING/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
├── 04_AUGMENTATION/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
├── 05_TRAIN_VAL_SPLIT/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
├── 06_FEATURE_EXTRACTION/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
├── 07_MODEL_TRAINING/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
├── 08_VALIDATION/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
├── 09_CHECKPOINT_SELECTION/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
├── 10_INFERENCE_SUBMISSION/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
└── 11_ENSEMBLE/
    ├── 00_FLOW.md
    ├── 01_DESIGN.md
    ├── 02_CODE_ANALOGIES.md
    └── 03_DEPENDENCIES.md
```

---

## Execution Order (Recommended)

1. **Notebook 00:** Data Loading & Validation + EDA
2. **Notebook 01:** Preprocessing + Augmentation + Train/Val Split
3. **Notebook 02:** Feature Extraction (optional, for feature-based models)
4. **Notebook 03:** Model Training & Validation (per model architecture)
5. **Notebook 04:** Checkpoint Selection & Best Model Analysis
6. **Notebook 05:** Inference & Submission
7. **Notebook 06 (Optional):** Ensemble strategies

---

## Success Criteria

- ✅ **EDA:** Understand class distributions, identify easy/hard samples, analyze post-processing impact
- ✅ **Reproducibility:** Fixed seeds, versioned data, logged hyperparameters
- ✅ **Validation:** 5-10 fold CV with stratification, per-class metrics, generalization gap < 5%
- ✅ **Logging:** All training events logged (epoch, loss, metrics, learning rate, checkpoint path)
- ✅ **Checkpoints:** All epochs saved + metadata (metrics, config, fold assignment)
- ✅ **Selection:** Best checkpoint chosen by generalization score across all folds
- ✅ **Submission:** Correct CSV format, no row misalignment, prediction confidence recorded
