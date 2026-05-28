# EXECUTION GUIDE: Step-by-Step Notebook Implementation

## Overview

This guide shows exactly how to structure your Jupyter notebooks to implement all pipelines in order. You will create **6-7 notebooks** that build on each other.

---

## Timeline & Notebooks

### **Notebook 00: Data Loading, Validation & EDA**
**Time:** ~30 minutes  
**What:** Load data, verify integrity, understand competition

```
Input: Data/ folder
Output: 
  - train_df, test_df (DataFrames)
  - data_stats.json
  - eda_report.html
  - eda_insights.json
```

**Key cells:**
```python
# Cell 1: Imports & Setup
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('notebook_00')

# Cell 2: Load & Validate Data
train_df = pd.read_csv('Data/training.csv')
test_df = pd.read_csv('Data/test.csv')
# ... validation code from 01_DATA_LOADING_VALIDATION pipeline ...

# Cell 3: EDA - Class Distribution
# ... code from 02_EDA pipeline ...

# Cell 4: Save Results
train_df.to_parquet('processed/train_metadata.parquet')
test_df.to_parquet('processed/test_metadata.parquet')
```

---

### **Notebook 01: Preprocessing, Augmentation & CV Splits**
**Time:** ~20 minutes  
**What:** Prepare images, create fold splits

```
Input: train_df, test_df
Output:
  - X_train, X_test (numpy arrays, preprocessed)
  - fold_metadata.json
```

**Key cells:**
```python
# Cell 1: Load from previous notebook
train_df = pd.read_parquet('processed/train_metadata.parquet')
test_df = pd.read_parquet('processed/test_metadata.parquet')

# Cell 2: Preprocessing
# ... ImagePreprocessor code from 03_PREPROCESSING ...
X_train = preprocess_all_images(train_df)  # (7000, 224, 224, 3)
X_test = preprocess_all_images(test_df)    # (3000, 224, 224, 3)

# Cell 3: Stratified K-Fold Split
# ... StratifiedKFold code from 05_TRAIN_VAL_SPLIT ...
fold_metadata = create_stratified_folds(train_df, n_splits=5)

# Cell 4: Save
np.save('processed/X_train.npy', X_train)
np.save('processed/X_test.npy', X_test)
with open('processed/fold_metadata.json', 'w') as f:
    json.dump(fold_metadata, f)
```

---

### **Notebook 02: Model Training (One Architecture)**
**Time:** ~3-4 hours per architecture  
**What:** Train EfficientNet-B4 with proper logging and checkpointing

```
Input: X_train, X_test, fold_metadata
Output:
  - checkpoints/fold_*/epoch_*.pth (all epochs)
  - logs/training_log_fold_*.json
```

**Structure:**
```python
# Cell 1: Load preprocessed data
X_train = np.load('processed/X_train.npy')
fold_metadata = json.load(open('processed/fold_metadata.json'))

# Cell 2: Config
config = {
    'model_name': 'efficientnet_b4',
    'num_epochs': 100,
    'batch_size': 32,
    'lr': 1e-4,
    'seed': 42
}

# Cell 3: Training loop (for each fold)
for fold_idx in range(5):
    logger.info(f"Training Fold {fold_idx}...")
    
    train_idx = fold_metadata[fold_idx]['train_indices']
    val_idx = fold_metadata[fold_idx]['val_indices']
    
    # Get fold data
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    # ... rest of training code ...
    
    # Save checkpoints
    torch.save(checkpoint, f'checkpoints/fold_{fold_idx}/epoch_{epoch:03d}.pth')
```

**Output:** checkpoints directory with all epochs

---

### **Notebook 03: Validation & Checkpoint Selection**
**Time:** ~30 minutes  
**What:** Evaluate all checkpoints, select best

```
Input: All checkpoints from Notebook 02
Output:
  - final_models/fold_*.pth (best per fold)
  - validation_metrics.csv
  - checkpoint_selection.json
```

**Key cells:**
```python
# Cell 1: Load checkpoint and validate
checkpoint = torch.load(f'checkpoints/fold_{fold_idx}/epoch_{epoch:03d}.pth')
model.load_state_dict(checkpoint['model_state_dict'])

metrics = evaluate_checkpoint(model, X_val, y_val)
gen_score = metrics['val_acc'] - abs(metrics['train_acc'] - metrics['val_acc'])

# Cell 2: Select best per fold (by Gen_Score)
best_checkpoints = select_best_checkpoints(validation_results)

# Cell 3: Copy to final_models
for fold_idx, ckpt_info in best_checkpoints.items():
    shutil.copy(
        ckpt_info['checkpoint_path'],
        f'final_models/fold_{fold_idx}_best.pth'
    )
```

---

### **Notebook 04: Inference & Submission**
**Time:** ~10 minutes  
**What:** Generate test predictions, create submission

```
Input: X_test, final_models/fold_*.pth
Output:
  - submission.csv (to Kaggle!)
  - submission_metadata.json
  - prediction_confidence.csv
```

**Key cells:**
```python
# Cell 1: Ensemble inference
all_probs = []
for fold_idx in range(5):
    model = load_model(f'final_models/fold_{fold_idx}_best.pth')
    probs = model.predict(X_test)
    all_probs.append(probs)

ensemble_probs = np.mean(all_probs, axis=0)
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# Cell 2: Format submission CSV
submission_df = test_df[['ID']].copy()
submission_df['TARGET'] = ensemble_preds
submission_df.to_csv('submission.csv', index=False)

# Cell 3: Verify
assert len(submission_df) == 3000
assert submission_df['TARGET'].isin(range(10)).all()
logger.info("✓ Submission ready!")
```

---

### **Notebook 05: Analysis & Insights (Optional)**
**Time:** ~1 hour  
**What:** Deep dive into what model learned

```
Optional cells:
- Analyze prediction confidence per class
- Find hard examples (low confidence)
- Compare models' predictions
- Visualize learned features
```

---

## Directory Structure After All Notebooks

```
project/
├── docs/                          # (Documentation you created)
├── Data/
│   ├── Training/                  # 7000 training images
│   ├── Test/                      # 3000 test images
│   ├── training.csv
│   ├── test.csv
│   └── sources.txt
├── processed/                     # Created by Notebook 01
│   ├── X_train.npy               # (7000, 224, 224, 3)
│   ├── X_test.npy                # (3000, 224, 224, 3)
│   ├── train_metadata.parquet
│   ├── test_metadata.parquet
│   └── fold_metadata.json
├── checkpoints/                   # Created by Notebook 02
│   ├── fold_0/
│   │   ├── epoch_000.pth
│   │   ├── epoch_001.pth
│   │   └── ... (100 epochs)
│   ├── fold_1/
│   └── ... (5 folds total)
├── logs/                          # Created by Notebook 02
│   ├── training_log_fold_0.json
│   ├── training_log_fold_1.json
│   └── ... (5 folds)
├── final_models/                  # Created by Notebook 03
│   ├── fold_0_best.pth
│   ├── fold_1_best.pth
│   └── ... (5 folds)
├── outputs/                       # Created by Notebooks 00, 03, 04
│   ├── eda/
│   │   ├── eda_report.html
│   │   ├── eda_insights.json
│   │   └── plots/
│   ├── validation/
│   │   ├── validation_metrics.csv
│   │   └── validation_report.html
│   └── inference/
│       ├── submission.csv         # ← Upload to Kaggle!
│       ├── submission_metadata.json
│       └── prediction_confidence.csv
└── Notebook_00_*.ipynb             # Your notebooks
    Notebook_01_*.ipynb
    Notebook_02_*.ipynb
    Notebook_03_*.ipynb
    Notebook_04_*.ipynb
```

---

## Timeline Estimate

| Notebook | Task | Time |
|----------|------|------|
| 00 | EDA & Data Loading | 30 min |
| 01 | Preprocessing & Splits | 20 min |
| 02 | Training (first model) | 3-4 hrs |
| 03 | Validation & Selection | 30 min |
| 04 | Inference & Submission | 10 min |
| **Total** | **First complete submission** | **~5 hours** |

**Then:**
- Train 2nd model architecture (Notebook 02b): +3-4 hrs
- Ensemble both models (Notebook 04b): +10 min
- Further optimization: +2-4 hrs each

---

## Key Principles to Follow

### 1. **Immutable Data Flow**
```
Raw Data → Notebook 00 → Preprocessed Data → Notebook 02 → ...

Each notebook outputs files the next notebook reads.
Never modify source files.
```

### 2. **Logging Everything**
```python
# Every notebook should log:
logger = logging.getLogger(__name__)

logger.info(f"Loaded {len(X_train)} training samples")
logger.info(f"Shape: {X_train.shape}")
logger.info(f"Dtype: {X_train.dtype}")
```

### 3. **Checkpointing Progress**
```python
# Save intermediate results
train_df.to_parquet('processed/train_metadata.parquet')
X_train = np.load('processed/X_train.npy')  # Don't recompute
```

### 4. **Reproducibility**
```python
# Set seed everywhere
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

---

## First Submission Checklist

- [ ] Notebook 00: EDA complete, data validated
- [ ] Notebook 01: Preprocessing done, X_train/X_test saved
- [ ] Notebook 02: At least 1 model trained (10 epochs for quick test)
- [ ] Notebook 03: Best checkpoint selected
- [ ] Notebook 04: submission.csv created and verified
- [ ] submission.csv format correct (ID, TARGET columns)
- [ ] submission.csv has 3000 rows
- [ ] All TARGET values in [0-9]
- [ ] Upload to Kaggle!

---

## Expected Performance (Rough Estimates)

| Model | Expected CV Acc | Expected LB Acc |
|-------|---|---|
| EfficientNet-B4 (1 fold) | 70-75% | 65-70% |
| EfficientNet-B4 (5-fold avg) | 72-77% | 70-72% |
| EfficientNet-B5 (5-fold) | 73-78% | 71-73% |
| Vision Transformer (5-fold) | 74-79% | 72-74% |
| Ensemble (3 diverse models) | 75-80% | 73-75% |

*Actual results depend on hyperparameters, augmentation, post-processing robustness*

---

## Optimization After First Submission

1. **Hyperparameter Tuning**
   - Learning rate: Try [5e-5, 1e-4, 2e-4, 5e-4]
   - Batch size: Try [16, 32, 64]
   - Augmentation strength: Adjust rotation, brightness ranges

2. **Data Strategy**
   - Add test-time augmentation (TTA)
   - Implement mixup or cutmix augmentation
   - Try different preprocessing (e.g., crop/pad to 256×256)

3. **Model Strategy**
   - Train ViT-Base (modern, robust)
   - Try larger EfficientNet (B5, B6)
   - Ensemble multiple models

4. **Ensemble Strategy**
   - Use weighted ensemble (by CV performance)
   - Try stacking (meta-model on top)
   - Snapshot ensemble (multiple epochs per model)

5. **Robustness**
   - Add post-processing simulation in augmentation
   - Test against JPEG artifacts specifically
   - Analyze confusion matrix for weakest pairs

---

## Success Metrics

**Target benchmarks:**
- ✅ First submission: 65%+ accuracy
- ✅ Optimized single model: 72-75%
- ✅ Ensemble: 74-78%
- ✅ Top competitive score: 78%+

**Progress tracking:**
- Track CV accuracy across folds
- Monitor generalization gap (should be < 5%)
- Compare LB (public) score to CV estimate

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size to 16 or 8 |
| Training too slow | Use AMP (automatic mixed precision), reduce image size to 192×192 |
| Low CV accuracy | Check preprocessing (maybe removing normalization?), verify labels |
| Overfitting (high gap) | Add more augmentation, use dropout/weight decay |
| Submission rejected | Check format: exactly 2 columns (ID, TARGET), no header issues |

---

## Next Steps After This Design

1. **Start Notebook 00** - Load data, verify, run EDA
2. **Review EDA results** - Understand what you're solving
3. **Run Notebook 01** - Preprocess and split
4. **Run Notebook 02** - Train (start small, 5 epochs to test)
5. **Run Notebook 03-04** - Validate and submit
6. **Analyze results** - See where to improve
7. **Iterate** - Train more models, optimize

**Happy competition! 🚀**
