# PROJECT SUMMARY & QUICK START

## 🎯 Competition at a Glance

| Aspect | Details |
|--------|---------|
| **Task** | 10-class image classification (synthetic image attribution) |
| **Data** | 7,000 training + 3,000 test face images (1024×1024) |
| **Classes** | 10 text-to-image generators (AuraFlow, Freepik, Lumina, ..., Tencent Hunyuan) |
| **Metric** | Classification Accuracy (assume F1-Macro for validation) |
| **Challenge** | Test images have 1-3 random post-processing ops (JPEG, crop, blur, etc.) |
| **Leaderboard** | 50% public / 50% private (released after competition ends) |

---

## 📊 Data Overview

### Training Set (7,000 images)
- **Balanced:** Exactly 1,000 samples per generator
- **Format:** PNG, RGB, ~1024×1024 pixels
- **Avg File Size:** ~250 KB
- **Access:** All labels provided (train.csv with ID, path, y)

### Test Set (3,000 images)
- **Unlabeled:** No ground truth provided
- **Post-Processing:** Each image has 1-3 transformations applied
- **Challenge:** Must be robust to:
  - JPEG/WebP compression
  - Cropping and resizing
  - Rotation with crop
  - Contrast/brightness adjustment
  - Blur, grayscale conversion
  - AI super-resolution
  - Combination of above

---

## 🔧 Solution Architecture (SOA Pipelines)

Your solution will follow **11 interconnected pipelines**, executed through **5-6 Jupyter notebooks**:

### Pipeline Sequence

```
Notebook 00: Data Loading + EDA
  ├─ 01_Data_Loading_Validation    [Read CSVs, verify integrity, image stats]
  └─ 02_EDA                        [Analyze distributions, generator fingerprints]
                    │
                    ▼
Notebook 01: Preprocessing + Splits
  ├─ 03_Preprocessing              [Normalize 224×224, ImageNet stats]
  ├─ 04_Augmentation               [Rotation, flip, brightness, blur, etc.]
  └─ 05_Train_Val_Split            [5-fold StratifiedKFold, save fold_metadata]
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
Notebook 02a/b/c: Model Training (per architecture)
  ├─ 06_Feature_Extraction         [Optional: ResNet features for baseline]
  └─ 07_Model_Training             [Train EfficientNet/ViT per fold, save all epochs]
                    │
                    ▼
Notebook 03: Validation & Selection
  ├─ 08_Validation                 [Evaluate all checkpoints, compute Gen_Score]
  └─ 09_Checkpoint_Selection       [Select best epoch per fold (by Gen_Score)]
                    │
                    ▼
Notebook 04: Inference
  └─ 10_Inference_Submission       [Ensemble 5-fold predictions → submission.csv]

(Optional)
Notebook 05: Ensemble
  └─ 11_Ensemble                   [Combine multiple model architectures]
```

---

## 📈 Key Formulas

### Generalization Score (Checkpoint Selection)

$$\text{Gen\_Score} = \text{val\_acc} - |\text{train\_acc} - \text{val\_acc}|$$

**Interpretation:**
- Balances validation accuracy with generalization gap
- High Gen_Score = good validation + minimal overfitting
- Use for selecting best checkpoint per fold

**Example:**
- Epoch 50: train=0.95, val=0.92 → Gen = 0.92 - 0.03 = **0.89** ✓ (good generalization)
- Epoch 75: train=0.98, val=0.91 → Gen = 0.91 - 0.07 = **0.84** ✗ (overfitting)

### Cross-Validation Metrics

**Per Fold:**
- Accuracy = (# correct) / (# total)
- F1-Macro = Average F1 across 10 classes
- Per-class precision, recall

**Aggregated (5-fold CV):**
- Mean accuracy ± std across folds
- Report: "72.5% ± 1.2% accuracy on 5-fold CV"

---

## 📁 Pipeline Documentation Structure

```
docs/
├── 00_PIPELINE_OVERVIEW.md              ← START HERE
├── EXECUTION_GUIDE.md                   ← Step-by-step notebook guide
├── 01_DATA_LOADING_VALIDATION/
│   ├── 00_FLOW.md                       [Flow diagram & operations]
│   ├── 01_DESIGN.md                     [Architecture & patterns]
│   ├── 02_CODE_ANALOGIES.md             [Pseudocode examples]
│   └── 03_DEPENDENCIES.md               [Libraries, inputs, outputs]
├── 02_EDA/
│   ├── 00_FLOW.md
│   ├── 01_DESIGN.md
│   ├── 02_CODE_ANALOGIES.md
│   └── 03_DEPENDENCIES.md
├── 03_PREPROCESSING/
│   └── 00_FLOW.md                       [Resize, normalize to 224×224]
├── 04_AUGMENTATION/
│   └── 00_FLOW.md                       [Train-time augmentation strategy]
├── 05_TRAIN_VAL_SPLIT/
│   └── 00_FLOW.md                       [StratifiedKFold 5-fold]
├── 06_FEATURE_EXTRACTION/
│   └── 00_FLOW.md                       [Optional: ResNet features]
├── 07_MODEL_TRAINING/
│   └── 00_FLOW.md                       [Core: EfficientNet/ViT training loop]
├── 08_VALIDATION/
│   └── 00_FLOW.md                       [Evaluate all checkpoints, metrics]
├── 09_CHECKPOINT_SELECTION/
│   └── 00_FLOW.md                       [Select best by Gen_Score]
├── 10_INFERENCE_SUBMISSION/
│   └── 00_FLOW.md                       [Ensemble & generate submission.csv]
└── 11_ENSEMBLE/
    └── 00_FLOW.md                       [Optional: Multi-model ensemble]
```

**Reading Guide:**
1. Start with **EXECUTION_GUIDE.md** (how to organize notebooks)
2. Read **00_PIPELINE_OVERVIEW.md** (big picture)
3. For each pipeline, read in order: **00_FLOW → 01_DESIGN → 02_CODE → 03_DEPENDENCIES**

---

## 🚀 Quick Start (5-Minute Overview)

### What You'll Do

```python
# Notebook 00: Load & analyze
train_df = load_data()  # 7000 images
plot_eda()              # Understand distributions

# Notebook 01: Preprocess
X_train = preprocess_images()  # 224×224, normalized
create_cv_splits()             # 5-fold StratifiedKFold

# Notebook 02: Train
for fold in range(5):
    model = EfficientNetB4()
    train(model, X_train[fold])
    save_checkpoints()  # ALL epochs

# Notebook 03: Select best
best_epoch = select_by_generalization_score()
copy_to_final_models()

# Notebook 04: Submit
ensemble_predictions = average_5_folds()
submission = format_csv(ensemble_predictions)
submission.to_csv('submission.csv')  # Upload to Kaggle!
```

### Expected Timeline

| Stage | Time | Output |
|-------|------|--------|
| Data Loading + EDA | 30 min | eda_report.html |
| Preprocessing + Splits | 20 min | X_train.npy, fold_metadata.json |
| Model Training (1st) | 3-4 hrs | checkpoints/fold_*/epoch_*.pth |
| Validation + Selection | 30 min | final_models/fold_*_best.pth |
| Inference + Submission | 10 min | **submission.csv** ✓ |
| **Total (1st submission)** | **~5 hours** | **Ready to submit!** |

---

## 💡 Key Design Decisions Explained

### 1. Why 5-Fold Cross-Validation?

| Strategy | Pros | Cons |
|----------|------|------|
| **No CV** (train/val split) | Fast, simple | Unstable estimates, high variance |
| **5-Fold CV** | Good balance of stability & speed | Moderate compute |
| **10-Fold CV** | More stable estimates | Slower (10× more training) |

**Recommendation:** Start with 5-fold for speed, use 10-fold for final submission if time permits.

### 2. Why Generalization Score for Checkpoint Selection?

**Problem:** Using highest validation accuracy can lead to overfitting
- Model A: train=0.99, val=0.80 → overfitted (gap=0.19)
- Model B: train=0.92, val=0.90 → well-generalized (gap=0.02)
- Max(val_acc) picks A (wrong!)

**Solution:** Generalization Score balances both
- Model A: 0.80 - 0.19 = **0.61** (penalized for overfitting)
- Model B: 0.90 - 0.02 = **0.88** (rewarded for generalization)
- Gen_Score picks B (correct!) ✓

### 3. Why Fold-Based Ensemble Instead of Single Model?

**Advantages:**
- Better utilization of training data (different validation sets)
- More robust to random seed variations
- Natural ensemble (average 5 models)

**Result:** 72% single model → 73-74% with 5-fold ensemble

### 4. Why All Checkpoints Instead of Just Best?

**Benefit:** Analyze training dynamics
- See if early stopping would help
- Detect overfitting patterns
- Compute validation curve plots
- Flexible selection strategy

**Cost:** Disk space (5 folds × 100 epochs × ~500MB = 250GB) ← use compression!

---

## ✅ Success Criteria

Your solution should:

- ✅ **Data Integrity:** Verify all 10K images loadable, no missing classes
- ✅ **EDA Insights:** Understand class distributions, post-processing impact
- ✅ **Reproducibility:** Fixed seed, logged hyperparameters, version control
- ✅ **Proper Validation:** 5-fold CV, stratified splits, per-class metrics
- ✅ **Checkpoint Management:** Save all epochs, metadata per checkpoint
- ✅ **Smart Selection:** Use Generalization Score, not just max accuracy
- ✅ **Logging:** Every epoch logged (loss, metrics, LR, checkpoint path)
- ✅ **Ensemble:** Combine 5-fold predictions (average probabilities)
- ✅ **Submission Format:** Correct CSV (ID, TARGET columns, 3000 rows)

---

## 📊 Expected Performance Benchmarks

| Approach | CV Accuracy | Expected LB | Notes |
|----------|---|---|---|
| Random baseline | 10% | 10% | Predict any class equally |
| Simple CNN | 55-60% | 50-55% | Without proper training |
| EfficientNet-B4 (proper training) | 72-76% | 70-72% | Good baseline |
| EfficientNet-B5 + ViT ensemble | 75-80% | 73-76% | Strong solution |
| Optimized ensemble (3+ models) | 78-82% | 76-79% | Competitive |

**Note:** Real scores depend heavily on:
- Hyperparameter tuning
- Augmentation effectiveness
- Post-processing robustness (hardest part!)
- Training time investment

---

## 🎓 Learning Outcomes

By implementing this solution, you'll master:

1. **End-to-end ML Pipeline**
   - Data loading → Training → Validation → Submission

2. **Proper Cross-Validation**
   - Why stratification matters
   - How to aggregate results
   - Generalization gap analysis

3. **Deep Learning Best Practices**
   - Checkpoint management
   - Metric computation
   - Logging & reproducibility

4. **Computer Vision Fundamentals**
   - Image preprocessing (normalization, resizing)
   - Data augmentation strategies
   - Transfer learning (pretrained models)

5. **Robustness to Adversarial Effects**
   - Post-processing simulation
   - Ensemble methods
   - Domain shift handling

---

## 🔗 Document Navigation

- **For complete pipeline overview:** [00_PIPELINE_OVERVIEW.md](00_PIPELINE_OVERVIEW.md)
- **For notebook-by-notebook guide:** [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)
- **For data loading details:** [01_DATA_LOADING_VALIDATION/](01_DATA_LOADING_VALIDATION/)
- **For EDA strategy:** [02_EDA/](02_EDA/)
- **For training details:** [07_MODEL_TRAINING/](07_MODEL_TRAINING/)
- **For inference/submission:** [10_INFERENCE_SUBMISSION/](10_INFERENCE_SUBMISSION/)

---

## ❓ FAQ

**Q: Should I train multiple models or optimize one?**  
A: Start with one architecture (EfficientNet-B4), get baseline, then add ViT if time permits. Ensemble is powerful.

**Q: How much data augmentation?**  
A: Moderate - rotate ±5°, flip H, brightness 0.8-1.2x. Test-time augmentation helps more than train-time for post-processing robustness.

**Q: What if my CV score is 60%? Is that bad?**  
A: Not necessarily! Post-processing is hard. Check if all classes learn equally. 60% + good generalization is better than 75% + overfitting.

**Q: Can I use pre-computed features (ResNet)?**  
A: Yes! Pipeline 06 shows how. But end-to-end fine-tuning usually works better for this competition.

**Q: How do I handle post-processing robustness?**  
A: Add post-processing simulation to augmentation (JPEG quality variations, crop, blur). Analyze confusion matrix for weak generator pairs.

---

## 📝 Checklist Before First Submission

- [ ] Read EXECUTION_GUIDE.md
- [ ] Understand all 11 pipelines (at least their purpose)
- [ ] Create Notebook 00, run EDA
- [ ] Create Notebook 01, verify X_train shape is (7000, 224, 224, 3)
- [ ] Create Notebook 02, train 5 epochs as test
- [ ] Create Notebook 03-04, generate submission.csv
- [ ] Verify submission format: 3000 rows, ID + TARGET columns
- [ ] Submit to Kaggle!
- [ ] Review LB feedback
- [ ] Optimize based on results

---

**Ready to build? Start with Notebook 00 in EXECUTION_GUIDE.md! 🚀**
