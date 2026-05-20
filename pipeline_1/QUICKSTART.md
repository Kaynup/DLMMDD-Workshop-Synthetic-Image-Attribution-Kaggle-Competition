# Quick Start Guide - Pipeline 1

## ⚡ 5-Minute Setup

### 1. Install Python & Dependencies
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Data Structure
Your `Data/` directory should exist at the parent level:
```
../
├── Data/
│   ├── Training/         # 7,000 images
│   ├── Test/            # 3,000 images
│   ├── training.csv     # Labels
│   ├── test.csv         # Test split
│   └── sources.txt      # Class names
└── pipeline_1/          # This project
```

### 3. Run the Pipeline
```bash
# Navigate to pipeline_1
cd pipeline_1

# Start Jupyter
jupyter notebook

# Open notebooks in this order:
# 1. 00_data_loading_validation_eda.ipynb         (~5 min)
# 2. 01_preprocessing_augmentation_splits.ipynb   (~15 min)
# 3. 02_model_training.ipynb                      (~2-4 hours)
# 4. 03_validation_checkpoint_selection.ipynb     (~5 min)
# 5. 04_inference_submission.ipynb                (~10 min)
```

---

## 📊 Expected Timeline

| Stage | Duration | Description |
|-------|----------|-------------|
| 00 | ~5 min | Load data, verify, analyze |
| 01 | ~15 min | Preprocess, augment, split |
| 02 | ~2-4 hrs | Train models on 5 folds |
| 03 | ~5 min | Select best checkpoints |
| 04 | ~10 min | Inference & submission |
| **Total** | **~2.5-4.5 hrs** | **Full pipeline** |

---

## 🔑 Key Configuration Options

Edit `config.toml` to customize:

```toml
# Model choice
[model]
architecture = "efficientnet_b4"  # Options: resnet50, vit_base, densenet121

# Training
[training]
num_epochs = 100
batch_size = 32          # Reduce if CUDA OOM
learning_rate = 1e-4

# Data augmentation
[augmentation]
enabled = true           # Set to false to disable augmentation

# Cross-validation
[train_val_split]
num_splits = 5           # 5-fold is recommended
```

---

## 💡 Tips for Better Results

1. **More Epochs:** Increase `num_epochs` to 150-200 for better convergence
2. **Larger Batch Size:** If GPU memory allows, use `batch_size = 64` for stability
3. **Better Model:** Try `efficientnet_b7` or `vit_base_patch16_224` for higher accuracy
4. **Lower Learning Rate:** Use `1e-5` for fine-tuning, `1e-4` for training from scratch
5. **Ensemble:** Train multiple architectures and average predictions in Stage 05

---

## 🐛 Common Issues & Fixes

### Issue: "CUDA out of memory"
```toml
[training]
batch_size = 16  # Reduce from 32
```

### Issue: "Cannot find Data directory"
```bash
# Make sure you're in pipeline_1/ directory
# and Data/ exists at ../Data/
ls ../Data/training.csv
```

### Issue: "Module not found" errors
```bash
# Ensure you're in pipeline_1/ when running notebooks
cd pipeline_1
jupyter notebook
```

### Issue: "Permission denied" on Linux/Mac
```bash
chmod +x notebooks/*.ipynb
```

---

## 📈 Monitoring Training

**During Training (Notebook 02):**
- Watch accuracy improve each epoch
- Loss should steadily decrease
- If loss explodes, reduce learning rate

**Check Logs:**
```bash
tail -f logs/pipeline.log
tail -f logs/training_fold_0.log
```

---

## 🎯 Success Indicators

✅ **Stage 00:** Shows 7,000 training samples with balanced classes  
✅ **Stage 01:** Preprocesses ~500 images/sec, creates 5 folds  
✅ **Stage 02:** Trains at ~10-50 iterations/sec, saves checkpoints  
✅ **Stage 03:** Selects best epoch per fold  
✅ **Stage 04:** Generates `submission_final.csv` with 3,000 predictions  

---

## 📝 Output Files

After running all stages, you'll have:

```
pipeline_1/
├── processed/
│   ├── train_metadata.parquet     # Full training metadata
│   ├── test_metadata.parquet      # Full test metadata
│   ├── X_train.npy                # Preprocessed training images (7000, 3, 224, 224)
│   ├── X_test.npy                 # Preprocessed test images (3000, 3, 224, 224)
│   ├── y_train.npy                # Training labels (7000,)
│   ├── fold_metadata.json         # 5-fold assignments
│   └── source_mapping.json        # Class ID → Generator name
│
├── checkpoints/
│   ├── fold_0/
│   │   ├── epoch_000.pth          # Checkpoint from epoch 0
│   │   ├── epoch_001.pth
│   │   └── ... (up to epoch 99)
│   ├── fold_1/
│   └── ... (fold 2-4)
│
├── logs/
│   ├── pipeline_00.log            # Data loading logs
│   ├── pipeline_01.log            # Preprocessing logs
│   ├── training_fold_0.log        # Training logs
│   ├── training_fold_1.log
│   └── ...
│
└── submissions/
    ├── submission_final.csv       # Final predictions
    └── submission_ensemble.csv    # Ensemble predictions (if Stage 05 run)
```

---

## 🚀 Next Steps

1. **Submit:** Upload `submission_final.csv` to Kaggle competition
2. **Iterate:** Adjust config and retrain to improve score
3. **Ensemble:** Train multiple models and combine in Stage 05
4. **Postprocessing:** Add TTA (Test-Time Augmentation) for robustness

---

## 📚 Resources

- **Competition:** [DLMMDD 2026](https://icann2026.org/)
- **PyTorch Docs:** https://pytorch.org/docs/stable/index.html
- **TorchVision Models:** https://pytorch.org/vision/stable/models.html
- **EfficientNet:** https://github.com/lukemelas/EfficientNet-PyTorch

---

**Good luck with the competition! 🎉**
