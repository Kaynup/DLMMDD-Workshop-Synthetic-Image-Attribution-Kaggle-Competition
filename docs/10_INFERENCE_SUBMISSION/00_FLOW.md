# Pipeline 10: Inference & Submission

## Flow Diagram

```
Input: X_test (preprocessed)
       final_models/fold_*.pth (5 best checkpoints)
       test_df (metadata, IDs)
       ↓
┌──────────────────────────────────────┐
│ For Each Best Checkpoint:            │
│ - Load model                         │
│ - Forward pass on test set           │
│ - Get prediction probabilities       │
└────────────┬─────────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Ensemble Predictions        │
      │ avg_probs = mean(all_probs) │
      │ pred = argmax(avg_probs)    │
      │ confidence = max(avg_probs) │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Format Submission CSV       │
      │ - ID | TARGET               │
      │ - Columns: ID from test_df  │
      │ - TARGET: predicted class   │
      │ - No headers required       │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Save Metadata               │
      │ - Prediction confidence     │
      │ - Per-model predictions     │
      │ - Ensemble weights          │
      │ - Inference time            │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Output:                     │
      │ - submission.csv (required) │
      │ - submission_metadata.json  │
      │ - prediction_confidence.csv │
      └──────────────────────────────┘
```

## Key Steps

### 1. **Load Test Data**
```python
X_test, test_df = load_and_preprocess_test(test_df, config)
# Shape: (3000, 224, 224, 3)
```

### 2. **Ensemble Prediction (Fold-Based)**
```python
all_predictions = []

for fold_idx in range(5):
    model = load_checkpoint(f'final_models/fold_{fold_idx}_best.pth')
    probs = model.predict(X_test)  # shape: (3000, 10)
    all_predictions.append(probs)

ensemble_probs = np.mean(all_predictions, axis=0)  # Average probabilities
ensemble_preds = np.argmax(ensemble_probs, axis=1)  # shape: (3000,)
```

### 3. **Format Submission CSV**
```csv
ID,TARGET
7000,2
7001,5
7002,1
...
9999,8
```

**Critical:** Row order must match test.csv!

## Code Pattern

```python
def inference_and_submit(test_df, X_test, final_models_dir, output_dir):
    """Main inference pipeline."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("Starting inference on test set...")
    
    # === Stage 1: Collect predictions from all folds ===
    all_probs = []
    all_preds = []
    
    for fold_idx in range(5):
        logger.info(f"Fold {fold_idx}: Loading and predicting...")
        
        ckpt_path = Path(final_models_dir) / f'fold_{fold_idx}_best.pth'
        
        # Load model
        model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=10)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Predict
        fold_probs = []
        with torch.no_grad():
            for i in range(0, len(X_test), 32):
                X_batch = torch.FloatTensor(X_test[i:i+32])
                X_batch = X_batch.permute(0, 3, 1, 2).to(device)  # (B, H, W, C) → (B, C, H, W)
                
                logits = model(X_batch)
                probs = torch.softmax(logits, dim=1)
                fold_probs.append(probs.cpu().numpy())
        
        fold_probs = np.vstack(fold_probs)  # shape: (3000, 10)
        all_probs.append(fold_probs)
        logger.info(f"  Fold {fold_idx} predictions shape: {fold_probs.shape}")
    
    # === Stage 2: Ensemble predictions ===
    logger.info("Ensembling predictions across folds...")
    ensemble_probs = np.mean(all_probs, axis=0)  # shape: (3000, 10)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)  # shape: (3000,)
    ensemble_confidence = np.max(ensemble_probs, axis=1)  # shape: (3000,)
    
    logger.info(f"Ensemble predictions shape: {ensemble_preds.shape}")
    logger.info(f"Average confidence: {ensemble_confidence.mean():.4f}")
    
    # === Stage 3: Format submission ===
    logger.info("Formatting submission CSV...")
    submission_df = test_df[['ID']].copy()
    submission_df['TARGET'] = ensemble_preds
    
    # Verify row alignment
    assert len(submission_df) == 3000, "Submission must have 3000 rows"
    assert submission_df['ID'].min() >= 7000, "IDs should start from 7000"
    
    # === Stage 4: Save outputs ===
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save submission CSV (CRITICAL: exact format required by Kaggle)
    submission_path = output_path / 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'num_models': 5,
        'ensemble_type': 'average_probability',
        'avg_confidence': float(ensemble_confidence.mean()),
        'min_confidence': float(ensemble_confidence.min()),
        'max_confidence': float(ensemble_confidence.max()),
        'num_predictions': len(ensemble_preds),
        'class_distribution': np.bincount(ensemble_preds).tolist()
    }
    
    with open(output_path / 'submission_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save confidence scores for analysis
    confidence_df = test_df[['ID']].copy()
    confidence_df['predicted_class'] = ensemble_preds
    confidence_df['confidence'] = ensemble_confidence
    confidence_df.to_csv(output_path / 'prediction_confidence.csv', index=False)
    
    logger.info("Inference complete!")
    return submission_df, metadata


def verify_submission(submission_df, test_df):
    """Verify submission format before upload."""
    
    checks = {
        'num_rows': len(submission_df) == 3000,
        'required_columns': set(submission_df.columns) == {'ID', 'TARGET'},
        'id_matches': (submission_df['ID'].sort_values().values == test_df['ID'].sort_values().values).all(),
        'valid_classes': submission_df['TARGET'].isin(range(10)).all(),
        'no_missing_values': not submission_df.isnull().any().any()
    }
    
    logger.info("Submission Verification:")
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {check_name}")
    
    if not all(checks.values()):
        raise ValueError("Submission verification failed!")
    
    logger.info("✓ Submission format is valid!")
    return checks
```

## Submission File Format

**CRITICAL: Exact format required by Kaggle**

```csv
ID,TARGET
7000,2
7001,5
7002,1
...
9999,8
```

**Rules:**
- CSV format (text file)
- Two columns: `ID` and `TARGET`
- Header row MUST be present
- No extra spaces or quotes
- IDs can be in any order (Kaggle will match by ID)
- TARGET values: integers 0-9 (generator class)

## Metadata Output Example

```json
{
  "timestamp": "2026-05-19T12:00:00",
  "num_models": 5,
  "ensemble_type": "average_probability",
  "avg_confidence": 0.6234,
  "min_confidence": 0.1234,
  "max_confidence": 0.9876,
  "num_predictions": 3000,
  "class_distribution": [320, 310, 315, 305, 315, 310, 320, 315, 310, 300]
}
```

## Prediction Confidence CSV

```csv
ID,predicted_class,confidence
7000,2,0.89
7001,5,0.72
7002,1,0.65
...
9999,8,0.83
```

**Use confidence scores to:**
- Identify uncertain predictions
- Analyze per-class confidence
- Compare different models

## Ensemble Strategy Options

### Strategy 1: Average Probabilities (Recommended)
```python
ensemble_probs = np.mean([fold_0_probs, fold_1_probs, ...], axis=0)
```
**Pro:** Simple, often works best
**Con:** Assumes equal fold quality

### Strategy 2: Weighted Average
```python
weights = [val_acc_fold_0, val_acc_fold_1, ...]
ensemble_probs = np.average([...], axis=0, weights=weights)
```
**Pro:** Uses fold quality information
**Con:** More complex, marginal improvement

### Strategy 3: Voting
```python
predictions = [fold_0_pred, fold_1_pred, ...]
ensemble_pred = np.bincount(predictions, axis=0).argmax(axis=1)
```
**Pro:** Robust to outliers
**Con:** Loses probability information

## Performance Analysis

**Post-submission analysis:**
```python
# After getting public LB score:
- Compare LB score to estimated test score
- Analyze if post-processing hurt some classes more
- Check confidence distribution per predicted class
- Review hard examples (low confidence)
```

## Dependencies
- `torch` - Model loading
- `timm` - Model architecture
- `numpy` - Array operations
- `pandas` - DataFrame formatting

## Integration
**Inputs:** X_test, test_df, final_models/
**Outputs:** submission.csv (to Kaggle)
**Outputs:** submission_metadata.json, prediction_confidence.csv
