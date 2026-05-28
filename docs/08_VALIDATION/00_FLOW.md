# Pipeline 08: Validation

## Flow Diagram

```
Input: All saved checkpoints (all epochs, all folds)
       Training logs (metrics per epoch)
       val_data (from fold splits)
       ↓
┌──────────────────────────────────────┐
│ For Each Fold:                       │
│ - Load all saved checkpoints         │
│ - Evaluate each on validation set    │
│ - Compute metrics per checkpoint     │
└────────────┬─────────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Compute Metrics Per Epoch:  │
      │ - Accuracy (overall)        │
      │ - F1-Macro (per-class avg)  │
      │ - Per-class precision       │
      │ - Per-class recall          │
      │ - Confusion matrix          │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Compute Generalization Score│
      │ Gen = val_acc -             │
      │       |train_acc - val_acc| │
      │ (Best: minimal overfitting) │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Aggregate Across Folds:     │
      │ - Per-epoch metrics         │
      │ - Fold-wise aggregation     │
      │ - Mean ± Std across folds   │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Generate Validation Report: │
      │ - Per-fold best epochs      │
      │ - Overall best epoch        │
      │ - Generalization gap plot   │
      │ - Per-class confusion       │
      │ - Metrics table             │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Output:                     │
      │ - validation_report.html    │
      │ - validation_metrics.csv    │
      │ - per_fold_metrics.json     │
      └──────────────────────────────┘
```

## Key Metrics

### 1. **Accuracy**
```
Accuracy = (# correct predictions) / (# total predictions)
Range: [0, 1]
Interpretation: Overall correctness
```

### 2. **F1-Macro**
```
F1-Macro = (1/n_classes) * Σ F1_i

Where F1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)

Range: [0, 1]
Interpretation: Average class performance (handles class imbalance)
Note: Since dataset is balanced, F1-Macro ≈ Accuracy, but F1-Macro is more robust
```

### 3. **Generalization Score (Custom)**
```
Gen_Score = val_acc - |train_acc - val_acc|

Intuition:
- If train=0.95, val=0.90 → gap=0.05 → Gen = 0.90 - 0.05 = 0.85 (good)
- If train=0.99, val=0.80 → gap=0.19 → Gen = 0.80 - 0.19 = 0.61 (overfitting)

Use for: Selecting checkpoints that generalize well (not just highest val_acc)
```

### 4. **Per-Class Metrics**
```python
For each class i:
  - Precision_i = TP_i / (TP_i + FP_i)
  - Recall_i = TP_i / (TP_i + FN_i)
  - F1_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)

Interpretation: Which generators are hardest to distinguish?
```

## Code Pattern

```python
def evaluate_checkpoint(checkpoint_path, X_val, y_val, device='cuda'):
    """Evaluate single checkpoint on validation set."""
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path)
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=10)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Forward pass
    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(X_val), 32):
            X_batch = torch.FloatTensor(X_val[i:i+32]).to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = (all_preds == y_val).mean()
    f1_macro = f1_score(y_val, all_preds, average='macro')
    
    # Per-class metrics
    per_class_precision = precision_score(y_val, all_preds, average=None)
    per_class_recall = recall_score(y_val, all_preds, average=None)
    per_class_f1 = f1_score(y_val, all_preds, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, all_preds, labels=range(10))
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs
    }


def validate_all_folds(fold_metadata, X_train, y_train, checkpoint_dir):
    """Validate all checkpoints across all folds."""
    
    validation_results = {}
    
    for fold_idx in range(len(fold_metadata)):
        logger.info(f"\nValidating Fold {fold_idx}...")
        
        # Get validation data
        val_idx = fold_metadata[fold_idx]['val_indices']
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        # Get training data for generalization gap
        train_idx = fold_metadata[fold_idx]['train_indices']
        y_train_fold = y_train[train_idx]
        
        fold_results = {}
        
        # Evaluate all epochs
        ckpt_dir = Path(checkpoint_dir) / f'fold_{fold_idx}'
        for ckpt_file in sorted(ckpt_dir.glob('epoch_*.pth')):
            epoch = int(ckpt_file.stem.split('_')[1])
            
            metrics = evaluate_checkpoint(str(ckpt_file), X_val, y_val)
            
            # Load training metrics from checkpoint
            ckpt = torch.load(ckpt_file)
            train_acc = ckpt['metrics']['train_acc']
            val_acc = metrics['accuracy']
            
            # Compute generalization score
            gen_score = val_acc - abs(train_acc - val_acc)
            
            fold_results[epoch] = {
                'val_accuracy': val_acc,
                'val_f1_macro': metrics['f1_macro'],
                'train_accuracy': train_acc,
                'generalization_score': gen_score,
                'per_class_f1': metrics['per_class_f1'].tolist(),
                'confusion_matrix': metrics['confusion_matrix'].tolist()
            }
            
            logger.info(f"  Epoch {epoch:3d}: Val_Acc={val_acc:.4f}, "
                       f"Gen_Score={gen_score:.4f}, F1={metrics['f1_macro']:.4f}")
        
        validation_results[fold_idx] = fold_results
    
    return validation_results


def aggregate_cv_results(validation_results):
    """Aggregate results across all folds."""
    
    # For each epoch, aggregate across folds
    epoch_results = {}
    
    for fold_idx, fold_data in validation_results.items():
        for epoch, metrics in fold_data.items():
            if epoch not in epoch_results:
                epoch_results[epoch] = {'folds': []}
            
            epoch_results[epoch]['folds'].append(metrics)
    
    # Compute mean ± std
    for epoch, fold_list in epoch_results.items():
        accs = [m['val_accuracy'] for m in fold_list['folds']]
        f1s = [m['val_f1_macro'] for m in fold_list['folds']]
        gen_scores = [m['generalization_score'] for m in fold_list['folds']]
        
        epoch_results[epoch]['mean_accuracy'] = np.mean(accs)
        epoch_results[epoch]['std_accuracy'] = np.std(accs)
        epoch_results[epoch]['mean_f1'] = np.mean(f1s)
        epoch_results[epoch]['std_f1'] = np.std(f1s)
        epoch_results[epoch]['mean_gen_score'] = np.mean(gen_scores)
        epoch_results[epoch]['std_gen_score'] = np.std(gen_scores)
    
    return epoch_results
```

## Visualization Strategy

1. **Metric curves over epochs** - Plot val_acc, f1_macro, gen_score vs epoch
2. **Per-fold comparison** - Box plot of metric across folds per epoch
3. **Confusion matrix heatmap** - Best epoch overall
4. **Generalization gap analysis** - (train_acc - val_acc) vs epoch

## Output Format

```csv
# validation_metrics.csv
epoch,fold,train_accuracy,val_accuracy,val_f1_macro,generalization_score
0,0,0.1089,0.1543,0.1423,0.1298
0,1,0.1156,0.1621,0.1498,0.1341
...
```

```json
# per_fold_metrics.json
{
  "0": {
    "0": {
      "val_accuracy": 0.1543,
      "val_f1_macro": 0.1423,
      "train_accuracy": 0.1089,
      "generalization_score": 0.1298,
      "per_class_f1": [0.12, 0.14, ...]
    },
    ...
  }
}
```

## Dependencies
- `torch` - Model loading
- `timm` - Model architecture
- `scikit-learn` - Metrics computation
- `numpy` - Array operations
- `matplotlib` - Visualization

## Integration
**Inputs:** All checkpoints, fold_metadata
**Outputs:** validation_metrics.csv, validation_report.html
**Consumed by:** Checkpoint Selection Pipeline
