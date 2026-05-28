# Pipeline 09: Checkpoint Selection

## Flow Diagram

```
Input: validation_metrics.csv
       All trained checkpoints
       Strategy: Generalization Score
       ↓
┌──────────────────────────────────────┐
│ For Each Fold:                       │
│ - Find epoch with max Gen_Score      │
│ - Gen = val_acc - |train_acc - val_acc|
│ - Store (best_epoch, fold_id)        │
└────────────┬─────────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Select Checkpoints          │
      │ Strategy 1: Best Gen_Score  │
      │ (Recommended)               │
      │ Strategy 2: Best Val_Acc    │
      │ (Simple, may overfit)       │
      │ Strategy 3: Ensemble        │
      │ (Use top-3 per fold)        │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Copy Selected Checkpoints   │
      │ To: final_models/           │
      │ - fold_0_best.pth           │
      │ - fold_1_best.pth           │
      │ - ...                       │
      │ - fold_4_best.pth           │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Generate Selection Report:  │
      │ - Per-fold selection summary
      │ - Why each was selected     │
      │ - Expected test performance │
      │ - Confidence estimate       │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Output:                     │
      │ - final_models/fold_*.pth   │
      │ - checkpoint_selection.json │
      │ - selection_report.txt      │
      └──────────────────────────────┘
```

## Selection Strategy: Generalization Score (Recommended)

### Why Not Just Best Validation Accuracy?

**Problem with max(val_acc):**
```
Epoch 50: train_acc=0.95, val_acc=0.92 → Strong generalization ✓
Epoch 75: train_acc=0.98, val_acc=0.91 → Overfitting ✗ (but selected if max rule)
```

**Solution with Generalization Score:**
```
Gen_Score = val_acc - |train_acc - val_acc|

Epoch 50: Gen = 0.92 - |0.95 - 0.92| = 0.92 - 0.03 = 0.89 ✓✓
Epoch 75: Gen = 0.91 - |0.98 - 0.91| = 0.91 - 0.07 = 0.84 ✗
```

### Interpretation
- **High Gen_Score:** Good validation performance with minimal overfitting
- **Low Gen_Score:** Either low validation performance OR high gap (overfitting)
- **Ideal:** Gen_Score close to val_acc (no gap)

## Code Pattern

```python
def select_best_checkpoints(validation_results, strategy='generalization_score'):
    """Select best checkpoint per fold based on strategy."""
    
    best_checkpoints = {}
    selection_details = {}
    
    for fold_idx, fold_data in validation_results.items():
        if strategy == 'generalization_score':
            # Find epoch with max Gen_Score
            best_epoch = max(fold_data.items(),
                           key=lambda x: x[1]['generalization_score'])[0]
        elif strategy == 'val_accuracy':
            # Find epoch with max Val_Acc
            best_epoch = max(fold_data.items(),
                           key=lambda x: x[1]['val_accuracy'])[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        metrics = fold_data[best_epoch]
        
        best_checkpoints[fold_idx] = {
            'fold': fold_idx,
            'best_epoch': best_epoch,
            'checkpoint_path': f'checkpoints/fold_{fold_idx}/epoch_{best_epoch:03d}.pth',
            'val_accuracy': metrics['val_accuracy'],
            'val_f1_macro': metrics['val_f1_macro'],
            'generalization_score': metrics['generalization_score'],
            'train_accuracy': metrics['train_accuracy']
        }
        
        logger.info(f"Fold {fold_idx}: Selected epoch {best_epoch}")
        logger.info(f"  Val_Acc: {metrics['val_accuracy']:.4f}")
        logger.info(f"  Gen_Score: {metrics['generalization_score']:.4f}")
    
    return best_checkpoints


def copy_selected_checkpoints(best_checkpoints, output_dir='final_models'):
    """Copy selected checkpoints to final_models directory."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fold_idx, ckpt_info in best_checkpoints.items():
        src = Path(ckpt_info['checkpoint_path'])
        dst = output_path / f'fold_{fold_idx}_best.pth'
        
        shutil.copy(str(src), str(dst))
        logger.info(f"Copied {src} → {dst}")
    
    return output_path


def estimate_test_performance(best_checkpoints):
    """Estimate performance on private test set based on CV results."""
    
    accs = [c['val_accuracy'] for c in best_checkpoints.values()]
    f1s = [c['val_f1_macro'] for c in best_checkpoints.values()]
    
    estimated_test_acc = np.mean(accs)
    test_acc_std = np.std(accs)
    
    estimated_test_f1 = np.mean(f1s)
    test_f1_std = np.std(f1s)
    
    # 95% confidence interval
    ci_lower_acc = estimated_test_acc - 1.96 * test_acc_std / np.sqrt(len(accs))
    ci_upper_acc = estimated_test_acc + 1.96 * test_acc_std / np.sqrt(len(accs))
    
    estimation = {
        'estimated_test_accuracy': estimated_test_acc,
        'test_accuracy_std': test_acc_std,
        'test_accuracy_ci': (ci_lower_acc, ci_upper_acc),
        'estimated_test_f1': estimated_test_f1,
        'test_f1_std': test_f1_std,
        'num_folds': len(accs)
    }
    
    logger.info(f"\nEstimated Test Performance:")
    logger.info(f"  Accuracy: {estimated_test_acc:.4f} ± {test_acc_std:.4f}")
    logger.info(f"  95% CI: [{ci_lower_acc:.4f}, {ci_upper_acc:.4f}]")
    logger.info(f"  F1-Macro: {estimated_test_f1:.4f} ± {test_f1_std:.4f}")
    
    return estimation


def generate_selection_report(best_checkpoints, estimation):
    """Generate human-readable selection report."""
    
    report_lines = [
        "=" * 70,
        "CHECKPOINT SELECTION REPORT",
        "=" * 70,
        f"Strategy: Generalization Score",
        f"Date: {datetime.now().isoformat()}",
        "",
        "SELECTED CHECKPOINTS",
        "-" * 70
    ]
    
    for fold_idx, ckpt_info in best_checkpoints.items():
        report_lines.append(
            f"Fold {fold_idx}:"
            f"  Epoch {ckpt_info['best_epoch']:3d} "
            f"(Val_Acc: {ckpt_info['val_accuracy']:.4f}, "
            f"Gen_Score: {ckpt_info['generalization_score']:.4f})"
        )
    
    report_lines.extend([
        "",
        "ESTIMATED TEST PERFORMANCE",
        "-" * 70,
        f"Estimated Accuracy: {estimation['estimated_test_accuracy']:.4f} ± "
        f"{estimation['test_accuracy_std']:.4f}",
        f"95% Confidence Interval: "
        f"[{estimation['test_accuracy_ci'][0]:.4f}, "
        f"{estimation['test_accuracy_ci'][1]:.4f}]",
        f"Estimated F1-Macro: {estimation['estimated_test_f1']:.4f} ± "
        f"{estimation['test_f1_std']:.4f}",
        "",
        "NOTE: Actual test performance may differ due to:",
        "  • Different post-processing on test set",
        "  • Domain shift between public/private test",
        "  • Ensemble effects if using multiple models",
        "=" * 70
    ])
    
    return "\n".join(report_lines)
```

## Selection Results Example

```json
{
  "strategy": "generalization_score",
  "timestamp": "2026-05-19T11:30:00",
  "selected_checkpoints": {
    "0": {
      "fold": 0,
      "best_epoch": 45,
      "checkpoint_path": "checkpoints/fold_0/epoch_045.pth",
      "val_accuracy": 0.7234,
      "val_f1_macro": 0.7189,
      "generalization_score": 0.6898,
      "train_accuracy": 0.7892
    },
    "1": {...},
    ...
    "4": {...}
  },
  "estimated_test_performance": {
    "estimated_test_accuracy": 0.7145,
    "test_accuracy_std": 0.0087,
    "test_accuracy_ci": [0.6974, 0.7316]
  }
}
```

## Key Insights from Selection

**Metrics to examine:**
- Which fold selected which epoch? (consistency = good sign)
- Average generalization gap across selected checkpoints (< 5% = good)
- Variance in val_acc across folds (low = stable)

## Dependencies
- `numpy` - Array operations
- `shutil` - File copying
- `json` - Serialization
- `pathlib` - Path operations

## Integration
**Inputs:** validation_results, all checkpoints
**Outputs:** final_models/fold_*.pth, checkpoint_selection.json
**Consumed by:** Inference Pipeline
