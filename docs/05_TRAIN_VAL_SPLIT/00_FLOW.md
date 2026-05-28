# Pipeline 05: Train/Val Stratified Split

## Flow Diagram

```
Input: train_df (7000 samples, 10 classes)
       y_labels (7000 labels)
       ↓
┌──────────────────────────────────────┐
│ Create Stratified K-Fold Split       │
│ - n_splits = 5 or 10                 │
│ - Stratification on y (class labels) │
│ - Random seed for reproducibility    │
│ - Shuffle = True                     │
└────────────┬─────────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ For Each Fold i (0 to 4):   │
      │ - train_idx: 5600 (80%)     │
      │ - val_idx: 1400 (20%)       │
      │ - Save fold_assignment      │
      │ - Log fold statistics       │
      │ - Verify stratification     │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Verify Stratification       │
      │ - Check each fold balanced  │
      │ - No class < 100 in fold    │
      │ - Log class dist per fold   │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Output:                     │
      │ - fold_assignments: list    │
      │ - fold_metadata: dict       │
      │ - per-fold statistics       │
      └──────────────────────────────┘
```

## Key Design

### 1. **Stratification Strategy**
- **Why:** Ensure each fold has balanced class distribution
- **Method:** Use sklearn.model_selection.StratifiedKFold
- **Benefit:** Stable validation estimates across folds

### 2. **Fold Count**
- **5-Fold:** Faster, good for quick iteration (4K train, 1.4K val per fold)
- **10-Fold:** More stable, slower, better for final validation (6.3K train, 700 val per fold)
- **Recommendation:** Start with 5-fold, switch to 10-fold for final submission

### 3. **No Leakage**
- **Train/val split happens here**
- **Test set NEVER touches training**
- **Each fold is independent**

## Code Pattern

```python
from sklearn.model_selection import StratifiedKFold

def create_stratified_folds(train_df, n_splits=5, random_state=42):
    """Create stratified K-fold cross-validation splits."""
    
    y = train_df['y'].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_assignments = []
    fold_metadata = {}
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
        # Get fold data
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]
        
        # Verify stratification
        train_class_dist = train_fold_df['y'].value_counts().sort_index()
        val_class_dist = val_fold_df['y'].value_counts().sort_index()
        
        fold_metadata[fold_idx] = {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_class_dist': train_class_dist.to_dict(),
            'val_class_dist': val_class_dist.to_dict(),
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist()
        }
        
        logger.info(f"Fold {fold_idx}: Train {len(train_idx)}, Val {len(val_idx)}")
        logger.info(f"  Train distribution: {train_class_dist.to_dict()}")
        logger.info(f"  Val distribution: {val_class_dist.to_dict()}")
    
    return fold_metadata

# Save for reproducibility
import json
with open('fold_assignments.json', 'w') as f:
    json.dump(fold_metadata, f)
```

## Output Format

```python
fold_metadata = {
    0: {
        'train_size': 5600,
        'val_size': 1400,
        'train_class_dist': {0: 800, 1: 800, ..., 9: 800},
        'val_class_dist': {0: 200, 1: 200, ..., 9: 200},
        'train_indices': [0, 1, 5, 7, ...],  # 5600 indices
        'val_indices': [2, 3, 4, 6, ...]      # 1400 indices
    },
    1: {...},
    ...
    4: {...}
}
```

## Dependencies
- `scikit-learn` - StratifiedKFold
- `pandas` - DataFrame operations
- `numpy` - Array operations
- `json` - Serialization

## Integration
**Inputs:** train_df (from Data Loading)
**Outputs:** fold_metadata, fold_assignments.json
**Consumed by:** Model Training (per-fold), Validation Pipeline
