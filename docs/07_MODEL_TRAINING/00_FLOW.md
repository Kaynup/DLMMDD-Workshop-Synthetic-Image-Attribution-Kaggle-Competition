# Pipeline 07: Model Training (Core)

## Flow Diagram

```
Input: X_train, y_train (preprocessed)
       fold_metadata (fold assignments)
       config (hyperparameters)
       ↓
┌──────────────────────────────────────┐
│ For Each Fold (0 to 4):              │
└────────────┬─────────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Get Train/Val Split (fold) │
      │ - train_idx from fold_meta │
      │ - val_idx from fold_meta   │
      │ - X_train_fold, X_val_fold │
      │ - y_train_fold, y_val_fold │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Initialize Model & Optimizer│
      │ - Model: EfficientNet/ViT   │
      │ - Optimizer: AdamW          │
      │ - LR scheduler: CosineAneal │
      │ - Loss: CrossEntropyLoss    │
      │ - Device: GPU               │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Create DataLoaders          │
      │ - train: batch_size=32      │
      │ - val: batch_size=32        │
      │ - Apply augmentations (train)
      │ - shuffle=True (train)      │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Training Loop (per epoch):  │
      │ 1. Forward pass             │
      │ 2. Compute loss             │
      │ 3. Backward & optimize      │
      │ 4. Log metrics              │
      │ 5. Save checkpoint          │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Validation Loop (per epoch):│
      │ 1. Forward pass (no grad)   │
      │ 2. Compute metrics          │
      │ 3. Log to metrics dict      │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Save Checkpoint             │
      │ - model.state_dict()        │
      │ - epoch, loss, metrics      │
      │ - optimizer state (opt)     │
      │ - fold number               │
      │ - Path: checkpoints/fold_{f}│
      │        /epoch_{e}.pth       │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Check Early Stopping (opt)  │
      │ - Monitor val_loss          │
      │ - Patience: 10-20 epochs    │
      │ - Stop if no improvement    │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ End of Fold:                │
      │ - Save training log (JSON)  │
      │ - All checkpoints saved     │
      │ - Move to next fold         │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Output (per fold):          │
      │ - checkpoints/fold_{f}/     │
      │ - training_log_fold_{f}.json│
      │ - metrics_fold_{f}.csv      │
      └──────────────────────────────┘
```

## Key Components

### 1. **Model Architecture**
**Recommended choices:**
- **EfficientNet-B4 or B5** - Fast, accurate, good at robustness
- **Vision Transformer (ViT-Base)** - Modern, robust to post-processing
- **ResNet-50** - Baseline, well-understood

**Initialization:**
```python
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=10)
model = model.to(device)
```

### 2. **Optimizer & Learning Rate**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

### 3. **Loss Function**
```python
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 4. **Training Metrics**
```python
Track per epoch:
- train_loss (avg)
- train_accuracy (overall)
- train_f1_macro (per-class)
- val_loss (avg)
- val_accuracy (overall)
- val_f1_macro (per-class)
- learning_rate (from scheduler)
```

### 5. **Checkpoint Structure**
```
checkpoints/
├── fold_0/
│   ├── epoch_000.pth
│   ├── epoch_001.pth
│   └── ...
│   └── epoch_100.pth
├── fold_1/
│   ├── epoch_000.pth
│   └── ...
└── fold_4/
    └── ...
```

**Each .pth file contains:**
```python
{
    'epoch': 0,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'train_loss': 2.3,
    'train_acc': 0.1,
    'val_loss': 2.2,
    'val_acc': 0.15,
    'val_f1_macro': 0.12,
    'fold': 0
}
```

## Code Pattern (Training Loop)

```python
def train_fold(fold_idx, train_idx, val_idx, X_train, y_train, config):
    """Train model on single fold."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split data
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]
    
    # Create dataloaders
    train_loader = create_dataloader(X_train_fold, y_train_fold, augment=True, batch_size=32)
    val_loader = create_dataloader(X_val_fold, y_val_fold, augment=False, batch_size=32)
    
    # Model
    model = timm.create_model(config['model_name'], pretrained=True, num_classes=10)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training loop
    metrics_history = []
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (logits.argmax(1) == y_batch).float().mean().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item()
                preds = logits.argmax(1)
                val_acc += (preds == y_batch).float().mean().item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1_macro': val_f1,
            'lr': optimizer.param_groups[0]['lr']
        }
        metrics_history.append(metrics)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'fold': fold_idx
        }
        ckpt_path = f'checkpoints/fold_{fold_idx}/epoch_{epoch:03d}.pth'
        torch.save(checkpoint, ckpt_path)
        
        logger.info(f"Fold {fold_idx} Epoch {epoch}: "
                   f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
                   f"Val loss={val_loss:.4f}, acc={val_acc:.4f}, F1={val_f1:.4f}")
        
        scheduler.step()
    
    # Save training log
    log_path = f'logs/training_log_fold_{fold_idx}.json'
    with open(log_path, 'w') as f:
        json.dump(metrics_history, f)
    
    return metrics_history


def run_all_folds(X_train, y_train, fold_metadata, config):
    """Train on all folds."""
    
    all_histories = {}
    
    for fold_idx in range(config['n_folds']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Fold {fold_idx}/{config['n_folds']-1}")
        logger.info(f"{'='*60}")
        
        train_idx = fold_metadata[fold_idx]['train_indices']
        val_idx = fold_metadata[fold_idx]['val_indices']
        
        history = train_fold(fold_idx, train_idx, val_idx, X_train, y_train, config)
        all_histories[fold_idx] = history
    
    return all_histories
```

## Key Hyperparameters

```python
config = {
    'model_name': 'efficientnet_b4',
    'num_classes': 10,
    'num_epochs': 100,
    'batch_size': 32,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'warmup_epochs': 5,
    'label_smoothing': 0.1,
    'gradient_clip': 1.0,
    'early_stopping_patience': 20,
    'seed': 42
}
```

## Logging Strategy

**Log every epoch:**
```
[2026-05-19 10:35:22] Fold 0 Epoch 0: Train loss=2.3015, acc=0.1089 | Val loss=2.2891, acc=0.1543, F1=0.1423
[2026-05-19 10:36:15] Fold 0 Epoch 1: Train loss=2.1245, acc=0.2156 | Val loss=2.0876, acc=0.2678, F1=0.2534
...
```

## Dependencies
- `torch` - PyTorch
- `timm` - Model zoo
- `scikit-learn` - Metrics
- `numpy` - Array operations

## Integration
**Inputs:** X_train, y_train, fold_metadata
**Outputs:** checkpoints/fold_*/epoch_*.pth, training logs
**Consumed by:** Validation Pipeline, Checkpoint Selection
