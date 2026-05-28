# Pipeline 06: Feature Extraction (Optional)

## Flow Diagram

```
Input: X_train, X_test (preprocessed images)
       ↓
┌──────────────────────────────────┐
│ Select Pretrained Model Backbone │
│ - ResNet-50 (common)             │
│ - ViT-Base (modern)              │
│ - EfficientNet (lightweight)      │
│ - Remove classification head     │
└────────────┬─────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Load Pretrained Weights     │
      │ - From timm or torchvision  │
      │ - ImageNet pretrained       │
      │ - Freeze all weights        │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Extract Features (no gradients)
      │ - Forward pass: image → feature
      │ - Batch processing for speed │
      │ - Store feature vectors     │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Output Feature Matrices:    │
      │ - X_train_features: (7000, 2048)
      │ - X_test_features: (3000, 2048) │
      │ - Feature metadata          │
      └──────────────────────────────┘
```

## When to Use

**Feature Extraction is optional because:**
- Deep learning models learn features automatically
- End-to-end fine-tuning often works better
- But useful for:
  - Classical models (SVM, Random Forest)
  - Quick baseline comparisons
  - Understanding learned representations

## Code Pattern (Feature Extraction)

```python
import torch
import timm

def extract_features(X_images, model_name='resnet50', device='cuda'):
    """Extract features using pretrained backbone."""
    
    # Load pretrained model
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()
    
    # Extract features in batches
    features = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(X_images), batch_size):
            batch = X_images[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            
            # Reorder for PyTorch: (B, H, W, C) → (B, C, H, W)
            batch_tensor = batch_tensor.permute(0, 3, 1, 2)
            
            feat = model(batch_tensor)
            features.append(feat.cpu().numpy())
    
    # Concatenate all batches
    X_features = np.vstack(features)  # shape: (7000, 2048)
    return X_features

# Use with classical models
X_train_feat = extract_features(X_train, 'resnet50')
X_test_feat = extract_features(X_test, 'resnet50')

# Train SVM on features
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=1.0)
clf.fit(X_train_feat, y_train)
pred = clf.predict(X_test_feat)
```

## Dependencies
- `torch` - PyTorch
- `timm` - PyTorch Image Models
- `numpy` - Array operations

## Integration
**Inputs:** X_train, X_test (preprocessed)
**Outputs:** X_train_features, X_test_features
**Optional:** Use for baseline models, skip for deep learning
