# Pipeline 11: Ensemble (Optional Advanced)

## Flow Diagram

```
Input: Multiple trained models
       Different architectures or hyperparams
       ↓
┌──────────────────────────────────────┐
│ Train Diverse Models:                │
│ - Model 1: EfficientNet-B4           │
│ - Model 2: Vision Transformer        │
│ - Model 3: ResNet-50                 │
│ - Model 4: EfficientNet-B5           │
│ - Model 5: Ensemble from pipeline 10 │
└────────────┬─────────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Collect All Predictions     │
      │ (3000, 10) per model        │
      │ Stack: (5, 3000, 10)        │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Select Ensemble Method:     │
      │ - Average (equal weights)   │
      │ - Weighted (by CV perf)     │
      │ - Stacking (train meta-m)   │
      │ - Voting (class voting)     │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Generate Final Predictions  │
      │ - Higher diversity → better  │
      │ - Compare to single model   │
      └──────┬──────────────────────┘
             │
      ┌──────▼──────────────────────┐
      │ Output Ensemble Submission  │
      │ - submission_ensemble.csv   │
      │ - ensemble_analysis.json    │
      └──────────────────────────────┘
```

## When Ensemble Helps

**Ensemble improves over single model when:**
✅ Models are diverse (different architectures, training)  
✅ Base models have similar performance  
✅ Base models make different mistakes  

**Ensemble doesn't help when:**
❌ All models are identical (same predictions)  
❌ One model dominates others (use just the best)  
❌ Models correlate highly (redundant)

## Types of Ensemble

### 1. **Average Probabilities (Simplest)**
```python
ensemble_probs = np.mean([model1_probs, model2_probs, ...], axis=0)
```
**Best for:** Quick iteration, similar model quality  

### 2. **Weighted Average (Balanced)**
```python
weights = np.array([0.25, 0.25, 0.25, 0.25])  # Based on CV perf
ensemble_probs = np.average([...], axis=0, weights=weights)
```
**Best for:** Combining models of different strengths  

### 3. **Stacking (Advanced)**
```python
# Train meta-model on predictions from base models
meta_model.fit(base_predictions_train, y_train)
final_pred = meta_model.predict(base_predictions_test)
```
**Best for:** Squeezing maximum performance  
**Downside:** Complex, requires holdout validation set

### 4. **Voting (Robust)**
```python
ensemble_pred = np.bincount([p1, p2, p3, ...], minlength=10).argmax()
```
**Best for:** Robust to individual model failures

## Code Pattern (Weighted Ensemble)

```python
def create_diverse_ensemble(test_df, X_test, models_config):
    """
    Create ensemble from diverse models.
    
    models_config = [
        {'name': 'efficientnet_b4', 'path': 'models/effb4.pth', 'weight': 0.30},
        {'name': 'vit_base', 'path': 'models/vit.pth', 'weight': 0.25},
        {'name': 'resnet50', 'path': 'models/resnet.pth', 'weight': 0.25},
        {'name': 'efficientnet_b5', 'path': 'models/effb5.pth', 'weight': 0.20},
    ]
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_probs = []
    all_weights = []
    
    for model_cfg in models_config:
        logger.info(f"Loading {model_cfg['name']}...")
        
        # Load model
        if 'vit' in model_cfg['name'].lower():
            model = timm.create_model(model_cfg['name'], pretrained=False, num_classes=10)
        elif 'resnet' in model_cfg['name'].lower():
            model = timm.create_model(model_cfg['name'], pretrained=False, num_classes=10)
        else:
            model = timm.create_model(model_cfg['name'], pretrained=False, num_classes=10)
        
        ckpt = torch.load(model_cfg['path'], map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        
        model.to(device)
        model.eval()
        
        # Predict
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_test), 32):
                X_batch = torch.FloatTensor(X_test[i:i+32])
                X_batch = X_batch.permute(0, 3, 1, 2).to(device)
                
                logits = model(X_batch)
                prob = torch.softmax(logits, dim=1)
                probs.append(prob.cpu().numpy())
        
        model_probs = np.vstack(probs)
        all_probs.append(model_probs)
        all_weights.append(model_cfg['weight'])
        
        logger.info(f"  {model_cfg['name']} predictions: {model_probs.shape}")
    
    # Ensemble
    logger.info("Creating weighted ensemble...")
    all_probs = np.array(all_probs)
    all_weights = np.array(all_weights)
    
    # Verify weights sum to 1
    all_weights = all_weights / all_weights.sum()
    
    ensemble_probs = np.average(all_probs, axis=0, weights=all_weights)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    logger.info(f"Ensemble shape: {ensemble_probs.shape}")
    logger.info(f"Ensemble predictions: {ensemble_preds.shape}")
    
    return ensemble_preds, ensemble_probs


def analyze_ensemble_diversity(all_probs):
    """Analyze diversity of ensemble members."""
    
    # Pairwise agreement
    diversity = {}
    num_models = len(all_probs)
    
    for i in range(num_models):
        for j in range(i+1, num_models):
            preds_i = np.argmax(all_probs[i], axis=1)
            preds_j = np.argmax(all_probs[j], axis=1)
            agreement = (preds_i == preds_j).mean()
            diversity[f'model_{i}_vs_{j}'] = agreement
    
    mean_agreement = np.mean(list(diversity.values()))
    
    logger.info(f"Ensemble Diversity Analysis:")
    logger.info(f"  Average pairwise agreement: {mean_agreement:.2%}")
    logger.info(f"  (Lower = more diverse = better ensemble)")
    
    return diversity
```

## When to Use Ensemble

**Worth the effort if:**
- You have trained multiple different models
- They each score 70%+ on validation
- They disagree on different samples

**Not worth it if:**
- You only have one model trained
- Just use single model pipeline

## Expected Improvement

```
Single best model: ~72% CV accuracy
2-model ensemble (if diverse): ~73% CV accuracy
5-model ensemble (if diverse): ~74% CV accuracy

Real test set improvement: Usually +0.5-1% if models are truly diverse
```

## Diversity Metrics

**Check ensemble diversity:**
```python
# Pairwise prediction agreement
agree_01 = (preds_model1 == preds_model2).mean()

# Good ensemble: ~60-80% agreement (not identical, not random)
# Bad ensemble: >90% agreement (essentially same model)
# Bad ensemble: <40% agreement (too diverse, random)
```

## Integration
**Inputs:** Multiple trained models, X_test
**Outputs:** submission_ensemble.csv
**Note:** Optional, only if you have time for multiple models
