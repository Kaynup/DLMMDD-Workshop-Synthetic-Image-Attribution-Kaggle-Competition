# EDA - Design Deep Dive

## Architectural Patterns

### Pattern 1: Section-Based Organization
```
EDA organized into sections:
- Class Distribution Analysis
- Image Characteristics
- Generator Fingerprinting
- Post-Processing Impact

Each section:
- Computes metrics
- Generates visualizations
- Returns insights dict

Benefits:
- Modular, testable
- Easy to add/remove sections
- Clear output structure
```

### Pattern 2: Dual Visualization Strategy
```
For each metric: Provide 2-3 views

Example - File Size Analysis:
1. Box plot (overall distribution, outliers)
2. Histogram (density view)
3. Per-class violin plot (class-wise comparison)

Benefits:
- Multiple perspectives reveal different insights
- Complementary visual information
- Robust to interpretation errors
```

### Pattern 3: Template Matching (Generator Fingerprints)
```
Hypothesis: Each generator has unique "fingerprint"

Method:
1. For each class (generator):
   - Compute mean image (average pixel values)
   - Compute std image (pixel variance)
   - Compute quantile images (25%, 50%, 75%)

2. Visualization:
   - Show 10 mean images in grid
   - Show differences between generators

Insight Value:
- Can models detect generator signatures?
- Are some generators more/less "natural"?
- What artifacts are common per generator?
```

---

## Analysis Components

### Component 1: Class Distribution Analyzer

```python
def analyze_class_distribution(train_df):
    """
    Analyze label distribution.
    
    Checks:
    - Is dataset balanced (1000 per class)?
    - Are there outliers?
    - What's the most/least common class?
    """
    
    class_counts = train_df['y'].value_counts().sort_index()
    
    return {
        'counts': class_counts.to_dict(),
        'is_balanced': (class_counts == 1000).all(),
        'min_count': class_counts.min(),
        'max_count': class_counts.max(),
        'std_count': class_counts.std(),
        'classes_with_imbalance': class_counts[class_counts != 1000].index.tolist()
    }
```

### Component 2: Image Characteristics Analyzer

```python
def analyze_image_characteristics(train_df, test_df):
    """
    Analyze pixel dimensions and file sizes.
    
    Metrics:
    - Height: min, max, mean, std
    - Width: min, max, mean, std
    - Aspect ratio
    - File size distribution
    - Format distribution
    - Color mode distribution
    """
    
    train_stats = {
        'height': {
            'mean': train_df['height'].mean(),
            'std': train_df['height'].std(),
            'min': train_df['height'].min(),
            'max': train_df['height'].max(),
            'unique_values': train_df['height'].nunique()
        },
        'width': {
            'mean': train_df['width'].mean(),
            'std': train_df['width'].std(),
            'min': train_df['width'].min(),
            'max': train_df['width'].max(),
            'unique_values': train_df['width'].nunique()
        },
        'aspect_ratio': {
            'mean': (train_df['width'] / train_df['height']).mean(),
            'std': (train_df['width'] / train_df['height']).std()
        },
        'file_size_mb': {
            'mean': (train_df['file_size_bytes'] / 1e6).mean(),
            'std': (train_df['file_size_bytes'] / 1e6).std(),
            'min': (train_df['file_size_bytes'] / 1e6).min(),
            'max': (train_df['file_size_bytes'] / 1e6).max()
        },
        'formats': train_df['format'].value_counts().to_dict(),
        'color_modes': train_df['color_mode'].value_counts().to_dict()
    }
    
    test_stats = {
        'height': {...},  # Same structure
        'width': {...},
        # ... etc
    }
    
    return {'train': train_stats, 'test': test_stats}
```

### Component 3: Generator Signature Extractor

```python
def extract_generator_signatures(train_df, image_dir='Data/Training'):
    """
    Compute mean/std templates per generator.
    
    Process:
    1. For each class (generator)
    2. Load all images for that class
    3. Compute mean image (averaged pixels)
    4. Compute std image (pixel variance)
    5. Compute percentile images (25%, 75%)
    """
    
    signatures = {}
    
    for class_id in range(10):
        class_images = train_df[train_df['y'] == class_id]['full_path'].tolist()
        
        # Load images as numpy arrays
        img_arrays = []
        for img_path in class_images[:100]:  # Limit to 100 for speed
            img = PIL.Image.open(img_path)
            img_arrays.append(np.array(img, dtype='float32'))
        
        img_arrays = np.stack(img_arrays, axis=0)
        
        signatures[class_id] = {
            'mean_image': img_arrays.mean(axis=0).astype('uint8'),
            'std_image': img_arrays.std(axis=0).astype('uint8'),
            'percentile_25': np.percentile(img_arrays, 25, axis=0).astype('uint8'),
            'percentile_75': np.percentile(img_arrays, 75, axis=0).astype('uint8'),
            'brightness': img_arrays.mean(),
            'contrast': img_arrays.std()
        }
    
    return signatures
```

### Component 4: Post-Processing Impact Analyzer

```python
def analyze_postprocessing_impact(train_df, test_df):
    """
    Infer what post-processing was applied to test set.
    
    Observations:
    - Test images: Smaller file size? → Compression
    - Test images: Different dimensions? → Cropping/resizing
    - Test images: Different aspect ratio? → Skewing
    - Test images: Grayscale? → Color conversion
    - Test images: More JPEG? → JPEG compression
    
    Output: Hypothesis about likely post-processing
    """
    
    # Compare file size distributions
    train_file_size = train_df['file_size_bytes'].mean()
    test_file_size = test_df['file_size_bytes'].mean()
    size_reduction = (train_file_size - test_file_size) / train_file_size * 100
    
    # Compare dimensions
    train_aspect = (train_df['width'] / train_df['height']).mean()
    test_aspect = (test_df['width'] / test_df['height']).mean()
    aspect_change = abs(train_aspect - test_aspect)
    
    # Infer post-processing
    likely_operations = []
    if size_reduction > 10:
        likely_operations.append('compression')
    if aspect_change > 0.01:
        likely_operations.append('cropping_or_resizing')
    
    return {
        'train_avg_file_size_mb': train_file_size / 1e6,
        'test_avg_file_size_mb': test_file_size / 1e6,
        'file_size_reduction_pct': size_reduction,
        'likely_postprocessing': likely_operations,
        'confidence': 'medium'  # Qualitative
    }
```

---

## Visualization Functions

### Visualization 1: Class Distribution Bar Chart
```python
def plot_class_distribution(train_df, ax=None):
    """Bar chart of samples per class."""
    class_counts = train_df['y'].value_counts().sort_index()
    ax = ax or plt.gca()
    ax.bar(class_counts.index, class_counts.values)
    ax.set_xlabel('Generator Class')
    ax.set_ylabel('Sample Count')
    ax.set_title('Class Distribution (Training Set)')
    ax.axhline(1000, color='r', linestyle='--', label='Expected (1000)')
    ax.legend()
    return ax
```

### Visualization 2: Dimension Comparison
```python
def plot_dimension_distributions(train_df, test_df):
    """Compare height/width distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Height
    axes[0].hist(train_df['height'], bins=20, alpha=0.5, label='Train')
    axes[0].hist(test_df['height'], bins=20, alpha=0.5, label='Test')
    axes[0].set_xlabel('Height (pixels)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Height Distribution')
    axes[0].legend()
    
    # Width
    axes[1].hist(train_df['width'], bins=20, alpha=0.5, label='Train')
    axes[1].hist(test_df['width'], bins=20, alpha=0.5, label='Test')
    axes[1].set_xlabel('Width (pixels)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Width Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    return fig, axes
```

### Visualization 3: Generator Sample Grid
```python
def plot_generator_samples(train_df, image_dir='Data/Training'):
    """Show 1 sample per generator class."""
    fig, axes = plt.subplots(5, 2, figsize=(10, 12))
    axes = axes.flatten()
    
    for class_id in range(10):
        sample_path = train_df[train_df['y'] == class_id]['full_path'].iloc[0]
        img = PIL.Image.open(sample_path)
        
        ax = axes[class_id]
        ax.imshow(img)
        ax.set_title(f'Class {class_id}: {SOURCES[class_id]}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig, axes
```

---

## Output Report Format

### HTML Report Structure
```html
<html>
  <head>
    <title>EDA Report</title>
    <style>/* Embedded CSS */</style>
  </head>
  <body>
    <h1>Synthetic Image Attribution - EDA Report</h1>
    <p>Generated: 2026-05-19 10:30:00</p>
    
    <h2>1. Executive Summary</h2>
    <ul>
      <li>Dataset: 7,000 training, 3,000 test images</li>
      <li>Classes: 10 generators (balanced)</li>
      <li>Dimensions: All 1024×1024</li>
      <li>Key Finding: ...</li>
    </ul>
    
    <h2>2. Class Distribution</h2>
    <img src="data:image/png;base64,..." />
    <table>...</table>
    
    <h2>3. Image Characteristics</h2>
    <img src="data:image/png;base64,..." />
    
    <h2>4. Generator Fingerprints</h2>
    <img src="data:image/png;base64,..." />
    
    <h2>5. Post-Processing Analysis</h2>
    <img src="data:image/png;base64,..." />
    <p>Inference: ...</p>
    
  </body>
</html>
```

---

## Insights Summary

The EDA should answer:

| Question | Answer Source | Importance |
|----------|---|-----------|
| Is dataset balanced? | Class distribution | High |
| Are all images same size? | Dimension stats | High |
| What post-processing? | Test vs Train comparison | High |
| Can we spot generator differences visually? | Sample grid + templates | Medium |
| Are there corrupted images? | File size outliers | Medium |
| What format are images? | Format distribution | Low |

---

## Integration Notes

**Inputs:** train_df, test_df, data_stats (from Data Loading)  
**Outputs:** eda_report.html, eda_insights.json, plots/  
**Dependencies:** matplotlib, seaborn, numpy, scipy, PIL  
**Runtime:** ~2-5 minutes (depending on template generation)  
**Memory:** ~4 GB (for loading 100 images × 10 classes simultaneously)
