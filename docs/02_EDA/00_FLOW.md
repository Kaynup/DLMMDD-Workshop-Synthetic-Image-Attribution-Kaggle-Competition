# Pipeline 02: EDA (Exploratory Data Analysis)

## Flow Diagram

```
┌──────────────────────────────────┐
│ Input: train_df, test_df,       │
│        data_stats               │
└──────────────┬───────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Class Distribution      │
        │ - Pie chart (training)  │
        │ - Count histogram       │
        │ - Verify balance        │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Image Characteristics   │
        │ - Size distribution     │
        │ - Format breakdown      │
        │ - Color mode analysis   │
        │ - File size stats       │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Format/Quality Impact   │
        │ - Group by format       │
        │ - Class per format      │
        │ - Potential artifacts   │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Sample Visualization    │
        │ - 1 sample/class grid   │
        │ - Check visual quality  │
        │ - Detect issues         │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Size/Dimension Stats    │
        │ - Height distribution   │
        │ - Width distribution    │
        │ - Aspect ratio analysis │
        │ - Outlier detection     │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Test Set Analysis       │
        │ - Dimension comparison  │
        │ - Format comparison     │
        │ - Size distribution     │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Generator Fingerprints  │
        │ - Per-class mean image  │
        │ - Per-class std image   │
        │ - Visual differences    │
        │ - Statistical profile   │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Generate Report HTML    │
        │ - All plots embedded    │
        │ - Statistics tables     │
        │ - Insights summary      │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Outputs:                │
        │ - eda_report.html       │
        │ - plots/ folder         │
        │ - eda_insights.json     │
        └──────────────────────────┘
```

---

## Design Philosophy

### 1. **Comprehensive Visual Analysis**
- Every aspect of the data visualized
- Multiple perspectives on same data (histogram, box plot, violin plot)
- Statistical summaries alongside plots

### 2. **Generator Fingerprinting**
- Analyze unique characteristics of each generator
- Can models "memorize" generator signatures?
- Compute per-class mean/std images (template approach)

### 3. **Post-Processing Impact Speculation**
- Analyze test set format/size distribution
- Hypothesize which post-processing was applied most
- Design augmentation strategy accordingly

### 4. **Self-Contained Report**
- Single HTML file with embedded plots
- No external dependencies for viewing
- Can be shared easily

---

## Key Insights to Investigate

### Insight 1: Class Balance Check
**Expected:** Exactly 1000 samples per class  
**Action:** Verify, visualize, confirm balanced learning problem

### Insight 2: Image Dimensions
**Expected:** All 1024×1024 (or very similar)  
**Action:** Check for outliers, variations
**Impact:** Affects preprocessing pipeline

### Insight 3: Format Distribution
**Expected:** All PNG or mostly PNG  
**Action:** Identify if mixed formats exist
**Impact:** Different post-processing sensitivity

### Insight 4: Generator Signatures
**Question:** Can we visually distinguish generators?  
**Method:** Compute mean image per class, inspect artifacts
**Impact:** Informs model architecture choice

### Insight 5: Test Set Differences
**Question:** Are test images notably different from training?  
**Method:** Compare dimensions, formats, file sizes
**Impact:** Signals domain shift or heavy post-processing

### Insight 6: File Size Anomalies
**Question:** Any suspiciously small files (corrupted)?  
**Method:** Box plot with outlier detection
**Impact:** Data quality assessment

### Insight 7: Corruption/Readability
**Expected:** All images readable by PIL  
**Action:** Check is_readable column, report failures
**Impact:** Data integrity

---

## Visualizations to Generate

### 1. Class Distribution Plot
```
Type: Bar chart (count per class)
X-axis: Generator name (0-9)
Y-axis: Sample count
Expected: All bars at 1000
Color: Different per class
```

### 2. Image Dimension Distribution
```
Type: Dual histogram
Subplot 1: Height distribution
Subplot 2: Width distribution
Expected: Single spike at 1024
Color: Different for height/width
Overlay: Mean, std lines
```

### 3. Format Distribution Pie Chart
```
Type: Pie chart
Categories: PNG, JPEG, WebP, etc.
Expected: Mostly PNG
Color: Auto
```

### 4. File Size Box Plot
```
Type: Box plot (by class)
X-axis: Generator class
Y-axis: File size (MB)
Expected: Similar distributions per class
Overlay: Individual points (alpha transparency)
Outlier detection: Mark extreme values
```

### 5. Generator Sample Grid
```
Type: 10×3 image grid (or 5×2)
Each row: One generator
Show: 3 representative samples
Action: Manual inspection for artifacts/quality
```

### 6. Test Set Comparison
```
Type: Histograms (overlay train vs test)
Subplots: Height, Width, File size
Color: Train vs Test different colors
Title: "Train vs Test Characteristics"
```

### 7. Mean/Std Template Images
```
Type: 10 image grid (one per class)
Each cell: Computed mean image per class
Action: Visual inspection for generator signatures
Explanation: Averaged pixel values across all samples
```

### 8. Per-Class Statistics Table
```
Columns: Class, Generator, Count, Avg Width, Avg Height, Avg File Size, Format
Rows: 10 (one per class)
Format: HTML table
Sortable: By any column
```

---

## Key Metrics to Compute

### Class-Level
```
For each class i in [0, 9]:
  - Sample count (should be 1000)
  - Average image width
  - Average image height
  - Average file size (bytes)
  - Format distribution (if mixed)
  - Color mode distribution (if mixed)
  - Min/max dimensions
  - Std dev of dimensions
```

### Overall Dataset
```
- Total training samples: 7000
- Total test samples: 3000
- Unique image widths: [list]
- Unique image heights: [list]
- Unique formats: [list]
- Unique color modes: [list]
- Images with dimensions != 1024×1024: [count]
- Corrupted/unreadable: [count]
```

### Test-Specific
```
- Average test file size vs training
- Test format distribution
- Test dimension distribution
- Potential post-processing inference
```

---

## Code Structure

### Main EDA Function
```python
def exploratory_data_analysis(train_df, test_df, data_stats, output_dir):
    """
    Main EDA pipeline.
    
    Returns:
        eda_report: dict with all insights
    """
    
    eda_report = {
        'timestamp': datetime.now().isoformat(),
        'sections': {}
    }
    
    # Section 1: Class Distribution
    eda_report['sections']['class_distribution'] = analyze_class_distribution(train_df)
    
    # Section 2: Image Characteristics
    eda_report['sections']['image_characteristics'] = analyze_image_characteristics(train_df, test_df)
    
    # Section 3: Generator Fingerprints
    eda_report['sections']['generator_fingerprints'] = extract_generator_signatures(train_df)
    
    # Section 4: Post-Processing Inference
    eda_report['sections']['postprocessing_analysis'] = analyze_postprocessing_impact(train_df, test_df)
    
    # Generate Visualizations
    plots = generate_all_plots(train_df, test_df, eda_report, output_dir)
    
    # Compile HTML Report
    html_report = compile_html_report(eda_report, plots, output_dir)
    
    # Save report
    save_report(eda_report, output_dir)
    
    return eda_report, plots
```

---

## Insights Output Format

```json
{
  "timestamp": "2026-05-19T10:30:00",
  "class_distribution": {
    "is_balanced": true,
    "counts_per_class": [1000, 1000, ..., 1000],
    "min_count": 1000,
    "max_count": 1000,
    "std_dev": 0.0
  },
  "image_characteristics": {
    "dimension_stats": {
      "height": {"mean": 1024, "std": 0, "min": 1024, "max": 1024},
      "width": {"mean": 1024, "std": 0, "min": 1024, "max": 1024},
      "aspect_ratio": 1.0
    },
    "format_distribution": {"PNG": 7000, "JPEG": 0},
    "color_mode_distribution": {"RGB": 7000}
  },
  "generator_profiles": {
    "0_AuraFlow": {
      "avg_width": 1024,
      "avg_height": 1024,
      "avg_file_size_mb": 0.25,
      "sample_count": 1000,
      "visual_profile": "description"
    }
    // ... 1-9
  },
  "test_analysis": {
    "dimension_stats_vs_train": "no significant difference",
    "postprocessing_inference": "moderate compression likely"
  }
}
```

---

## Dependencies

**Inputs:**
- `train_df` from Data Loading pipeline
- `test_df` from Data Loading pipeline
- `data_stats` from Data Loading pipeline

**Libraries:**
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `numpy` - Numerical operations
- `scipy` - Statistical tests
- `PIL` - Load actual images for templates
- `plotly` or `altair` - Interactive plots (optional)

**Outputs:**
- `eda_report.html` - Interactive HTML report
- `eda_insights.json` - Machine-readable insights
- `plots/` - Individual plot files
