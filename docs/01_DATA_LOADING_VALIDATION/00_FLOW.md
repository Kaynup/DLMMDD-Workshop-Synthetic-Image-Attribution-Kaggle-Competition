# Pipeline 01: Data Loading & Validation

## Flow Diagram

```
┌──────────────────────────────┐
│ Read training.csv            │
│ (columns: ID, path, y)       │
└──────────────┬───────────────┘
               │
        ┌──────▼────────┐
        │ Read test.csv │
        │ (columns: ID, path)
        └──────┬────────┘
               │
        ┌──────▼──────────────────┐
        │ Verify file existence   │
        │ - Check all image paths │
        │ - Log missing files     │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Load image metadata     │
        │ - Image dimensions      │
        │ - File sizes            │
        │ - Color modes (RGB/etc) │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ Validate class distribution     │
        │ - Check 1000 per class (train)  │
        │ - Check balanced stratification │
        │ - Log any anomalies             │
        └──────┬──────────────────────────┘
               │
        ┌──────▼─────────────────────┐
        │ Generate data statistics   │
        │ - Class counts             │
        │ - Image size distribution  │
        │ - Format distribution      │
        │ - Corrupted image check    │
        └──────┬─────────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Save validation report  │
        │ - JSON statistics file  │
        │ - Summary text file     │
        └──────┬──────────────────┘
               │
        ┌──────▼──────────────────┐
        │ Output:                 │
        │ - train_df (pandas)     │
        │ - test_df (pandas)      │
        │ - data_stats (dict)     │
        └──────────────────────────┘
```

## Design Principles

### 1. **Immutability**
- Load CSVs and image metadata once
- Store as immutable DataFrames (frozen state for reproducibility)
- Create copies before any transformations in downstream pipelines

### 2. **Validation First**
- Verify all image files exist before proceeding
- Check label integrity (values in [0, 9])
- Confirm class balance (exactly 1000 per class in training set)
- Log all validation failures with file paths

### 3. **Metadata Tracking**
- Store image dimensions, file sizes, formats
- Track corrupted/unreadable images
- Record load timestamp and MD5 checksums for reproducibility

### 4. **Error Resilience**
- Graceful handling of missing/corrupted images
- Detailed logging of issues with paths
- Continue loading with warnings (don't fail on single bad image)
- Generate validation report for manual inspection

### 5. **Reproducibility**
- Fixed random seed (though no randomness in loading)
- Record data version/collection date
- Store exact file paths (relative or absolute)
- Generate data checksum for version tracking

---

## Data Structure

### Input CSVs
```
training.csv:
  ID | path         | y
  0  | 0.png        | 0
  1  | 1.png        | 5
  ... (7000 rows)

test.csv:
  ID | path
  7000 | 6.png
  7001 | 12.png
  ... (3000 rows)
```

### Output DataFrames

**train_df** (pandas.DataFrame)
```
  ID   path           y   img_width  img_height  file_size  format
  0    0.png          0   1024       1024        250KB      PNG
  1    1.png          5   1024       1024        240KB      PNG
  ... (7000 rows)
```

**test_df** (pandas.DataFrame)
```
  ID   path         img_width  img_height  file_size  format
  7000 6.png        1024       1024        245KB      PNG
  7001 12.png       1024       1024        255KB      PNG
  ... (3000 rows)
```

### Statistics (data_stats)
```python
{
  "train": {
    "n_samples": 7000,
    "n_classes": 10,
    "class_distribution": [1000, 1000, ..., 1000],
    "missing_files": [],
    "corrupted_files": [],
    "avg_img_height": 1024.0,
    "avg_img_width": 1024.0,
    "avg_file_size_mb": 0.25
  },
  "test": {
    "n_samples": 3000,
    "missing_files": [],
    "corrupted_files": [],
    "avg_img_height": 1024.0,
    "avg_img_width": 1024.0,
    "avg_file_size_mb": 0.25
  },
  "sources": {
    0: "AuraFlow",
    1: "Freepik",
    ...
    9: "Tencent Hunyuan"
  },
  "timestamp": "2026-05-19 10:30:00",
  "data_version": "v1.0"
}
```

---

## Key Operations

### 1. Load CSVs
```
train_df = pd.read_csv(data_path / 'training.csv')
test_df = pd.read_csv(data_path / 'test.csv')
```

### 2. Verify Files Exist
```
For each row in train_df:
  - Check if path exists (full path = data_path / 'Training' / path)
  - Log missing files
  - Try to open as image (PIL/cv2) - catch corruption

For each row in test_df:
  - Check if path exists (full path = data_path / 'Test' / path)
  - Log missing files
```

### 3. Extract Image Metadata
```
For each image:
  - Open with PIL.Image.open()
  - Record width, height, format
  - Check color mode (should be RGB)
  - Record file size (os.path.getsize)
```

### 4. Validate Class Distribution
```
class_counts = train_df['y'].value_counts().sort_index()
Assert all counts == 1000
Assert no missing classes (0-9)
Log distribution
```

### 5. Load Source Mapping
```
sources.txt format:
0|AuraFlow
1|Freepik
...
9|Tencent Hunyuan

Parse and create dict: {0: 'AuraFlow', 1: 'Freepik', ...}
Merge into train_df as 'source_name' column
```

---

## Validation Checks

| Check | Expected | Action if Failed |
|-------|----------|-----------------|
| `len(train_df)` | 7000 | Log error, continue |
| `len(test_df)` | 3000 | Log error, continue |
| `train_df['y'].nunique()` | 10 | Log error, continue |
| All training file paths exist | True | Log missing files, continue |
| All test file paths exist | True | Log missing files, continue |
| All images readable (PIL) | True | Mark as corrupted, continue |
| `train_df['y'].value_counts().min()` | 1000 | Log imbalance, continue |
| All images RGB or grayscale | True | Log format mismatches |

---

## Error Handling

### File Not Found
```
Log: "Missing training image: Data/Training/1234.png (ID: 1234)"
Continue processing, add to missing_files list
Report at end: "Found 0 missing files"
```

### Corrupted Image (can't open with PIL)
```
Log: "Corrupted image: Data/Training/5678.png (ID: 5678)"
Continue processing, add to corrupted_files list
Report at end: "Found 0 corrupted files"
```

### Class Distribution Anomaly
```
Log warning: "Class 3 has 999 samples (expected 1000)"
Continue processing
Recommend checking data source
```

---

## Output Files

**Saved Outputs:**
1. `data_stats.json` - All statistics in JSON format (for programmatic access)
2. `data_validation_report.txt` - Human-readable validation summary
3. Optionally: `train_df.parquet`, `test_df.parquet` (if large)

---

## Dependencies

- `pandas` - Data loading and manipulation
- `PIL` (Pillow) - Image loading and metadata extraction
- `pathlib.Path` - Cross-platform path handling
- `json` - Statistics serialization
- `logging` - Validation logging

---

## Notes

- Images are NOT loaded into memory at this stage (just metadata)
- Actual image loading happens in Preprocessing pipeline
- This pipeline is fast (< 1 minute for full dataset)
- Focus is on data integrity verification before downstream processing
