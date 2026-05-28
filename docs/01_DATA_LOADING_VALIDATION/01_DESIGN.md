# Data Loading & Validation - Design Deep Dive

## Architecture Pattern: Pipeline Stage Gateway

This pipeline acts as the **gateway** between raw files and the rest of the system. It ensures:
- ✅ All downstream pipelines work with validated, consistent data
- ✅ Early detection of data integrity issues
- ✅ Reproducible file references (paths, checksums)
- ✅ Clear error reporting for manual intervention

---

## Component Design

### Component 1: CSV Loader
**Purpose:** Load training and test metadata CSVs

**Key Decisions:**
- Use `pd.read_csv()` with explicit dtype specification
  - `ID`: int64
  - `path`: string
  - `y`: int64 (training only)
- No column renaming - keep original names
- Store as-is: don't drop or reorder columns yet

**Why Explicit Dtypes:**
- Prevent pandas auto-inference bugs (e.g., treating ID as float)
- Ensure consistent behavior across runs
- Catch malformed data early

---

### Component 2: Path Resolver
**Purpose:** Convert relative paths from CSV to absolute file paths

**Key Decision:** Store both relative and absolute paths
```
train_df columns after resolution:
  ID | path (original) | full_path (absolute) | y
```

**Why Store Both:**
- Original `path`: For reproducibility (as in original CSV)
- `full_path`: For actual file operations
- Enables debugging (clear reference to source)

---

### Component 3: Image Inspector
**Purpose:** Non-invasive image metadata extraction without loading full image into memory

**What to Extract:**
- `width`, `height` - Tensor shape info
- `format` - PNG/JPEG/WebP (important for post-processing impact analysis)
- `file_size_bytes` - Detect anomalies (corrupted = much smaller)
- `color_mode` - RGB/RGBA/Grayscale
- `is_readable` - Boolean flag (can PIL open it?)

**Implementation Pattern:**
```
def inspect_image(img_path):
    try:
        img = PIL.Image.open(img_path)
        img.verify()  # Lightweight check
        width, height = img.size
        return {
            'width': width,
            'height': height,
            'format': img.format,
            'color_mode': img.mode,
            'file_size_bytes': os.path.getsize(img_path),
            'is_readable': True,
            'error': None
        }
    except Exception as e:
        return {
            'width': None,
            'height': None,
            'format': None,
            'color_mode': None,
            'file_size_bytes': os.path.getsize(img_path) if exists else None,
            'is_readable': False,
            'error': str(e)
        }
```

**Why `image.verify()` instead of full load:**
- `verify()` checks file integrity without loading pixel data to RAM
- Much faster for 10K images
- Detects truncated/corrupted files

---

### Component 3: Distribution Validator
**Purpose:** Ensure training set is properly balanced

**Validation Rules:**
1. **Exact Count:** `train_df['y'].value_counts()[i] == 1000` for all i in [0, 9]
2. **Complete Coverage:** No missing classes in train_df['y'].unique()
3. **Test Set:** No labels (as expected)

**Failure Handling:**
- Log as warning, NOT error (don't block pipeline)
- Recommend manual inspection
- Continue processing

**Why Warning, Not Error:**
- In real competitions, you sometimes get imperfect data
- Graceful degradation better than hard crash
- Allows user to make informed decisions

---

### Component 4: Source Mapping
**Purpose:** Add human-readable generator names

**Input:** `sources.txt`
```
0|AuraFlow
1|Freepik
2|Lumina
...
9|Tencent Hunyuan
```

**Output:** `source_name` column added to train_df
```
  y | source_name
  0 | AuraFlow
  5 | Playground v2.5
  ...
```

---

### Component 5: Statistics Aggregator
**Purpose:** Compute summary statistics for EDA context

**Statistics to Track:**

**Training Set:**
- Total samples: 7000
- Class distribution: [count_0, count_1, ..., count_9]
- Image dimensions: min/max/mean height, width
- Format distribution: {PNG: 7000} (or mixed if present)
- Color mode distribution: {RGB: 7000}
- File size stats: min/max/mean
- Missing files: []
- Corrupted files: []

**Test Set:**
- Total samples: 3000
- Image dimensions: min/max/mean height, width
- Format distribution
- Missing files: []
- Corrupted files: []

**Metadata:**
- Load timestamp
- Data directory path
- Data version (if tracked)
- Validation checksum (optional)

---

## Data Flow Diagram

```
CSV Files (training.csv, test.csv)
    │
    ▼
┌─────────────────────────────────┐
│ CSV Loader                      │
│ - Parse with explicit dtypes    │
│ - Validate columns present      │
│ - Validate no NaN values        │
└──────────┬──────────────────────┘
           │
    ┌──────▼──────────┐
    │ Path Resolver   │
    │ - Full path     │
    │ - Verify exists │
    └──────┬──────────┘
           │
    ┌──────▼─────────────────┐
    │ Image Inspector        │
    │ (parallel per image)   │
    │ - Dimensions           │
    │ - Format               │
    │ - File size            │
    │ - Readability check    │
    └──────┬─────────────────┘
           │
    ┌──────▼──────────────────┐
    │ Source Mapping          │
    │ (training only)         │
    │ - Add source_name col   │
    └──────┬──────────────────┘
           │
    ┌──────▼──────────────────┐
    │ Validators              │
    │ - Class distribution    │
    │ - File integrity        │
    │ - Image readability     │
    └──────┬──────────────────┘
           │
    ┌──────▼──────────────────┐
    │ Statistics Aggregator   │
    │ - Compute all stats     │
    │ - Generate report       │
    └──────┬──────────────────┘
           │
    ┌──────▼──────────────────┐
    │ Output                  │
    │ - train_df (DataFrame)  │
    │ - test_df (DataFrame)   │
    │ - data_stats (dict)     │
    └──────────────────────────┘
```

---

## Key Design Decisions

### Why NOT Load Images in Memory?
- 10,000 images × 1024×1024 × 3 bytes = ~30 GB RAM
- Metadata extraction is lightweight
- Actual loading deferred to Preprocessing pipeline
- Allows memory-efficient batch processing

### Why Store Both Original and Full Paths?
- **Original path:** Enables auditing ("which file caused this?")
- **Full path:** Enables actual file operations
- Redundancy allows catching path resolution bugs

### Why Validate Images but Continue on Errors?
- Real-world data is messy
- Single corrupted image shouldn't block entire pipeline
- User can manually handle reported issues
- Transparent logging supports debugging

### Why Include File Size in Metadata?
- Detect significantly smaller files (early corruption indicator)
- Anomaly detection (suspicious tiny files)
- Useful for storage analysis

---

## Error Handling Patterns

### Pattern: Permissive Logging
```
- Log all issues (missing files, corrupted images)
- Store issues list in output
- Continue processing
- User reviews report and decides action
```

### Pattern: Immutable Data
```
- CSV data loaded once
- Stored in immutable DataFrame
- Downstream pipelines create copies
- Original always available for reference
```

### Pattern: Checksum Tracking (Optional)
```
- Compute MD5 of data_stats.json
- Store in report
- Enables detecting if data changed between runs
```

---

## Performance Characteristics

| Operation | Complexity | Time (10K images) |
|-----------|-----------|------------------|
| Load CSVs | O(n) | < 1s |
| Resolve paths | O(n) | < 1s |
| Inspect images | O(n) | ~10-20s (parallel) |
| Validate distribution | O(n) | < 1s |
| Aggregate stats | O(n) | < 1s |
| **Total** | **O(n)** | **~20s** |

**n = 10,000 images**

---

## Reproducibility Guarantees

✅ **Deterministic:** No randomness, always same result  
✅ **Versioned:** Data paths and formats recorded  
✅ **Auditable:** Full error log and validation report  
✅ **Replayable:** Can re-run with same data snapshot  

---

## Integration Points

**Inputs from:**
- File system (Data/Training, Data/Test directories)
- CSV files (training.csv, test.csv, sources.txt)

**Outputs to:**
- EDA Pipeline (train_df, test_df, data_stats)
- Preprocessing Pipeline (train_df, test_df)
- Logging system (validation_report.txt)
