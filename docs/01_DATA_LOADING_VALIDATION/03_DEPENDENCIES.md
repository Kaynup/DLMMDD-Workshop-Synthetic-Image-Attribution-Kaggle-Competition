# Data Loading & Validation - Dependencies

## External Libraries

| Library | Version | Purpose | Used For |
|---------|---------|---------|----------|
| `pandas` | >=1.3.0 | Data manipulation | CSV loading, metadata merging |
| `Pillow` (PIL) | >=8.0.0 | Image processing | Image metadata extraction |
| `numpy` | >=1.20.0 | Numerical operations | Statistics computation |
| `pathlib` | (stdlib) | Path handling | Cross-platform file operations |
| `logging` | (stdlib) | Logging | Pipeline event tracking |
| `json` | (stdlib) | JSON serialization | Statistics storage |
| `os` | (stdlib) | OS operations | File size, existence checks |
| `datetime` | (stdlib) | Timestamps | Report timestamping |

### Optional (for performance)
| Library | Purpose |
|---------|---------|
| `pandarallel` | Parallel apply() operations |
| `joblib` | Distributed parallel processing |
| `tqdm` | Progress bars |

---

## Internal Dependencies (from this project)

```
None (self-contained pipeline)
```

**Note:** This pipeline has NO dependencies on other project pipelines.
It's the entry point - everything depends on it, but it depends on nothing.

---

## Data Dependencies

### Inputs Required
```
Data/
├── training.csv
│   Columns: ID (int), path (str), y (int)
│   Rows: 7000
├── test.csv
│   Columns: ID (int), path (str)
│   Rows: 3000
├── sources.txt
│   Format: "idx|name" per line
│   Lines: 10
├── Training/
│   Files: 0.png, 1.png, ..., 6999.png (7000 images)
└── Test/
    Files: Random integer names (3000 images)
```

### Data Assumptions
- All training images exist at `Data/Training/{path}`
- All test images exist at `Data/Test/{path}`
- All training images are 1024×1024 pixels (or thereabouts)
- All images are readable by PIL (RGB, PNG/JPEG format)
- Class distribution is perfectly balanced (1000 per class)

---

## Output Dependencies (consumed by)

### Direct Consumers
1. **EDA Pipeline** - Uses `train_df`, `test_df`, `data_stats`
2. **Preprocessing Pipeline** - Uses `train_df`, `test_df`
3. **Logging System** - Consumes `data_validation_report.txt`

### Indirect Consumers (through downstream pipelines)
- All subsequent pipelines transitively depend on this one
- Train/Val Split pipeline depends on `train_df`
- Model Training depends on train/val splits

---

## File I/O Dependencies

### Read Operations
| File | Purpose | Read Mode |
|------|---------|-----------|
| `training.csv` | Load metadata | Text (CSV) |
| `test.csv` | Load metadata | Text (CSV) |
| `sources.txt` | Load class mappings | Text |
| `Data/Training/*.png` | Image metadata | Binary (PIL open) |
| `Data/Test/*.png` | Image metadata | Binary (PIL open) |

### Write Operations
| File | Purpose | Write Mode | Overwrite |
|------|---------|-----------|-----------|
| `data_validation_report.txt` | Validation summary | Text | Yes |
| `data_stats.json` | Statistics JSON | Text | Yes |
| `train_df.parquet` (optional) | Checkpoint | Binary | Yes |
| `test_df.parquet` (optional) | Checkpoint | Binary | Yes |

---

## Environment Dependencies

### Python Version
- **Minimum:** Python 3.8+
- **Recommended:** Python 3.9+
- **Tested:** Python 3.10, 3.11

### System Resources
| Resource | Typical | Peak |
|----------|---------|------|
| Memory (RAM) | 2-4 GB | 4-8 GB (if not parallel) |
| Disk Space | ~50 MB (input) + 5 MB (output) | 55 MB |
| CPU Cores | 1 (sequential) | 8 (parallel recommended) |
| Time (sequential) | ~3 minutes | N/A |
| Time (parallel, 8 cores) | ~20 seconds | N/A |

### Working Directory
- Must be in project root where `Data/` folder is accessible
- Can be relative or absolute path

---

## Configuration Dependencies

### Required Config Keys
```python
config = {
    'data_dir': Path('Data'),
    'parallel_workers': 8,  # -1 for CPU count
    'verify_checksums': False,
    'save_parquet': False,  # Optional
}
```

### Config Sources (in priority order)
1. Function argument (highest priority)
2. Environment variables
3. `config.yaml` or `config.toml`
4. Defaults in code

---

## Logging Dependencies

### Logging Setup Required (in main notebook)
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Logger Names
- `data_loading_validation` - Main pipeline logger
- `data_loading_validation.loader` - CSV loader
- `data_loading_validation.inspector` - Image inspector
- `data_loading_validation.validator` - Validation checks

---

## Downstream Pipeline Contracts

### What Downstream Pipelines Expect

**EDA Pipeline expects:**
```python
train_df: pd.DataFrame with columns:
  - ID, path, y, source_name, full_path
  - width, height, format, color_mode, is_readable
  
test_df: pd.DataFrame with columns:
  - ID, path, full_path
  - width, height, format, color_mode, is_readable
  
data_stats: dict with keys:
  - 'timestamp', 'train', 'test', 'validation'
```

**Preprocessing Pipeline expects:**
```python
train_df: pd.DataFrame with columns:
  - ID, full_path, y (at minimum)
  
test_df: pd.DataFrame with columns:
  - ID, full_path (at minimum)
```

---

## Compatibility Notes

### Compatibility with Different Data Layouts
```
Current structure assumes:
Data/
├── Training/
└── Test/

Can adapt to:
├── train/images/
└── test/images/

By modifying path resolution in Component 2 (Path Resolver)
```

### Compatibility with Different CSV Formats
```
If training.csv has different columns, modify:
- load_csv_with_validation() dtype specification
- Class validation checks
- Statistics aggregation
```

---

## Reproducibility Dependencies

### Fixed Seeds (N/A - no randomness)
- Pipeline is deterministic
- No random operations

### Version Tracking
- Record data version in statistics
- Store Python package versions in `requirements.txt`:
  ```
  pandas>=1.3.0
  Pillow>=8.0.0
  numpy>=1.20.0
  ```

---

## Testing Dependencies

### Unit Test Requirements
```python
# For testing:
import pytest
import tempfile
import shutil

# Create mock data structure:
tmp_dir = tempfile.mkdtemp()
# Create Data/Training and Data/Test subdirectories
# Create sample CSV files
# Create sample PNG images
# Run pipeline tests
```

---

## Integration Checklist

Before running this pipeline, ensure:

- [ ] `Data/` folder exists with correct structure
- [ ] All required CSV files present (training.csv, test.csv, sources.txt)
- [ ] All image files accessible at specified paths
- [ ] Python 3.8+ installed
- [ ] Required libraries installed: `pip install pandas Pillow numpy`
- [ ] Logging configured in main notebook
- [ ] Output directory writable
- [ ] Sufficient disk space (100 MB recommended)

---

## Troubleshooting Dependencies

### "No such file or directory: Data/training.csv"
- Check working directory: `print(os.getcwd())`
- Verify relative paths from project root
- Use absolute paths if needed

### "PIL cannot open image: Data/Training/1234.png"
- Image format not supported or file corrupted
- Check file is readable: `file Data/Training/1234.png`
- Try opening in image viewer

### "ModuleNotFoundError: No module named 'pandas'"
- Install: `pip install pandas`
- Check environment: `python --version`, `pip --version`

### "Memory Error: list index out of range"
- Reduce parallel workers: `parallel_workers=4`
- Switch to sequential: `parallel_workers=1`
- Increase system RAM if available

---

## Version Compatibility

### Known Working Versions
- pandas 1.3.5, 1.4.x, 1.5.x
- Pillow 8.4.0, 9.x, 10.x
- numpy 1.21.x, 1.22.x, 1.23.x

### Known Issues
- pandas 2.0+ may have different memory behavior
- Pillow <8.0 doesn't support all image formats
- Python 3.7 and earlier not supported (f-strings)
