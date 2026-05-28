# EDA - Dependencies

## Python Libraries

| Library | Version | Purpose | Used For |
|---------|---------|---------|----------|
| `pandas` | >=1.3.0 | Data manipulation | DataFrame operations |
| `numpy` | >=1.20.0 | Numerical operations | Array operations, statistics |
| `matplotlib` | >=3.4.0 | Plotting | Static plots (PNG) |
| `seaborn` | >=0.11.0 | Statistical visualization | Enhanced plots with stats |
| `scipy` | >=1.7.0 | Scientific computing | Statistical tests |
| `PIL` (Pillow) | >=8.0.0 | Image processing | Load images for templates |
| `logging` | (stdlib) | Logging | Pipeline event tracking |
| `datetime` | (stdlib) | Timestamps | Report timestamping |
| `json` | (stdlib) | Serialization | Save insights JSON |
| `base64` | (stdlib) | Encoding | Embed images in HTML |

### Optional Libraries

| Library | Purpose |
|---------|---------|
| `plotly` | Interactive HTML plots (alternative to matplotlib) |
| `altair` | Declarative visualization (alternative) |
| `tqdm` | Progress bars for long operations |

---

## Input Dependencies

### Data Inputs
```python
# From Data Loading & Validation pipeline:
train_df: pd.DataFrame
  Columns: ID, path, y, source_name, full_path,
           width, height, format, color_mode, is_readable, file_size_bytes
  Rows: 7000

test_df: pd.DataFrame
  Columns: ID, path, full_path,
           width, height, format, color_mode, is_readable, file_size_bytes
  Rows: 3000

data_stats: dict
  Keys: timestamp, train, test, validation, sources
```

### File Inputs
```
Data/Training/*.png  # Images for template extraction
  (Optional - only needed if compute_generator_templates=True)
```

---

## Configuration Dependencies

```python
config = {
    'output_dir': 'outputs',
    'compute_generator_templates': False,  # If True, slow but informative
    'template_samples_per_class': 100,     # Limit for speed
    'data_dir': Path('Data'),              # For image loading
    'figure_dpi': 100,
    'figure_style': 'seaborn',
}
```

---

## Output Dependencies

### Files Generated
| File | Type | Format | Purpose |
|------|------|--------|---------|
| `eda_report.html` | Report | HTML | Main self-contained report |
| `eda_insights.json` | Data | JSON | Machine-readable statistics |
| `class_distribution.png` | Plot | PNG | Embedded in HTML |
| `image_characteristics.png` | Plot | PNG | Embedded in HTML |
| `postprocessing_analysis.png` | Plot | PNG | Embedded in HTML |
| `generator_templates.png` | Plot | PNG | (Optional) Embedded in HTML |

### Data Format (eda_insights.json)
```json
{
  "timestamp": "2026-05-19T10:30:00",
  "class_distribution": {
    "counts": {0: 1000, 1: 1000, ...},
    "is_balanced": true,
    "min_count": 1000,
    "max_count": 1000
  },
  "image_characteristics": {
    "train": {
      "dimensions": {
        "height": {"mean": 1024, "std": 0, "min": 1024, "max": 1024},
        "width": {"mean": 1024, "std": 0, "min": 1024, "max": 1024}
      },
      "file_size_mb": {"mean": 0.25, "std": 0.01, ...},
      "formats": {"PNG": 7000},
      "color_modes": {"RGB": 7000}
    },
    "test": {...}
  },
  "postprocessing": {
    "file_size_reduction_pct": 5.2,
    "height_change_pct": 0.1,
    "likely_operations": ["JPEG compression"],
    "confidence": 0.65
  }
}
```

---

## Consumer Dependencies

### Direct Consumers
1. **Feature Extraction Pipeline** - Uses insights about image characteristics
2. **Model Training Pipeline** - Uses class distribution for loss weighting
3. **Report & Documentation** - Embeds EDA findings

### Indirect Consumers
- All downstream pipelines benefit from EDA insights for hyperparameter tuning

---

## Environment Dependencies

### Python Version
- **Minimum:** Python 3.8+
- **Recommended:** Python 3.9+

### System Resources
| Resource | Typical | Peak |
|----------|---------|------|
| Memory (RAM) | 4 GB | 8 GB (if compute_generator_templates=True) |
| Disk Space | ~10 MB output | 50 MB (if saving all templates) |
| CPU Cores | 1 (sequential) | 4+ (for parallel operations) |
| Runtime (without templates) | 1-2 minutes | N/A |
| Runtime (with templates) | 5-10 minutes | N/A |

---

## Logging Dependencies

### Logger Configuration Required
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('eda')
```

### Logger Names Used
- `eda` - Main pipeline logger
- `eda.class_distribution` - Class distribution analysis
- `eda.image_characteristics` - Image characteristic analysis
- `eda.postprocessing` - Post-processing inference
- `eda.visualization` - Visualization generation

---

## Performance Dependencies

### Plotting Libraries
- **matplotlib:** Better for static plots, embeds well in HTML
- **seaborn:** Prettier defaults, requires matplotlib
- **plotly:** Better for interactive plots, but larger file size

### Parallel Processing (Optional)
```python
from joblib import Parallel, delayed

# Can parallelize template extraction:
results = Parallel(n_jobs=-1)(
    delayed(extract_class_template)(class_id)
    for class_id in range(10)
)
```

---

## Integration Checklist

Before running EDA:

- [ ] Data Loading & Validation pipeline complete
- [ ] train_df, test_df, data_stats available
- [ ] All required libraries installed: `pip install pandas numpy matplotlib seaborn scipy`
- [ ] Output directory writable
- [ ] Sufficient disk space (50 MB recommended)
- [ ] Logging configured

---

## Version Compatibility

### Tested Versions
- pandas 1.3.5 ✅
- numpy 1.21.0 ✅
- matplotlib 3.4.2 ✅
- seaborn 0.11.1 ✅
- scipy 1.7.0 ✅

### Known Issues
- matplotlib <3.4: May have different default styles
- seaborn <0.11: Different color palettes
- numpy 2.0: May have breaking changes (not tested)

---

## Memory Profile

For different operations:

| Operation | Memory Used | Duration |
|-----------|-------------|----------|
| Load train/test metadata | 50 MB | <1s |
| Generate basic plots | 200 MB | 10s |
| Extract 100 templates per class (10×100 images) | 4 GB | 2 min |
| Compile HTML report | 100 MB | 5s |

**Recommendation:** Set `compute_generator_templates=False` for first run (fast feedback), then enable for deeper insights.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'seaborn'"
```bash
pip install seaborn matplotlib scipy
```

### "PNG images not found" (for templates)
- Check `Data/Training/` directory exists
- Set `compute_generator_templates=False` if not needed

### "Memory Error when extracting templates"
- Reduce `template_samples_per_class` (e.g., 50 instead of 100)
- Or disable template generation

### Plots not showing/saving
- Check output directory: `output_dir = Path(config['output_dir']) / 'eda'`
- Ensure directory is writable: `os.makedirs(output_dir, exist_ok=True)`

---

## Reproducibility

### Reproducible Settings
```python
# EDA is deterministic (no randomness)
config = {
    'output_dir': 'outputs',
    'compute_generator_templates': False,
    'template_samples_per_class': 100,
    'data_dir': Path('Data'),
    'figure_dpi': 100,
    'figure_style': 'seaborn-v0_8-darkgrid',  # Fixed style
}
```

### Seeded Dependencies
- No random seed needed (EDA is deterministic)
- Plots will be identical across runs (same config, same data)

---

## Documentation References

### matplotlib tutorials
- https://matplotlib.org/stable/tutorials/index.html

### seaborn gallery
- https://seaborn.pydata.org/examples.html

### PIL/Pillow image operations
- https://pillow.readthedocs.io/

### Helpful Patterns
- **Embedding images in HTML:** Use base64 encoding
- **Creating multi-subplot figures:** Use `plt.subplots(rows, cols)`
- **Per-class statistics:** Use `groupby()` and `agg()`
