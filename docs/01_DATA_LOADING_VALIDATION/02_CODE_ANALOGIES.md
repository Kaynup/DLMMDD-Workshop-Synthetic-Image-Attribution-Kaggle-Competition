# Data Loading & Validation - Code Analogies

## Pseudocode: Main Pipeline

```python
def load_and_validate_data(data_dir, config):
    """
    Main pipeline function.
    
    Args:
        data_dir: Path to Data/ directory
        config: Configuration dict with settings
    
    Returns:
        train_df, test_df, data_stats
    """
    
    # === STAGE 1: Load CSVs ===
    train_df = load_csv_with_validation(
        data_dir / 'training.csv',
        expected_cols=['ID', 'path', 'y'],
        dtype={'ID': 'int64', 'path': 'str', 'y': 'int64'}
    )
    
    test_df = load_csv_with_validation(
        data_dir / 'test.csv',
        expected_cols=['ID', 'path'],
        dtype={'ID': 'int64', 'path': 'str'}
    )
    
    sources_map = load_sources_mapping(data_dir / 'sources.txt')
    
    # === STAGE 2: Resolve Paths ===
    train_df['full_path'] = train_df['path'].apply(
        lambda p: data_dir / 'Training' / p
    )
    test_df['full_path'] = test_df['path'].apply(
        lambda p: data_dir / 'Test' / p
    )
    
    # === STAGE 3: Inspect Images (Parallel) ===
    train_metadata = train_df['full_path'].parallel_apply(inspect_image)
    test_metadata = test_df['full_path'].parallel_apply(inspect_image)
    
    # Merge metadata back
    train_df = pd.concat([train_df, train_metadata], axis=1)
    test_df = pd.concat([test_df, test_metadata], axis=1)
    
    # === STAGE 4: Add Source Names ===
    train_df['source_name'] = train_df['y'].map(sources_map)
    
    # === STAGE 5: Validate ===
    validation_report = validate_all(train_df, test_df)
    
    # === STAGE 6: Aggregate Statistics ===
    data_stats = aggregate_statistics(
        train_df, test_df, sources_map, validation_report
    )
    
    # === STAGE 7: Save Report ===
    save_validation_report(validation_report, data_dir / 'validation_report.txt')
    save_json(data_stats, data_dir / 'data_stats.json')
    
    return train_df, test_df, data_stats


def load_csv_with_validation(path, expected_cols, dtype):
    """Load CSV with explicit column and dtype validation."""
    df = pd.read_csv(path, dtype=dtype)
    
    # Validate columns
    assert set(df.columns) == set(expected_cols), \
        f"Expected columns {expected_cols}, got {list(df.columns)}"
    
    # Validate no NaN
    assert not df.isnull().any().any(), \
        f"Found NaN values in {path}"
    
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def inspect_image(img_path):
    """Extract lightweight metadata from image without loading into RAM."""
    try:
        # Only open, don't load pixels
        with PIL.Image.open(img_path) as img:
            img.verify()  # Lightweight integrity check
            width, height = img.size
            
            return pd.Series({
                'width': width,
                'height': height,
                'format': img.format,
                'color_mode': img.mode,
                'file_size_bytes': os.path.getsize(img_path),
                'is_readable': True,
                'error': None
            })
    
    except Exception as e:
        logger.warning(f"Failed to inspect {img_path}: {e}")
        return pd.Series({
            'width': None,
            'height': None,
            'format': None,
            'color_mode': None,
            'file_size_bytes': os.path.getsize(img_path) if os.path.exists(img_path) else None,
            'is_readable': False,
            'error': str(e)
        })


def load_sources_mapping(sources_path):
    """Load generator source mapping from sources.txt."""
    sources = {}
    with open(sources_path, 'r') as f:
        for line in f:
            idx, name = line.strip().split('|')
            sources[int(idx)] = name
    
    assert len(sources) == 10, f"Expected 10 sources, got {len(sources)}"
    logger.info(f"Loaded {len(sources)} source mappings")
    return sources


def validate_all(train_df, test_df):
    """Run all validation checks."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }
    
    # Check 1: Row counts
    check_1 = len(train_df) == 7000 and len(test_df) == 3000
    report['checks']['row_counts'] = {
        'passed': check_1,
        'train_n': len(train_df),
        'test_n': len(test_df),
        'expected_train': 7000,
        'expected_test': 3000
    }
    
    # Check 2: Class distribution
    class_counts = train_df['y'].value_counts().sort_index()
    class_balanced = all(count == 1000 for count in class_counts.values)
    report['checks']['class_balance'] = {
        'passed': class_balanced,
        'counts': class_counts.to_dict(),
        'expected_per_class': 1000
    }
    
    # Check 3: File integrity
    missing_train = train_df[~train_df['is_readable']]['path'].tolist()
    missing_test = test_df[~test_df['is_readable']]['path'].tolist()
    report['checks']['file_integrity'] = {
        'passed': len(missing_train) == 0 and len(missing_test) == 0,
        'missing_training': missing_train,
        'missing_test': missing_test
    }
    
    # Log results
    for check_name, result in report['checks'].items():
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        logger.info(f"{status}: {check_name}")
    
    return report


def aggregate_statistics(train_df, test_df, sources_map, validation_report):
    """Compute comprehensive statistics for logging and EDA context."""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'train': {
            'n_samples': len(train_df),
            'n_classes': train_df['y'].nunique(),
            'class_distribution': train_df['y'].value_counts().sort_index().to_dict(),
            'class_names': sources_map,
            'image_stats': {
                'avg_width': train_df['width'].mean(),
                'avg_height': train_df['height'].mean(),
                'min_width': train_df['width'].min(),
                'max_width': train_df['width'].max(),
                'avg_file_size_mb': (train_df['file_size_bytes'].sum() / 1e6),
            },
            'format_distribution': train_df['format'].value_counts().to_dict(),
            'color_mode_distribution': train_df['color_mode'].value_counts().to_dict(),
            'readable_count': train_df['is_readable'].sum(),
            'corrupted_count': (~train_df['is_readable']).sum(),
        },
        'test': {
            'n_samples': len(test_df),
            'image_stats': {
                'avg_width': test_df['width'].mean(),
                'avg_height': test_df['height'].mean(),
                'avg_file_size_mb': (test_df['file_size_bytes'].sum() / 1e6),
            },
            'format_distribution': test_df['format'].value_counts().to_dict(),
            'readable_count': test_df['is_readable'].sum(),
            'corrupted_count': (~test_df['is_readable']).sum(),
        },
        'validation': validation_report['checks']
    }
    
    return stats
```

---

## Code Pattern: Parallel Image Inspection

### Approach 1: Using pandas apply (simple)
```python
# Sequential
train_metadata = train_df['full_path'].apply(inspect_image)
```

### Approach 2: Using parallel_apply (faster)
```python
# Parallel with pandarallel
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=8)

train_metadata = train_df['full_path'].parallel_apply(inspect_image)
```

### Approach 3: Using joblib (most control)
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(inspect_image)(path) for path in train_df['full_path']
)
train_metadata = pd.DataFrame(results)
```

---

## Code Pattern: Error Aggregation

```python
class ValidationReport:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def add_issue(self, category, message, severity='ERROR'):
        """Add a validation issue."""
        self.issues.append({
            'severity': severity,
            'category': category,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_stat(self, key, value):
        """Add a statistic."""
        self.stats[key] = value
    
    def to_dict(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'issues': self.issues,
            'warnings': self.warnings,
            'stats': self.stats,
            'summary': {
                'total_issues': len(self.issues),
                'total_warnings': len(self.warnings),
                'passed': len(self.issues) == 0
            }
        }
    
    def to_text(self):
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "DATA VALIDATION REPORT",
            "=" * 60,
            f"Timestamp: {datetime.now()}",
            "",
            "SUMMARY",
            "-" * 60,
            f"Issues: {len(self.issues)}",
            f"Warnings: {len(self.warnings)}",
            f"Status: {'PASS' if len(self.issues) == 0 else 'FAIL'}",
            "",
            "ISSUES",
            "-" * 60,
        ]
        
        for issue in self.issues:
            lines.append(f"[{issue['severity']}] {issue['category']}: {issue['message']}")
        
        if not self.issues:
            lines.append("No issues found!")
        
        lines.append("")
        lines.append("STATISTICS")
        lines.append("-" * 60)
        for key, value in self.stats.items():
            lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
```

---

## Code Pattern: Data Immutability

```python
# Approach 1: Return frozen DataFrames
class FrozenDataFrame:
    def __init__(self, df):
        self._df = df.copy()
        self._df.flags.writeable = False
    
    def get(self):
        return self._df.copy()  # Always return copies

# Approach 2: Use tuple of tuples (more immutable)
train_df = pd.DataFrame(...)
train_tuple = tuple(train_df.itertuples(index=False))
# Downstream can convert back: pd.DataFrame(train_tuple)

# Approach 3: Store as parquet (immutable on disk)
train_df.to_parquet('train_data.parquet')
train_df_loaded = pd.read_parquet('train_data.parquet')  # Always fresh copy
```

---

## Integration Example

```python
# In Jupyter notebook
import pandas as pd
import logging
from pathlib import Path
from data_loading_validation import (
    load_and_validate_data,
    load_csv_with_validation,
    inspect_image,
    validate_all,
    aggregate_statistics
)

# Setup
logging.basicConfig(level=logging.INFO)
data_dir = Path('Data')
config = {'parallel_workers': 8}

# Execute pipeline
train_df, test_df, data_stats = load_and_validate_data(data_dir, config)

# Inspect results
print(f"Train shape: {train_df.shape}")
print(f"Train columns: {train_df.columns.tolist()}")
print(f"\nClass distribution:")
print(train_df['y'].value_counts().sort_index())
print(f"\nData stats:")
import json
print(json.dumps(data_stats['train']['image_stats'], indent=2))
```

---

## Key Takeaways

1. **Immutability:** Never modify source CSVs, always work with copies
2. **Validation First:** Check data integrity before downstream processing
3. **Detailed Logging:** Log each validation step with context
4. **Graceful Degradation:** Continue on errors, log for manual review
5. **Metadata Tracking:** Store image info for reproducibility and debugging
6. **Parallel Processing:** Use parallel_apply for 10K images (20s vs 3min sequential)
