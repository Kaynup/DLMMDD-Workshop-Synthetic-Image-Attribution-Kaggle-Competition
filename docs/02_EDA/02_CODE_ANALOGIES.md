# EDA - Code Analogies & Patterns

## Main EDA Function

```python
def run_eda(train_df, test_df, data_stats, config):
    """
    Main EDA pipeline orchestrator.
    
    Args:
        train_df: Training metadata (from Data Loading)
        test_df: Test metadata (from Data Loading)
        data_stats: Statistics dict (from Data Loading)
        config: Configuration dict
    
    Returns:
        eda_report: Comprehensive analysis report
        eda_plots: Dict of matplotlib figures
    """
    
    logger.info("Starting EDA...")
    
    # Initialize report
    eda_report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'analyses': {}
    }
    
    output_dir = Path(config['output_dir']) / 'eda'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === Analysis 1: Class Distribution ===
    logger.info("Analyzing class distribution...")
    eda_report['analyses']['class_distribution'] = analyze_class_distribution(
        train_df, output_dir
    )
    
    # === Analysis 2: Image Characteristics ===
    logger.info("Analyzing image characteristics...")
    eda_report['analyses']['image_characteristics'] = analyze_image_characteristics(
        train_df, test_df, output_dir
    )
    
    # === Analysis 3: Generator Signatures (Optional, slow) ===
    if config.get('compute_generator_templates', False):
        logger.info("Extracting generator signatures (this may take a while)...")
        eda_report['analyses']['generator_signatures'] = extract_generator_signatures(
            train_df, config['data_dir'] / 'Training', output_dir
        )
    
    # === Analysis 4: Post-Processing Analysis ===
    logger.info("Analyzing post-processing impact...")
    eda_report['analyses']['postprocessing'] = analyze_postprocessing_impact(
        train_df, test_df, output_dir
    )
    
    # === Generate All Visualizations ===
    logger.info("Generating visualizations...")
    eda_plots = generate_all_visualizations(train_df, test_df, eda_report, output_dir)
    
    # === Compile HTML Report ===
    logger.info("Compiling HTML report...")
    html_report = compile_html_report(eda_report, eda_plots, output_dir)
    
    # === Save Results ===
    logger.info("Saving results...")
    save_eda_report(eda_report, output_dir / 'eda_report.json')
    with open(output_dir / 'eda_report.html', 'w') as f:
        f.write(html_report)
    
    logger.info(f"EDA complete! Report saved to {output_dir / 'eda_report.html'}")
    
    return eda_report, eda_plots


# === Analysis Functions ===

def analyze_class_distribution(train_df, output_dir):
    """Analyze training set class balance."""
    
    # Compute statistics
    class_counts = train_df['y'].value_counts().sort_index()
    class_names = ['AuraFlow', 'Freepik', 'Lumina', 'Photon', 'Pixart',
                   'Playground', 'SD3', 'SD3.5', 'SDXLTurbo', 'Hunyuan']
    
    stats = {
        'counts': class_counts.to_dict(),
        'counts_with_names': {class_names[i]: count for i, count in enumerate(class_counts)},
        'is_balanced': (class_counts == 1000).all(),
        'min_count': int(class_counts.min()),
        'max_count': int(class_counts.max()),
        'mean_count': float(class_counts.mean()),
        'std_count': float(class_counts.std())
    }
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(10), class_counts.values, color='steelblue')
    ax.axhline(1000, color='red', linestyle='--', linewidth=2, label='Expected (1000)')
    ax.set_xlabel('Generator Class', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Training Set Class Distribution (n=7000)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(10))
    ax.set_xticklabels([f'{i}\n{class_names[i][:10]}' for i in range(10)], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'class_distribution.png', dpi=100)
    plt.close()
    
    stats['plot_path'] = 'class_distribution.png'
    return stats


def analyze_image_characteristics(train_df, test_df, output_dir):
    """Analyze image dimensions, file sizes, formats."""
    
    # Compute statistics
    train_stats = {
        'dimensions': {
            'height': {
                'mean': float(train_df['height'].mean()),
                'std': float(train_df['height'].std()),
                'min': int(train_df['height'].min()),
                'max': int(train_df['height'].max()),
                'unique_values': int(train_df['height'].nunique())
            },
            'width': {
                'mean': float(train_df['width'].mean()),
                'std': float(train_df['width'].std()),
                'min': int(train_df['width'].min()),
                'max': int(train_df['width'].max()),
                'unique_values': int(train_df['width'].nunique())
            }
        },
        'file_size_mb': {
            'mean': float((train_df['file_size_bytes'] / 1e6).mean()),
            'std': float((train_df['file_size_bytes'] / 1e6).std()),
            'min': float((train_df['file_size_bytes'] / 1e6).min()),
            'max': float((train_df['file_size_bytes'] / 1e6).max())
        },
        'formats': train_df['format'].value_counts().to_dict(),
        'color_modes': train_df['color_mode'].value_counts().to_dict()
    }
    
    test_stats = {
        'dimensions': {
            'height': {
                'mean': float(test_df['height'].mean()),
                'std': float(test_df['height'].std()),
                'min': int(test_df['height'].min()),
                'max': int(test_df['height'].max())
            },
            'width': {
                'mean': float(test_df['width'].mean()),
                'std': float(test_df['width'].std()),
                'min': int(test_df['width'].min()),
                'max': int(test_df['width'].max())
            }
        },
        'file_size_mb': {
            'mean': float((test_df['file_size_bytes'] / 1e6).mean()),
            'std': float((test_df['file_size_bytes'] / 1e6).std()),
            'min': float((test_df['file_size_bytes'] / 1e6).min()),
            'max': float((test_df['file_size_bytes'] / 1e6).max())
        },
        'formats': test_df['format'].value_counts().to_dict(),
        'color_modes': test_df['color_mode'].value_counts().to_dict()
    }
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Height distribution
    axes[0, 0].hist(train_df['height'], bins=20, alpha=0.7, label='Train', color='blue')
    axes[0, 0].hist(test_df['height'], bins=20, alpha=0.7, label='Test', color='orange')
    axes[0, 0].set_xlabel('Height (pixels)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Height Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Subplot 2: Width distribution
    axes[0, 1].hist(train_df['width'], bins=20, alpha=0.7, label='Train', color='blue')
    axes[0, 1].hist(test_df['width'], bins=20, alpha=0.7, label='Test', color='orange')
    axes[0, 1].set_xlabel('Width (pixels)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Width Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Subplot 3: File size distribution
    axes[1, 0].hist(train_df['file_size_bytes'] / 1e6, bins=20, alpha=0.7, label='Train', color='blue')
    axes[1, 0].hist(test_df['file_size_bytes'] / 1e6, bins=20, alpha=0.7, label='Test', color='orange')
    axes[1, 0].set_xlabel('File Size (MB)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('File Size Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Subplot 4: Format pie chart
    format_counts = train_df['format'].value_counts()
    axes[1, 1].pie(format_counts.values, labels=format_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Image Format Distribution (Train)')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'image_characteristics.png', dpi=100)
    plt.close()
    
    return {'train': train_stats, 'test': test_stats, 'plot_path': 'image_characteristics.png'}


def analyze_postprocessing_impact(train_df, test_df, output_dir):
    """Infer likely post-processing on test set."""
    
    # Compute metrics
    train_file_size_mb = (train_df['file_size_bytes'] / 1e6).mean()
    test_file_size_mb = (test_df['file_size_bytes'] / 1e6).mean()
    size_reduction_pct = (train_file_size_mb - test_file_size_mb) / train_file_size_mb * 100
    
    train_height_mean = train_df['height'].mean()
    test_height_mean = test_df['height'].mean()
    height_change_pct = abs(train_height_mean - test_height_mean) / train_height_mean * 100
    
    train_width_mean = train_df['width'].mean()
    test_width_mean = test_df['width'].mean()
    width_change_pct = abs(train_width_mean - test_width_mean) / train_width_mean * 100
    
    # Infer post-processing
    likely_operations = []
    confidence_score = 0
    
    if size_reduction_pct > 15:
        likely_operations.append('JPEG or WebP compression')
        confidence_score += 0.4
    
    if height_change_pct > 2 or width_change_pct > 2:
        likely_operations.append('Cropping or resizing')
        confidence_score += 0.3
    
    if test_df['format'].apply(lambda x: x == 'JPEG' if x else False).sum() > 0:
        likely_operations.append('JPEG format applied')
        confidence_score += 0.3
    
    analysis = {
        'train_avg_file_size_mb': float(train_file_size_mb),
        'test_avg_file_size_mb': float(test_file_size_mb),
        'file_size_reduction_pct': float(size_reduction_pct),
        'height_change_pct': float(height_change_pct),
        'width_change_pct': float(width_change_pct),
        'likely_operations': likely_operations,
        'confidence': float(min(confidence_score, 1.0))
    }
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    categories = ['Average\nFile Size (MB)', 'Average\nHeight (px)']
    train_vals = [train_file_size_mb, train_height_mean]
    test_vals = [test_file_size_mb, test_height_mean]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, train_vals, width, label='Train', color='steelblue')
    axes[0].bar(x + width/2, test_vals, width, label='Test', color='orange')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Train vs Test Characteristics')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].axis('off')
    inference_text = (
        f"Post-Processing Inference:\n\n"
        f"File size reduction: {size_reduction_pct:.1f}%\n"
        f"Height change: {height_change_pct:.2f}%\n"
        f"Width change: {width_change_pct:.2f}%\n\n"
        f"Likely operations:\n" +
        "\n".join([f"  • {op}" for op in likely_operations]) +
        f"\n\nConfidence: {confidence_score:.1%}"
    )
    axes[1].text(0.1, 0.5, inference_text, fontsize=11, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(output_dir / 'postprocessing_analysis.png', dpi=100)
    plt.close()
    
    return {**analysis, 'plot_path': 'postprocessing_analysis.png'}
```

---

## HTML Report Compilation

```python
def compile_html_report(eda_report, eda_plots, output_dir):
    """Compile all analyses and plots into single HTML file."""
    
    class_dist = eda_report['analyses']['class_distribution']
    img_chars = eda_report['analyses']['image_characteristics']
    postproc = eda_report['analyses']['postprocessing']
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EDA Report - Synthetic Image Attribution</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            h1 {{ color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
            h2 {{ color: #0066cc; margin-top: 30px; }}
            .section {{ background: white; padding: 15px; margin: 15px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #0066cc; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric {{ font-weight: bold; color: #0066cc; }}
            img {{ max-width: 100%; height: auto; margin: 15px 0; }}
        </style>
    </head>
    <body>
        <h1>📊 Synthetic Image Attribution - Exploratory Data Analysis</h1>
        <p><strong>Generated:</strong> {eda_report['timestamp']}</p>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <ul>
                <li>Dataset: <span class="metric">7,000</span> training images, <span class="metric">3,000</span> test images</li>
                <li>Classes: <span class="metric">10 generators</span> (perfectly balanced at 1,000 each)</li>
                <li>Image dimensions: <span class="metric">~{img_chars['train']['dimensions']['height']['mean']:.0f}×{img_chars['train']['dimensions']['width']['mean']:.0f} pixels</span></li>
                <li>Primary finding: {postproc['likely_operations'][0] if postproc['likely_operations'] else 'Minimal post-processing expected'}</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>1. Class Distribution Analysis</h2>
            <p>Verification that all generator classes are equally represented.</p>
            <img src="data:image/png;base64,{encode_image_to_base64(output_dir / class_dist['plot_path'])}" />
            <p><strong>Status:</strong> {'✅ Perfectly Balanced' if class_dist['is_balanced'] else '⚠️ Imbalanced'}</p>
        </div>
        
        <div class="section">
            <h2>2. Image Characteristics</h2>
            <p>Analysis of image dimensions, file sizes, and formats.</p>
            <img src="data:image/png;base64,{encode_image_to_base64(output_dir / img_chars['plot_path'])}" />
            
            <h3>Training Set Statistics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Height</th>
                    <th>Width</th>
                    <th>File Size (MB)</th>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{img_chars['train']['dimensions']['height']['mean']:.0f}</td>
                    <td>{img_chars['train']['dimensions']['width']['mean']:.0f}</td>
                    <td>{img_chars['train']['file_size_mb']['mean']:.2f}</td>
                </tr>
                <tr>
                    <td>Std Dev</td>
                    <td>{img_chars['train']['dimensions']['height']['std']:.2f}</td>
                    <td>{img_chars['train']['dimensions']['width']['std']:.2f}</td>
                    <td>{img_chars['train']['file_size_mb']['std']:.2f}</td>
                </tr>
                <tr>
                    <td>Min</td>
                    <td>{img_chars['train']['dimensions']['height']['min']}</td>
                    <td>{img_chars['train']['dimensions']['width']['min']}</td>
                    <td>{img_chars['train']['file_size_mb']['min']:.2f}</td>
                </tr>
                <tr>
                    <td>Max</td>
                    <td>{img_chars['train']['dimensions']['height']['max']}</td>
                    <td>{img_chars['train']['dimensions']['width']['max']}</td>
                    <td>{img_chars['train']['file_size_mb']['max']:.2f}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>3. Post-Processing Analysis</h2>
            <p>Inference of post-processing operations applied to test set.</p>
            <img src="data:image/png;base64,{encode_image_to_base64(output_dir / postproc['plot_path'])}" />
            
            <p><strong>File Size Reduction:</strong> {postproc['file_size_reduction_pct']:.1f}%</p>
            <p><strong>Height Change:</strong> {postproc['height_change_pct']:.2f}%</p>
            <p><strong>Likely Operations:</strong></p>
            <ul>
                {(''.join([f'<li>{op}</li>' for op in postproc['likely_operations']]) 
                  if postproc['likely_operations'] else '<li>Minimal processing</li>')}
            </ul>
            <p><strong>Confidence:</strong> {postproc['confidence']:.0%}</p>
        </div>
    </body>
    </html>
    """
    
    return html


def encode_image_to_base64(image_path):
    """Encode PNG image to base64 for embedding in HTML."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()
```

---

## Key Output

**EDA produces:**
1. `eda_report.html` - Self-contained HTML with all plots embedded
2. `eda_insights.json` - Machine-readable statistics
3. `plots/` folder - Individual PNG files for each plot

**Used by:** Downstream pipelines for feature engineering insights
