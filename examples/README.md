# Didactic Engine Examples

This directory contains comprehensive examples and tutorials for using Didactic Engine.

## Quick Start

### 1. Interactive Tutorials

#### Option A: Jupyter Notebook (Traditional)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakuzadave/didactic-engine/blob/main/examples/tutorial.ipynb)

```bash
jupyter notebook tutorial.ipynb
```

#### Option B: Marimo Notebook (Pure Python, Git-Friendly)
```bash
pip install marimo
marimo edit tutorial_marimo.py
```

See [MARIMO_README.md](MARIMO_README.md) for details on Marimo advantages.

### 2. Basic Usage Examples

**[example_usage.py](example_usage.py)** - Simple API usage patterns

```bash
python examples/example_usage.py
```

Covers:
- Single file processing
- Advanced configuration with Essentia
- Batch processing basics
- Custom configuration (time signatures, sample rates)
- Using individual components (ingester, analyzer, preprocessor)

### 3. Batch Processing Examples

**[batch_processing_example.py](batch_processing_example.py)** - Advanced batch workflows

```bash
# Install dependencies
pip install -e ".[batch]"

# Run examples
python examples/batch_processing_example.py
```

The `batch` extra includes:
- `tqdm` for progress bars
- `polars` (recommended) for fast Parquet aggregation/statistics in large batch runs

Demonstrates:
- **Simple batch**: Process multiple files with `AudioPipeline.process_batch()`
- **Parallel processing**: Use multiprocessing to process files concurrently
- **Progress tracking**: Add tqdm progress bars for long-running batches
- **Directory tree processing**: Recursively process nested directories
- **Retry logic**: Automatic retry for failed files
- **Results serialization**: Save batch results to JSON

Example output:
```
Processed 8/10 files successfully
Failed: 2 files
Results saved to output/batch_parallel/results.json
```

### 4. Complete Workflow (Batch + Visualizations)

**[workflow_batch_viz.py](workflow_batch_viz.py)** - Production-ready workflow

```bash
# Process all WAV files in a directory
python examples/workflow_batch_viz.py sample_audio/ --output results/

# Process specific files with 8 parallel workers
python examples/workflow_batch_viz.py song1.wav song2.wav song3.wav --workers 8

# Process without visualizations (faster)
python examples/workflow_batch_viz.py sample_audio/*.wav --no-viz
```

This complete workflow:
1. **Batch processes** multiple files in parallel
2. **Aggregates** results across all songs
3. **Creates visualizations**: batch summary, feature comparisons, correlations
4. **Exports reports**: JSON results, CSV datasets, statistics

Output:
- `batch_results.json` - Processing results for all files
- `aggregated_features.csv` - Combined feature data
- `feature_statistics.csv` - Summary statistics
- `batch_summary.html` - Interactive dashboard
- `feature_comparisons.html` - Feature distributions
- `feature_correlations.html` - Correlation heatmap

### 5. Visualization Examples

**[visualization_plotly.py](visualization_plotly.py)** - Interactive Plotly charts

```bash
# Install dependencies
pip install -e ".[viz]"

# Run visualizations
python examples/visualization_plotly.py

# Open generated files in browser
open output/visualizations/*.html
```

Creates interactive HTML visualizations:

#### Audio Analysis
- **Waveform plots**: Time-domain view of audio signals
- **Spectrograms**: Frequency content over time
- **Tempo and beats**: Visual beat detection with BPM gauge

#### Feature Analysis
- **Feature distributions**: Histograms of RMS, spectral centroid, etc.
- **Feature timelines**: How features evolve across bars
- **Feature heatmaps**: Normalized feature values per bar

#### Stem Comparison
- **Multi-stem waveforms**: Side-by-side comparison of vocals, drums, bass, other

#### MIDI Visualization
- **Piano roll**: Interactive MIDI note display with velocity

#### Batch Results
- **Success rate**: Pie chart of successful vs failed files
- **Tempo distribution**: Histogram of detected tempos
- **Duration distribution**: File length statistics
- **Bars per song**: Bar chart showing song structures

Example visualization:

```python
from visualization_plotly import plot_tempo_and_beats, load_analysis_results

# Load analysis results
analysis = load_analysis_results("my_song", Path("output"))

# Create interactive plot
fig = plot_tempo_and_beats(analysis)
fig.show()  # Opens in browser
fig.write_html("tempo_beats.html")  # Save for sharing
```

## Installation Options

### Minimal (Core features only)
```bash
pip install -e .
```

### With batch processing support
```bash
pip install -e ".[batch]"
```

### With visualization support
```bash
pip install -e ".[viz]"
```

### With all optional features
```bash
pip install -e ".[all,batch,viz]"
```

## Sample Audio

The `sample_audio/` directory should contain WAV files for testing. You can:

1. **Add your own files**: Copy any WAV files to `sample_audio/`
2. **Use test files**: Download from [freesound.org](https://freesound.org) or similar
3. **Generate test audio**: Use the synthetic audio in test files

Example directory structure:
```
sample_audio/
├── song1.wav
├── song2.wav
└── jazz/
    └── waltz.wav
```

## Output Structure

All examples write to `output/` or `examples/output/`:

```
output/
├── visualizations/          # HTML plots from visualization_plotly.py
│   ├── sample_song_tempo_beats.html
│   ├── sample_song_feature_distributions.html
│   ├── sample_song_piano_roll.html
│   └── batch_results_summary.html
├── batch_parallel/          # Parallel batch processing results
│   └── results.json
├── batch_progress/          # Progress-tracked batch results
│   └── results.json
└── analysis/               # Per-song analysis outputs
    └── sample_song/
        └── combined.json
```

## Common Patterns

### Pattern 1: Process and Visualize Single Song

```python
from pathlib import Path
from didactic_engine import AudioPipeline, PipelineConfig
from visualization_plotly import plot_tempo_and_beats, load_analysis_results

# Process audio
cfg = PipelineConfig(
    song_id="my_song",
    input_wav=Path("sample_audio/song.wav"),
    out_dir=Path("output"),
)
pipeline = AudioPipeline(cfg)
results = pipeline.run()

# Visualize
analysis = load_analysis_results("my_song", Path("output"))
fig = plot_tempo_and_beats(analysis)
fig.write_html("output/my_song_analysis.html")
```

### Pattern 2: Batch Process with Progress

```python
from pathlib import Path
from tqdm import tqdm
from didactic_engine import AudioPipeline, PipelineConfig

input_files = list(Path("sample_audio").glob("*.wav"))

for wav_path in tqdm(input_files, desc="Processing"):
    cfg = PipelineConfig(
        song_id=wav_path.stem,
        input_wav=wav_path,
        out_dir=Path("output/batch"),
    )
    pipeline = AudioPipeline(cfg)
    results = pipeline.run()
```

### Pattern 3: Parallel Processing

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from didactic_engine.pipeline import AudioPipeline

def process_file(wav_path):
    return AudioPipeline.process_batch(
        [wav_path],
        Path("output"),
        analysis_sr=22050
    )

input_files = list(Path("sample_audio").glob("*.wav"))

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, input_files))
```

### Pattern 4: Visualize Batch Results

```python
from pathlib import Path
from visualization_plotly import plot_batch_results_summary

# After batch processing
fig = plot_batch_results_summary(Path("output/batch/results.json"))
fig.write_html("output/batch_summary.html")
```

## Tips and Best Practices

### Performance Optimization

1. **Use parallel processing** for multiple files (see `batch_processing_example.py`)
2. **Lower analysis_sr** (e.g., 16000 instead of 22050) for faster processing
3. **Disable write_bar_chunks** if you don't need per-bar audio files
4. **Skip Essentia features** unless you need advanced audio analysis

### Memory Management

1. Audio is held in memory as `float32` arrays ($4$ bytes/sample)
2. A 3-minute mono track at 44.1kHz is ~31.7 MB in memory
3. Stem separation can temporarily create ~4× audio copies (vocals/drums/bass/other)
4. For very long recordings (>10 minutes), consider chunking/splitting upstream, lowering `analysis_sr`, and/or disabling `write_bar_chunks` to reduce I/O

Tip: For very large batches, `polars` can speed up aggregating Parquet datasets and computing
summary statistics (used automatically by `workflow_batch_viz.py` when installed).

### Visualization Tips

1. **Save to HTML** for interactive exploration and sharing
2. **Use fig.show()** for immediate viewing in Jupyter/browser
3. **Limit time range** for long audio (use `duration` or `end_time` parameters)
4. **Normalize features** before heatmaps for better visual comparison

### Error Handling

1. **Check file existence** before processing
2. **Use try/except blocks** around pipeline runs
3. **Log errors to file** for batch processing
4. **Implement retry logic** for transient failures (network, disk)

## Troubleshooting

### Issue: "No module named 'plotly'"
```bash
pip install plotly
# or
pip install -e ".[viz]"
```

### Issue: "No module named 'tqdm'"
```bash
pip install tqdm
# or
pip install -e ".[batch]"
```

### Issue: "FFmpeg not found"
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Issue: "Demucs command not found"
```bash
pip install demucs
```

### Issue: Visualization HTML files won't open
- Check browser security settings (some browsers block local HTML)
- Try opening with `python -m http.server` in output directory
- Use `fig.show()` instead of `write_html()` for immediate viewing

## Contributing

To add new examples:

1. Create a new Python file in `examples/`
2. Follow existing naming convention: `{feature}_example.py`
3. Add docstrings explaining the example
4. Update this README with the new example
5. Add sample output or screenshots if applicable

## Next Steps

After exploring the examples:

1. Read [../docs/00_README_DEV.md](../docs/00_README_DEV.md) for architecture overview
2. Check [../docs/02_KEY_FLOWS.md](../docs/02_KEY_FLOWS.md) for detailed flow diagrams
3. See [../docs/03_DEBUGGING.md](../docs/03_DEBUGGING.md) for troubleshooting tips
4. Review [../docs/04_CONTRIBUTING.md](../docs/04_CONTRIBUTING.md) to contribute

## Questions or Issues?

- Open an issue on GitHub: [github.com/yakuzadave/didactic-engine/issues](https://github.com/yakuzadave/didactic-engine/issues)
- Check existing documentation in [../docs/](../docs/)
- Review test files in [../tests/](../tests/) for more usage patterns
