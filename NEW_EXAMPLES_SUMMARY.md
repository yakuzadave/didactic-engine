# New Examples and Workflows - Summary

This document summarizes the new batch processing and visualization examples added to Didactic Engine.

## Files Added

### 1. **examples/batch_processing_example.py**
Advanced batch processing patterns with various strategies.

**Features:**
- Simple batch processing using `AudioPipeline.process_batch()`
- Parallel processing with `ProcessPoolExecutor` (4-8 workers)
- Progress tracking with tqdm progress bars
- Recursive directory tree processing
- Retry logic for transient failures
- Results serialization to JSON

**Usage:**
```bash
python examples/batch_processing_example.py
```

**Key Functions:**
- `example_simple_batch()` - Basic batch processing
- `example_parallel_batch()` - Multi-process parallelization
- `example_batch_with_progress()` - tqdm progress bars
- `example_batch_from_directory_tree()` - Recursive processing
- `example_batch_with_retry()` - Automatic retry logic

---

### 2. **examples/visualization_plotly.py**
Interactive Plotly visualizations for analysis results.

**Visualizations:**
- Audio waveforms (downsampled for performance)
- Spectrograms (frequency content over time)
- Tempo and beat detection (with BPM gauge)
- Feature distributions (histograms)
- Feature timelines (line plots across bars)
- Stem comparisons (side-by-side waveforms)
- MIDI piano rolls (interactive note display)
- Batch results summary (pie charts, histograms, bar charts)
- Feature heatmaps (normalized values per bar)
- Correlation matrices (feature relationships)

**Usage:**
```bash
pip install -e ".[viz]"
python examples/visualization_plotly.py
# Open output/visualizations/*.html in browser
```

**Key Functions:**
- `plot_waveform()` - Time-domain audio plot
- `plot_spectrogram()` - Frequency analysis
- `plot_tempo_and_beats()` - Beat detection results
- `plot_feature_distributions()` - Feature histograms
- `plot_feature_timeline()` - Feature evolution
- `plot_stem_comparison()` - Multi-stem view
- `plot_midi_piano_roll()` - MIDI visualization
- `plot_batch_results_summary()` - Batch statistics
- `plot_bar_features_heatmap()` - Feature matrix

---

### 3. **examples/quickstart_demo.py** ⭐
Quick start script for new users.

**What it does:**
1. Process a single WAV file
2. Generate all visualizations
3. Display summary statistics
4. Show output directory structure

**Usage:**
```bash
# Basic usage
python examples/quickstart_demo.py path/to/audio.wav

# Custom output directory
python examples/quickstart_demo.py audio.wav --output my_results/

# Skip visualizations (faster)
python examples/quickstart_demo.py audio.wav --no-viz

# Custom song ID
python examples/quickstart_demo.py audio.wav --song-id my_track
```

**Output:**
```
[1/4] Configuring pipeline...
[2/4] Running audio processing pipeline...
[3/4] Processing Summary:
  Duration: 180.5 seconds
  Tempo: 120.4 BPM
  Number of bars: 48
[4/4] Creating visualizations...
  ✓ Tempo and beats: output/visualizations/my_track_tempo_beats.html
  ✓ Feature timeline: output/visualizations/my_track_feature_timeline.html
  ✓ Feature heatmap: output/visualizations/my_track_feature_heatmap.html
```

---

### 4. **examples/workflow_batch_viz.py** ⭐⭐
Production-ready batch workflow with aggregation and visualization.

**Complete Workflow:**
1. **Batch process** multiple files in parallel (configurable workers)
2. **Aggregate** feature datasets across all songs
3. **Generate visualizations**: summary dashboard, comparisons, correlations
4. **Export reports**: JSON, CSV, statistics

**Usage:**
```bash
# Process directory
python examples/workflow_batch_viz.py sample_audio/ --output results/

# Process specific files with 8 workers
python examples/workflow_batch_viz.py song1.wav song2.wav --workers 8

# Fast mode (no chunks, no viz)
python examples/workflow_batch_viz.py sample_audio/*.wav --no-chunks --no-viz
```

**Output Structure:**
```
results/
├── analysis/           # Per-song analysis
├── datasets/           # Per-song Parquet datasets
├── reports/
│   ├── batch_results.json          # All processing results
│   ├── aggregated_features.csv     # Combined feature data
│   └── feature_statistics.csv      # Summary statistics
└── visualizations/
    ├── batch_summary.html          # Dashboard (success rate, tempos, durations)
    ├── feature_comparisons.html    # Violin plots across songs
    └── feature_correlations.html   # Correlation heatmap
```

---

### 5. **examples/README.md**
Comprehensive documentation for all examples.

**Sections:**
- Quick Start (all 4 tutorial/example types)
- Installation options (minimal, batch, viz, all)
- Sample audio setup
- Output structure
- Common patterns (4 practical examples)
- Tips and best practices
- Troubleshooting guide

---

## Updated Files

### **README.md**
- Added "Quickstart Demo" section as first usage example
- Added "Batch Processing Examples" section
- Added "Visualization Examples" section
- Updated dependency documentation

### **pyproject.toml**
Added new optional dependency groups:
- `viz = ["plotly>=5.0.0", "kaleido>=0.2.0"]`
- `batch = ["tqdm>=4.65.0"]`
- Updated `all` to include new dependencies

---

## Installation

### For batch processing:
```bash
pip install -e ".[batch]"
```

Adds: tqdm (progress bars)

### For visualizations:
```bash
pip install -e ".[viz]"
```

Adds: plotly, kaleido (chart rendering)

### For everything:
```bash
pip install -e ".[all,batch,viz]"
```

---

## Common Use Cases

### Use Case 1: Quick Single File Analysis
```bash
python examples/quickstart_demo.py my_song.wav
# Opens visualizations in browser automatically
```

### Use Case 2: Batch Process Multiple Files
```bash
python examples/batch_processing_example.py
# Demonstrates 5 different batch patterns
```

### Use Case 3: Production Batch Workflow
```bash
python examples/workflow_batch_viz.py sample_audio/ --workers 8
# Processes, aggregates, visualizes, and exports reports
```

### Use Case 4: Custom Visualization
```python
from visualization_plotly import *

# Load your results
analysis = load_analysis_results("my_song", Path("output"))

# Create custom visualization
fig = plot_tempo_and_beats(analysis)
fig.show()  # Opens in browser
fig.write_html("my_viz.html")  # Save for sharing
```

---

## Performance Benchmarks

Approximate processing times (on typical hardware):

| Task | Duration | Notes |
|------|----------|-------|
| Single 3-min song | 30-60s | With all features |
| Single 3-min song | 15-30s | Minimal config |
| 10 songs (parallel, 4 workers) | 2-4 min | Linear speedup |
| Visualization generation | 5-10s | Per song |

**Optimization tips:**
- Use `--no-chunks` to skip writing per-bar files
- Lower `--sr` (e.g., 16000) for faster analysis
- Increase `--workers` on multi-core machines
- Use `--no-viz` for data processing only

---

## Next Steps

1. **Try the quickstart**: `python examples/quickstart_demo.py your_file.wav`
2. **Explore batch processing**: `python examples/batch_processing_example.py`
3. **Run the complete workflow**: `python examples/workflow_batch_viz.py sample_audio/`
4. **Customize visualizations**: Edit `visualization_plotly.py` functions
5. **Read the docs**: See [docs/00_README_DEV.md](../docs/00_README_DEV.md)

---

## Questions or Issues?

- Check [examples/README.md](README.md) for detailed documentation
- Review [docs/03_DEBUGGING.md](../docs/03_DEBUGGING.md) for troubleshooting
- Open an issue: [github.com/yakuzadave/didactic-engine/issues](https://github.com/yakuzadave/didactic-engine/issues)
