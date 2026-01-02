# Examples Quick Reference Guide

## Example Files Overview

```
examples/
├── 📓 Interactive Tutorials
│   ├── tutorial.ipynb           # Jupyter notebook (Colab-compatible)
│   └── tutorial_marimo.py        # Marimo reactive notebook
│
├── 🚀 Quick Start Scripts
│   ├── quickstart_demo.py        # ⭐ Start here! Process + visualize single file
│   └── example_usage.py          # Basic API usage patterns
│
├── 🔄 Batch Processing
│   ├── batch_processing_example.py   # 5 batch processing patterns
│   └── workflow_batch_viz.py         # ⭐ Complete production workflow
│
├── 📊 Visualization
│   └── visualization_plotly.py   # Interactive Plotly charts library
│
└── 📖 Documentation
    ├── README.md                 # Full examples documentation
    ├── MARIMO_README.md          # Marimo-specific guide
    └── MARIMO_QUICKSTART.md      # Quick Marimo setup
```

---

## Decision Tree: Which Example Should I Use?

### 🎯 I want to...

#### ...learn the basics interactively
→ **tutorial.ipynb** or **tutorial_marimo.py**
- Best for: Learning concepts step-by-step
- Format: Jupyter or Marimo notebook
- Time: 15-30 minutes

#### ...quickly process one file
→ **quickstart_demo.py** ⭐
```bash
python examples/quickstart_demo.py my_audio.wav
```
- Best for: Testing the pipeline, quick results
- Output: Analysis + visualizations
- Time: 1-2 minutes

#### ...understand the API
→ **example_usage.py**
- Best for: Code reference, copy-paste patterns
- Shows: 5 different usage patterns
- Format: Runnable Python script

#### ...process many files
→ **batch_processing_example.py** → **workflow_batch_viz.py** ⭐
- **batch_processing_example.py**: Learn patterns
  - 5 different batch strategies
  - Progress tracking, retry logic, etc.
  
- **workflow_batch_viz.py**: Production use
  ```bash
  python examples/workflow_batch_viz.py sample_audio/ --workers 8
  ```
  - Processes + aggregates + visualizes
  - Complete reports and dashboards

#### ...create custom visualizations
→ **visualization_plotly.py**
- Best for: Custom charts, exploration
- Format: Reusable function library
- Usage: Import and call functions

---

## Feature Comparison Matrix

| Feature | quickstart | example_usage | batch_example | workflow | viz_plotly |
|---------|------------|---------------|---------------|----------|------------|
| **Single file** | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ |
| **Batch processing** | ❌ | ⚠️ | ✅ | ✅ | ❌ |
| **Parallel processing** | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Progress tracking** | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Visualizations** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Aggregation** | ❌ | ❌ | ❌ | ✅ | ⚠️ |
| **Reports export** | ❌ | ❌ | ⚠️ | ✅ | ❌ |
| **CLI interface** | ✅ | ❌ | ❌ | ✅ | ❌ |
| **Library usage** | ⚠️ | ✅ | ✅ | ⚠️ | ✅ |

Legend: ✅ Full support | ⚠️ Partial/demonstrates | ❌ Not included

---

## Workflow Diagrams

### Quick Start Workflow
```
quickstart_demo.py
       ↓
[Load WAV] → [Process] → [Analyze] → [Visualize] → [Display Summary]
                                           ↓
                                    HTML files (output/visualizations/)
```

### Batch Processing Workflow
```
batch_processing_example.py
       ↓
Pattern 1: Simple batch         → Process all files sequentially
Pattern 2: Parallel             → ProcessPoolExecutor (4 workers)
Pattern 3: Progress bars        → tqdm integration
Pattern 4: Directory tree       → Recursive processing
Pattern 5: Retry logic          → Auto-retry on failure
       ↓
results.json (batch results)
```

### Complete Production Workflow
```
workflow_batch_viz.py
       ↓
[Find WAV files] → [Parallel Process] → [Aggregate] → [Visualize] → [Export]
                          ↓                  ↓             ↓           ↓
                   Per-song analysis    Combined CSV   HTML charts  Reports
                   
Output:
  ├── analysis/          # Per-song JSON
  ├── datasets/          # Per-song Parquet
  ├── reports/
  │   ├── batch_results.json
  │   ├── aggregated_features.csv
  │   └── feature_statistics.csv
  └── visualizations/
      ├── batch_summary.html
      ├── feature_comparisons.html
      └── feature_correlations.html
```

### Visualization Library Usage
```python
from visualization_plotly import *

# Pattern 1: Load and visualize
analysis = load_analysis_results("song_id", Path("output"))
fig = plot_tempo_and_beats(analysis)
fig.show()

# Pattern 2: Feature analysis
df = pd.read_parquet("output/datasets/song/bar_features.parquet")
fig = plot_feature_timeline(df, ['rms', 'spectral_centroid'])
fig.write_html("my_chart.html")

# Pattern 3: Batch results
fig = plot_batch_results_summary(Path("results.json"))
fig.show()
```

---

## Installation Quick Reference

### Minimal (core only)
```bash
pip install -e .
```

### With batch processing
```bash
pip install -e ".[batch]"  # Adds: tqdm
```

### With visualizations
```bash
pip install -e ".[viz]"    # Adds: plotly, kaleido
```

### Everything
```bash
pip install -e ".[all,batch,viz]"
```

---

## Command Quick Reference

### Single File Processing
```bash
# Quickest way
python examples/quickstart_demo.py audio.wav

# Without visualizations (faster)
python examples/quickstart_demo.py audio.wav --no-viz

# CLI tool
didactic-engine --wav audio.wav --song-id my_song --out data/
```

### Batch Processing
```bash
# Learn patterns (demonstrates 5 approaches)
python examples/batch_processing_example.py

# Production batch workflow
python examples/workflow_batch_viz.py sample_audio/ --workers 8

# Fast mode (no chunks, no viz)
python examples/workflow_batch_viz.py *.wav --no-chunks --no-viz

# Process directory tree
python examples/workflow_batch_viz.py sample_audio/ --output results/
```

### Visualization Only
```bash
# Generate all visualizations for processed data
python examples/visualization_plotly.py
```

---

## Common Patterns (Copy-Paste Ready)

### Pattern 1: Quick Single File
```python
from pathlib import Path
from didactic_engine import AudioPipeline, PipelineConfig

cfg = PipelineConfig(
    song_id="my_song",
    input_wav=Path("audio.wav"),
    out_dir=Path("output"),
)
pipeline = AudioPipeline(cfg)
results = pipeline.run()
print(f"Tempo: {results['analysis']['tempo_bpm']:.1f} BPM")
```

### Pattern 2: Batch with Progress
```python
from pathlib import Path
from tqdm import tqdm
from didactic_engine import AudioPipeline, PipelineConfig

files = list(Path("audio").glob("*.wav"))
for wav in tqdm(files):
    cfg = PipelineConfig(
        song_id=wav.stem,
        input_wav=wav,
        out_dir=Path("output"),
    )
    AudioPipeline(cfg).run()
```

### Pattern 3: Parallel Processing
```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from batch_processing_example import process_single_file

files = list(Path("audio").glob("*.wav"))
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(
        lambda f: process_single_file(f, Path("output"), {}),
        files
    ))
```

### Pattern 4: Custom Visualization
```python
from visualization_plotly import *

# Load results
analysis = load_analysis_results("my_song", Path("output"))
df = pd.read_parquet("output/datasets/my_song/bar_features.parquet")

# Create charts
fig1 = plot_tempo_and_beats(analysis)
fig2 = plot_feature_timeline(df, ['rms', 'spectral_centroid'])

# Display or save
fig1.show()
fig2.write_html("timeline.html")
```

---

## Troubleshooting Quick Fixes

| Error | Solution |
|-------|----------|
| `No module named 'plotly'` | `pip install plotly` or `pip install -e ".[viz]"` |
| `No module named 'tqdm'` | `pip install tqdm` or `pip install -e ".[batch]"` |
| `FFmpeg not found` | Ubuntu: `sudo apt install ffmpeg`<br>Mac: `brew install ffmpeg` |
| `Demucs command not found` | `pip install demucs` |
| Empty visualizations | Run pipeline first to generate data |
| "Permission denied" on scripts | `chmod +x examples/*.py` |

---

## Performance Tips

### For single files:
- Use `quickstart_demo.py --no-viz` to skip visualizations
- Lower `--sr 16000` for faster processing
- Disable chunks: set `write_bar_chunks=False`

### For batch processing:
- Increase workers: `--workers 8` (match CPU cores)
- Use `workflow_batch_viz.py` instead of running files sequentially
- Add `--no-chunks` flag to save I/O time
- Use `--no-viz` for data extraction only

### For visualizations:
- Limit time range: `plot_midi_piano_roll(midi, end_time=30)`
- Downsample waveforms (automatic in viz functions)
- Use `write_html()` instead of `show()` for batch generation

---

## Next Steps

1. ✅ **Try quickstart**: `python examples/quickstart_demo.py your_file.wav`
2. ✅ **Explore tutorials**: Open `tutorial.ipynb` or run `marimo edit tutorial_marimo.py`
3. ✅ **Learn batch patterns**: `python examples/batch_processing_example.py`
4. ✅ **Run production workflow**: `python examples/workflow_batch_viz.py sample_audio/`
5. 📚 **Read architecture docs**: [docs/01_ARCHITECTURE.md](../docs/01_ARCHITECTURE.md)

---

*For complete documentation, see [examples/README.md](README.md)*
