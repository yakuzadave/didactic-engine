# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Didactic Engine** is a Python 3.11+ audio processing pipeline for music analysis. It orchestrates WAV ingestion → preprocessing → stem separation → audio analysis → MIDI transcription → beat/bar alignment → per-bar chunking → feature extraction → dataset generation (Parquet).

**Core principle**: All audio processing uses **1D mono numpy.float32 arrays** internally. Configuration is **immutable** (frozen dataclass).

## Essential Commands

### Development Setup

```bash
# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate                 # PowerShell: .venv\Scripts\Activate.ps1

# Install for development
pip install -e ".[dev]"                   # CPU build with tests/linting

# GPU acceleration (NVIDIA drivers required)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -e ".[ml-gpu,dev]"            # ONNX Runtime GPU + Demucs + dev tools
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# WSL: avoid Windows temp dir issues
TMPDIR=/tmp TEMP=/tmp TMP=/tmp pytest -v

# Skip heavy/optional stacks
pytest -m "not optional_deps and not integration"

# Only GPU-specific tests
pytest -m gpu

# With coverage
pytest tests/ --cov=didactic_engine

# Run specific test class
pytest tests/test_pipeline.py::TestAudioAnalyzer -v

# Run specific test file
pytest tests/test_pipeline.py -v
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking (if configured)
mypy src/
```

### Running the Pipeline

```bash
# Single file
didactic-engine --wav input.wav --song-id my_song --out data/

# Batch processing
didactic-engine --wav *.wav --out data/

# Custom configuration
didactic-engine --wav input.wav --out data/ --sr 22050 --ts-num 4 --ts-den 4

# View all options
didactic-engine --help
```

### Example Scripts

```bash
# Quickstart demo with visualizations
python examples/quickstart_demo.py path/to/audio.wav

# Batch processing workflow
python examples/workflow_batch_viz.py *.wav --workers 8 --output data/

# Interactive tutorials
marimo edit examples/tutorial_marimo.py
jupyter notebook examples/tutorial.ipynb
```

## Architecture

### Pipeline Flow (14 Steps)

The pipeline is orchestrated by `AudioPipeline.run()` in `src/didactic_engine/pipeline.py`:

```
1. Load/validate WAV (mono @ analysis_sr)
2. Optional: Preprocess with pydub (resample, normalize, trim silence)
3. Audio analysis (tempo, beats, spectral features via librosa)
4. Optional: Stem separation (Demucs CLI)
5. Optional: MIDI transcription (Basic Pitch CLI)
6. Note alignment to beat grid
7. Beat → bar segmentation
8. Per-stem per-bar audio chunking
9. Feature extraction per bar
10. Dataset generation (Parquet: events, beats, bars, bar_features)
11. Markdown report export
12. ABC notation export
```

### Key Files

**Core orchestration**:
- `pipeline.py`: Main orchestrator - read `AudioPipeline.run()` first
- `config.py`: Frozen dataclass with all settings and output path properties

**Processing modules**:
- `ingestion.py`: WAV loading and validation (converts to mono float32)
- `preprocessing.py`: Optional pydub preprocessing
- `analysis.py`: Tempo, beats, spectral features (librosa/essentia)
- `separation.py`: Demucs stem separation (CLI wrapper)
- `transcription.py`: Basic Pitch MIDI transcription (CLI wrapper)
- `align.py`: Align MIDI notes to beat grid
- `segmentation.py`: Beat → bar segmentation, per-bar chunking
- `features.py`: Feature extraction per bar
- `export_md.py`, `export_abc.py`: Report generation

**Supporting modules**:
- `cli.py`: Command-line interface
- `midi_parser.py`: MIDI file parsing
- `bar_chunker.py`: Per-bar audio chunking
- `onnx_inference.py`: ONNX Runtime helpers (Python 3.12+ compatible)
- `essentia_features.py`: Extended audio features (optional AGPL dependency)

### Configuration Pattern

**All pipeline behavior is controlled via immutable `PipelineConfig`**:

```python
from didactic_engine import PipelineConfig, AudioPipeline
from pathlib import Path

cfg = PipelineConfig(
    song_id="track1",
    input_wav=Path("audio.wav"),
    out_dir=Path("data/"),
    analysis_sr=22050,           # Audio analysis sample rate
    time_signature_num=4,        # Beats per bar
    use_pydub_preprocess=True,   # Enable preprocessing
    write_bar_chunks=True,       # Save per-bar WAV files
)

# NEVER modify cfg after creation (frozen=True)
# To change config, use replace():
new_cfg = cfg.replace(analysis_sr=44100)
```

**Output paths are deterministic properties**: `cfg.stems_dir`, `cfg.midi_dir`, `cfg.dataset_dir`, etc.

### Output Directory Structure

```
out_dir/
├── input/<song_id>/original_copy.wav
├── preprocessed/<song_id>/<song_id>.wav
├── stems/<song_id>/{vocals,drums,bass,other}.wav
├── chunks/<song_id>/<stem>/bar_NNNN.wav
├── midi/<song_id>/{vocals,drums,bass,other}.mid
├── analysis/<song_id>/combined.json
├── reports/<song_id>/{midi_markdown.md,*.abc}
└── datasets/<song_id>/{events,beats,bars,bar_features}.parquet
```

## Critical Conventions

### 1. Audio Format Invariant
All processing expects `np.ndarray` shape `(n_samples,)` dtype `float32`:
- Validated by `WAVIngester.validate()` (checks 1D, float, no NaN/Inf, valid sr)
- Mono conversion happens at ingestion time
- Never pass multi-channel or non-float arrays to processing modules

### 2. Beat/Bar Alignment
Notes are aligned to a **fixed beat grid** based on:
- Detected tempo (BPM) from `librosa.beat.beat_track`
- Time signature (e.g., 4/4 = 4 beats per bar)
- Aligns MIDI notes to nearest beat using `align.py`

### 3. Graceful Degradation
Optional dependencies (Demucs, Basic Pitch, Essentia) are **late-bound imports** with CLI checks:
- If Demucs unavailable: uses `full_mix` stem (original audio)
- If Basic Pitch unavailable: skips MIDI transcription
- If Essentia unavailable: skips extended audio features

Pattern in `separation.py:55-70`:
```python
def check_demucs_availability() -> bool:
    try:
        result = subprocess.run(["demucs", "--help"], ...)
        return result.returncode == 0
    except FileNotFoundError:
        return False
```

### 4. Error Handling Conventions
- **Optional dependencies**: Catch import/CLI errors, log warning, continue with degraded functionality
- **Invalid input**: Raise `ValueError` with clear message about what's wrong and valid range
- **File I/O errors**: Let `IOError`/`FileNotFoundError` propagate (caller handles)
- **Unexpected data**: Raise `RuntimeError` with context about what was expected vs. received

### 5. Testing Strategy
- Unit tests in `tests/test_pipeline.py` use **synthetic audio** (sine waves, noise)
- Integration test in `tests/test_examples.py` uses `sample_audio/`
- Tests should pass with or without optional dependencies
- Run `pytest tests/ -v` (19 tests) before commits

### 6. Documentation Standard
Google-style docstrings (see `STYLE_GUIDE.md`):
- Always include: Args, Returns, Raises, Side Effects
- Add examples for public API functions
- Cross-link related functions/docs ("See also: …")

## Developer Workflows

### Adding New Features

1. **New audio processing step**: Add method to `pipeline.py` `AudioPipeline.run()`
2. **New feature extractor**: Extend `features.py` `FeatureExtractor`
3. **New export format**: Create new file like `export_md.py`

**Pattern**: Each module has a single-responsibility class (e.g., `AudioAnalyzer`, `StemSeparator`) instantiated in `AudioPipeline.__init__`

### Debugging Scenarios

**Pipeline stops at stem separation**:
- Check: `demucs --help` returns successfully
- Check: Sufficient disk space in temp directory (~500MB per song)
- Workaround: Set `TMPDIR` env var to location with more space

**MIDI transcription produces empty results**:
- Check: Audio has actual note content (not just percussion/noise)
- Check: `basic-pitch --help` works and version ≥ 0.2.0
- Debug: Check `out_dir/midi/<song_id>/*.mid` file size (should be > 0 bytes)

**Feature extraction fails with KeyError**:
- Check: Audio analysis completed successfully
- Check: Beat detection found beats (`len(beat_times) > 0`)
- Common cause: Audio too short (< 2 seconds) or no clear beat

**Memory errors during processing**:
- Reduce `analysis_sr` (try 16000 instead of 22050)
- Disable `write_bar_chunks=False` to reduce I/O
- Process shorter audio segments (split before processing)

**Tempo detection gives wrong BPM**:
- Librosa's beat tracker works best for 80-160 BPM range
- Very fast (>180 BPM) or slow (<60 BPM) music may detect double/half tempo
- Alternative: Use Essentia's tempo detection (`use_essentia_features=True`)

## Dependencies & Python Version

### Python Version
- **Python 3.11**: Full support including Basic Pitch MIDI transcription (TensorFlow backend)
- **Python 3.12+**: Full support with ONNX Runtime (no Basic Pitch due to TensorFlow incompatibility)

### Core Dependencies
- numpy (<2.0), pandas, pyarrow
- librosa, soundfile, pydub
- pretty-midi, music21

### Optional Dependencies
- `demucs`: Stem separation (install: `pip install demucs`)
- `basic-pitch`: MIDI transcription (Python 3.11 only)
- `onnxruntime` / `onnxruntime-gpu`: ONNX model inference
- `essentia`: Extended audio features (AGPL license)
- `plotly`, `kaleido`: Visualizations
- `tqdm`, `polars`: Batch processing helpers

### Installation Extras
```bash
pip install -e ".[ml]"         # Demucs, Basic Pitch (3.11), ONNX Runtime
pip install -e ".[ml-gpu]"     # GPU-accelerated ML dependencies
pip install -e ".[essentia]"   # Essentia features
pip install -e ".[viz]"        # Plotly visualizations
pip install -e ".[batch]"      # Batch processing helpers
pip install -e ".[dev]"        # pytest, ruff, mypy, marimo
pip install -e ".[all]"        # All CPU extras
pip install -e ".[all-gpu]"    # All GPU extras
```

## Common Gotchas

1. **FFmpeg required**: pydub operations fail without FFmpeg on PATH
2. **Demucs writes to temp dir**: Automatically copies results to `cfg.stems_dir`
3. **Basic Pitch CLI**: Transcribes to temp MIDI file, then loads with pretty_midi
4. **Mono conversion**: Happens at load time via `ingestion.py`, not preprocessing
5. **Sample rate handling**: `analysis_sr` (default 22050) for feature extraction, `preprocess_target_sr` (default 44100) for output
6. **Time signature**: Defaults to 4/4, affects bar boundary computation in `segmentation.py`
7. **Windows MAX_PATH**: Keep paths short to avoid issues when Demucs writes stems
8. **WSL temp directories**: Set `TMPDIR=/tmp` to avoid Windows temp dir issues in pytest

## Performance Optimization

### Sample Rate Selection
- Use `analysis_sr=22050` (default) for most music analysis
- Use `analysis_sr=44100` for high-fidelity spectral analysis
- Lower sample rates = faster processing, less memory, smaller files

### Optional Features
- Disable `use_pydub_preprocess=False` if input audio is already clean
- Disable `write_bar_chunks=False` to skip writing individual bar files (saves I/O time)
- Enable `use_essentia_features=True` only when needed (adds ~20% processing time)

### Memory Considerations
- Audio held in memory as float32 numpy arrays (4 bytes per sample)
- 3-minute song @ 44.1kHz mono = ~31.7 MB
- Stem separation creates 4x copies (vocals, drums, bass, other)
- Consider processing in chunks for very long recordings (>10 minutes)

## Key Documentation Files

- `README.md`: User-facing quickstart and API reference
- `.github/copilot-instructions.md`: Detailed architectural knowledge, workflows, debugging guides
- `STYLE_GUIDE.md`: Docstring and comment standards
- `docs/environment.md`: Cross-platform setup (Windows, WSL, GPU)
- `docs/01_ARCHITECTURE.md`: Visual diagrams and component descriptions
- `docs/02_KEY_FLOWS.md`: Step-by-step sequence diagrams
- `TASKS.md`, `STATUS.md`, `DOC_INVENTORY.md`: Documentation tracking files

## File References

When referencing specific code locations, use the pattern `file_path:line_number`:
- Configuration handling: `src/didactic_engine/config.py:64-70`
- Pipeline orchestration: `src/didactic_engine/pipeline.py:89-100`
- Beat alignment: `src/didactic_engine/align.py`
- Demucs availability check: `src/didactic_engine/separation.py:55-70`
