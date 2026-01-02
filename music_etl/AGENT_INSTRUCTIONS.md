# Agent Instructions for Music ETL Pipeline

## Purpose

This document provides implementation guidance for AI agents working on the Music ETL pipeline. It describes the architecture, design constraints, and common patterns used throughout the codebase.

## Project Architecture

### Module Organization

The project follows a single-responsibility module design:

- **`config.py`**: Configuration dataclass with computed properties for output paths
- **`separate.py`**: Demucs wrapper for stem separation
- **`audio_preprocess.py`**: pydub-based audio preprocessing
- **`wav_features.py`**: librosa feature extraction
- **`essentia_features.py`**: Optional Essentia enrichment (safe imports)
- **`transcribe.py`**: Basic Pitch MIDI transcription wrapper
- **`midi_features.py`**: pretty_midi parsing
- **`align.py`**: Beat/bar grid alignment for MIDI events
- **`bar_chunker.py`**: Bar boundary computation and audio segmentation
- **`export_md.py`**: Markdown report generation
- **`export_abc.py`**: ABC notation export via music21
- **`datasets.py`**: Parquet dataset writing
- **`utils_flatten.py`**: Dictionary flattening utilities
- **`pipeline.py`**: Main orchestrator tying everything together

### Design Constraints

1. **Python 3.11+ Typing**
   - Use `list[str]`, `dict[str, float]`, not `typing.List` or `typing.Dict`
   - Type hints on all public functions
   - Return types must be explicit

2. **Path Handling**
   - All paths use `pathlib.Path`
   - No string path concatenation
   - Output paths derived from `PipelineConfig` properties

3. **Error Handling**
   - CLI tool absence raises `RuntimeError` with installation instructions
   - Optional features (Essentia) fail gracefully
   - ABC export errors logged but don't crash pipeline

4. **Robustness**
   - Demucs output discovery via `rglob("*.wav")` (handles varying folder structures)
   - Basic Pitch MIDI discovery by modification time (handles unpredictable filenames)
   - Beat grid extrapolation for alignment edge cases

## Key Implementation Patterns

### 1. CLI Tool Checking

```python
import shutil

if shutil.which("demucs") is None:
    raise RuntimeError(
        "demucs command not found. Please install Demucs:\n"
        "  pip install demucs\n"
        "and ensure it's on your PATH."
    )
```

### 2. Safe Optional Imports

```python
try:
    import essentia.standard as es
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

def extract_features(path):
    if not AVAILABLE:
        return {"available": False, "error": "..."}
    # ... feature extraction
```

### 3. Beat Grid Extrapolation

When beat tracking produces too few beats or notes extend beyond the grid:

```python
# Compute beat interval
if len(beat_array) > 1:
    beat_interval = np.median(np.diff(beat_array))
else:
    beat_interval = 60.0 / tempo_bpm

# Extend to cover duration
while beat_array[-1] < max_needed:
    beat_array = np.append(beat_array, beat_array[-1] + beat_interval)
```

### 4. Parquet Column Flattening

Essentia features contain nested dicts. Flatten before adding to DataFrame:

```python
from music_etl.utils_flatten import flatten_dict

essentia_feats = extract_essentia_features(path)
flat = flatten_dict(
    {k: v for k, v in essentia_feats.items() if not isinstance(v, list)},
    parent_key="essentia"
)
row.update(flat)  # Now safe for DataFrame
```

### 5. Empty DataFrame Handling

```python
if not df.empty:
    df.to_parquet(path, index=False)
else:
    print("Skipped dataset (no data)")
```

## Common Failure Modes and Fixes

### 1. Demucs Output Paths Vary

**Problem**: Demucs creates varying folder structures based on model/version.

**Solution**: Search recursively for WAV files:
```python
wav_files = list(out_dir.rglob("*.wav"))
```

### 2. Basic Pitch Unpredictable MIDI Filenames

**Problem**: Basic Pitch may use input filename or generate new names.

**Solution**: Find newest MIDI file and copy to canonical path:
```python
midi_files = list(output_dir.rglob("*.mid"))
newest = max(midi_files, key=lambda p: p.stat().st_mtime)
shutil.copy2(newest, canonical_path)
```

### 3. Beat Grid Too Short

**Problem**: Beat tracking may end before audio ends or notes extend beyond beats.

**Solution**: Extrapolate beats using tempo (see pattern 3 above).

### 4. Bar Segmentation Tail Problems

**Problem**: Last bar may be incomplete or extend beyond audio.

**Solution**: Clamp end times and skip zero-length bars:
```python
end_s = min(end_s, audio_duration_s)
if end_s > start_s:
    bars.append((bar_idx, start_s, end_s))
```

### 5. Nested Dicts in Parquet

**Problem**: Pandas/PyArrow can't handle nested dicts in DataFrame columns.

**Solution**: Flatten before inserting (see pattern 4 above).

### 6. ABC Export Errors

**Problem**: music21 may fail on complex/malformed MIDI.

**Solution**: Wrap in try/except, log error, write error file:
```python
try:
    score.write("abc", fp=output_path)
except Exception as e:
    print(f"Warning: ABC export failed: {e}")
    with open(output_path, "w") as f:
        f.write(f"% ABC export failed: {e}\n")
```

## Pipeline Flow

The `pipeline.py` orchestrator follows this order:

1. Create all output directories
2. Run Demucs → stem WAV map
3. (Optional) Preprocess stems with pydub
4. Analyze each stem with librosa + optional Essentia
5. Write per-bar chunks and analyze each chunk
6. Transcribe stems to MIDI with Basic Pitch
7. Parse MIDI files with pretty_midi
8. Align notes to beat/bar grid
9. Combine all notes into single DataFrame
10. Write combined JSON summary
11. Export Markdown report
12. Export ABC notation per stem
13. Write Parquet datasets

## Testing Strategy

### Import Checks
```bash
python -c "from music_etl.pipeline import run_all"
```

### CLI Help
```bash
python scripts/run_pipeline.py --help
```

### Missing CLI Handling
```bash
# Should show friendly error:
python scripts/run_pipeline.py --wav test.wav --song-id test
# (if demucs/basic-pitch not installed)
```

### Optional Dependency Handling
```python
# Should not crash:
from music_etl.essentia_features import extract_essentia_features
result = extract_essentia_features(Path("test.wav"))
assert "available" in result
```

## Code Quality Standards

### Linting
```bash
ruff check src/
```

### Type Checking
```bash
mypy src/ --strict
```

### Formatting
- Max line length: 100
- Use double quotes for strings
- One import per line

## Extending the Pipeline

### Adding a New Feature Extractor

1. Create new module (e.g., `src/music_etl/new_features.py`)
2. Implement extraction function with clear type hints
3. Import and call in `pipeline.py` step 3 or 5
4. Add results to appropriate dataset in step 11

### Adding a New Export Format

1. Create new module (e.g., `src/music_etl/export_new.py`)
2. Implement export function taking DataFrame/dict and output path
3. Call in `pipeline.py` between steps 9-12
4. Document in README output structure section

### Adding a New Dataset

1. Define schema in `datasets.py` docstring
2. Build DataFrame in `pipeline.py` from appropriate intermediate data
3. Call `to_parquet()` in `datasets.py:write_datasets()`
4. Document schema in README

## Common Development Tasks

### Running the Pipeline Locally

```bash
# With a test WAV file
python scripts/run_pipeline.py \
  --wav data/input/test.wav \
  --song-id test_song \
  --no-chunks  # Skip chunks for faster testing
```

### Debugging a Specific Module

```python
from pathlib import Path
from music_etl.wav_features import extract_wav_features

features = extract_wav_features(Path("test.wav"), sr=22050)
print(features)
```

### Inspecting Parquet Outputs

```python
import pandas as pd

events = pd.read_parquet("data/datasets/song_id/events.parquet")
print(events.head())
print(events.columns)
```

## Glossary

- **Stem**: Individual audio component (vocals, drums, bass, other)
- **Beat**: Single pulse in the music rhythm
- **Bar**: Musical measure (group of beats defined by time signature)
- **Time Signature**: Fraction like 4/4 defining beats per bar
- **Chroma**: 12-dimensional pitch class representation
- **MFCC**: Mel-frequency cepstral coefficients (timbral features)
- **Spectral Centroid/Rolloff/Bandwidth**: Frequency distribution features
- **ZCR**: Zero-crossing rate (related to noisiness)

## References

- [librosa documentation](https://librosa.org/doc/latest/)
- [Demucs GitHub](https://github.com/facebookresearch/demucs)
- [Basic Pitch GitHub](https://github.com/spotify/basic-pitch)
- [pretty_midi documentation](https://craffel.github.io/pretty-midi/)
- [music21 documentation](https://web.mit.edu/music21/doc/)
- [Essentia documentation](https://essentia.upf.edu/documentation/)
