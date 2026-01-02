# Agent Instructions: Didactic Engine Maintenance Guide

This document provides instructions for maintaining and extending the didactic-engine audio processing toolkit.

## Project Overview

**Purpose**: Didactic Engine is a comprehensive audio processing pipeline that:
- Ingests and validates WAV files
- Separates audio into stems (Demucs)
- Extracts audio features (librosa + optional Essentia)
- Transcribes audio to MIDI (Basic Pitch)
- Aligns MIDI notes to beat/bar grid
- Segments audio into per-bar chunks
- Generates Parquet datasets
- Exports human-readable reports (Markdown, ABC notation)

**Target Users**: Music researchers, ML practitioners, audio engineers

## Architecture

```
src/didactic_engine/
├── __init__.py          # Package exports
├── config.py            # PipelineConfig dataclass
├── pipeline.py          # Main AudioPipeline class
├── cli.py              # Command-line interface
├── ingestion.py        # WAVIngester class
├── preprocessing.py    # AudioPreprocessor class
├── separation.py       # StemSeparator class
├── analysis.py         # AudioAnalyzer class
├── essentia_features.py # Optional Essentia integration
├── transcription.py    # BasicPitchTranscriber class
├── midi_parser.py      # MIDIParser class
├── align.py            # Note-to-beat alignment
├── segmentation.py     # Bar segmentation functions
├── bar_chunker.py      # Per-bar chunking utilities
├── features.py         # FeatureExtractor class
├── export_md.py        # Markdown export
├── export_abc.py       # ABC notation export
└── utils_flatten.py    # Dictionary flattening utilities
```

## Key Design Decisions

### 1. Frozen Configuration
`PipelineConfig` is a frozen dataclass to ensure configuration immutability during pipeline execution. All output directories are computed as properties.

### 2. Additive Essentia Integration
Essentia is optional. When not installed, features gracefully return `{"available": False, ...}`. Never fail if Essentia is missing.

### 3. CLI Tool Checks
Demucs and Basic Pitch availability is checked via `shutil.which()`. Clear error messages explain installation if missing.

### 4. Mono Audio Standard
All internal processing uses mono audio (1D numpy arrays). The ingester always converts to mono.

### 5. Parquet Datasets
Output datasets use Parquet format for efficient storage and compatibility with pandas/pyarrow.

## Adding New Features

### Adding a New Feature Extractor

1. **Create extraction function** in `features.py` or a new module:
   ```python
   def extract_new_feature(audio: np.ndarray, sr: int) -> Dict[str, Any]:
       # Extract features
       return {"feature_name": value, ...}
   ```

2. **Update `FeatureExtractor` class** with a new method:
   ```python
   def extract_new_feature(self, audio, sample_rate):
       return extract_new_feature(audio, sample_rate)
   ```

3. **Integrate into pipeline** in `pipeline.py`:
   ```python
   new_features = self.feature_extractor.extract_new_feature(audio, sr)
   ```

4. **Write tests** in `tests/test_pipeline.py`:
   ```python
   def test_extract_new_feature(self):
       extractor = FeatureExtractor()
       result = extractor.extract_new_feature(audio, sr)
       assert "feature_name" in result
   ```

5. **Update config** if the feature needs configuration options.

### Adding a New Export Format

1. Create `export_<format>.py` in `src/didactic_engine/`
2. Implement export function with graceful fallback if dependencies missing
3. Add to `__init__.py` exports
4. Call from `pipeline.py` in the export step
5. Add tests

### Adding CLI Options

1. Add argument in `cli.py`:
   ```python
   parser.add_argument("--new-option", help="Description")
   ```

2. Update `PipelineConfig` in `config.py` if needed
3. Pass to config in CLI main function
4. Update `--help` examples in docstring

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single test file
pytest tests/test_pipeline.py -v

# Specific test class
pytest tests/test_pipeline.py::TestFeatureExtractor -v

# With coverage
pytest tests/ --cov=didactic_engine --cov-report=html
```

### Test Guidelines

- Tests should not require Demucs or Basic Pitch installed
- Mock external tool calls with `unittest.mock`
- Skip tests when optional dependencies are missing:
  ```python
  @pytest.mark.skipif(not ESSENTIA_AVAILABLE, reason="Essentia not installed")
  ```
- Use synthetic audio for feature tests
- Use `tempfile.TemporaryDirectory()` for file operations

## Interpreting Test Output

- **PASSED**: Test succeeded
- **FAILED**: Assertion failed - fix the code
- **ERROR**: Exception during test - check for bugs
- **SKIPPED**: Test requirements not met (OK)
- **XFAIL**: Expected failure - known issue

## Known Failure Modes

### 1. FFmpeg Not Found
**Symptom**: `RuntimeWarning: Couldn't find ffmpeg`
**Solution**: Install FFmpeg: `apt install ffmpeg` or `brew install ffmpeg`

### 2. Demucs RuntimeError
**Symptom**: `RuntimeError: Demucs is not installed`
**Solution**: Install Demucs: `pip install demucs`

### 3. Basic Pitch Not Found
**Symptom**: `RuntimeError: basic-pitch command not found`
**Solution**: Install Basic Pitch: `pip install basic-pitch`

### 4. Empty Beat Detection
**Symptom**: Zero beats detected
**Cause**: Silent audio or very short clips
**Solution**: Check audio has content > 1 second

### 5. Parquet Write Errors
**Symptom**: `ArrowInvalid: Cannot mix list and non-list values`
**Cause**: Inconsistent column types
**Solution**: Flatten nested dicts before creating DataFrame

## Updating Dependencies

1. Update version in `pyproject.toml`
2. Run `pip install -e .`
3. Run full test suite
4. Update README if breaking changes

## Performance Optimization

- Use `librosa.load()` with `sr=None` to avoid unnecessary resampling
- Process stems in parallel if I/O bound
- Cache beat detection results for repeated access
- Use numpy vectorized operations over loops

## Code Style

- Follow PEP 8
- Use type hints consistently
- Docstrings in Google style
- Maximum line length: 100 characters
- Run `ruff check src/` before committing

## Release Checklist

1. Update version in `pyproject.toml` and `__init__.py`
2. Run full test suite
3. Update README with new features
4. Create git tag: `git tag v0.x.0`
5. Build package: `python -m build`
6. Upload to PyPI (if applicable)
