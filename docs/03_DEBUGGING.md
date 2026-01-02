# Debugging Guide

This document helps troubleshoot common issues with didactic-engine.

---

## Quick Diagnostics

### Check Installation

```bash
# Verify package is installed
python -c "import didactic_engine; print(didactic_engine.__version__)"
# Expected: 0.1.0

# Verify CLI is available
didactic-engine --version
# Expected: didactic-engine 0.1.0
```

### Check Dependencies

```bash
# Core dependencies
python -c "import numpy, pandas, librosa, soundfile, pydub, pretty_midi, music21"

# Optional: Demucs
demucs --help

# Optional: Basic Pitch
basic-pitch --help

# Optional: Essentia
python -c "import essentia; print('Essentia available')"
```

---

## Common Issues

### Issue: "FFmpeg not found" warning

**Symptom:**
```
RuntimeWarning: Couldn't find ffmpeg or avconv
```

**Cause:** pydub requires FFmpeg for many audio operations.

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/ and add to PATH
```

### Issue: "Demucs is not installed"

**Symptom:**
```
RuntimeError: Demucs is not installed. Install with: pip install demucs
```

**Cause:** Demucs CLI not found on PATH.

**Solution:**
```bash
pip install demucs

# Verify
demucs --help
```

**Note:** If you don't need stem separation, the pipeline will continue
using the original audio as a single "full_mix" stem.

### Issue: "basic-pitch command not found"

**Symptom:**
```
RuntimeError: basic-pitch command not found. Install with: pip install basic-pitch
```

**Cause:** Basic Pitch CLI not found on PATH.

**Solution:**
```bash
pip install basic-pitch

# Verify
basic-pitch --help
```

**Note:** If Basic Pitch is unavailable, MIDI transcription is skipped.

### Issue: "Invalid audio data"

**Symptom:**
```
ValueError: Invalid audio data
```

**Cause:** Audio file is corrupted, empty, or in an unsupported format.

**Diagnosis:**
```python
from didactic_engine.ingestion import WAVIngester

ingester = WAVIngester()
try:
    audio, sr = ingester.load("problem_file.wav")
    print(f"Loaded: shape={audio.shape}, sr={sr}, dtype={audio.dtype}")
    print(f"Valid: {ingester.validate(audio, sr)}")
except Exception as e:
    print(f"Load failed: {e}")
```

**Common causes:**
- File is not a valid WAV (check with `file problem_file.wav`)
- File contains NaN or Inf values
- File is empty (0 samples)
- Sample rate is invalid

### Issue: Empty beat detection

**Symptom:**
```
Found 0 beats
Computed 0 bars
```

**Cause:** librosa couldn't detect beats in the audio.

**Diagnosis:**
```python
import librosa

audio, sr = librosa.load("problem_file.wav", sr=22050)
print(f"Duration: {len(audio)/sr:.2f}s")
print(f"Max amplitude: {audio.max():.4f}")

tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
print(f"Tempo: {tempo}, Beats: {len(beats)}")
```

**Common causes:**
- Audio is too short (< 1 second)
- Audio is silent or near-silent
- Audio has no rhythmic content (ambient/drone)

**Workaround:** The pipeline synthesizes beats using tempo if none detected.

### Issue: Parquet write errors

**Symptom:**
```
ArrowInvalid: Cannot mix list and non-list values
```

**Cause:** DataFrame contains nested structures that Parquet can't serialize.

**Solution:** The `utils_flatten.py` module provides flattening utilities:
```python
from didactic_engine.utils_flatten import flatten_dict_for_parquet

flat = flatten_dict_for_parquet(nested_dict)
```

### Issue: ABC export fails

**Symptom:**
```
Warning: ABC export failed for stem.mid: ...
```

**Cause:** music21 couldn't convert the MIDI file.

**Common causes:**
- MIDI has no notes
- MIDI has unusual time signatures
- MIDI has complex polyphony

**Note:** ABC export continues on failure, writing an error message to the .abc file.

---

## Logging and Verbose Output

### Enable Verbose Pipeline Output

The pipeline prints step-by-step progress by default:

```
Step 1: Copying input file...
Step 2: Ingesting WAV file...
  Loaded audio: duration=180.25s, sr=22050
Step 3: Preprocessing audio...
...
```

### Debug Individual Components

```python
import numpy as np
from didactic_engine.ingestion import WAVIngester
from didactic_engine.analysis import AudioAnalyzer

# Test ingestion
ingester = WAVIngester(sample_rate=22050)
audio, sr = ingester.load("song.wav")
print(f"Audio: shape={audio.shape}, sr={sr}, dtype={audio.dtype}")
print(f"Valid: {ingester.validate(audio, sr)}")
print(f"Range: [{audio.min():.4f}, {audio.max():.4f}]")

# Test analysis
analyzer = AudioAnalyzer(use_essentia=False)
features = analyzer.analyze(audio, sr)
print(f"Tempo: {features['tempo']:.1f} BPM")
print(f"Beats: {len(features['beat_times'])}")
```

---

## Performance Issues

### Slow Stem Separation

Demucs is computationally intensive. Solutions:
- Use GPU if available: Demucs auto-detects CUDA
- Use a lighter model: `--demucs-model htdemucs_ft` (faster but lower quality)
- Skip separation: Pipeline falls back to single "full_mix" stem

### Slow MIDI Transcription

Basic Pitch is also computationally intensive. Solutions:
- Use GPU if available
- Process shorter audio segments
- Skip transcription if not needed

### Large Output Files

Per-bar chunks can consume significant disk space. Solutions:
- Use `--no-bar-chunks` to skip chunk generation
- Process stems individually
- Clean up chunks after feature extraction

---

## Testing Specific Components

### Test Ingestion

```python
import pytest
from didactic_engine.ingestion import WAVIngester

def test_load_valid_wav():
    ingester = WAVIngester()
    audio, sr = ingester.load("test.wav")
    assert audio.ndim == 1  # Mono
    assert audio.dtype == np.float32
    assert ingester.validate(audio, sr)
```

### Test Analysis

```python
from didactic_engine.analysis import AudioAnalyzer

def test_analysis():
    analyzer = AudioAnalyzer(use_essentia=False)
    audio = np.random.randn(22050 * 5).astype(np.float32)  # 5 seconds
    features = analyzer.analyze(audio, 22050)
    assert "tempo" in features
    assert "beat_times" in features
```

---

## Getting Help

### Collect Diagnostic Info

When reporting issues, include:

```bash
# Python version
python --version

# Package version
python -c "import didactic_engine; print(didactic_engine.__version__)"

# Installed packages
pip list | grep -E "(numpy|pandas|librosa|soundfile|pydub|demucs|basic-pitch|essentia|music21)"

# System info
uname -a  # Linux/macOS
```

### Minimal Reproduction

Create a minimal script that reproduces the issue:

```python
from didactic_engine import PipelineConfig, AudioPipeline
from pathlib import Path

cfg = PipelineConfig(
    song_id="test",
    input_wav=Path("problem_file.wav"),
    out_dir=Path("test_output"),
)

try:
    pipeline = AudioPipeline(cfg)
    results = pipeline.run()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

---

## See Also

- [Architecture](01_ARCHITECTURE.md) - System design
- [Key Flows](02_KEY_FLOWS.md) - Pipeline execution flows
- [Contributing](04_CONTRIBUTING.md) - How to report bugs
