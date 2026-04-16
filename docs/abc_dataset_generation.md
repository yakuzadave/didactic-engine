# ABC Notation Dataset Generation

This guide explains how to use didactic-engine to generate datasets for training audio generation models conditioned on ABC notation.

## Overview

The pipeline creates training-ready datasets where each audio clip is paired with ABC notation text as a prompt. This enables "steerable" audio generation where you can control musical content via symbolic notation.

## Features

### 1. MIDI Quantization

Quantization cleans up timing jitter from transcription, producing more readable ABC notation.

**Config options:**
```python
config = PipelineConfig(
    quantize_midi=True,           # Enable quantization
    quantize_division=16,         # 16=sixteenth notes (default)
    quantize_min_duration=None,   # Optional minimum note duration
)
```

**Division values:**
- `4`: Quarter notes (very coarse, simple notation)
- `8`: Eighth notes (coarse grid)
- `12`: Triplet-friendly (1/12 notes)
- `16`: Sixteenth notes (**default**, suitable for most music)
- `24`: Twenty-fourth notes (fine grid, preserves more detail)

### 2. ABC Text Export

ABC notation is exported both as files and as text strings for metadata.

**Output files:**
- `reports/{song_id}/{stem}.abc` - ABC notation files
- Results dict includes `abc_text_{stem}` keys with raw ABC text

### 3. Metadata Export

Generates `metadata.jsonl` compatible with HuggingFace AudioFolder:

```python
config = PipelineConfig(
    export_metadata_jsonl=True,
    abc_trigger_token="abcstyle",  # Conditioning token
    abc_max_chars=2500,            # Max ABC length in prompts
)
```

**Output:**
- `datasets/{song_id}/metadata.jsonl` - One JSON per audio chunk

**Format:**
```json
{
  "file_name": "chunks/song1/vocals/bar_0001.wav",
  "text": "abcstyle music matching this ABC notation: <abc> X:1\nM:4/4\n... </abc>",
  "abc_raw": "X:1\nM:4/4\n...",
  "source_track": "song1.wav",
  "stem_used": "vocals",
  "start_sec": 0.0,
  "end_sec": 3.2,
  "tempo_bpm": 120.0,
  "sr": 22050,
  "bar_index": 1,
  "rms_energy": 0.042
}
```

### 4. Stem Selection (Not Yet Integrated)

The `stem_selector.py` module provides functions to automatically select the best stem for melody transcription using pitch salience analysis.

**Usage (manual):**
```python
from didactic_engine.stem_selector import select_best_melody_stem

# After stem separation
stem_paths = {
    "vocals": Path("vocals.wav"),
    "other": Path("other.wav"),
    "bass": Path("bass.wav"),
}

best_stem, best_path, scores = select_best_melody_stem(
    stem_paths,
    sample_rate=22050,
    candidate_stems=("vocals", "other", "bass"),
)

print(f"Best: {best_stem} (score: {scores[best_stem]:.3f})")
# Then use best_path for transcription
```

## Quick Start

### Using the Example Script

```bash
# Basic usage with default settings
python examples/abc_dataset_example.py song.wav output/

# Full dataset generation with quantization and metadata
python examples/abc_dataset_example.py song.wav output/ --quantize --metadata

# Customize quantization (eighth notes for simpler notation)
python examples/abc_dataset_example.py song.wav output/ --quantize --division 8

# Custom time signature
python examples/abc_dataset_example.py song.wav output/ --time-sig 3/4 --quantize
```

### Using the Python API

```python
from pathlib import Path
from didactic_engine.config import PipelineConfig
from didactic_engine.pipeline import AudioPipeline

# Configure for ABC dataset generation
config = PipelineConfig(
    song_id="my_song",
    input_wav=Path("input.wav"),
    out_dir=Path("output"),
    # Enable ABC features
    quantize_midi=True,
    quantize_division=16,
    export_metadata_jsonl=True,
    abc_trigger_token="abcstyle",
    # Optional: faster processing
    write_bar_chunk_wavs=False,  # Don't write individual bar files
)

# Run pipeline
pipeline = AudioPipeline(config)
results = pipeline.run()

# Access results
print(f"ABC notation: {results['abc_vocals']}")
print(f"Metadata: {results['metadata_jsonl']}")
print(f"Entries: {results['metadata_entry_count']}")
```

### Loading with HuggingFace Datasets

```python
from datasets import load_dataset

# Load the generated dataset
ds = load_dataset(
    "audiofolder",
    data_dir="output/datasets/my_song"
)

# Inspect an entry
print(ds["train"][0])
# {
#   'audio': {'array': [...], 'sampling_rate': 22050},
#   'text': 'abcstyle music matching this ABC notation: <abc> ... </abc>',
#   'abc_raw': 'X:1\nM:4/4\n...',
#   ...
# }
```

## Workflow

The typical ABC dataset generation workflow:

1. **Audio Ingestion** → WAV file loaded and validated
2. **Preprocessing** → Normalize, trim silence (optional)
3. **Stem Separation** → Demucs creates vocals/drums/bass/other
4. **Audio Analysis** → Detect tempo, beats, bars
5. **Bar Segmentation** → Split into per-bar audio chunks
6. **MIDI Transcription** → Basic Pitch transcribes each stem
7. **MIDI Quantization** → Snap notes to rhythmic grid (**optional**)
8. **ABC Export** → Convert quantized MIDI to ABC notation
9. **Metadata Generation** → Create metadata.jsonl (**optional**)

## Output Structure

```
output/
├── datasets/{song_id}/
│   ├── metadata.jsonl           # HuggingFace AudioFolder metadata
│   ├── events.parquet           # Event-level features
│   ├── beats.parquet            # Beat-level features
│   ├── bars.parquet             # Bar-level features
│   └── bar_features.parquet     # Bar audio features
├── chunks/{song_id}/{stem}/
│   ├── bar_0000.wav             # Per-bar audio chunks
│   ├── bar_0001.wav
│   └── ...
├── reports/{song_id}/
│   ├── vocals.abc               # ABC notation files
│   ├── other.abc
│   ├── bass.abc
│   └── midi_markdown.md         # Human-readable report
├── midi/{song_id}/
│   ├── vocals.mid               # Transcribed MIDI
│   ├── vocals_quantized.mid     # Quantized MIDI (if enabled)
│   └── ...
└── stems/{song_id}/
    ├── vocals.wav
    ├── drums.wav
    ├── bass.wav
    └── other.wav
```

## Best Practices

### 1. Choose Appropriate Quantization

- **Most music**: `division=16` (sixteenth notes)
- **Triplet-heavy**: `division=12` 
- **Simple melodies**: `division=8` (eighth notes)
- **Preserve detail**: `division=24` (finer grid)

### 2. Prompt Consistency

Keep the `abc_trigger_token` consistent across your dataset. The model learns to associate this token with ABC-conditioned generation.

```python
# Use the same trigger token for all songs
config = PipelineConfig(
    abc_trigger_token="abcstyle",  # Consistent across dataset
    ...
)
```

### 3. ABC Quality Filtering

Not all transcriptions are perfect. Consider filtering:

```python
import pandas as pd

# Load metadata
metadata = pd.read_json("output/datasets/song/metadata.jsonl", lines=True)

# Filter by ABC length (avoid empty or too-long)
good_entries = metadata[
    (metadata["abc_raw"] != "NA") &
    (metadata["abc_raw"].str.len() > 50) &
    (metadata["abc_raw"].str.len() < 2000)
]

print(f"Kept {len(good_entries)}/{len(metadata)} entries")
```

### 4. Stem Selection

For best results, transcribe the **melody-dominant stem**:

- **Vocal music**: Usually `vocals` stem
- **Instrumental**: Usually `other` stem (lead guitar/synth/etc.)
- **Bass lines**: Use `bass` stem

The `stem_selector` module can automate this (manual use for now).

## Limitations

### ABC Export

- Works best on **monophonic** or simple polyphonic content
- Complex rhythms may not be perfectly represented
- **Drums produce poor results** (no melodic content)

### Transcription Quality

- Basic Pitch is imperfect - expect some transcription errors
- Very quiet or noisy stems transcribe poorly
- Quantization helps but doesn't fix bad transcriptions

### Dataset Size

- Per-bar chunks create many small files (can be slow on some filesystems)
- Use `write_bar_chunk_wavs=False` if you only need metadata
- Consider WSL users: writing to `/mnt/c/...` is slower than native Linux paths

## Advanced: Manual Stem Selection

Until automatic stem selection is integrated, you can manually choose stems:

```python
# Run pipeline first to get stems
results1 = pipeline.run()

# Manually inspect which stem has the best melody
# Then re-run MIDI transcription on just that stem

from didactic_engine.stem_selector import score_melody_stem
import librosa

# Score each stem
for stem_name in ["vocals", "other", "bass"]:
    stem_path = config.stems_dir / f"{stem_name}.wav"
    audio, sr = librosa.load(stem_path, sr=22050, mono=True)
    score = score_melody_stem(audio, sr)
    print(f"{stem_name}: {score:.3f}")

# Highest score = best melody stem
```

## Troubleshooting

### "No metadata entries to export"

- Ensure `write_bar_chunks=True` and `write_bar_chunk_wavs=True`
- Check that beats were detected (look at logs)
- Verify audio is longer than a few seconds

### ABC files contain "% ABC export failed"

- music21 not installed: `pip install music21`
- MIDI file is empty (no notes transcribed)
- Check transcription logs for errors

### Quantized MIDI not created

- Ensure `quantize_midi=True` in config
- Check logs for "MIDI quantization failed" warnings
- Verify tempo was detected correctly

### Metadata.jsonl has wrong file paths

- File paths are relative to `out_dir`
- Check that chunk files actually exist on disk
- Verify `chunks_dir` structure matches expectations

## References

- [HuggingFace AudioFolder](https://huggingface.co/docs/datasets/audio_dataset)
- [ABC Notation Standard](http://abcnotation.com/)
- [Basic Pitch (Spotify)](https://github.com/spotify/basic-pitch)
- [Demucs (Meta)](https://github.com/facebookresearch/demucs)
