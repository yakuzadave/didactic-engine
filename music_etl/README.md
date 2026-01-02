# Music ETL Pipeline

A comprehensive Python 3.11+ audio processing pipeline that extracts structured data from audio files through stem separation, feature analysis, MIDI transcription, and bar-level segmentation.

## What This Project Does

The Music ETL pipeline processes audio files through the following stages:

1. **Stem Separation**: Splits audio into individual stems (vocals, drums, bass, other) using Demucs
2. **Preprocessing**: Optionally normalizes, trims silence, and enhances stems with pydub
3. **Audio Analysis**: Extracts comprehensive features using librosa (tempo, beats, spectral features, MFCCs, chroma)
4. **Essentia Enrichment**: Optionally adds advanced features from Essentia
5. **MIDI Transcription**: Converts audio to MIDI using Spotify's Basic Pitch
6. **MIDI Parsing**: Extracts note events, tempo maps, and instrument information
7. **Beat/Bar Alignment**: Aligns MIDI events to detected beat and bar boundaries
8. **Bar Chunking**: Segments stems into per-bar audio chunks for detailed analysis
9. **Feature Extraction**: Computes evolving bar-level features for each segment
10. **Export**: Outputs to JSON summaries, Markdown reports, ABC notation, and Parquet datasets

## Prerequisites

### Python Environment
- **Python 3.11 or higher** (tested on 3.11 and 3.12)

### Required Command-Line Tools

The pipeline requires these tools to be installed and available on your PATH:

1. **Demucs** (stem separation)
   ```bash
   pip install demucs
   ```

2. **Basic Pitch** (MIDI transcription)
   ```bash
   pip install basic-pitch
   ```

### Optional Dependencies

- **Essentia** (advanced audio analysis):
  ```bash
  pip install essentia
  ```
  If not installed, the pipeline will skip Essentia features and continue normally.

- **FFmpeg** (for pydub to handle non-WAV formats):
  Most distributions include this, but pydub works with WAV files without it.

## Installation

### 1. Clone or extract this repository

```bash
cd music_etl
```

### 2. Install the package

Basic installation (without Essentia):
```bash
pip install -e .
```

With Essentia support:
```bash
pip install -e .[essentia]
```

For development (includes testing tools):
```bash
pip install -e .[dev]
```

### 3. Verify CLI tools

Check that required tools are available:
```bash
demucs --help
basic-pitch --help
```

## Usage

### Basic Command

```bash
python scripts/run_pipeline.py --wav data/input/song.wav --song-id my_song
```

### Command-Line Options

```
Required:
  --wav PATH        Path to input WAV file
  --song-id ID      Unique identifier for this song

Optional:
  --out DIR         Output base directory (default: data)
  --demucs-model    Demucs model name (default: htdemucs)
  --sr RATE         Sample rate for analysis (default: 22050)
  --hop LENGTH      Hop length for STFT (default: 512)
  --ts-num N        Time signature numerator (default: 4)
  --ts-den N        Time signature denominator (default: 4)
  --no-preprocess   Skip pydub preprocessing
  --no-essentia     Skip Essentia feature extraction
  --no-chunks       Skip per-bar audio chunk writing
```

### Examples

**Process with custom output directory:**
```bash
python scripts/run_pipeline.py \
  --wav audio/song.wav \
  --song-id song1 \
  --out output/
```

**Process a 3/4 time signature (waltz):**
```bash
python scripts/run_pipeline.py \
  --wav waltz.wav \
  --song-id waltz \
  --ts-num 3 \
  --ts-den 4
```

**Skip preprocessing and chunk writing:**
```bash
python scripts/run_pipeline.py \
  --wav song.wav \
  --song-id song1 \
  --no-preprocess \
  --no-chunks
```

## Output Structure

After running the pipeline, the output directory will contain:

```
data/
├── stems/<song_id>/          # Separated audio stems
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
│
├── preprocessed/<song_id>/   # Preprocessed stems (if enabled)
│   └── *.wav
│
├── chunks/<song_id>/         # Per-bar audio chunks (if enabled)
│   ├── vocals/
│   │   ├── bar_0000.wav
│   │   ├── bar_0001.wav
│   │   └── ...
│   └── drums/
│       └── ...
│
├── midi/<song_id>/           # Transcribed MIDI files
│   ├── vocals.mid
│   ├── drums.mid
│   └── ...
│
├── analysis/<song_id>/       # Analysis results
│   └── combined.json         # Summary JSON with tempo, features, etc.
│
├── reports/<song_id>/        # Human-readable reports
│   ├── midi_markdown.md      # Markdown table of MIDI events by bar/stem
│   ├── vocals.abc            # ABC notation per stem
│   └── ...
│
└── datasets/<song_id>/       # Parquet datasets for analysis
    ├── events.parquet        # All MIDI note events with alignment
    ├── beats.parquet         # Beat times per stem
    ├── bars.parquet          # Aggregated bar-level note statistics
    └── bar_features.parquet  # Detailed bar-level audio features
```

### Dataset Schemas

**events.parquet** - MIDI note events:
- `song_id`, `stem`, `bar_index`, `beat_index`, `beat_in_bar`
- `pitch`, `velocity`, `start_s`, `end_s`, `dur_s`
- `instrument_index`, `instrument_name`, `program`, `is_drum`
- `start_beat_float`, `end_beat_float`

**beats.parquet** - Beat grid:
- `song_id`, `stem`, `beat_index`, `time_s`, `tempo_bpm`

**bars.parquet** - Bar-level note aggregations:
- `song_id`, `stem`, `bar_index`
- `num_notes`, `mean_velocity`, `pitch_min`, `pitch_max`
- `start_s`, `end_s`

**bar_features.parquet** - Detailed audio features per bar:
- `song_id`, `stem`, `bar_index`, `start_s`, `end_s`, `duration_s`
- `tempo_bpm`, `chunk_path`
- `chroma_mean_00` through `chroma_mean_11`
- `mfcc_00_mean`, `mfcc_00_std`, ... through `mfcc_12_std`
- `spectral_centroid_mean`, `spectral_centroid_std`, ...
- `zcr_mean`, `zcr_std`, ...
- Optional: `essentia.*` fields (if Essentia enabled)

## Notes and Limitations

### Bar Segmentation
- Assumes constant time signature throughout the song
- Beat tracking is approximate and may drift for complex rhythms
- Bars are extrapolated to cover full audio duration

### MIDI Transcription
- Works best on monophonic or simple polyphonic material
- Drums and percussive sounds may produce noisy transcriptions
- Complex chords may be partially captured

### ABC Notation
- Best suited for monophonic or simple melodic lines
- Complex polyphonic MIDI may not render well in ABC format
- Export failures are logged but don't stop the pipeline

### Performance
- Full pipeline can take 5-20 minutes depending on audio length and system
- Demucs separation is the most time-intensive step
- Use `--no-chunks` to skip bar segmentation if not needed

## Troubleshooting

### "demucs command not found"
Install Demucs: `pip install demucs`

### "basic-pitch command not found"
Install Basic Pitch: `pip install basic-pitch`

### Essentia errors
Essentia is optional. If you see Essentia-related warnings, the pipeline will continue without those features. Install with `pip install essentia` if desired.

### Audio format issues
The pipeline expects WAV files. If you have MP3/FLAC/etc., convert first:
```bash
ffmpeg -i input.mp3 output.wav
```

### Out of memory errors
Reduce the sample rate: `--sr 16000`
Or skip bar chunking: `--no-chunks`

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
ruff check src/
```

### Type Checking
```bash
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite the underlying tools:
- [Demucs](https://github.com/facebookresearch/demucs) for stem separation
- [Basic Pitch](https://github.com/spotify/basic-pitch) for MIDI transcription
- [librosa](https://librosa.org/) for audio analysis
- [Essentia](https://essentia.upf.edu/) for advanced features (optional)

## Contributing

Contributions welcome! Please open an issue or pull request.

## Contact

For questions or issues, please open a GitHub issue.
