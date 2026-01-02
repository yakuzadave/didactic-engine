# Didactic Engine

A comprehensive Python 3.11+ audio processing toolkit for music analysis, stem separation, MIDI transcription, and dataset generation.

## What Does It Do?

Didactic Engine is an end-to-end audio processing pipeline that:

1. **Ingests** WAV files and validates their format
2. **Preprocesses** audio (resampling, mono conversion, normalization, silence trimming)
3. **Separates stems** using Demucs (vocals, drums, bass, other)
4. **Extracts features** (tempo, beats, spectral features, MFCCs, chroma)
5. **Transcribes to MIDI** using Spotify's Basic Pitch
6. **Aligns notes** to a beat/bar grid based on a configurable time signature
7. **Segments audio** into per-bar WAV chunks
8. **Generates datasets** in Parquet format (events, beats, bars, bar features)
9. **Exports reports** in Markdown and ABC notation

## Installation

### Prerequisites

- Python 3.11 or higher
- FFmpeg (for audio processing with pydub)
- Demucs CLI (for stem separation) - optional
- Basic Pitch CLI (for MIDI transcription) - optional

### Install from Source

```bash
git clone https://github.com/yakuzadave/didactic-engine.git
cd didactic-engine
pip install -e .
```

### Install Optional Dependencies

For Essentia features:
```bash
pip install -e ".[essentia]"
```

For ML features (Demucs, Basic Pitch, and ONNX inference):
```bash
pip install -e ".[ml]"
```

For all extras:
```bash
pip install -e ".[all]"
```

### Python Version Notes

- **Python 3.11**: Full support including Basic Pitch MIDI transcription (uses TensorFlow backend)
- **Python 3.12+**: Full support with ONNX Runtime as the ML inference backend. Basic Pitch is not available on Python 3.12+ due to TensorFlow compatibility requirements (Basic Pitch requires TensorFlow < 2.15.1, which is not available for Python 3.12).

The `ml` extras include `onnxruntime` which provides:
- ONNX model loading and inference
- GPU acceleration support (CUDA, TensorRT, CoreML)
- Cross-platform compatibility

```python
from didactic_engine.onnx_inference import (
    is_onnxruntime_available,
    create_inference_session,
)

# Check availability
if is_onnxruntime_available():
    # Load and run ONNX model
    session = create_inference_session("model.onnx", prefer_gpu=True)
    outputs = session.run({"input": audio_data})
```

## Usage

### Command Line Interface

Process a WAV file:

```bash
didactic-engine --wav input.wav --song-id my_song --out data/
```

With custom options:

```bash
didactic-engine --wav input.wav --song-id my_song --out data/ \
    --sr 44100 \
    --ts-num 4 \
    --ts-den 4 \
    --use-essentia
```

View all options:

```bash
didactic-engine --help
```

### Python API

```python
from didactic_engine import AudioPipeline, PipelineConfig
from pathlib import Path

# Create configuration
cfg = PipelineConfig(
    song_id="my_song",
    input_wav=Path("input.wav"),
    out_dir=Path("data"),
    analysis_sr=22050,
    time_signature_num=4,
    time_signature_den=4,
    use_essentia_features=False,
    write_bar_chunks=True,
)

# Run pipeline
pipeline = AudioPipeline(cfg)
results = pipeline.run()
```

### Using Individual Components

```python
from didactic_engine.ingestion import WAVIngester
from didactic_engine.analysis import AudioAnalyzer
from didactic_engine.preprocessing import AudioPreprocessor

# Load and validate audio
ingester = WAVIngester(sample_rate=22050)
audio, sr = ingester.load("input.wav")

# Analyze
analyzer = AudioAnalyzer(use_essentia=False)
analysis = analyzer.analyze(audio, sr)
print(f"Tempo: {analysis['tempo']:.2f} BPM")
print(f"Beats: {len(analysis['beat_times'])}")

# Preprocess
preprocessor = AudioPreprocessor()
normalized = preprocessor.normalize(audio, sr)
```

## Output Directory Structure

```
data/
├── input/<song_id>/
│   └── original_copy.wav
├── preprocessed/<song_id>/
│   └── <song_id>.wav
├── stems/<song_id>/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
├── chunks/<song_id>/<stem>/
│   ├── bar_0000.wav
│   ├── bar_0001.wav
│   └── ...
├── midi/<song_id>/
│   ├── vocals.mid
│   ├── drums.mid
│   └── ...
├── analysis/<song_id>/
│   └── combined.json
├── reports/<song_id>/
│   ├── midi_markdown.md
│   ├── vocals.abc
│   └── ...
└── datasets/<song_id>/
    ├── events.parquet
    ├── beats.parquet
    ├── bars.parquet
    └── bar_features.parquet
```

## Features

### Audio Analysis
- Tempo detection
- Beat tracking
- Spectral features (centroid, bandwidth, rolloff)
- MFCCs (13 coefficients)
- Chroma features
- Zero crossing rate
- RMS energy

### Optional Essentia Features
- Stevens loudness
- EBU R128 loudness
- Additional MFCC analysis
- Spectral centroid variations

### Dataset Outputs
- **events.parquet**: Individual note events with timing
- **beats.parquet**: Beat grid information
- **bars.parquet**: Bar-level aggregations
- **bar_features.parquet**: Audio features per bar

## Limitations and Caveats

- **Demucs** must be installed separately (`pip install demucs`)
- **Basic Pitch** must be installed separately (`pip install basic-pitch`)
- **Essentia** has AGPL license considerations if distributed
- Beat detection assumes **steady tempo** and works best with rhythmic music
- Bar detection assumes a **simple time signature** (default: 4/4)
- Basic Pitch may mis-transcribe polyphonic audio; stem separation helps

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
ruff check src/
```

## Dependencies

### Core
- numpy, pandas, pyarrow
- librosa, soundfile, pydub
- pretty-midi, music21

### Optional
- essentia (for advanced audio features)
- demucs (for stem separation)
- basic-pitch (for MIDI transcription, Python 3.11 only)
- onnxruntime (for ONNX model inference, Python 3.12+ compatible)

## License

MIT License

## Acknowledgments

Built with:
- [Demucs](https://github.com/facebookresearch/demucs) for stem separation
- [Basic Pitch](https://github.com/spotify/basic-pitch) for MIDI transcription
- [librosa](https://librosa.org/) for audio analysis
- [Essentia](https://essentia.upf.edu/) for advanced features
- [music21](https://web.mit.edu/music21/) for ABC notation export