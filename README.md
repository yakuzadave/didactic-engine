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

### Windows / WSL environment setup

Full cross-platform setup (CPU and NVIDIA GPU) is documented in **[docs/environment.md](docs/environment.md)**. Quick start:

```bash
python3.11 -m venv .venv
source .venv/bin/activate                 # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"                   # CPU build with tests/linting
```

For GPU acceleration (NVIDIA drivers required):

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -e ".[ml-gpu,dev]"            # ONNX Runtime GPU + Demucs + dev tools
```

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

### Quickstart Demo

The fastest way to get started:

```bash
# Process a single file and generate visualizations
python examples/quickstart_demo.py path/to/audio.wav

# Process with custom output directory
python examples/quickstart_demo.py audio.wav --output my_output/

# Process without visualizations (faster)
python examples/quickstart_demo.py audio.wav --no-viz
```

This will:
1. Process the audio file (ingestion, analysis, feature extraction)
2. Generate interactive HTML visualizations
3. Display summary statistics
4. Show output directory structure

### Command Line Interface

#### Process a single WAV file:

```bash
didactic-engine --wav input.wav --song-id my_song --out data/
```

#### Process multiple WAV files (batch mode):

```bash
# Process multiple files with auto-generated song IDs
didactic-engine --wav song1.wav song2.wav song3.wav --out data/

# Process all WAV files in a directory
didactic-engine --wav *.wav --out data/
```

Note: CLI batch processing runs files sequentially by default. For parallel processing, use an
external tool like GNU parallel or the provided multiprocessing workflow script:

```bash
# GNU parallel (example)
parallel didactic-engine --wav {} --out data/ ::: *.wav

# Python multiprocessing workflow (example)
python examples/workflow_batch_viz.py *.wav --workers 8 --output data/
```

With custom options:

```bash
didactic-engine --wav input.wav --song-id my_song --out data/ \
    --sr 22050 \
    --ts-num 4 \
    --ts-den 4 \
    --use-essentia
```

View all options:

```bash
didactic-engine --help
```

### Python API

#### Single file processing:

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

#### Batch processing:

```python
from didactic_engine.pipeline import AudioPipeline
from pathlib import Path

# Process multiple files
input_files = [
    Path("song1.wav"),
    Path("song2.wav"),
    Path("song3.wav"),
]

results = AudioPipeline.process_batch(
    input_files,
    out_dir=Path("data"),
    analysis_sr=22050,
    use_essentia_features=False,
)

print(f"Processed {results['success_count']}/{results['total']} files")
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

## Tutorial and Examples

### Interactive Tutorials

We provide comprehensive interactive tutorials in multiple formats:

#### Jupyter Notebook (Google Colab Compatible)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakuzadave/didactic-engine/blob/main/examples/tutorial.ipynb)

**[examples/tutorial.ipynb](examples/tutorial.ipynb)** - Traditional Jupyter notebook format

#### Marimo Notebook (Pure Python, Git-Friendly)

**[examples/tutorial_marimo.py](examples/tutorial_marimo.py)** - Lightweight, reactive notebook

Run with: `marimo edit examples/tutorial_marimo.py`

Learn more: **[examples/MARIMO_README.md](examples/MARIMO_README.md)**

Both tutorials cover:
- Installation (local and Google Colab)
- Basic usage examples
- Individual component usage
- Full pipeline execution
- Output structure and dataset exploration
- Troubleshooting tips

### Python Examples

See **[examples/example_usage.py](examples/example_usage.py)** for additional code examples including:
- Basic usage
- Advanced usage with Essentia
- Batch processing
- Custom configuration
- Individual component access

### Batch Processing Examples

See **[examples/batch_processing_example.py](examples/batch_processing_example.py)** for advanced batch processing patterns:
- Simple batch processing with default settings
- Parallel processing with multiprocessing
- Progress tracking with tqdm
- Processing from directory trees
- Retry logic for failed files
- Batch results serialization

Install batch processing dependencies:
```bash
pip install -e ".[batch]"
```

### Visualization Examples

See **[examples/visualization_plotly.py](examples/visualization_plotly.py)** for interactive Plotly visualizations:
- Audio waveforms and spectrograms
- Tempo and beat detection results
- Feature distributions and timelines
- Stem comparison charts
- MIDI piano roll visualization
- Batch processing statistics
- Feature heatmaps

Install visualization dependencies:
```bash
pip install -e ".[viz]"
```

Run visualizations:
```bash
python examples/visualization_plotly.py
# Open generated HTML files in output/visualizations/
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
TMPDIR=/tmp TEMP=/tmp TMP=/tmp pytest -v   # WSL: avoid Windows temp dir issues
pytest -m "not optional_deps and not integration"    # Skip heavy/optional stacks
pytest -m gpu                                        # Only GPU-specific tests
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
- torchcodec (required by torchaudio 2.9+ for audio encoding/saving)

## License

MIT License

## Acknowledgments

Built with:
- [Demucs](https://github.com/facebookresearch/demucs) for stem separation
- [Basic Pitch](https://github.com/spotify/basic-pitch) for MIDI transcription
- [librosa](https://librosa.org/) for audio analysis
- [Essentia](https://essentia.upf.edu/) for advanced features
- [music21](https://web.mit.edu/music21/) for ABC notation export
