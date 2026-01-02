# Didactic Engine

A comprehensive Python 3.11+ audio processing pipeline for music analysis and manipulation.

## Features

- **WAV Ingestion**: Load and validate WAV audio files with automatic resampling
- **Stem Separation**: Separate audio into individual stems (vocals, drums, bass, other) using Demucs
- **Audio Preprocessing**: Normalize, compress, and process stems with pydub
- **Audio Analysis**: Comprehensive analysis using librosa and optional Essentia
  - Tempo and beat detection
  - Spectral feature extraction
  - Onset detection
  - Chroma and MFCCs
- **MIDI Transcription**: Convert audio to MIDI using Spotify's Basic Pitch
- **MIDI Parsing**: Parse and manipulate MIDI data with pretty_midi
- **Beat/Bar Alignment**: Align MIDI events to detected beat and bar grids
- **Stem Segmentation**: Split stems into per-bar WAV chunks
- **Feature Extraction**: Extract evolving bar-level features for analysis

## Installation

### Requirements
- Python 3.11 or higher
- FFmpeg (for audio processing)

### Install from source

```bash
git clone https://github.com/yakuzadave/didactic-engine.git
cd didactic-engine
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

Process a single audio file:

```bash
didactic-engine input.wav -o output/
```

Process multiple files:

```bash
didactic-engine file1.wav file2.wav file3.wav -o output/
```

With custom options:

```bash
didactic-engine input.wav -o output/ \
    --sample-rate 48000 \
    --use-essentia \
    --beats-per-bar 4
```

### Python API

```python
from didactic_engine.pipeline import AudioPipeline

# Initialize pipeline
pipeline = AudioPipeline(
    sample_rate=44100,
    use_essentia=False,
    preprocess_stems=True,
    beats_per_bar=4,
)

# Process audio file
results = pipeline.process("input.wav", "output/")

# Access results
print(f"Tempo: {results['analysis']['tempo']:.2f} BPM")
print(f"Bars detected: {len(results['bar_times'])}")
print(f"Stems: {results['stem_names']}")
```

### Using Individual Components

```python
from didactic_engine.ingestion import WAVIngester
from didactic_engine.analysis import AudioAnalyzer
from didactic_engine.separation import StemSeparator

# Load audio
ingester = WAVIngester(sample_rate=44100)
audio, sr = ingester.load("input.wav")

# Analyze
analyzer = AudioAnalyzer()
analysis = analyzer.analyze(audio, sr)

# Separate stems
separator = StemSeparator()
stems = separator.separate(audio, sr)
```

## Pipeline Architecture

The complete pipeline consists of the following stages:

1. **Ingestion**: Load WAV files and validate audio data
2. **Analysis**: Extract tempo, beats, bars, and spectral features
3. **Separation**: Split audio into instrumental stems
4. **Preprocessing**: Normalize and enhance stem audio quality
5. **Transcription**: Convert audio to MIDI notation
6. **MIDI Parsing**: Extract note events and timing information
7. **Alignment**: Align MIDI events to beat/bar grid
8. **Segmentation**: Split stems into per-bar chunks
9. **Feature Extraction**: Extract bar-level audio features

## Output Structure

```
output/
├── pipeline_results.json       # Complete pipeline results
├── transcription.mid          # Transcribed MIDI file
├── stems/                     # Separated audio stems
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
├── segments/                  # Per-bar audio chunks
│   ├── vocals/
│   │   ├── vocals_bar_0000.wav
│   │   ├── vocals_bar_0001.wav
│   │   └── ...
│   └── drums/
│       ├── drums_bar_0000.wav
│       └── ...
└── features/                  # Extracted features
    ├── vocals_features.json
    ├── drums_features.json
    └── ...
```

## API Documentation

### AudioPipeline

Main pipeline class that orchestrates the complete workflow.

**Parameters:**
- `sample_rate` (int): Target sample rate for processing (default: 44100)
- `use_essentia` (bool): Enable Essentia for advanced analysis (default: False)
- `preprocess_stems` (bool): Apply preprocessing to stems (default: True)
- `beats_per_bar` (int): Number of beats per bar (default: 4)

**Methods:**
- `process(input_wav_path, output_dir)`: Process a single WAV file
- `process_batch(input_wav_paths, output_base_dir)`: Process multiple files

### Individual Components

- `WAVIngester`: Load and validate WAV files
- `StemSeparator`: Separate audio into stems using Demucs
- `AudioPreprocessor`: Normalize and process audio with pydub
- `AudioAnalyzer`: Analyze audio with librosa and Essentia
- `MIDITranscriber`: Transcribe audio to MIDI using Basic Pitch
- `MIDIParser`: Parse and manipulate MIDI data
- `StemSegmenter`: Segment audio into time-based chunks
- `FeatureExtractor`: Extract audio features

## Examples

See the `examples/` directory for detailed usage examples:

- `example_usage.py`: Comprehensive examples of all major features

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
ruff check src/
```

## Dependencies

### Core Dependencies
- `numpy`: Numerical computing
- `scipy`: Scientific computing
- `librosa`: Audio analysis
- `soundfile`: Audio I/O
- `pydub`: Audio processing
- `demucs`: Stem separation
- `basic-pitch`: MIDI transcription
- `pretty-midi`: MIDI manipulation
- `essentia`: Advanced audio analysis (optional)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{didactic_engine,
  title = {Didactic Engine: Audio Processing Pipeline},
  author = {Yakuza Dave},
  year = {2026},
  url = {https://github.com/yakuzadave/didactic-engine}
}
```

## Acknowledgments

This project builds upon several excellent open-source libraries:
- [Demucs](https://github.com/facebookresearch/demucs) for stem separation
- [Basic Pitch](https://github.com/spotify/basic-pitch) for MIDI transcription
- [librosa](https://librosa.org/) for audio analysis
- [Essentia](https://essentia.upf.edu/) for advanced audio features