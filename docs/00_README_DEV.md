# Developer Guide: Navigating the Didactic-Engine Codebase

Welcome to the didactic-engine developer documentation! This guide will help you
understand, navigate, and contribute to the codebase.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/yakuzadave/didactic-engine.git
cd didactic-engine
pip install -e .

# Run tests
pytest tests/ -v

# Run the CLI
didactic-engine --help
```

---

## Documentation Map

| Document | Purpose |
|----------|---------|
| [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | System design and module relationships |
| [02_KEY_FLOWS.md](02_KEY_FLOWS.md) | End-to-end pipeline flow |
| [03_DEBUGGING.md](03_DEBUGGING.md) | Troubleshooting and common issues |
| [04_CONTRIBUTING.md](04_CONTRIBUTING.md) | How to contribute |
| [05_GLOSSARY.md](05_GLOSSARY.md) | Domain terms and definitions |
| [06_API_REFERENCE.md](06_API_REFERENCE.md) | API reference for modules/classes/functions |
| [07_DESIGN_PATTERNS_AND_DECISIONS.md](07_DESIGN_PATTERNS_AND_DECISIONS.md) | Design patterns, anti-patterns, ADR-style decisions |
| [07_DESIGN_PATTERNS.md](07_DESIGN_PATTERNS.md) | Catalog of design patterns used |
| [08_ANTI_PATTERNS.md](08_ANTI_PATTERNS.md) | Risks, footguns, and mitigations |
| [adr/README.md](adr/README.md) | Architecture Decision Records (ADR index) |
| [knowledge_graph/README.md](knowledge_graph/README.md) | Generated knowledge graph artifacts + how to regenerate |
| [09_DEPENDENCY_MAPS.md](09_DEPENDENCY_MAPS.md) | Visual module dependency maps + coupling hotspots |

---

## Project Structure

```
didactic-engine/
├── src/
│   └── didactic_engine/        # Main package
│       ├── __init__.py         # Package exports
│       ├── config.py           # PipelineConfig dataclass
│       ├── pipeline.py         # AudioPipeline orchestrator
│       ├── cli.py              # Command-line interface
│       │
│       │ # Audio Processing
│       ├── ingestion.py        # WAV file loading
│       ├── preprocessing.py    # Audio normalization/trimming
│       ├── analysis.py         # Feature extraction (librosa)
│       ├── separation.py       # Stem separation (Demucs)
│       │
│       │ # MIDI Processing
│       ├── transcription.py    # Audio→MIDI (Basic Pitch)
│       ├── midi_parser.py      # MIDI parsing (pretty_midi)
│       ├── align.py            # Note-to-beat alignment
│       │
│       │ # Segmentation & Features
│       ├── segmentation.py     # Bar boundary computation
│       ├── bar_chunker.py      # Per-bar audio chunking
│       ├── features.py         # Bar-level feature extraction
│       │
│       │ # Export
│       ├── export_md.py        # Markdown report generation
│       ├── export_abc.py       # ABC notation export
│       │
│       │ # Optional/Utilities
│       ├── essentia_features.py # Essentia integration
│       ├── onnx_inference.py    # ONNX Runtime inference
│       └── utils_flatten.py    # Dictionary flattening
│
├── tests/
│   └── test_pipeline.py        # Unit tests
│
├── docs/                       # Developer documentation
│
├── examples/
│   └── example_usage.py        # Usage examples
│
├── pyproject.toml              # Package configuration
├── README.md                   # User documentation
└── AGENT_INSTRUCTIONS.md       # Maintenance guide
```

---

## Key Entry Points

### 1. CLI Entry Point

```bash
didactic-engine --wav song.wav --song-id my_song --out data/
```

**Code path:** `cli.py:main()` → `pipeline.py:run_all()` → `AudioPipeline.run()`

### 2. Python API Entry Point

```python
from didactic_engine import PipelineConfig, AudioPipeline

cfg = PipelineConfig(
    song_id="my_song",
    input_wav="song.wav",
    out_dir="data",
)
pipeline = AudioPipeline(cfg)
results = pipeline.run()
```

---

## Module Categories

### Public API (stable interface)
- `__init__.py` - Package exports
- `config.py` - Configuration dataclass
- `pipeline.py` - Main orchestrator
- `cli.py` - Command-line interface

### Core Audio Processing
- `ingestion.py` - Load/validate WAV files
- `preprocessing.py` - Normalize, trim, resample
- `analysis.py` - Extract librosa features
- `separation.py` - Demucs stem separation

### MIDI Processing
- `transcription.py` - Basic Pitch transcription
- `midi_parser.py` - Parse MIDI to DataFrame
- `align.py` - Align notes to beat grid

### Feature Extraction
- `segmentation.py` - Compute bar boundaries
- `bar_chunker.py` - Write per-bar chunks
- `features.py` - Extract bar-level features

### Export
- `export_md.py` - Markdown reports
- `export_abc.py` - ABC notation (music21)

### ML Inference
- `onnx_inference.py` - ONNX Runtime inference (TensorFlow-free alternative for ML models)

---

## Common Development Tasks

### Adding a New Feature Extractor

1. Add extraction logic to `features.py`
2. Update `pipeline.py` to call the new extractor
3. Add to Parquet output in datasets section
4. Write tests in `test_pipeline.py`

### Adding CLI Options

1. Add argument in `cli.py` parser
2. Add field to `PipelineConfig` if needed
3. Wire up in `PipelineConfig` creation
4. Update README with usage example

### Supporting a New Audio Format

1. Update `ingestion.py` to handle format
2. May need additional dependencies in `pyproject.toml`
3. Document in README

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=didactic_engine

# Run specific test
pytest tests/test_pipeline.py::TestWAVIngester -v
```

---

## See Also

- [Architecture Overview](01_ARCHITECTURE.md)
- [Pipeline Flows](02_KEY_FLOWS.md)
- [STYLE_GUIDE.md](../STYLE_GUIDE.md) - Documentation standards
