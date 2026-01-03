# API Reference (didactic-engine)

This document is a **module-by-module API reference** for the current codebase.
It is intended as a practical map of *what you can import and call*.

If you’re new to the system, start with:
- [01_ARCHITECTURE.md](01_ARCHITECTURE.md)
- [02_KEY_FLOWS.md](02_KEY_FLOWS.md)

---

## Scope

This repository contains two related Python packages:

1. **`didactic_engine/`** (primary package)
   - Installed from the repository root (`pyproject.toml`)
   - CLI entrypoint: `didactic-engine`

2. **`music_etl/`** (secondary package)
   - A sibling project under `music_etl/` with its own `pyproject.toml`
   - Similar concepts and naming, but a separate implementation

This document focuses on **`didactic_engine`**, with a short section on **`music_etl`** at the end.

---

## Public package exports (`didactic_engine`)

The package exports a small “public surface” via `didactic_engine/__init__.py`:

- `didactic_engine.AudioPipeline`
- `didactic_engine.PipelineConfig`
- `didactic_engine.export_midi_markdown`
- `didactic_engine.export_full_report`
- `didactic_engine.export_abc`

Version:
- `didactic_engine.__version__`

---

## CLI (`didactic_engine.cli`)

Entry point:
- `didactic_engine.cli.main()`

Script entrypoint (from `pyproject.toml`):
- `didactic-engine = didactic_engine.cli:main`

### Key CLI arguments

Required:
- `--wav PATH [PATH ...]`: one or more WAVs (batch mode when multiple)
- `--song-id ID`: required for single-file mode

Common options:
- `--out DIR`: output root (default: `data`)
- `--sr INT`: analysis sample rate (default: `22050`)
- `--ts-num INT`, `--ts-den INT`: time signature
- `--use-essentia`: enable optional Essentia features
- `--no-preprocess`: skip pydub preprocessing
- `--no-bar-chunks`: skip bar chunking and bar feature extraction

Performance / acceleration options:
- `--demucs-device {cpu,cuda,cuda:0,...}`: run Demucs on GPU when available
- `--no-chunk-wavs`: compute bar features but do **not** write per-bar WAVs
- `--basic-pitch-backend {tf,onnx,tflite,coreml}`: choose Basic Pitch backend

Notes:
- Batch mode uses `AudioPipeline.process_batch(...)`.
- In single-file mode, the CLI builds `PipelineConfig` then calls `run_all(cfg)`.

---

## Pipeline orchestration (`didactic_engine.pipeline`)

### `AudioPipeline`

Primary orchestrator class.

Constructor:
- `AudioPipeline(cfg: PipelineConfig, logger: logging.Logger | None = None)`

Primary method:
- `AudioPipeline.run() -> dict[str, Any]`

Key step behavior in `run()`:
1. Copies input to the output tree
2. Ingests audio (mono float32, typically resampled to `cfg.analysis_sr`)
3. Optional preprocessing (pydub)
4. Analysis (librosa + optional Essentia)
5. Optional stem separation (Demucs; graceful fallback to `full_mix`)
6. Compute bar boundaries from beat times
7. Process each stem:
   - optional bar chunking + bar feature extraction
   - MIDI transcription (Basic Pitch)
   - align notes to beat/bar grid
8. Build pandas datasets (events/beats/bars/bar_features)
9. Write Parquet datasets
10. Export reports (Markdown + ABC)
11. Write summary JSON

Convenience functions:
- `run_all(cfg: PipelineConfig, logger: logging.Logger | None = None) -> dict[str, Any]`

Batch processing:
- `AudioPipeline.process_batch(input_files: list[Path], out_dir: Path, song_ids: list[str] | None = None, logger: logging.Logger | None = None, **kwargs) -> dict[str, Any]`

Side effects:
- Writes outputs under `cfg.out_dir` (see `PipelineConfig` paths)
- Runs external CLIs if available (`demucs`, `basic-pitch`)

---

## Configuration (`didactic_engine.config`)

### `PipelineConfig`

Frozen dataclass that controls pipeline behavior and output locations.

Important fields:
- `song_id: str`
- `input_wav: Path`
- `out_dir: Path`
- `demucs_model: str` (default: `htdemucs`)
- `demucs_device: str` (default: `cpu`; use `cuda` to accelerate)
- `analysis_sr: int` (default: `22050`)
- `hop_length: int`
- time signature: `time_signature_num`, `time_signature_den`
- preprocessing flags: `use_pydub_preprocess`, plus preprocess parameters
- Essentia: `use_essentia_features`
- bar chunking:
  - `write_bar_chunks: bool`
  - `write_bar_chunk_wavs: bool` (skip per-bar WAVs to reduce I/O)
- transcription:
  - `basic_pitch_backend: str` (e.g., `tf` or `onnx`)

Computed output paths:
- `cfg.stems_dir`, `cfg.preprocess_dir`, `cfg.chunks_dir`, `cfg.midi_dir`,
  `cfg.analysis_dir`, `cfg.reports_dir`, `cfg.datasets_dir`, `cfg.input_dir`

Directory creation:
- `cfg.create_directories()`

---

## Audio ingestion (`didactic_engine.ingestion`)

### `WAVIngester`

Loads audio files with soundfile, converts to mono float32, optionally resamples.

Methods:
- `load(path) -> tuple[np.ndarray, int]`
- `validate(audio, sample_rate) -> bool`
- `save(audio, sample_rate, output_path) -> None`

Contract:
- Internal pipeline convention is **mono 1D `np.float32`**.

---

## Preprocessing (`didactic_engine.preprocessing`)

### `AudioPreprocessor`

Responsible for optional audio cleanup (normalization, trimming silence, resampling) using pydub.

Entry point:
- `preprocess(audio: np.ndarray, sr: int, cfg: PipelineConfig) -> tuple[np.ndarray, int]`

---

## Analysis (`didactic_engine.analysis`)

### `AudioAnalyzer`

Computes tempo/beat tracking and core audio features.

Constructor:
- `AudioAnalyzer(use_essentia: bool = False)`

Common methods:
- `analyze(audio: np.ndarray, sr: int) -> dict[str, Any]`
- `extract_beat_times(audio: np.ndarray, sr: int) -> np.ndarray`

---

## Stem separation (`didactic_engine.separation`)

### `StemSeparator`

Wraps Demucs CLI.

Constructor:
- `StemSeparator(model: str = "htdemucs", device: str = "cpu")`

Methods:
- `separate(audio_path: str | Path, out_dir: str | Path) -> dict[str, Path]`

Notes:
- Uses `rglob("*.wav")` to find Demucs outputs.
- Raises `RuntimeError` with installation instructions if Demucs is missing.

---

## Transcription (`didactic_engine.transcription`)

### `BasicPitchTranscriber`

Wraps the `basic-pitch` CLI.

Constructor:
- `BasicPitchTranscriber(model_serialization: str = "tf")`

Methods:
- `transcribe(stem_wav: str | Path, out_dir: str | Path) -> Path`

Notes:
- The transcriber canonicalizes the newest `.mid` file to `{stem_name}.mid`.
- `model_serialization` is forwarded as `--model-serialization`.

---

## MIDI parsing (`didactic_engine.midi_parser`)

### `MIDIParser`

Parses MIDI into structured Python + pandas forms.

Common methods:
- `parse(midi_path: Path) -> dict[str, Any]` (includes `notes_df`)
- `create_from_notes(...) -> pretty_midi.PrettyMIDI`

---

## Beat/bar alignment (`didactic_engine.align`)

Key function:
- `align_notes_to_beats(notes_df: pd.DataFrame, beat_times: list[float], tempo_bpm: float, ts_num: int, ts_den: int) -> pd.DataFrame`

Adds alignment columns such as:
- `beat_index`, `bar_index`, `beat_in_bar`

---

## Segmentation (`didactic_engine.segmentation`)

Core functions:
- `segment_beats_into_bars(beat_times, tempo_bpm, ts_num, ts_den, audio_duration) -> list[tuple[int,float,float]]`
- `segment_audio_by_bars(audio_path, boundaries, out_dir) -> list[dict[str, Any]]`

Class-based helper:
- `StemSegmenter`

---

## Bar chunking helper (`didactic_engine.bar_chunker`)

High-level wrapper around segmentation:
- `compute_bar_boundaries(...)`
- `write_bar_chunks(...)`
- `write_bar_chunks_with_features(...)`

Note:
- The pipeline’s bar-chunking path is implemented in `AudioPipeline._process_stems()`.

---

## Feature extraction (`didactic_engine.features`)

### `FeatureExtractor`

Produces pandas datasets and per-bar audio features.

Common methods:
- `extract_bar_features_from_audio(audio: np.ndarray, sample_rate: int) -> dict[str, Any]`
- `extract_bar_features_from_file(audio_path: str | Path, sample_rate: int = 22050) -> dict[str, Any]`
- `extract_events(all_notes_df: pd.DataFrame) -> pd.DataFrame`
- `extract_beats(beat_times, tempo_bpm, stem, song_id) -> pd.DataFrame`
- `extract_bars(all_notes_df: pd.DataFrame, song_id: str) -> pd.DataFrame`

---

## Essentia integration (`didactic_engine.essentia_features`)

Function:
- `extract_essentia_features(wav_path: str | Path, sr: int = 44100) -> dict`

Contract:
- Returns `{"available": False, "error": ...}` when Essentia isn’t installed.

---

## ONNX runtime utilities (`didactic_engine.onnx_inference`)

Functions:
- `is_onnxruntime_available() -> bool`
- `get_onnxruntime_version() -> str | None`
- `get_available_providers() -> list[str]`

Class:
- `ONNXInferenceSession(model_path, providers=None)`

---

## Exports (`didactic_engine.export_md`, `didactic_engine.export_abc`)

Markdown export:
- `export_midi_markdown(...) -> None`
- `export_full_report(results: dict[str, Any], output_path: str) -> None`

ABC export:
- `export_abc(midi_path: str | Path, output_path: str | Path) -> None`

---

## Utilities (`didactic_engine.utils_flatten`)

Function:
- `flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict`

---

## Package: `music_etl` (secondary)

`music_etl` is a separate package located under `music_etl/` with its own pipeline.
It has similarly named concepts (config, pipeline, separation, transcription), but it is not the same code path as `didactic_engine`.

Entry points to start with:
- `music_etl.pipeline` (or `music_etl/scripts/run_pipeline.py`)
- `music_etl.config.PipelineConfig`

If you want a full `music_etl` API reference too, we can either:
- extend this document with a dedicated `music_etl` module-by-module section, or
- generate a separate `music_etl/docs/API_REFERENCE.md`.
