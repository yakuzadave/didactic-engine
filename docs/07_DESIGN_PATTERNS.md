# Design Patterns (didactic-engine)

This document lists design patterns used intentionally across the **didactic-engine** package, with pointers to where they appear.

See also:
- [01_ARCHITECTURE.md](01_ARCHITECTURE.md)
- [02_KEY_FLOWS.md](02_KEY_FLOWS.md)
- [06_API_REFERENCE.md](06_API_REFERENCE.md)
- [07_DESIGN_PATTERNS_AND_DECISIONS.md](07_DESIGN_PATTERNS_AND_DECISIONS.md) (index)

---

## 1) Orchestrator / Facade

**What:** A single object coordinates subsystems behind a simple interface.

**Where:** `didactic_engine.pipeline.AudioPipeline`

**How it shows up:**
- `AudioPipeline.run()` is the canonical “do the whole thing” entrypoint.
- The pipeline owns subsystem instances (`WAVIngester`, `AudioAnalyzer`, `StemSeparator` wrapper usage, etc.).

**Why it was chosen:** simplifies CLI and batch invocation.

---

## 2) Strategy via configuration flags

**What:** Behavior selection by a configuration object.

**Where:** `didactic_engine.config.PipelineConfig` and conditionals in `AudioPipeline`.

Examples:
- `use_pydub_preprocess`
- `use_essentia_features`
- `write_bar_chunks`
- `write_bar_chunk_wavs`
- `basic_pitch_backend`
- `demucs_device`

**Why it was chosen:** lets one pipeline support multiple “profiles” (fast runs, feature-heavy runs, CPU vs GPU).

---

## 3) Adapter / Wrapper around external tools

**What:** Small wrappers normalize an external interface.

**Where:**
- `didactic_engine.separation.StemSeparator` (Demucs CLI)
- `didactic_engine.transcription.BasicPitchTranscriber` (Basic Pitch CLI)

**Traits:**
- centralizes availability checks
- centralizes error messages and actionable install hints
- normalizes output conventions (e.g., canonical MIDI filenames)

---

## 4) Graceful degradation (capability-based behavior)

**What:** If optional dependencies are missing or fail, the pipeline degrades into a reduced feature set rather than crashing.

**Where:**
- `_separate_stems()` falls back to `{"full_mix": input_wav}`
- `_process_stems()` records transcription errors per stem and continues
- `essentia_features` returns `{"available": False, "error": ...}`

**Why it was chosen:** keeps the system usable in constrained environments and makes tests less brittle.

---

## 5) Late binding for optional dependencies

**What:** Optional imports are guarded to avoid import-time failures.

**Where:**
- `didactic_engine.essentia_features` imports Essentia inside the function
- `didactic_engine.onnx_inference` sets `ONNXRUNTIME_AVAILABLE` based on import

**Why it was chosen:** enables “install-light” workflows.

---

## 6) Deterministic output layout

**What:** Output directories are derived from `out_dir` + `song_id`.

**Where:** `PipelineConfig` properties (`stems_dir`, `midi_dir`, `datasets_dir`, etc.).

**Why it was chosen:** makes outputs predictable, scriptable, and easier to debug.

---

## 7) Boundary validation

**What:** Validate inputs at module boundaries.

**Where:**
- `WAVIngester.validate()` checks type, sample rate, NaN/Inf, etc.
- CLI checks missing input files before running pipeline

**Why it was chosen:** fail early and produce actionable error messages.

---

## 8) Structured step logging

**What:** Consistent step start/stop logging with durations.

**Where:** `_PipelineStep` context manager in `didactic_engine.pipeline`.

**Why it was chosen:** operational clarity when runs take minutes.
