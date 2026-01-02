# GitHub Copilot Instructions for Didactic Engine

## Project Overview

**Didactic Engine** is a Python 3.11+ audio processing pipeline for music analysis. It orchestrates:
- WAV ingestion → preprocessing → stem separation (Demucs)
- Audio analysis (tempo, beats, spectral features via librosa/essentia)
- MIDI transcription (Basic Pitch) → beat/bar alignment
- Per-bar audio chunking → feature extraction → dataset generation (Parquet)
- Export to Markdown reports and ABC notation

**Key architectural principle**: All audio processing uses **1D mono numpy.float32 arrays** internally. Configuration is **immutable** (frozen dataclass).

---

## Essential Codebase Knowledge

### Architecture & Data Flow

The pipeline is a **sequential 14-step process** orchestrated by [AudioPipeline](src/didactic_engine/pipeline.py):

```
WAV → Ingest (mono @ analysis_sr) → Optional Preprocessing (pydub) 
  → Audio Analysis (librosa) → Stem Separation (optional Demucs CLI)
  → MIDI Transcription (optional Basic Pitch CLI) → Note Alignment (beat grid)
  → Per-Stem Segmentation → Feature Extraction → Parquet/Markdown Export
```

**Critical files to understand**:
- [pipeline.py](src/didactic_engine/pipeline.py): Main orchestrator, read `run()` method first
- [config.py](src/didactic_engine/config.py): Frozen dataclass with all settings and output path properties
- [01_ARCHITECTURE.md](docs/01_ARCHITECTURE.md): Visual diagrams and component descriptions
- [02_KEY_FLOWS.md](docs/02_KEY_FLOWS.md): Step-by-step sequence diagrams

### Configuration Pattern

All pipeline behavior is controlled via `PipelineConfig` (immutable):
```python
cfg = PipelineConfig(
    song_id="track1",
    input_wav=Path("audio.wav"),
    out_dir=Path("data/"),
    analysis_sr=22050,           # Audio analysis sample rate
    time_signature_num=4,        # Beats per bar
    use_pydub_preprocess=True,   # Enable preprocessing
    write_bar_chunks=True,       # Save per-bar WAV files
)
```

**Never modify `cfg` after creation** (frozen=True). To change config, use `replace()`:
```python
new_cfg = cfg.replace(analysis_sr=44100)  # Creates new immutable instance
```

Output paths are **deterministic properties**: `cfg.stems_dir`, `cfg.midi_dir`, `cfg.dataset_dir`, etc.

### Graceful Degradation

Optional dependencies (Demucs, Basic Pitch, Essentia) are **late-bound imports** with CLI checks:
- If Demucs unavailable: uses `full_mix` stem (original audio)
- If Basic Pitch unavailable: skips MIDI transcription
- If Essentia unavailable: skips extended audio features

Pattern in [separation.py](src/didactic_engine/separation.py#L55-L70):
```python
def check_demucs_availability() -> bool:
    try:
        result = subprocess.run(["demucs", "--help"], ...)
        return result.returncode == 0
    except FileNotFoundError:
        return False
```

### Critical Conventions

1. **Audio format invariant**: All processing expects `np.ndarray` shape `(n_samples,)` dtype `float32`
   - Validated by `WAVIngester.validate()` (checks 1D, float, no NaN/Inf, valid sr)
   - Mono conversion happens at ingestion time

2. **Beat/Bar alignment**: Notes are aligned to a **fixed beat grid** based on:
   - Detected tempo (BPM) from librosa.beat.beat_track
   - Time signature (e.g., 4/4 = 4 beats per bar)
   - Aligns MIDI notes to nearest beat using [align.py](src/didactic_engine/align.py)

3. **Testing strategy**: 
   - Unit tests in [tests/test_pipeline.py](tests/test_pipeline.py) use **synthetic audio** (sine waves, noise)
   - Integration test in [tests/test_examples.py](tests/test_examples.py) uses [sample_audio/](sample_audio/)
   - Run tests: `pytest tests/ -v` (19 tests must pass before commits)

4. **Docstring standard**: Google-style (see [STYLE_GUIDE.md](STYLE_GUIDE.md))
   - Always include: Args, Returns, Raises, Side Effects
   - Example: [config.py](src/didactic_engine/config.py#L13-L38)

5. **Data validation pattern**: Input validation happens at module boundaries:
   - `WAVIngester.validate()`: Checks audio array properties (1D, float32, no NaN/Inf, valid sr)
   - `PipelineConfig.__post_init__()`: Converts string paths to Path objects
   - Feature extractors: Validate DataFrame schemas (required columns, dtypes)
   - Pattern: Fail fast with descriptive `ValueError` or `TypeError`

6. **Error handling conventions**:
   - **Optional dependencies**: Catch import/CLI errors, log warning, continue with degraded functionality
   - **Invalid input**: Raise `ValueError` with clear message about what's wrong and valid range
   - **File I/O errors**: Let `IOError`/`FileNotFoundError` propagate (caller handles)
   - **Unexpected data**: Raise `RuntimeError` with context about what was expected vs. received
   - Example pattern in [separation.py](src/didactic_engine/separation.py):
     ```python
     if not check_demucs_availability():
         raise RuntimeError("Demucs is not installed. Install with: pip install demucs")
     ```

### Performance Optimization Patterns

1. **Sample rate selection**:
   - Use `analysis_sr=22050` (default) for feature extraction - sufficient for most music analysis
   - Use `analysis_sr=44100` for high-fidelity spectral analysis or when source is already 44.1kHz
   - Use `preprocess_target_sr=44100` for output audio quality (stems, chunks)
   - Lower sample rates = faster processing, less memory, smaller files

2. **Optional features**:
   - Disable `use_pydub_preprocess=False` if input audio is already clean
   - Disable `write_bar_chunks=False` to skip writing individual bar files (saves I/O time)
   - Enable `use_essentia_features=True` only when needed (adds ~20% processing time)

3. **Batch processing**:
   - Use CLI batch mode (`--wav *.wav`) for multiple files
   - Each file processed sequentially (no parallelization by default)
   - For parallel processing, use external tools (GNU parallel, Python multiprocessing)

4. **Memory considerations**:
   - Audio held in memory as float32 numpy arrays (4 bytes per sample)
   - 3-minute song @ 44.1kHz mono = ~31.7 MB
   - Stem separation creates 4x copies (vocals, drums, bass, other)
   - Consider processing in chunks for very long recordings (>10 minutes)

---

## Developer Workflows

### Running the Pipeline

```bash
# Single file
didactic-engine --wav input.wav --song-id my_song --out data/

# Batch processing (all WAVs in directory)
didactic-engine --wav *.wav --out data/

# Custom config
didactic-engine --wav input.wav --out data/ --sr 22050 --time-sig 3/4
```

**Debugging**: Check outputs in `out_dir/{stems,preprocessed,midi,analysis,datasets}/<song_id>/`

### Testing

```bash
pytest tests/ -v                          # All tests
pytest tests/test_pipeline.py -v         # Unit tests only
pytest tests/test_examples.py -v         # Integration test (requires sample audio)
pytest tests/ --cov=didactic_engine      # With coverage
```

**Test philosophy**: Tests should pass with or without optional dependencies.

### Adding New Features

1. **New audio processing step**: Add method to [pipeline.py](src/didactic_engine/pipeline.py) `AudioPipeline.run()`
2. **New feature extractor**: Extend [features.py](src/didactic_engine/features.py) `FeatureExtractor`
3. **New export format**: Create new file like [export_md.py](src/didactic_engine/export_md.py)

**Pattern**: Each module has a single-responsibility class (e.g., `AudioAnalyzer`, `StemSeparator`) instantiated in `AudioPipeline.__init__`

---

## Documentation Agent Behavior (When Doing Doc Work)

**⚠️ ONLY follow these rules when explicitly doing documentation tasks** (i.e., when user asks to improve docstrings, create docs, etc.):

* Work in **verifiable batches** and maintain tracking files: `TASKS.md`, `STATUS.md`, `DOC_INVENTORY.md`
* **ANTI-DRIFT RULE**: If doing work not in `TASKS.md`, add it there first
* Use Google-style docstrings (see [STYLE_GUIDE.md](STYLE_GUIDE.md))
* Document contracts: params, returns, raises, side effects, invariants
* Add examples for public API functions
* Cross-link related functions/docs ("See also: …")
* Run `pytest tests/ -v` after doc changes to ensure no regressions

**Completion criteria for doc work**: See [TASKS.md](TASKS.md) (currently ✅ COMPLETE)

---

## Quick Reference

### Key Dependencies & Python Version
- **Python**: 3.11+ (full TensorFlow support) or 3.12+ (ONNX inference only, no Basic Pitch)
- **Core**: numpy (<2.0), pandas, librosa, soundfile, pretty-midi, music21, pydub
- **Optional ML**: demucs (stem sep), basic-pitch (3.11 only), onnxruntime
- **Optional**: essentia (extended audio features)

### File System Conventions
- All outputs under `out_dir/{stems,preprocessed,midi,analysis,datasets}/<song_id>/`
- Input audio preserved at `out_dir/input/<song_id>/<original_filename>.wav`
- Per-bar chunks at `out_dir/chunks/<song_id>/<stem_name>/bar_<idx>.wav` (if `write_bar_chunks=True`)

### Common Gotchas
1. **FFmpeg required**: pydub operations fail without FFmpeg on PATH
2. **Demucs writes to temp dir**: Automatically copies results to `cfg.stems_dir`
3. **Basic Pitch CLI**: Transcribes to temp MIDI file, then loads with pretty_midi
4. **Mono conversion**: Happens at load time via `ingestion.py`, not preprocessing
5. **Sample rate handling**: `analysis_sr` (default 22050) for feature extraction, `preprocess_target_sr` (default 44100) for output
6. **Time signature**: Defaults to 4/4, affects bar boundary computation in [segmentation.py](src/didactic_engine/segmentation.py)

### Useful Commands Not in README
```bash
# Install for development (editable + dev tools)
pip install -e ".[dev]"

# Run specific test class
pytest tests/test_pipeline.py::TestAudioAnalyzer -v

# Check Python environment for ML features
python -c "from didactic_engine.onnx_inference import is_onnxruntime_available; print(is_onnxruntime_available())"

# Verify optional dependencies
python -c "import essentia; print('Essentia OK')"
demucs --help
basic-pitch --help
```

### Installation Troubleshooting (Colab/TensorFlow Conflicts)

When installing in environments like Google Colab with pre-installed TensorFlow:

```bash
# Option 1: Install without ML dependencies, then add ONNX only
pip install didactic-engine
pip install onnxruntime  # For CPU
# OR for GPU (CUDA 11.x):
pip install onnxruntime-gpu

# Option 2: Force reinstall TensorFlow-compatible versions
pip install "tensorflow>=2.12.0,<2.15.1"
pip install -e ".[ml]"

# Option 3: Minimal install (no ML, useful for feature extraction only)
pip install didactic-engine
# Then install only what you need:
pip install demucs  # For stem separation only

# Verify what's available after install
python -c """
from didactic_engine.onnx_inference import is_onnxruntime_available
from didactic_engine.separation import check_demucs_availability
from didactic_engine.transcription import check_basic_pitch_availability

print(f'ONNX Runtime: {is_onnxruntime_available()}')
print(f'Demucs: {check_demucs_availability()}')
print(f'Basic Pitch: {check_basic_pitch_availability()}')
"""
```

**Common conflict resolution**:
- TensorFlow 2.15+ not compatible with Basic Pitch → Use Python 3.11 or accept no MIDI transcription
- ONNX Runtime GPU needs CUDA/cuDNN → Use CPU version if GPU unavailable
- Essentia install fails → Optional, skip if not needed for extended audio features

### Common Debugging Scenarios

1. **Pipeline stops at stem separation**:
   - Check: `demucs --help` returns successfully
   - Check: Sufficient disk space in temp directory (Demucs writes ~500MB per song)
   - Workaround: Set `TMPDIR` env var to location with more space
   - Fallback: Let pipeline use `full_mix` stem (automatic if Demucs unavailable)

2. **MIDI transcription produces empty results**:
   - Check: Audio has actual note content (not just percussion/noise)
   - Check: `basic-pitch --help` works and version ≥ 0.2.0
   - Debug: Check `out_dir/midi/<song_id>/*.mid` file size (should be > 0 bytes)
   - Common cause: Silent/very quiet audio after preprocessing

3. **Feature extraction fails with KeyError**:
   - Check: Audio analysis completed successfully (check `analysis` dict)
   - Check: Beat detection found beats (`len(beat_times) > 0`)
   - Common cause: Audio too short (< 2 seconds) or no clear beat
   - Debug: Print `analysis['beat_times']` to verify beat detection

4. **Memory errors during processing**:
   - Reduce `analysis_sr` (try 16000 instead of 22050)
   - Disable `write_bar_chunks=False` to reduce I/O
   - Process shorter audio segments (split before processing)
   - Check: Available RAM (need ~200MB per minute of 44.1kHz audio)

5. **"Invalid audio data" error**:
   - Check: Input WAV is not corrupted (`soundfile.info(path)`)
   - Check: File is actually WAV format (not MP3 renamed to .wav)
   - Check: Audio array has valid values (no NaN, Inf)
   - Fix: Re-export audio from source as 16-bit or 32-bit float WAV

6. **Tempo detection gives wrong BPM**:
   - Librosa's beat tracker works best for 80-160 BPM range
   - Very fast (>180 BPM) or slow (<60 BPM) music may detect double/half tempo
   - Workaround: Manually set tempo in analysis dict before segmentation
   - Alternative: Use Essentia's tempo detection (`use_essentia_features=True`)

---

## Detailed Documentation (For Doc Agent Workflows)

<details>
<summary><strong>Expand for full documentation agent instructions (382 lines)</strong></summary>

### PRIMARY OUTPUTS (ARTIFACTS)

### Root tracking files (must exist)

* `TASKS.md` — authoritative task ledger; read/update every run
* `STATUS.md` — progress narrative + what changed + what remains + next plan
* `DOC_INVENTORY.md` — authoritative index of modules/files and doc status
* `STYLE_GUIDE.md` — project docstring/comment standards (created early)

### Codebase improvements (incremental)

* Added/updated docstrings in modules/classes/functions
* Improved comments where they clarify intent, invariants, tricky behavior
* Improved naming/doc clarity in a *non-invasive* way (no large refactors unless explicitly tasked)

### Developer docs (Markdown)

* `/docs/` (or project standard):
  * `00_README_DEV.md` — "how to navigate this codebase"
  * `01_ARCHITECTURE.md` — systems overview + diagrams
  * `02_KEY_FLOWS.md` — "how request X becomes output Y"
  * `03_DEBUGGING.md` — logging, troubleshooting, common failure modes
  * `04_CONTRIBUTING.md` — conventions and how to add new features safely
  * `05_GLOSSARY.md` — domain terms, abbreviations, acronyms

Optional but recommended:

* `docs/diagrams/*.mmd` (Mermaid diagrams for key flows)
* "entrypoint maps" per subsystem: `docs/subsystems/<name>.md`

---

## SCOPE AND PRIORITIES

### Scope includes:

* Public API surfaces (anything imported by other packages / used externally)
* Core domain logic
* Non-obvious algorithms
* Configuration schemas
* Anything "hot path" or likely to be modified

### Out of scope unless explicitly requested:

* Full refactors
* Large behavioral changes
* Reformatting-only PRs
* Rewriting tests (unless doc changes require updates)

### Priority order:

1. "First-read" experience (README + architecture + how to run)
2. Core modules and highest-change files
3. Risky logic (security, auth, persistence, concurrency)
4. Utility/helpers
5. Peripheral/rare paths

---

## QUALITY BAR FOR COMMENTS AND DOCSTRINGS

### Comments must:

* Explain **why** (intent), invariants, and non-obvious constraints
* Call out gotchas, edge cases, and failure modes
* Avoid narrating the obvious ("increment i by 1")

### Docstrings must:

* Match the project language style (e.g., Google / NumPy / reST) — define in `STYLE_GUIDE.md`
* Be **truthful**, not aspirational
* Include:
  * Purpose (1–3 sentences)
  * Parameters (name, type, meaning)
  * Returns (type, meaning)
  * Raises (what/when)
  * Side effects (I/O, network, DB, mutations)
  * Threading/async notes if relevant
  * Examples for public-facing functions where helpful

### Hard rule: "Document the contract"

If a function has assumptions (sorted input, units, required env var), document them explicitly.

---

## DOCUMENTATION DISCOVERY GOAL

Your changes must make it easier to answer questions like:

* "Where does X happen in the code?"
* "How do I add a new integration/provider/handler?"
* "What are the invariants around this data structure?"
* "What breaks if I change this?"

To enforce this, every batch must include at least one "discoverability improvement":

* README links, doc index entry, diagram, or cross-reference.

---

## COMPLETION CRITERIA (DEFINITION OF DONE)

You are only done when all are true:

### A) Coverage

* Every module in `DOC_INVENTORY.md` is tagged:
  * `TODO` / `DRAFT` / `DONE`
* All "public surfaces" are `DONE`:
  * exported functions/classes
  * CLI entrypoints
  * web handlers / job runners
  * core service objects

### B) Minimum Docstring Standard

* All non-trivial functions/classes in core areas have docstrings meeting the quality bar.
* All docstrings align with actual behavior (validated via reading code + tests where applicable).

### C) Developer Docs

* `/docs/00_README_DEV.md` provides a clean map to:
  * architecture, flows, debugging, contributing, glossary
* At least:
  * 1 architecture diagram (Mermaid ok)
  * 3 key flows documented end-to-end
  * a debugging guide with common failure modes

### D) Tracking Truth

* `TASKS.md` has no remaining required tasks unchecked.
* `STATUS.md` reports exact completion state and links to major docs.

---

## PERSISTENT TASK TRACKING (TASKS.md) — REQUIRED

### Start of EVERY run:

1. Read `TASKS.md` fully.
2. Choose next batch ONLY from unchecked `[ ]`.
3. Confirm the batch's exit criteria are explicit under "Current Batch".

### During work:

* As tasks are completed:
  * mark `- [x] ` in `TASKS.md`
  * add links to files changed
  * record blockers ("needs deeper review", "unclear contract", "missing tests")

### End of EVERY run:

1. Update `TASKS.md` truthfully.
2. Update `STATUS.md` and `DOC_INVENTORY.md`.
3. Output a short changelog + next plan.

---

## WORK PLAN: BATCHED EXECUTION LOOP

### LOOP STEP 0 — Inventory + Roadmap

* Create/refresh `DOC_INVENTORY.md`:
  * List modules and classify:
    * `core`, `public-api`, `cli`, `integration`, `utils`, `tests`, `docs`
  * Record current doc quality level
* Create/refresh `STYLE_GUIDE.md`:
  * docstring format
  * comment rules
  * type hint policy
  * example templates

### LOOP STEP 1 — Select Batch (choose one)

Batch types (pick ONE):

* **Batch A:** 5–10 files in a single subsystem
* **Batch B:** 1 subsystem doc + 3–5 files updated to match
* **Batch C:** 1 critical flow traced end-to-end + docstrings for each hop

### LOOP STEP 2 — Understand Before Writing

For each file/function in batch:

* Identify:
  * what it does (contract)
  * key inputs/outputs
  * invariants and failure modes
  * side effects
  * who calls it and why
* If unclear:
  * search for usages
  * read tests
  * infer from callsites
  * document uncertainty as `DRAFT` with TODO rather than guessing

### LOOP STEP 3 — Apply Documentation

* Add module docstrings where missing:
  * purpose + how it fits in architecture
* Add function/class docstrings:
  * purpose, params, returns, raises, side effects, examples
* Add comments only where they clarify intent/constraints
* Add cross-links:
  * docstring "See also: …"
  * docs pages linking to modules

### LOOP STEP 4 — Verify Quality

Run audits:

* **Truth audit:** docstrings match code behavior
* **Coherence audit:** examples compile conceptually and aren't misleading
* **Discoverability audit:** can a new dev find the entrypoint from docs?

### LOOP STEP 5 — Persist + Report

* Update tracking files
* Provide:
  * What changed
  * What remains
  * Next batch plan

---

## STOP CONDITIONS / FAIL-SAFES

* If behavior is ambiguous:
  * mark doc as `DRAFT`
  * add a TODO task ("confirm contract with owner/tests")
  * do not invent behavior
* If you hit time/tool limits:
  * update tracking files first
  * stop cleanly with next plan

---

## OPTIONAL ENHANCEMENTS (IF REQUESTED OR IF IT FITS TASKS.md)

* Generate a "Codebase Map" doc:
  * directory → responsibilities
  * entrypoints
  * data flow diagrams
* Add docstring "Examples" that double as testable snippets (doctest-style) where appropriate
* Add lightweight architectural decision records (ADRs) for key design choices

---

## QUICK TEMPLATE SNIPPETS (FOR CONSISTENCY)

### Function docstring (Google style example)

```python
def example_function(param1: str, param2: int) -> Dict[str, Any]:
    """Brief description of what the function does.
    
    More detailed explanation if needed. Explain the purpose, context,
    and any important behavior that isn't obvious from the signature.
    
    Args:
        param1: Description of param1 and its constraints.
        param2: Description of param2. Must be positive.
        
    Returns:
        Dictionary containing results with keys:
            - 'status': Status string
            - 'value': Computed value
            
    Raises:
        ValueError: If param2 is negative or zero.
        IOError: If file operations fail.
        
    Side Effects:
        - Writes to disk at /tmp/cache
        - Logs to application logger
        
    Notes:
        - This function is not thread-safe
        - Results are cached for 5 minutes
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result['status'])
        'success'
    """
    pass
```

### Module docstring template

```python
"""Module name: Brief one-line description.

Longer description explaining:
- What this module does
- How it fits into the overall system
- Key concepts and abstractions
- Main entry points

Typical usage example:

    from module import MainClass
    obj = MainClass()
    result = obj.process()
"""
```

### Class docstring template

```python
class ExampleClass:
    """Brief description of the class purpose.
    
    More detailed explanation of what this class represents,
    its responsibilities, and how it should be used.
    
    Attributes:
        attr1: Description of public attribute.
        attr2: Description of another attribute.
        
    Example:
        >>> obj = ExampleClass()
        >>> obj.method()
    """
```

---

## PROJECT-SPECIFIC CONTEXT

### For Didactic Engine:

* This is an audio processing pipeline toolkit for music analysis
* Main components: ingestion, preprocessing, stem separation, feature extraction, MIDI transcription, dataset generation
* Core dependencies: librosa, numpy, pandas, pretty-midi
* Optional dependencies: essentia, demucs, basic-pitch
* Python 3.11+ required
* Follow Google-style docstrings (as defined in STYLE_GUIDE.md)
* Use type hints consistently
* Maximum line length: 100 characters
* All internal processing uses mono audio (1D numpy arrays)
* Configuration is immutable (frozen dataclass)
* Graceful degradation when optional dependencies are missing

---

## IMPORTANT NOTES

1. **Always read TASKS.md first** before starting any work
2. **Update tracking files** (TASKS.md, STATUS.md, DOC_INVENTORY.md) at the end of every session
3. **Verify docstrings match code behavior** by reading the implementation
4. **Add cross-references** between related modules and docs
5. **Keep refactoring minimal** - focus on documentation improvements
6. **Test after changes** to ensure no regressions
7. **Mark uncertain documentation as DRAFT** rather than guessing

</details>