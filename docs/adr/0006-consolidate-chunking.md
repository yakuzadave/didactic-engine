# ADR 0006: Consolidate bar chunking into a single implementation

Date: 2026-01-02

## Context

The repository currently exposes more than one path for “bar chunking”:

- `didactic_engine.bar_chunker` provides helper functions that are largely file/pydub oriented.
- `didactic_engine.pipeline.AudioPipeline._process_stems()` performs pipeline-integrated chunking and per-bar feature extraction.

Both approaches address the same domain concept (per-bar slicing and metadata), but they differ in implementation details and performance characteristics.

This duplication creates a maintenance risk: improvements (performance, correctness, metadata schema) can be applied to one path but not the other.

## Decision

Consolidate bar chunking behind a single implementation that:

- Supports **in-memory slicing** of already-loaded audio (fast path)
- Optionally persists chunk WAVs to disk (for inspection/training)
- Exposes a consistent metadata schema (bar_index/start_s/end_s/duration_s/chunk_path)
- Provides a hook or helper for feature extraction (without forcing per-chunk decode/resample)

In practice, this means:

- The pipeline should call the consolidated implementation.
- The `bar_chunker` module should either:
  - become the consolidated implementation, or
  - become a thin facade around it.

## Alternatives

1. Keep both implementations
   - Pro: no refactor cost
   - Con: divergence over time is likely

2. Delete `bar_chunker` and keep logic only in the pipeline
   - Pro: fewer modules
   - Con: harder reuse outside pipeline; makes testing chunking harder

3. Keep `bar_chunker` as file-based and pipeline as in-memory (status quo)
   - Pro: both use cases covered
   - Con: maintenance duplication continues

## Consequences

- ✅ Reduced duplication and fewer “fix it twice” issues
- ✅ One canonical place to tune performance (especially important on WSL and network filesystems)
- ✅ Clearer API for chunking semantics and metadata
- ⚠️ Requires a refactor pass to unify behavior and update call sites
- ⚠️ Tests may need to be expanded to cover both persistence and no-persistence paths

## Related

- `didactic_engine.bar_chunker`
- `didactic_engine.segmentation.segment_beats_into_bars`
- `didactic_engine.pipeline.AudioPipeline._process_stems`
- `docs/08_ANTI_PATTERNS.md` (duplication of chunking logic)
