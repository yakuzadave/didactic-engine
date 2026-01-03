# ADR 0004: Bar-chunking correctness with performance tunability

Date: 2026-01-02

## Context

Bar chunking can generate hundreds/thousands of small WAV files. This can be slow (especially on Windows-mounted filesystems under WSL).

## Decision

Support bar-level feature extraction without requiring per-bar WAV persistence.

Mechanisms:
- `PipelineConfig.write_bar_chunks`: compute per-bar features
- `PipelineConfig.write_bar_chunk_wavs`: whether to write per-bar WAV files
- CLI flags: `--no-bar-chunks`, `--no-chunk-wavs`

## Alternatives

- Always write chunk WAV files
- Never write chunk WAV files (only features)

## Consequences

- ✅ Enables fast feature extraction in I/O-constrained environments
- ✅ Still supports chunk WAV persistence when needed for inspection/training
- ⚠️ Some workflows expect chunk WAVs to exist; they must check configuration

## Related

- `didactic_engine.pipeline.AudioPipeline._process_stems`
- `didactic_engine.features.FeatureExtractor.extract_bar_features_from_audio`
- `didactic_engine.segmentation.segment_beats_into_bars`
