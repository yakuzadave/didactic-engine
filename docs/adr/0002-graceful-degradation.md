# ADR 0002: Graceful degradation for optional tools (Demucs, Basic Pitch, Essentia)

Date: 2026-01-02

## Context

Some core capabilities rely on optional external tools or heavyweight dependencies:
- Demucs (stem separation)
- Basic Pitch (audio → MIDI transcription)
- Essentia (advanced audio features)

These may not be installed in all environments.

## Decision

Design the pipeline to continue running when optional tools are unavailable:
- If Demucs fails/unavailable: run with a single `full_mix` stem.
- If Basic Pitch fails/unavailable: skip transcription for that stem.
- If Essentia is not installed: return `available=False` and proceed.

## Alternatives

- Fail fast whenever any optional tool is missing
- Hard-require optional tooling in installation

## Consequences

- ✅ Pipeline can still produce partial outputs (analysis, reports, some datasets)
- ✅ Tests can run in environments without optional tooling
- ⚠️ Downstream consumers must handle missing stems/MIDI/features gracefully

## Related

- `didactic_engine.pipeline.AudioPipeline._separate_stems`
- `didactic_engine.pipeline.AudioPipeline._process_stems`
- `didactic_engine.separation.StemSeparator`
- `didactic_engine.transcription.BasicPitchTranscriber`
- `didactic_engine.essentia_features.extract_essentia_features`
