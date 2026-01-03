# ADR 0001: Immutable configuration (`PipelineConfig` is frozen)

Date: 2026-01-02

## Context

The pipeline has many stages (ingestion → analysis → separation → transcription → exports) that depend on consistent parameters and deterministic output paths.

## Decision

Use a frozen dataclass (`didactic_engine.config.PipelineConfig`) and derive output directories from `out_dir` and `song_id` via computed properties.

## Alternatives

- Mutable config object modified over time
- Dict-based configuration passed between steps

## Consequences

- ✅ Reduces "action at a distance" and improves reproducibility
- ✅ Output layout is deterministic and easy to script against
- ⚠️ Configuration changes require creating a new config instance (intentional)

## Related

- `didactic_engine.config.PipelineConfig`
- `didactic_engine.pipeline.AudioPipeline`
