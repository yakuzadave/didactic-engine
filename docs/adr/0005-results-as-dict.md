# ADR 0005: Pipeline results returned as a dictionary

Date: 2026-01-02

## Context

The pipeline produces heterogeneous outputs (files, DataFrames, summary metadata). The CLI and notebook usage benefit from a simple return type.

## Decision

Return results from `AudioPipeline.run()` as `dict[str, Any]`.

## Alternatives

- Typed results dataclass with strict fields
- Multiple return objects (e.g., `PipelineArtifacts`, `Datasets`, etc.)

## Consequences

- ✅ Simple integration surface for CLI and interactive use
- ✅ Easy to add new keys without breaking older runs
- ⚠️ Weaker typing; keys can drift over time

## Related

- `didactic_engine.pipeline.AudioPipeline.run`
- `docs/06_API_REFERENCE.md`
