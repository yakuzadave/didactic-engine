# ADR 0003: CLI-first integration for external ML tools

Date: 2026-01-02

## Context

Demucs and Basic Pitch are commonly installed and used as command-line tools, and their internal Python APIs change across versions.

## Decision

Invoke Demucs and Basic Pitch via subprocess calls to their CLIs and treat outputs as build artifacts.

## Alternatives

- Import and call Python APIs directly (tighter coupling)
- Vendor or pin tool internals in this repository

## Consequences

- ✅ Lower coupling to upstream implementation details
- ✅ Mirrors user expectations ("install CLI and run")
- ⚠️ Subprocess output discovery can be brittle
- ⚠️ Error handling must surface stderr/stdout for diagnosis

## Related

- `didactic_engine.separation.StemSeparator.separate`
- `didactic_engine.transcription.BasicPitchTranscriber.transcribe`
