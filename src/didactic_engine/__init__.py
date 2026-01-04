"""
Didactic Engine - Audio Processing Pipeline

A comprehensive audio processing pipeline for stem separation, analysis,
MIDI transcription, and feature extraction.

This package provides tools for:
- WAV file ingestion and validation
- Stem separation using Demucs
- Audio preprocessing with pydub (normalization, silence trimming)
- Feature extraction using librosa and optional Essentia
- MIDI transcription using Basic Pitch
- Beat and bar alignment
- Per-bar audio segmentation
- Dataset generation (Parquet format)
- Human-readable report generation (Markdown, ABC notation)
- Resilience patterns (retry, circuit breaker, health checks)
"""

__version__ = "0.1.0"

from didactic_engine.pipeline import AudioPipeline
from didactic_engine.config import PipelineConfig
from didactic_engine.export_md import export_midi_markdown, export_full_report
from didactic_engine.export_abc import export_abc
from didactic_engine.resilience import (
    retry_with_backoff,
    CircuitBreaker,
    run_all_health_checks,
    print_health_report,
    resource_cleanup,
    ProcessingCheckpoint,
)

__all__ = [
    "AudioPipeline",
    "PipelineConfig",
    "export_midi_markdown",
    "export_full_report",
    "export_abc",
    # Resilience utilities
    "retry_with_backoff",
    "CircuitBreaker",
    "run_all_health_checks",
    "print_health_report",
    "resource_cleanup",
    "ProcessingCheckpoint",
]
