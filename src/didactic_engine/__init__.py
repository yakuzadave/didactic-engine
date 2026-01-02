"""
Didactic Engine - Audio Processing Pipeline

A comprehensive audio processing pipeline for stem separation, analysis, 
MIDI transcription, and feature extraction.
"""

__version__ = "0.1.0"

from didactic_engine.pipeline import AudioPipeline
from didactic_engine.export_md import export_midi_markdown, export_full_report
from didactic_engine.export_abc import export_abc

__all__ = [
    "AudioPipeline",
    "export_midi_markdown",
    "export_full_report",
    "export_abc",
]
