"""
Music ETL Pipeline

A comprehensive audio processing pipeline that:
- Separates audio into stems using Demucs
- Preprocesses audio with pydub
- Analyzes audio with librosa and optional Essentia
- Transcribes to MIDI using Basic Pitch
- Aligns MIDI events to beat/bar grids
- Segments stems into per-bar chunks
- Extracts evolving bar-level features
- Exports to JSON, Markdown, ABC notation, and Parquet datasets
"""

__version__ = "0.1.0"

from music_etl.config import PipelineConfig

__all__ = ["PipelineConfig"]
