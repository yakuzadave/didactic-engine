"""
Bar chunking module.

Provides functions to compute bar boundaries and write per-bar audio chunks.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
import numpy as np
from pydub import AudioSegment

from didactic_engine.segmentation import segment_beats_into_bars, segment_audio_by_bars


def compute_bar_boundaries(
    beat_times: List[float],
    tempo_bpm: float,
    ts_num: int = 4,
    ts_den: int = 4,
    audio_duration_s: float = None,
) -> List[Tuple[int, float, float]]:
    """
    Compute bar boundaries from beat times.

    Uses beat extrapolation if beat_times doesn't cover the full audio duration.

    Args:
        beat_times: List of beat times in seconds.
        tempo_bpm: Tempo in beats per minute.
        ts_num: Time signature numerator.
        ts_den: Time signature denominator.
        audio_duration_s: Total audio duration in seconds.

    Returns:
        List of (bar_index, start_s, end_s) tuples.
    """
    if audio_duration_s is None and beat_times:
        audio_duration_s = max(beat_times) + (60.0 / tempo_bpm)
    elif audio_duration_s is None:
        audio_duration_s = 0.0

    return segment_beats_into_bars(
        beat_times, tempo_bpm, ts_num, ts_den, audio_duration_s
    )


def write_bar_chunks(
    wav_path: Union[str, Path],
    out_dir: Union[str, Path],
    beat_times: List[float],
    tempo_bpm: float,
    ts_num: int = 4,
    ts_den: int = 4,
) -> List[Dict[str, Any]]:
    """
    Write per-bar audio chunks from a WAV file.

    Args:
        wav_path: Path to input WAV file.
        out_dir: Output directory for bar chunks.
        beat_times: List of beat times in seconds.
        tempo_bpm: Tempo in beats per minute.
        ts_num: Time signature numerator.
        ts_den: Time signature denominator.

    Returns:
        List of metadata dicts with: bar_index, start_s, end_s,
        duration_s, chunk_path.
    """
    wav_path = Path(wav_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load audio to get duration
    audio = AudioSegment.from_file(str(wav_path))
    audio_duration_s = len(audio) / 1000.0

    # Compute bar boundaries
    boundaries = compute_bar_boundaries(
        beat_times, tempo_bpm, ts_num, ts_den, audio_duration_s
    )

    # Write chunks
    return segment_audio_by_bars(wav_path, boundaries, out_dir)


def write_bar_chunks_with_features(
    wav_path: Union[str, Path],
    out_dir: Union[str, Path],
    beat_times: List[float],
    tempo_bpm: float,
    ts_num: int = 4,
    ts_den: int = 4,
    sample_rate: int = 22050,
    use_essentia: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Write bar chunks and extract features for each.

    Args:
        wav_path: Path to input WAV file.
        out_dir: Output directory for bar chunks.
        beat_times: List of beat times in seconds.
        tempo_bpm: Tempo in beats per minute.
        ts_num: Time signature numerator.
        ts_den: Time signature denominator.
        sample_rate: Sample rate for analysis.
        use_essentia: Whether to extract Essentia features.

    Returns:
        Tuple of (chunks_meta, chunk_features) where chunk_features is a
        list of feature dicts for each chunk.
    """
    from didactic_engine.features import FeatureExtractor
    from didactic_engine.essentia_features import extract_essentia_features

    # Write chunks
    chunks_meta = write_bar_chunks(
        wav_path, out_dir, beat_times, tempo_bpm, ts_num, ts_den
    )

    # Extract features for each chunk
    extractor = FeatureExtractor()
    chunk_features: List[Dict[str, Any]] = []

    for meta in chunks_meta:
        chunk_path = meta["chunk_path"]

        # Extract librosa features
        features = extractor.extract_bar_features_from_file(chunk_path, sample_rate)

        # Optionally add Essentia features
        if use_essentia:
            essentia_feats = extract_essentia_features(chunk_path, sample_rate)
            if essentia_feats.get("available", False):
                features["essentia"] = essentia_feats

        features["bar_index"] = meta["bar_index"]
        features["start_s"] = meta["start_s"]
        features["end_s"] = meta["end_s"]
        features["duration_s"] = meta["duration_s"]
        features["chunk_path"] = chunk_path

        chunk_features.append(features)

    return chunks_meta, chunk_features
