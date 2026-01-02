"""
Bar chunking module.

This module provides high-level functions for computing bar boundaries and
writing per-bar audio chunks. It wraps the lower-level segmentation functions
and optionally integrates feature extraction.

Key Functions:
    - :func:`compute_bar_boundaries`: Calculate bar start/end times
    - :func:`write_bar_chunks`: Write bar audio chunks to disk
    - :func:`write_bar_chunks_with_features`: Chunks + feature extraction

Integration:
    Bar chunking is typically used after analysis (beat detection) and before
    feature extraction. The chunks enable bar-level analysis and ML training.

Example:
    >>> boundaries = compute_bar_boundaries(
    ...     beat_times, tempo_bpm=120, ts_num=4, ts_den=4,
    ...     audio_duration_s=60.0
    ... )
    >>> chunks_meta = write_bar_chunks(
    ...     "vocals.wav", "output/chunks", beat_times, 120.0
    ... )

See Also:
    - :mod:`didactic_engine.segmentation` for lower-level functions
    - :mod:`didactic_engine.features` for feature extraction
    - :mod:`didactic_engine.analysis` for beat detection
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
    """Compute bar boundaries from beat times.

    High-level wrapper around :func:`segmentation.segment_beats_into_bars`
    with sensible defaults and automatic duration handling.

    Args:
        beat_times: List of beat times in seconds from audio analysis.
        tempo_bpm: Tempo in beats per minute. Used to extrapolate beats
            if needed to cover the full audio duration.
        ts_num: Time signature numerator. Default 4 (for 4/4 time).
        ts_den: Time signature denominator. Default 4 (quarter note).
        audio_duration_s: Total audio duration in seconds. If None,
            inferred from beat_times plus one beat interval.

    Returns:
        List of (bar_index, start_s, end_s) tuples defining bar boundaries.
        Each tuple contains:
        - bar_index: 0-based bar number
        - start_s: Bar start time in seconds
        - end_s: Bar end time in seconds

    Example:
        >>> beat_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        >>> boundaries = compute_bar_boundaries(beat_times, 120)
        >>> print(len(boundaries))  # Two 4/4 bars
        2

    Note:
        If audio_duration_s is not provided, the function adds one beat
        interval to the last beat time to estimate duration.

    See Also:
        - :func:`write_bar_chunks` for using boundaries to slice audio
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
    """Write per-bar audio chunks from a WAV file.

    Computes bar boundaries from beat times and slices the audio file
    into individual bar chunks, saving each as a separate WAV file.

    Args:
        wav_path: Path to input WAV file.
        out_dir: Output directory for bar chunks. Created if needed.
        beat_times: List of beat times in seconds from analysis.
        tempo_bpm: Tempo in beats per minute.
        ts_num: Time signature numerator. Default 4.
        ts_den: Time signature denominator. Default 4.

    Returns:
        List of metadata dictionaries, one per bar:
        - ``bar_index``: Bar number (0-based)
        - ``start_s``: Chunk start time in seconds
        - ``end_s``: Chunk end time in seconds
        - ``duration_s``: Chunk duration in seconds
        - ``chunk_path``: Path to the saved WAV file

    Example:
        >>> chunks = write_bar_chunks(
        ...     "vocals.wav", "output/chunks",
        ...     beat_times=[0.0, 0.5, 1.0, 1.5, 2.0],
        ...     tempo_bpm=120
        ... )
        >>> print(chunks[0]["chunk_path"])
        output/chunks/bar_0000.wav

    Note:
        Audio duration is automatically detected from the input file.
        Files are named ``bar_XXXX.wav`` with zero-padded indices.

    See Also:
        - :func:`write_bar_chunks_with_features` for chunks + features
        - :func:`compute_bar_boundaries` for boundary computation only
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
    """Write bar chunks and extract audio features for each.

    Combines chunking with feature extraction in a single operation.
    Useful for building bar-level feature datasets.

    Args:
        wav_path: Path to input WAV file.
        out_dir: Output directory for bar chunks.
        beat_times: List of beat times in seconds.
        tempo_bpm: Tempo in beats per minute.
        ts_num: Time signature numerator. Default 4.
        ts_den: Time signature denominator. Default 4.
        sample_rate: Sample rate for feature extraction. Default 22050.
        use_essentia: If True, include Essentia features (requires
            Essentia installation). Default False.

    Returns:
        Tuple of (chunks_meta, chunk_features):
        
        - chunks_meta: List of chunk metadata dicts (same as write_bar_chunks)
        - chunk_features: List of feature dicts, one per chunk, containing:
            - Librosa features (spectral, MFCC, chroma)
            - Optional Essentia features (if use_essentia=True)
            - Bar metadata (bar_index, start_s, end_s, duration_s, chunk_path)

    Example:
        >>> meta, features = write_bar_chunks_with_features(
        ...     "vocals.wav", "output/chunks",
        ...     beat_times, tempo_bpm=120,
        ...     use_essentia=False
        ... )
        >>> print(features[0].keys())
        dict_keys(['rms', 'zcr', 'spectral_centroid_mean', ...])

    Note:
        Feature extraction adds significant processing time. For large
        files, consider chunking first and extracting features separately.

    See Also:
        - :func:`write_bar_chunks` for chunking without features
        - :class:`didactic_engine.features.FeatureExtractor` for feature details
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
