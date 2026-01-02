"""
Feature extraction module.

This module provides the ``FeatureExtractor`` class for extracting structured
features from audio and MIDI data at various levels of granularity: events,
beats, bars, and bar-level feature vectors.

Key Features:
    - Event extraction from aligned MIDI notes
    - Beat-level metadata generation
    - Bar-level aggregation (note counts, velocity, pitch range)
    - Per-bar audio feature extraction (spectral, MFCC, chroma)
    - Evolving feature extraction across segments

Integration:
    Feature extraction occurs after MIDI parsing and alignment. Results
    are typically exported to Parquet datasets for further analysis.

Output Levels:
    1. **Events**: One row per MIDI note with timing and alignment info
    2. **Beats**: One row per beat with position and tempo info
    3. **Bars**: Aggregated statistics per bar (note count, velocity, etc.)
    4. **Bar Features**: Audio features extracted from each bar chunk

Example:
    >>> extractor = FeatureExtractor()
    >>> events_df = extractor.extract_events(aligned_notes)
    >>> beats_df = extractor.extract_beats(beat_times, tempo, stem, song_id)
    >>> bars_df = extractor.extract_bars(aligned_notes, song_id)

See Also:
    - :mod:`didactic_engine.align` for note alignment
    - :mod:`didactic_engine.bar_chunker` for bar audio slicing
    - :mod:`didactic_engine.pipeline` for dataset generation
"""

from pathlib import Path
from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd
import librosa
import soundfile as sf


class FeatureExtractor:
    """Extract structured features at event, beat, bar, and segment levels.
    
    Provides methods to convert raw MIDI and audio data into structured
    DataFrames suitable for machine learning or analysis.
    
    Feature Levels:
        - **Events**: Raw MIDI notes with alignment info
        - **Beats**: Beat grid metadata
        - **Bars**: Aggregated bar statistics
        - **Bar Features**: Audio features per bar chunk
    
    Example:
        >>> extractor = FeatureExtractor()
        >>> 
        >>> # Extract event-level data
        >>> events = extractor.extract_events(aligned_notes_df)
        >>> 
        >>> # Extract bar-level aggregates
        >>> bars = extractor.extract_bars(aligned_notes_df, "song_001")
        >>> 
        >>> # Extract audio features from a chunk
        >>> features = extractor.extract_bar_features_from_audio(audio, sr)
    """

    def __init__(self):
        """Initialize the feature extractor.
        
        The extractor is stateless—all data is passed to methods directly.
        """
        pass

    def extract_events(self, aligned_notes: pd.DataFrame) -> pd.DataFrame:
        """Extract events DataFrame from aligned MIDI notes.

        Returns the aligned notes with any additional computed fields.
        This is primarily a pass-through that ensures required columns exist.

        Args:
            aligned_notes: DataFrame with aligned note events from
                :func:`didactic_engine.align.align_notes_to_beats`. Expected
                columns: start_s, end_s, pitch, velocity, bar_index, beat_index.

        Returns:
            DataFrame with same structure as input, plus computed columns:
            - ``dur_s``: Note duration (end_s - start_s) if not present
            
            Empty DataFrame is returned unchanged.

        Example:
            >>> extractor = FeatureExtractor()
            >>> events = extractor.extract_events(aligned_df)
            >>> events.to_parquet("events.parquet")

        See Also:
            - :func:`didactic_engine.align.align_notes_to_beats` for alignment
        """
        if aligned_notes.empty:
            return aligned_notes

        # Return with any additional computed fields
        result = aligned_notes.copy()

        # Ensure required columns exist
        if "dur_s" not in result.columns and "start_s" in result.columns and "end_s" in result.columns:
            result["dur_s"] = result["end_s"] - result["start_s"]

        return result

    def extract_beats(
        self,
        beat_times: List[float],
        tempo_bpm: float,
        stem: str,
        song_id: str,
    ) -> pd.DataFrame:
        """Generate a beats DataFrame with one row per beat.

        Creates a structured record of the beat grid for a given stem,
        useful for analysis and joins with event data.

        Args:
            beat_times: List of beat times in seconds, typically from
                audio analysis via :class:`didactic_engine.analysis.AudioAnalyzer`.
            tempo_bpm: Tempo in beats per minute. Used as metadata for
                each beat row (same value throughout).
            stem: Stem name (e.g., 'vocals', 'bass'). Identifies which
                audio source the beats correspond to.
            song_id: Song identifier for grouping/filtering in datasets.

        Returns:
            DataFrame with columns:
            - ``song_id``: Song identifier
            - ``stem``: Stem name
            - ``beat_index``: 0-based beat number
            - ``time_s``: Beat time in seconds
            - ``tempo_bpm``: Tempo at this beat

        Example:
            >>> extractor = FeatureExtractor()
            >>> beat_times = [0.5, 1.0, 1.5, 2.0]
            >>> beats_df = extractor.extract_beats(beat_times, 120.0, "vocals", "song1")
            >>> print(beats_df.columns.tolist())
            ['song_id', 'stem', 'beat_index', 'time_s', 'tempo_bpm']
        """
        data = []
        for idx, time_s in enumerate(beat_times):
            data.append({
                "song_id": song_id,
                "stem": stem,
                "beat_index": idx,
                "time_s": float(time_s),
                "tempo_bpm": float(tempo_bpm),
            })

        return pd.DataFrame(data)

    def extract_bars(
        self,
        aligned_notes: pd.DataFrame,
        song_id: str,
    ) -> pd.DataFrame:
        """Compute per-bar aggregate statistics from aligned notes.

        Groups notes by stem and bar_index, computing summary statistics
        useful for bar-level analysis.

        Args:
            aligned_notes: DataFrame with aligned note events. Required
                columns: bar_index, pitch, velocity, start_s, end_s.
                Optional: stem (defaults to 'all' if missing).
            song_id: Song identifier for the output DataFrame.

        Returns:
            DataFrame with one row per (stem, bar_index) combination:
            - ``song_id``: Song identifier
            - ``stem``: Stem name
            - ``bar_index``: Bar number (0-based)
            - ``num_notes``: Count of notes in this bar
            - ``mean_velocity``: Average velocity of notes
            - ``pitch_min``: Lowest pitch (MIDI number)
            - ``pitch_max``: Highest pitch (MIDI number)
            - ``start_s``: Earliest note start time in bar
            - ``end_s``: Latest note end time in bar
            
            Returns empty DataFrame if aligned_notes is empty.

        Example:
            >>> extractor = FeatureExtractor()
            >>> bars = extractor.extract_bars(aligned_notes, "song_001")
            >>> print(bars[['bar_index', 'num_notes', 'mean_velocity']].head())
               bar_index  num_notes  mean_velocity
            0          0          8           78.5
            1          1         12           82.3
        """
        if aligned_notes.empty:
            return pd.DataFrame()

        # Group by stem (if present) and bar_index
        if "stem" not in aligned_notes.columns:
            aligned_notes = aligned_notes.copy()
            aligned_notes["stem"] = "all"

        bars_data = []
        grouped = aligned_notes.groupby(["stem", "bar_index"])

        for (stem, bar_idx), group in grouped:
            bars_data.append({
                "song_id": song_id,
                "stem": stem,
                "bar_index": int(bar_idx),
                "num_notes": len(group),
                "mean_velocity": float(group["velocity"].mean()),
                "pitch_min": int(group["pitch"].min()),
                "pitch_max": int(group["pitch"].max()),
                "start_s": float(group["start_s"].min()),
                "end_s": float(group["end_s"].max()),
            })

        return pd.DataFrame(bars_data)

    def extract_bar_features(
        self,
        chunks_meta: List[Dict[str, Any]],
        chunk_features: List[Dict[str, Any]],
        song_id: str,
        stem: str,
    ) -> pd.DataFrame:
        """
        Merge chunk metadata with features into a bar features DataFrame.

        Args:
            chunks_meta: List of chunk metadata dicts.
            chunk_features: List of feature dicts for each chunk.
            song_id: Song identifier.
            stem: Stem name.

        Returns:
            DataFrame with bar-level features.
        """
        rows = []

        for meta, features in zip(chunks_meta, chunk_features):
            row = {
                "song_id": song_id,
                "stem": stem,
                "bar_index": meta.get("bar_index", 0),
                "start_s": meta.get("start_s", 0.0),
                "end_s": meta.get("end_s", 0.0),
                "duration_s": meta.get("duration_s", 0.0),
                "chunk_path": str(meta.get("chunk_path", "")),
            }

            # Flatten features and add to row
            for key, value in features.items():
                if isinstance(value, (int, float, bool)):
                    row[key] = value
                elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    # Expand list features
                    for i, v in enumerate(value):
                        row[f"{key}_{i:02d}"] = v

            rows.append(row)

        return pd.DataFrame(rows)

    def extract_bar_features_from_audio(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Extract features from a single bar/segment of audio.

        Args:
            audio: Input audio array (1D or 2D).
            sample_rate: Sample rate of the audio.

        Returns:
            Dictionary containing extracted features.
        """
        # Convert to mono for feature extraction
        if audio.ndim == 2:
            if audio.shape[0] <= 2:
                audio_mono = librosa.to_mono(audio)
            else:
                audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio.flatten()

        features: Dict[str, Any] = {}

        # Time-domain features
        features["rms"] = float(np.sqrt(np.mean(audio_mono**2)))
        features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(audio_mono)))
        features["energy"] = float(np.sum(audio_mono**2))

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_mono, sr=sample_rate)
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_mono, sr=sample_rate)
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=sample_rate)
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))

        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_mono, sr=sample_rate, n_mfcc=13)
        features["mfcc_mean"] = [float(np.mean(mfccs[i])) for i in range(mfccs.shape[0])]
        features["mfcc_std"] = [float(np.std(mfccs[i])) for i in range(mfccs.shape[0])]

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_mono, sr=sample_rate)
        features["chroma_mean"] = [float(np.mean(chroma[i])) for i in range(chroma.shape[0])]

        return features

    def extract_bar_features_from_file(
        self, audio_path: Union[str, Path], sample_rate: int = 22050
    ) -> Dict[str, Any]:
        """
        Extract features from an audio file.

        Args:
            audio_path: Path to audio file.
            sample_rate: Sample rate for analysis.

        Returns:
            Dictionary containing extracted features.
        """
        audio, sr = sf.read(str(audio_path))
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return self.extract_bar_features_from_audio(audio, sample_rate)

    # Legacy method names for backward compatibility
    def extract_bar_features(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Legacy alias for extract_bar_features_from_audio."""
        return self.extract_bar_features_from_audio(audio, sample_rate)

    def extract_features_from_chunks(
        self, chunk_paths: List[str], sample_rate: int
    ) -> List[Dict[str, Any]]:
        """
        Extract features from multiple audio chunks.

        Args:
            chunk_paths: List of paths to audio chunk files.
            sample_rate: Sample rate of the audio.

        Returns:
            List of feature dictionaries, one per chunk.
        """
        all_features = []

        for i, chunk_path in enumerate(chunk_paths):
            features = self.extract_bar_features_from_file(chunk_path, sample_rate)
            features["chunk_index"] = i
            features["chunk_path"] = chunk_path
            all_features.append(features)

        return all_features

    def extract_evolving_features(
        self, audio: np.ndarray, sample_rate: int, n_segments: int = 4
    ) -> Dict[str, Any]:
        """
        Extract evolving features by dividing audio into sub-segments.

        Args:
            audio: Input audio array.
            sample_rate: Sample rate of the audio.
            n_segments: Number of sub-segments.

        Returns:
            Dictionary containing evolving features.
        """
        # Ensure audio is 1D
        if audio.ndim == 2:
            audio = librosa.to_mono(audio) if audio.shape[0] <= 2 else np.mean(audio, axis=1)
        else:
            audio = audio.flatten()

        segment_length = len(audio) // n_segments
        evolving_features: Dict[str, Any] = {
            "n_segments": n_segments,
            "segments": [],
        }

        for i in range(n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length if i < n_segments - 1 else len(audio)
            segment = audio[start:end]

            segment_features = self.extract_bar_features_from_audio(segment, sample_rate)
            segment_features["segment_index"] = i
            segment_features["segment_start_time"] = start / sample_rate
            segment_features["segment_end_time"] = end / sample_rate

            evolving_features["segments"].append(segment_features)

        return evolving_features
