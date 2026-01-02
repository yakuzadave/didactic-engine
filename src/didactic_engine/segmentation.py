"""
Stem segmentation module.

This module provides functions and the ``StemSegmenter`` class for dividing
audio stems into per-bar WAV chunks aligned to a beat/bar grid derived from
audio analysis.

Key Features:
    - Beat-to-bar boundary computation with time signature support
    - Audio slicing using pydub for precise segment extraction
    - Support for multiple stems simultaneously
    - Arbitrary time interval segmentation

Integration:
    Segmentation occurs after analysis (to get beat grid) and separation
    (to get individual stems). The resulting chunks can be used for:
    - Per-bar feature extraction
    - Training ML models on bar-level data
    - Visual/audio inspection of specific bars

Example:
    >>> boundaries = segment_beats_into_bars(
    ...     beat_times=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    ...     tempo_bpm=120,
    ...     ts_num=4, ts_den=4,
    ...     audio_duration=10.0
    ... )
    >>> chunks_meta = segment_audio_by_bars("stem.wav", boundaries, "output/")

See Also:
    - :mod:`didactic_engine.analysis` for beat detection
    - :mod:`didactic_engine.bar_chunker` for higher-level chunking
    - :mod:`didactic_engine.features` for chunk feature extraction
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import soundfile as sf
from pydub import AudioSegment


def segment_beats_into_bars(
    beat_times: List[float],
    tempo_bpm: float,
    ts_num: int,
    ts_den: int,
    audio_duration: float,
) -> List[Tuple[int, float, float]]:
    """Compute bar boundaries from beat times and time signature.

    Takes a list of beat times and groups them into bars based on the
    time signature. Handles beat grid extension if needed to cover
    the full audio duration.

    Args:
        beat_times: List of beat times in seconds, typically from
            :meth:`didactic_engine.analysis.AudioAnalyzer.extract_beat_times`.
        tempo_bpm: Tempo in beats per minute. Used to extrapolate
            additional beats if beat_times doesn't cover full duration.
        ts_num: Time signature numerator (beats per bar). Common values:
            4 for 4/4, 3 for 3/4, 6 for 6/8.
        ts_den: Time signature denominator (beat unit). Common values:
            4 for quarter note, 8 for eighth note.
        audio_duration: Total audio duration in seconds. Used to ensure
            bar boundaries extend to cover the entire audio.

    Returns:
        List of tuples (bar_index, start_s, end_s) where:
        - bar_index: 0-based bar number
        - start_s: Bar start time in seconds
        - end_s: Bar end time in seconds (clamped to audio_duration)
        
        Zero-length bars are excluded from the output.

    Example:
        >>> beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        >>> boundaries = segment_beats_into_bars(
        ...     beats, tempo_bpm=120, ts_num=4, ts_den=4, audio_duration=4.0
        ... )
        >>> print(boundaries[0])  # First bar
        (0, 0.0, 2.0)
        >>> print(boundaries[1])  # Second bar
        (1, 2.0, 4.0)

    Note:
        The beats_per_bar formula is: ts_num * (4.0 / ts_den)
        - 4/4 time: 4 * (4/4) = 4 beats per bar
        - 3/4 time: 3 * (4/4) = 3 beats per bar
        - 6/8 time: 6 * (4/8) = 3 beats per bar

    See Also:
        - :func:`segment_audio_by_bars` for slicing audio at boundaries
    """
    # Convert to numpy array
    beat_array = np.array(beat_times, dtype=float)

    if len(beat_array) == 0:
        return []

    # Calculate beat interval
    if len(beat_array) > 1:
        beat_interval = float(np.median(np.diff(beat_array)))
    else:
        beat_interval = 60.0 / tempo_bpm if tempo_bpm > 0 else 0.5

    # Extend beat array to cover audio duration
    if beat_array[-1] < audio_duration:
        last_beat = beat_array[-1]
        num_extra = int(np.ceil((audio_duration - last_beat) / beat_interval)) + 1
        extra_beats = last_beat + np.arange(1, num_extra + 1) * beat_interval
        beat_array = np.concatenate([beat_array, extra_beats])

    # Calculate beats per bar
    beats_per_bar = ts_num * (4.0 / ts_den)

    # Build bar boundaries
    bars: List[Tuple[int, float, float]] = []
    bar_idx = 0
    beat_idx = 0

    while beat_idx < len(beat_array):
        start_beat_idx = int(beat_idx)
        end_beat_idx = int(beat_idx + beats_per_bar)

        if start_beat_idx >= len(beat_array):
            break

        start_s = float(beat_array[start_beat_idx])
        end_s = float(beat_array[min(end_beat_idx, len(beat_array) - 1)])

        # Clamp to audio duration
        end_s = min(end_s, audio_duration)

        # Skip zero-length bars
        if end_s > start_s:
            bars.append((bar_idx, start_s, end_s))

        bar_idx += 1
        beat_idx += beats_per_bar

    return bars


def segment_audio_by_bars(
    audio_path: Union[str, Path],
    boundaries: List[Tuple[int, float, float]],
    out_dir: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Segment an audio file into per-bar WAV chunks.

    Takes bar boundaries and slices the audio file into individual
    chunks, one per bar. Each chunk is saved as a separate WAV file.

    Args:
        audio_path: Path to input audio file. Supports WAV and other
            formats readable by pydub (requires ffmpeg for some formats).
        boundaries: List of (bar_index, start_s, end_s) tuples from
            :func:`segment_beats_into_bars`.
        out_dir: Output directory for chunk WAV files. Created if it
            doesn't exist.

    Returns:
        List of metadata dictionaries, one per bar chunk:
        - ``bar_index``: Bar number (matches input boundaries)
        - ``start_s``: Chunk start time in seconds
        - ``end_s``: Chunk end time in seconds
        - ``duration_s``: Chunk duration (end_s - start_s)
        - ``chunk_path``: Absolute path to the saved WAV file

    Example:
        >>> boundaries = [(0, 0.0, 2.0), (1, 2.0, 4.0)]
        >>> chunks = segment_audio_by_bars("vocals.wav", boundaries, "chunks/")
        >>> print(chunks[0]["chunk_path"])
        chunks/bar_0000.wav

    Note:
        Files are named ``bar_XXXX.wav`` where XXXX is the zero-padded
        bar index. pydub handles the conversion internally.

    See Also:
        - :func:`segment_beats_into_bars` for computing boundaries
        - :class:`StemSegmenter` for class-based segmentation
    """
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load audio with pydub
    audio = AudioSegment.from_file(str(audio_path))

    chunks_meta: List[Dict[str, Any]] = []

    for bar_idx, start_s, end_s in boundaries:
        # Convert to milliseconds
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)

        # Extract chunk
        chunk = audio[start_ms:end_ms]

        # Write chunk
        chunk_path = out_dir / f"bar_{bar_idx:04d}.wav"
        chunk.export(str(chunk_path), format="wav")

        chunks_meta.append({
            "bar_index": bar_idx,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "chunk_path": str(chunk_path),
        })

    return chunks_meta


class StemSegmenter:
    """Class-based interface for segmenting audio stems into chunks.
    
    Provides methods for segmenting audio by bar times, time intervals,
    or processing multiple stems at once.
    
    The class is stateless—all data is passed to methods directly.
    For function-based segmentation, see :func:`segment_beats_into_bars`
    and :func:`segment_audio_by_bars`.
    
    Example:
        >>> segmenter = StemSegmenter()
        >>> 
        >>> # Segment by bar times
        >>> bar_times = np.array([0.0, 2.0, 4.0, 6.0])
        >>> chunks = segmenter.segment_by_bars(
        ...     audio, sr, bar_times, "output/", stem_name="vocals"
        ... )
        >>> 
        >>> # Segment multiple stems
        >>> all_chunks = segmenter.segment_stems_by_bars(
        ...     stems_dict, sr, bar_times, "output/"
        ... )
    """

    def __init__(self):
        """Initialize the stem segmenter.
        
        The segmenter is stateless—configuration is passed per-method.
        """
        pass

    def segment_by_bars(
        self,
        audio: np.ndarray,
        sample_rate: int,
        bar_times: np.ndarray,
        output_dir: Union[str, Path],
        stem_name: str = "audio",
    ) -> List[str]:
        """Segment audio array into per-bar WAV chunks.

        Unlike :func:`segment_audio_by_bars`, this method works directly
        with numpy arrays instead of audio files.

        Args:
            audio: Input audio array (1D mono or 2D stereo).
            sample_rate: Sample rate in Hz.
            bar_times: Array of bar start times in seconds. Each
                consecutive pair defines a bar segment.
            output_dir: Directory for output WAV files.
            stem_name: Stem identifier for filenames.

        Returns:
            List of paths to saved chunk WAV files, in bar order.

        Example:
            >>> segmenter = StemSegmenter()
            >>> bar_times = np.array([0.0, 2.0, 4.0])
            >>> paths = segmenter.segment_by_bars(
            ...     audio, 44100, bar_times, "output/", "vocals"
            ... )
            >>> print(paths[0])
            output/vocals_bar_0000.wav

        Note:
            Files are named ``{stem_name}_bar_XXXX.wav``. The output
            directory is created if it doesn't exist.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []

        # Segment audio by bars
        for i in range(len(bar_times) - 1):
            start_time = bar_times[i]
            end_time = bar_times[i + 1]

            # Convert time to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Extract chunk
            if audio.ndim == 1:
                chunk = audio[start_sample:end_sample]
            else:
                # Handle both (channels, samples) and (samples, channels)
                if audio.shape[0] <= 2:
                    chunk = audio[:, start_sample:end_sample].T
                else:
                    chunk = audio[start_sample:end_sample, :]

            # Save chunk
            chunk_path = output_dir / f"{stem_name}_bar_{i:04d}.wav"
            sf.write(str(chunk_path), chunk, sample_rate)

            chunk_paths.append(str(chunk_path))

        return chunk_paths

    def segment_stems_by_bars(
        self,
        stems: Dict[str, np.ndarray],
        sample_rate: int,
        bar_times: np.ndarray,
        output_dir: Union[str, Path],
    ) -> Dict[str, List[str]]:
        """Segment multiple stems into per-bar chunks simultaneously.

        Convenience method for processing all stems from a separation
        result with the same bar grid.

        Args:
            stems: Dictionary mapping stem names to audio arrays.
                Typical keys: 'vocals', 'drums', 'bass', 'other'.
            sample_rate: Sample rate in Hz (same for all stems).
            bar_times: Array of bar start times in seconds.
            output_dir: Base output directory. Each stem gets a subdirectory.

        Returns:
            Dictionary mapping stem names to lists of chunk file paths.
            Structure mirrors the input stems dictionary.

        Example:
            >>> segmenter = StemSegmenter()
            >>> stems = {"vocals": vocals_audio, "bass": bass_audio}
            >>> all_chunks = segmenter.segment_stems_by_bars(
            ...     stems, 44100, bar_times, "output/"
            ... )
            >>> print(all_chunks["vocals"][0])
            output/vocals/vocals_bar_0000.wav
        """
        output_dir = Path(output_dir)
        segmented_stems: Dict[str, List[str]] = {}

        for stem_name, stem_audio in stems.items():
            stem_output_dir = output_dir / stem_name
            chunk_paths = self.segment_by_bars(
                stem_audio, sample_rate, bar_times, stem_output_dir, stem_name
            )
            segmented_stems[stem_name] = chunk_paths

        return segmented_stems

    def segment_by_time_intervals(
        self,
        audio: np.ndarray,
        sample_rate: int,
        intervals: List[Tuple[float, float]],
        output_dir: Union[str, Path],
        stem_name: str = "audio",
    ) -> List[str]:
        """Segment audio by arbitrary time intervals.

        More flexible than bar-based segmentation—allows any time
        boundaries. Useful for custom segmentation schemes.

        Args:
            audio: Input audio array (1D or 2D).
            sample_rate: Sample rate in Hz.
            intervals: List of (start_time, end_time) tuples in seconds.
                Intervals can overlap or have gaps.
            output_dir: Directory for output WAV files.
            stem_name: Stem identifier for filenames.

        Returns:
            List of paths to saved segment WAV files.

        Example:
            >>> segmenter = StemSegmenter()
            >>> intervals = [(0.0, 1.5), (2.0, 3.5), (4.0, 5.5)]
            >>> paths = segmenter.segment_by_time_intervals(
            ...     audio, 44100, intervals, "output/", "custom"
            ... )
            >>> print(paths[0])
            output/custom_segment_0000.wav

        Note:
            Files are named ``{stem_name}_segment_XXXX.wav``. Unlike
            bar segmentation, segment indices correspond to interval
            order, not musical position.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []

        for i, (start_time, end_time) in enumerate(intervals):
            # Convert time to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # Extract chunk
            if audio.ndim == 1:
                chunk = audio[start_sample:end_sample]
            else:
                if audio.shape[0] <= 2:
                    chunk = audio[:, start_sample:end_sample].T
                else:
                    chunk = audio[start_sample:end_sample, :]

            # Save chunk
            chunk_path = output_dir / f"{stem_name}_segment_{i:04d}.wav"
            sf.write(str(chunk_path), chunk, sample_rate)

            chunk_paths.append(str(chunk_path))

        return chunk_paths
