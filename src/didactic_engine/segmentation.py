"""
Stem segmentation module.

Segments audio stems into per-bar WAV chunks aligned to beat/bar grid.
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
    """
    Compute bar boundaries from beat times.

    Args:
        beat_times: List of beat times in seconds.
        tempo_bpm: Tempo in beats per minute.
        ts_num: Time signature numerator (beats per bar).
        ts_den: Time signature denominator (beat unit).
        audio_duration: Total audio duration in seconds.

    Returns:
        List of (bar_index, start_s, end_s) tuples.
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
    """
    Segment audio file into per-bar chunks.

    Args:
        audio_path: Path to audio file.
        boundaries: List of (bar_index, start_s, end_s) tuples.
        out_dir: Output directory for chunks.

    Returns:
        List of metadata dicts with: bar_index, start_s, end_s,
        duration_s, chunk_path.
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
    """Segment audio stems into per-bar chunks."""

    def __init__(self):
        """Initialize the stem segmenter."""
        pass

    def segment_by_bars(
        self,
        audio: np.ndarray,
        sample_rate: int,
        bar_times: np.ndarray,
        output_dir: Union[str, Path],
        stem_name: str = "audio",
    ) -> List[str]:
        """
        Segment audio into per-bar chunks.

        Args:
            audio: Input audio array (1D or 2D).
            sample_rate: Sample rate of the audio.
            bar_times: Array of bar start times in seconds.
            output_dir: Directory to save segmented chunks.
            stem_name: Name of the stem for output filenames.

        Returns:
            List of paths to saved chunk files.
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
        """
        Segment multiple stems into per-bar chunks.

        Args:
            stems: Dictionary mapping stem names to audio arrays.
            sample_rate: Sample rate of the audio.
            bar_times: Array of bar start times in seconds.
            output_dir: Directory to save segmented chunks.

        Returns:
            Dictionary mapping stem names to lists of chunk paths.
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
        """
        Segment audio by arbitrary time intervals.

        Args:
            audio: Input audio array.
            sample_rate: Sample rate of the audio.
            intervals: List of (start_time, end_time) tuples in seconds.
            output_dir: Directory to save segmented chunks.
            stem_name: Name of the stem for output filenames.

        Returns:
            List of paths to saved chunk files.
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
