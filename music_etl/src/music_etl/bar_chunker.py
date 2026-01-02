"""
Bar chunking and segmentation.
"""

from pathlib import Path
import numpy as np
from pydub import AudioSegment


def compute_bar_boundaries(
    beat_times: list[float],
    tempo_bpm: float,
    ts_num: int = 4,
    ts_den: int = 4,
    audio_duration_s: float = None,
) -> list[tuple[int, float, float]]:
    """
    Compute bar boundaries from beat times.

    Args:
        beat_times: List of beat times in seconds
        tempo_bpm: Tempo in BPM
        ts_num: Time signature numerator
        ts_den: Time signature denominator
        audio_duration_s: Audio duration to extend beats to

    Returns:
        List of (bar_index, start_s, end_s) tuples
    """
    # Convert to array
    beat_array = np.array(beat_times)

    if len(beat_array) == 0:
        return []

    # Compute beat interval
    if len(beat_array) > 1:
        beat_interval = np.median(np.diff(beat_array))
    else:
        beat_interval = 60.0 / tempo_bpm if tempo_bpm > 0 else 0.5

    # Extend beats to audio duration (vectorized for performance)
    if audio_duration_s and len(beat_array) > 0:
        last_beat = beat_array[-1]
        if last_beat < audio_duration_s:
            # Calculate number of beats needed
            num_extra_beats = int(np.ceil((audio_duration_s - last_beat) / beat_interval))
            # Generate extra beats using vectorized operation
            extra_beats = last_beat + np.arange(1, num_extra_beats + 1) * beat_interval
            beat_array = np.concatenate([beat_array, extra_beats])

    # Compute beats per bar
    beats_per_bar = ts_num * (4.0 / ts_den)

    # Generate bar boundaries
    bars = []
    bar_idx = 0
    beat_idx = 0

    while beat_idx < len(beat_array):
        start_beat_idx = int(beat_idx)
        end_beat_idx = int(beat_idx + beats_per_bar)

        if start_beat_idx >= len(beat_array):
            break

        start_s = beat_array[start_beat_idx]
        end_s = beat_array[min(end_beat_idx, len(beat_array) - 1)]

        # Clamp to audio duration
        if audio_duration_s:
            end_s = min(end_s, audio_duration_s)

        # Skip zero-length bars
        if end_s > start_s:
            bars.append((bar_idx, start_s, end_s))

        bar_idx += 1
        beat_idx += beats_per_bar

    return bars


def write_bar_chunks(
    wav_path: Path,
    out_dir: Path,
    beat_times: list[float],
    tempo_bpm: float,
    ts_num: int = 4,
    ts_den: int = 4,
) -> list[dict]:
    """
    Write per-bar audio chunks.

    Args:
        wav_path: Path to input WAV file
        out_dir: Output directory for chunks
        beat_times: List of beat times
        tempo_bpm: Tempo in BPM
        ts_num: Time signature numerator
        ts_den: Time signature denominator

    Returns:
        List of dictionaries with bar metadata
    """
    # Load audio
    audio = AudioSegment.from_file(str(wav_path))
    audio_duration_s = len(audio) / 1000.0

    # Compute bar boundaries
    bars = compute_bar_boundaries(beat_times, tempo_bpm, ts_num, ts_den, audio_duration_s)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write chunks
    chunk_metadata = []

    for bar_idx, start_s, end_s in bars:
        # Convert to milliseconds
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)

        # Extract chunk
        chunk = audio[start_ms:end_ms]

        # Write chunk
        chunk_path = out_dir / f"bar_{bar_idx:04d}.wav"
        chunk.export(str(chunk_path), format="wav")

        chunk_metadata.append({
            "bar_index": bar_idx,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "chunk_path": str(chunk_path),
        })

    return chunk_metadata
