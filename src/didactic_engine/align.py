"""
Note-to-beat alignment module.

Provides functions to align MIDI note events to a beat grid derived from
audio analysis.
"""

import bisect
from typing import List
import numpy as np
import pandas as pd


def align_notes_to_beats(
    notes: pd.DataFrame,
    beat_times: List[float],
    tempo_bpm: float,
    ts_num: int = 4,
    ts_den: int = 4,
) -> pd.DataFrame:
    """
    Align MIDI notes to a beat grid.

    If beat_times is shorter than needed, synthesizes additional beats
    at intervals of 60 / tempo_bpm until covering the max note end time.

    Args:
        notes: DataFrame with note events. Must have columns:
            - start_s: Note start time in seconds
            - end_s: Note end time in seconds
        beat_times: List of beat times in seconds from audio analysis.
        tempo_bpm: Tempo in beats per minute.
        ts_num: Time signature numerator (beats per bar).
        ts_den: Time signature denominator (beat unit).

    Returns:
        DataFrame with added columns:
            - beat_index: Index of the beat at or before note start
            - bar_index: Bar index (beat_index // beats_per_bar)
            - beat_in_bar: Beat position within the bar (0 to beats_per_bar-1)
            - start_beat_float: Fractional beat position of note start
            - end_beat_float: Fractional beat position of note end
    """
    if notes.empty:
        # Return empty DataFrame with expected columns
        return notes.assign(
            beat_index=pd.Series(dtype=int),
            bar_index=pd.Series(dtype=int),
            beat_in_bar=pd.Series(dtype=float),
            start_beat_float=pd.Series(dtype=float),
            end_beat_float=pd.Series(dtype=float),
        )

    # Convert beat_times to numpy array
    beat_array = np.array(beat_times, dtype=float)

    # Calculate beat interval
    if len(beat_array) > 1:
        beat_interval = float(np.median(np.diff(beat_array)))
    else:
        beat_interval = 60.0 / tempo_bpm if tempo_bpm > 0 else 0.5

    # Find max time we need to cover
    max_time = notes["end_s"].max()

    # Extend beat array if needed
    if len(beat_array) > 0 and beat_array[-1] < max_time:
        last_beat = beat_array[-1]
        num_extra_beats = int(np.ceil((max_time - last_beat) / beat_interval)) + 1
        extra_beats = last_beat + np.arange(1, num_extra_beats + 1) * beat_interval
        beat_array = np.concatenate([beat_array, extra_beats])
    elif len(beat_array) == 0:
        # Generate beats from scratch
        num_beats = int(np.ceil(max_time / beat_interval)) + 1
        beat_array = np.arange(num_beats) * beat_interval

    # Calculate beats per bar
    beats_per_bar = ts_num * (4.0 / ts_den)

    # Compute alignment for each note
    beat_indices = []
    bar_indices = []
    beats_in_bar = []
    start_beat_floats = []
    end_beat_floats = []

    for _, row in notes.iterrows():
        start_s = row["start_s"]
        end_s = row["end_s"]

        # Find beat index for start time using bisect
        # bisect_right returns index where start_s would be inserted
        # We want the beat at or before start_s
        beat_idx = bisect.bisect_right(beat_array, start_s) - 1
        beat_idx = max(0, beat_idx)

        # Calculate fractional beat position
        if beat_idx < len(beat_array) - 1:
            beat_start = beat_array[beat_idx]
            beat_end = beat_array[beat_idx + 1]
            beat_duration = beat_end - beat_start
            if beat_duration > 0:
                frac = (start_s - beat_start) / beat_duration
            else:
                frac = 0.0
            start_beat_float = beat_idx + frac
        else:
            start_beat_float = float(beat_idx)

        # Calculate end beat float
        end_beat_idx = bisect.bisect_right(beat_array, end_s) - 1
        end_beat_idx = max(0, end_beat_idx)
        if end_beat_idx < len(beat_array) - 1:
            beat_start = beat_array[end_beat_idx]
            beat_end = beat_array[end_beat_idx + 1]
            beat_duration = beat_end - beat_start
            if beat_duration > 0:
                frac = (end_s - beat_start) / beat_duration
            else:
                frac = 0.0
            end_beat_float = end_beat_idx + frac
        else:
            end_beat_float = float(end_beat_idx)

        # Calculate bar index and beat within bar
        bar_idx = int(start_beat_float / beats_per_bar)
        beat_in_bar = start_beat_float % beats_per_bar

        beat_indices.append(beat_idx)
        bar_indices.append(bar_idx)
        beats_in_bar.append(beat_in_bar)
        start_beat_floats.append(start_beat_float)
        end_beat_floats.append(end_beat_float)

    # Add columns to DataFrame
    result = notes.copy()
    result["beat_index"] = beat_indices
    result["bar_index"] = bar_indices
    result["beat_in_bar"] = beats_in_bar
    result["start_beat_float"] = start_beat_floats
    result["end_beat_float"] = end_beat_floats

    return result


def synthesize_beats(
    start_time: float,
    end_time: float,
    tempo_bpm: float,
) -> np.ndarray:
    """
    Synthesize a beat grid for a given time range.

    Args:
        start_time: Start time in seconds.
        end_time: End time in seconds.
        tempo_bpm: Tempo in beats per minute.

    Returns:
        Array of beat times in seconds.
    """
    if tempo_bpm <= 0:
        tempo_bpm = 120.0  # Default tempo

    beat_interval = 60.0 / tempo_bpm
    num_beats = int(np.ceil((end_time - start_time) / beat_interval)) + 1
    beats = start_time + np.arange(num_beats) * beat_interval

    return beats
