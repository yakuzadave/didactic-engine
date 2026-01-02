"""
Align MIDI notes to beat and bar grid.
"""

import numpy as np
import pandas as pd


def align_notes_to_grid(
    notes_df: pd.DataFrame,
    beat_times: list[float],
    tempo_bpm: float,
    ts_num: int = 4,
    ts_den: int = 4,
    audio_duration_s: float = None,
) -> pd.DataFrame:
    """
    Align MIDI notes to beat and bar grid.

    Args:
        notes_df: DataFrame with note events
        beat_times: List of beat times in seconds
        tempo_bpm: Tempo in BPM
        ts_num: Time signature numerator
        ts_den: Time signature denominator
        audio_duration_s: Audio duration for extrapolation

    Returns:
        DataFrame with added beat_index, bar_index, beat_in_bar columns
    """
    if notes_df.empty:
        return notes_df

    # Convert beat_times to array
    beat_array = np.array(beat_times)

    # Extrapolate beats if needed
    if len(beat_array) > 1:
        beat_interval = np.median(np.diff(beat_array))
    else:
        beat_interval = 60.0 / tempo_bpm if tempo_bpm > 0 else 0.5

    # Extend beat array to cover audio duration
    if audio_duration_s and len(beat_array) > 0:
        max_needed = audio_duration_s
        if notes_df["end_s"].max() > max_needed:
            max_needed = notes_df["end_s"].max()

        while beat_array[-1] < max_needed:
            beat_array = np.append(beat_array, beat_array[-1] + beat_interval)

    # Compute beats per bar
    beats_per_bar = ts_num * (4.0 / ts_den)

    # Align each note
    beat_indices = []
    bar_indices = []
    beats_in_bar = []
    start_beat_floats = []
    end_beat_floats = []

    for _, row in notes_df.iterrows():
        start_s = row["start_s"]
        end_s = row["end_s"]

        # Find beat index for start time
        beat_idx = np.searchsorted(beat_array, start_s)
        if beat_idx > 0 and beat_idx < len(beat_array):
            # Interpolate fractional beat
            prev_beat = beat_array[beat_idx - 1]
            next_beat = beat_array[beat_idx]
            beat_frac = (start_s - prev_beat) / (next_beat - prev_beat) if next_beat != prev_beat else 0
            beat_float = beat_idx - 1 + beat_frac
        else:
            beat_float = float(beat_idx)

        # Compute bar index
        bar_idx = int(beat_float / beats_per_bar)
        beat_in_bar = beat_float % beats_per_bar

        # Compute end beat float
        end_beat_idx = np.searchsorted(beat_array, end_s)
        if end_beat_idx > 0 and end_beat_idx < len(beat_array):
            prev_beat = beat_array[end_beat_idx - 1]
            next_beat = beat_array[end_beat_idx]
            end_frac = (end_s - prev_beat) / (next_beat - prev_beat) if next_beat != prev_beat else 0
            end_beat_float = end_beat_idx - 1 + end_frac
        else:
            end_beat_float = float(end_beat_idx)

        beat_indices.append(int(beat_float))
        bar_indices.append(bar_idx)
        beats_in_bar.append(beat_in_bar)
        start_beat_floats.append(beat_float)
        end_beat_floats.append(end_beat_float)

    # Add columns to DataFrame
    aligned_df = notes_df.copy()
    aligned_df["beat_index"] = beat_indices
    aligned_df["bar_index"] = bar_indices
    aligned_df["beat_in_bar"] = beats_in_bar
    aligned_df["start_beat_float"] = start_beat_floats
    aligned_df["end_beat_float"] = end_beat_floats

    return aligned_df
