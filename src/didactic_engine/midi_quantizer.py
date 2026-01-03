"""
MIDI note quantization for cleaner ABC notation.

This module provides functions to quantize MIDI notes to a rhythmic grid,
reducing timing jitter and producing cleaner, more readable ABC notation.

Key Functions:
    - :func:`quantize_notes`: Quantize note timings to a grid
    - :func:`quantize_midi_file`: Quantize a MIDI file in place

Quantization:
    Rounds note start and end times to the nearest grid step:
    - 1/16 note grid: suitable for most music (default)
    - 1/12 note grid: better for triplet-heavy music
    - 1/8 note grid: coarser, simpler notation
    
    Formula: quantized_time = round(time / grid_step) * grid_step

Example:
    >>> import pandas as pd
    >>> notes = pd.DataFrame({
    ...     'start_time': [0.0, 0.52, 1.03],
    ...     'end_time': [0.48, 0.97, 1.51],
    ...     'pitch': [60, 62, 64],
    ... })
    >>> quantized = quantize_notes(notes, tempo_bpm=120, division=16)
    >>> print(quantized[['start_time', 'end_time']])
        start_time  end_time
    0         0.0      0.5
    1         0.5      1.0
    2         1.0      1.5

See Also:
    - :mod:`didactic_engine.export_abc` for ABC notation export
    - :mod:`didactic_engine.midi_parser` for MIDI file parsing
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np


def quantize_notes(
    notes_df: pd.DataFrame,
    tempo_bpm: float = 120.0,
    division: int = 16,
    min_duration: Optional[float] = None,
) -> pd.DataFrame:
    """
    Quantize note timings to a rhythmic grid.
    
    Rounds note start and end times to the nearest grid step based on
    tempo and division. This reduces jitter from imperfect transcription
    and produces cleaner ABC notation.
    
    Args:
        notes_df: DataFrame with at least 'start_time' and 'end_time' columns
            (in seconds). Other columns are preserved.
        tempo_bpm: Tempo in beats per minute. Used to compute grid step size.
        division: Rhythmic division for the grid. Common values:
            - 16: 1/16 notes (default, suitable for most music)
            - 12: 1/12 notes (better for triplet-heavy music)
            - 8: 1/8 notes (coarser grid, simpler notation)
            - 4: quarter notes (very coarse)
        min_duration: Optional minimum note duration in seconds.
            If specified, notes shorter than this are extended to this duration.
            Useful to prevent very short notes that look like typos in ABC.
            
    Returns:
        DataFrame with quantized 'start_time' and 'end_time' columns.
        All other columns are preserved unchanged.
        
    Example:
        >>> import pandas as pd
        >>> notes = pd.DataFrame({
        ...     'start_time': [0.0, 0.52, 1.03, 1.48],
        ...     'end_time': [0.48, 0.97, 1.44, 1.99],
        ...     'pitch': [60, 62, 64, 65],
        ...     'velocity': [100, 100, 100, 100],
        ... })
        >>> quantized = quantize_notes(notes, tempo_bpm=120, division=16)
        >>> print(quantized[['start_time', 'end_time']])
            start_time  end_time
        0         0.0       0.5
        1         0.5       1.0
        2         1.0       1.5
        3         1.5       2.0
        
    Note:
        - Grid step size = 60 / (tempo_bpm * division / 4)
        - Example: 120 BPM, division=16 → grid step = 0.125s (eighth note)
        - Notes are quantized independently; overlaps may occur
        
    See Also:
        - :func:`quantize_midi_file` for file-based quantization
    """
    if notes_df.empty:
        return notes_df
    
    # Calculate grid step in seconds
    # For tempo_bpm and division:
    # - Quarter note duration = 60 / tempo_bpm
    # - Grid step = quarter_note_duration * (4 / division)
    #   e.g., division=16 means 1/16 notes, step = quarter/4
    quarter_note_duration = 60.0 / tempo_bpm
    grid_step = quarter_note_duration * (4.0 / division)
    
    # Copy the dataframe to avoid modifying the original
    result = notes_df.copy()
    
    # Quantize start times
    result['start_time'] = (
        np.round(result['start_time'] / grid_step) * grid_step
    )
    
    # Quantize end times
    result['end_time'] = (
        np.round(result['end_time'] / grid_step) * grid_step
    )
    
    # Ensure end_time is always after start_time
    # If quantization made them equal, add one grid step to end_time
    duration = result['end_time'] - result['start_time']
    too_short = duration <= 0
    result.loc[too_short, 'end_time'] = (
        result.loc[too_short, 'start_time'] + grid_step
    )
    
    # Apply minimum duration if specified
    if min_duration is not None:
        duration = result['end_time'] - result['start_time']
        too_short = duration < min_duration
        result.loc[too_short, 'end_time'] = (
            result.loc[too_short, 'start_time'] + min_duration
        )
    
    return result


def quantize_midi_file(
    midi_path: Union[str, Path],
    output_path: Union[str, Path],
    tempo_bpm: float = 120.0,
    division: int = 16,
) -> bool:
    """
    Quantize a MIDI file by adjusting note timings to a rhythmic grid.
    
    Loads a MIDI file, quantizes all note timings, and saves to a new file.
    
    Args:
        midi_path: Path to input MIDI file.
        output_path: Path for output quantized MIDI file.
        tempo_bpm: Tempo in beats per minute.
        division: Rhythmic division (16=sixteenth notes, 8=eighth notes, etc.).
        
    Returns:
        True if quantization succeeded, False if it failed.
        
    Example:
        >>> success = quantize_midi_file(
        ...     "vocals.mid",
        ...     "vocals_quantized.mid",
        ...     tempo_bpm=120,
        ...     division=16,
        ... )
        >>> if success:
        ...     print("MIDI file quantized successfully")
        
    Note:
        This function uses pretty_midi for MIDI I/O.
        Tempo changes in the MIDI file are ignored; a constant tempo is assumed.
        
    See Also:
        - :func:`quantize_notes` for DataFrame-based quantization
    """
    try:
        import pretty_midi
    except ImportError:
        print("Warning: pretty_midi not installed. Quantization skipped.")
        return False
    
    try:
        # Load MIDI file
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Calculate grid step
        quarter_note_duration = 60.0 / tempo_bpm
        grid_step = quarter_note_duration * (4.0 / division)
        
        # Quantize all notes in all instruments
        for instrument in midi.instruments:
            for note in instrument.notes:
                # Quantize start and end times
                quantized_start = round(note.start / grid_step) * grid_step
                quantized_end = round(note.end / grid_step) * grid_step
                
                # Ensure end > start
                if quantized_end <= quantized_start:
                    quantized_end = quantized_start + grid_step
                
                note.start = quantized_start
                note.end = quantized_end
        
        # Write quantized MIDI
        midi.write(str(output_path))
        return True
        
    except Exception as e:
        print(f"Warning: MIDI quantization failed: {e}")
        return False
