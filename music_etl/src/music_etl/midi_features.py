"""
MIDI parsing and feature extraction with pretty_midi.
"""

from pathlib import Path
import pandas as pd
import pretty_midi


def parse_midi(midi_path: Path) -> dict:
    """
    Parse MIDI file and extract features.

    Args:
        midi_path: Path to MIDI file

    Returns:
        Dictionary containing tempo map and notes DataFrame
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # Extract tempo changes
    tempo_times, tempos = pm.get_tempo_changes()
    tempo_map = [
        {"time_s": float(t), "tempo_bpm": float(tempo)}
        for t, tempo in zip(tempo_times, tempos)
    ]

    # Extract notes from all instruments
    notes_data = []
    for inst_idx, instrument in enumerate(pm.instruments):
        for note in instrument.notes:
            notes_data.append({
                "instrument_index": inst_idx,
                "instrument_name": instrument.name,
                "program": instrument.program,
                "is_drum": instrument.is_drum,
                "pitch": note.pitch,
                "velocity": note.velocity,
                "start_s": note.start,
                "end_s": note.end,
                "dur_s": note.end - note.start,
            })

    notes_df = pd.DataFrame(notes_data)

    return {
        "midi_path": str(midi_path),
        "tempo_map": tempo_map,
        "notes_df": notes_df,
    }
