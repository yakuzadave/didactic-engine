"""
MIDI parsing and manipulation module using pretty_midi.

Provides utilities for parsing, analyzing, and modifying MIDI data.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Any, Union
import pretty_midi
import pandas as pd
import numpy as np


class MIDIParser:
    """Parse and manipulate MIDI data using pretty_midi."""

    def __init__(self):
        """Initialize the MIDI parser."""
        pass

    def load(self, midi_path: Union[str, Path]) -> pretty_midi.PrettyMIDI:
        """
        Load a MIDI file.

        Args:
            midi_path: Path to the MIDI file.

        Returns:
            PrettyMIDI object.
        """
        return pretty_midi.PrettyMIDI(str(midi_path))

    def parse(self, midi_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse MIDI file and extract structured data.

        Args:
            midi_path: Path to the MIDI file.

        Returns:
            Dictionary containing:
            - notes_df: DataFrame with columns: pitch, velocity, start_s, end_s,
                        dur_s, instrument_index, instrument_name, program, is_drum
            - tempo_map: List of dicts with time_s and tempo_bpm
            - duration_s: Total duration in seconds
            - total_notes: Total number of notes
        """
        midi_path = Path(midi_path)
        pm = pretty_midi.PrettyMIDI(str(midi_path))

        # Extract tempo map
        tempo_times, tempos = pm.get_tempo_changes()
        tempo_map = [
            {"time_s": float(t), "tempo_bpm": float(tempo)}
            for t, tempo in zip(tempo_times, tempos)
        ]

        # Extract notes into a list of dicts
        notes_data = []
        for inst_idx, instrument in enumerate(pm.instruments):
            for note in instrument.notes:
                notes_data.append({
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "start_s": float(note.start),
                    "end_s": float(note.end),
                    "dur_s": float(note.end - note.start),
                    "instrument_index": inst_idx,
                    "instrument_name": instrument.name or f"Instrument_{inst_idx}",
                    "program": instrument.program,
                    "is_drum": instrument.is_drum,
                })

        # Create DataFrame
        notes_df = pd.DataFrame(notes_data)

        return {
            "notes_df": notes_df,
            "tempo_map": tempo_map,
            "duration_s": float(pm.get_end_time()),
            "total_notes": len(notes_data),
        }

    def parse_midi_object(self, midi_data: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        """
        Parse a PrettyMIDI object and extract structured data.

        Args:
            midi_data: PrettyMIDI object.

        Returns:
            Dictionary containing notes_df, tempo_map, etc.
        """
        # Extract tempo map
        tempo_times, tempos = midi_data.get_tempo_changes()
        tempo_map = [
            {"time_s": float(t), "tempo_bpm": float(tempo)}
            for t, tempo in zip(tempo_times, tempos)
        ]

        # Extract notes
        notes_data = []
        for inst_idx, instrument in enumerate(midi_data.instruments):
            for note in instrument.notes:
                notes_data.append({
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "start_s": float(note.start),
                    "end_s": float(note.end),
                    "dur_s": float(note.end - note.start),
                    "instrument_index": inst_idx,
                    "instrument_name": instrument.name or f"Instrument_{inst_idx}",
                    "program": instrument.program,
                    "is_drum": instrument.is_drum,
                })

        notes_df = pd.DataFrame(notes_data)

        return {
            "notes_df": notes_df,
            "tempo_map": tempo_map,
            "duration_s": float(midi_data.get_end_time()),
            "total_notes": len(notes_data),
        }

    def extract_notes(
        self, midi_data: pretty_midi.PrettyMIDI
    ) -> List[Tuple[float, float, int, int]]:
        """
        Extract all notes from MIDI data.

        Args:
            midi_data: PrettyMIDI object.

        Returns:
            List of tuples (start_time, end_time, pitch, velocity).
        """
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append((note.start, note.end, note.pitch, note.velocity))

        # Sort by start time
        notes.sort(key=lambda x: x[0])
        return notes

    def get_piano_roll(
        self, midi_data: pretty_midi.PrettyMIDI, fs: int = 100
    ) -> np.ndarray:
        """
        Get piano roll representation of MIDI data.

        Args:
            midi_data: PrettyMIDI object.
            fs: Sampling frequency for piano roll.

        Returns:
            Piano roll array (128 notes x time steps).
        """
        return midi_data.get_piano_roll(fs=fs)

    def align_to_grid(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        grid_times: np.ndarray,
        quantize: bool = True,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Align MIDI notes to a time grid (e.g., beat or bar grid).

        Args:
            midi_data: PrettyMIDI object.
            grid_times: Array of grid boundary times in seconds.
            quantize: Whether to quantize note start times to nearest grid point.

        Returns:
            Dictionary mapping grid index to list of notes in that segment.
        """
        aligned_notes: Dict[int, List[Dict[str, Any]]] = {}

        # Extract all notes
        all_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_notes.append({
                    "start": note.start,
                    "end": note.end,
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "instrument": instrument.name,
                })

        # Assign notes to grid segments
        for i in range(len(grid_times) - 1):
            segment_start = grid_times[i]
            segment_end = grid_times[i + 1]
            aligned_notes[i] = []

            for note in all_notes:
                note_start = note["start"]
                note_end = note["end"]

                # Check if note overlaps with this segment
                if note_start < segment_end and note_end > segment_start:
                    if quantize:
                        note_copy = note.copy()
                        note_copy["start"] = segment_start
                        note_copy["original_start"] = note_start
                        aligned_notes[i].append(note_copy)
                    else:
                        aligned_notes[i].append(note)

        return aligned_notes

    def create_from_notes(
        self,
        notes: List[Tuple[float, float, int, int]],
        program: int = 0,
        tempo: float = 120.0,
    ) -> pretty_midi.PrettyMIDI:
        """
        Create MIDI data from note list.

        Args:
            notes: List of tuples (start_time, end_time, pitch, velocity).
            program: MIDI program number for instrument.
            tempo: Tempo in BPM.

        Returns:
            PrettyMIDI object.
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=program)

        for start, end, pitch, velocity in notes:
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=end,
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)
        return midi
