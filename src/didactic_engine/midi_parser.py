"""
MIDI parsing and manipulation module using pretty_midi.

Provides utilities for parsing, analyzing, and modifying MIDI data.
"""

from typing import List, Dict, Tuple, Optional, Any
import pretty_midi
import numpy as np


class MIDIParser:
    """Parse and manipulate MIDI data using pretty_midi."""

    def __init__(self):
        """Initialize the MIDI parser."""
        pass

    def load(self, midi_path: str) -> pretty_midi.PrettyMIDI:
        """
        Load a MIDI file.

        Args:
            midi_path: Path to the MIDI file.

        Returns:
            PrettyMIDI object.
        """
        return pretty_midi.PrettyMIDI(midi_path)

    def parse(self, midi_data: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        """
        Parse MIDI data and extract information.

        Args:
            midi_data: PrettyMIDI object.

        Returns:
            Dictionary containing parsed MIDI information.
        """
        info = {
            "tempo_changes": [],
            "time_signature_changes": [],
            "key_signature_changes": [],
            "instruments": [],
            "total_notes": 0,
            "duration": midi_data.get_end_time(),
        }

        # Extract tempo changes
        tempo_times, tempos = midi_data.get_tempo_changes()
        info["tempo_changes"] = [
            {"time": float(t), "tempo": float(tempo)}
            for t, tempo in zip(tempo_times, tempos)
        ]

        # Extract time signatures
        for ts in midi_data.time_signature_changes:
            info["time_signature_changes"].append({
                "time": float(ts.time),
                "numerator": ts.numerator,
                "denominator": ts.denominator,
            })

        # Extract key signatures
        for ks in midi_data.key_signature_changes:
            info["key_signature_changes"].append({
                "time": float(ks.time),
                "key_number": ks.key_number,
            })

        # Extract instrument information
        for instrument in midi_data.instruments:
            inst_info = {
                "program": instrument.program,
                "is_drum": instrument.is_drum,
                "name": instrument.name,
                "notes": [],
            }
            
            for note in instrument.notes:
                inst_info["notes"].append({
                    "pitch": note.pitch,
                    "start": float(note.start),
                    "end": float(note.end),
                    "velocity": note.velocity,
                })
                info["total_notes"] += 1
            
            info["instruments"].append(inst_info)

        return info

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
        aligned_notes = {}
        
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
                    # Optionally quantize to grid
                    if quantize:
                        # Snap start time to nearest grid point
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
