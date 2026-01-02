"""
MIDI parsing and manipulation module using pretty_midi.

This module provides the ``MIDIParser`` class for loading, parsing, and
manipulating MIDI files. It converts MIDI data into structured formats
suitable for analysis and alignment.

Key Features:
    - Load and parse MIDI files to structured data
    - Extract notes as pandas DataFrames
    - Get tempo maps and timing information
    - Align notes to time grids
    - Create MIDI from note lists

Integration:
    MIDI parsing occurs after transcription (Basic Pitch) and before
    alignment. The parsed notes are aligned to the beat grid from audio
    analysis.

Example:
    >>> parser = MIDIParser()
    >>> result = parser.parse("vocals.mid")
    >>> notes_df = result["notes_df"]
    >>> print(f"Found {len(notes_df)} notes")

See Also:
    - :mod:`didactic_engine.transcription` for MIDI generation
    - :mod:`didactic_engine.align` for beat alignment
    - :mod:`didactic_engine.features` for note statistics
"""

from pathlib import Path
from typing import List, Dict, Tuple, Any, Union
import pretty_midi
import pandas as pd
import numpy as np


class MIDIParser:
    """Parse and manipulate MIDI data using pretty_midi.
    
    Provides methods for loading MIDI files, extracting structured note
    data, and creating new MIDI from note lists.
    
    The parser is stateless—each method operates on the provided input
    without maintaining internal state.
    
    Example:
        >>> parser = MIDIParser()
        >>> 
        >>> # Parse MIDI file to DataFrame
        >>> result = parser.parse("transcription.mid")
        >>> notes_df = result["notes_df"]
        >>> tempo_map = result["tempo_map"]
        >>> 
        >>> # Create MIDI from notes
        >>> notes = [(0.0, 0.5, 60, 100), (0.5, 1.0, 62, 100)]
        >>> midi = parser.create_from_notes(notes, tempo=120)
    """

    def __init__(self):
        """Initialize the MIDI parser.
        
        The parser is stateless—no configuration needed.
        """
        pass

    def load(self, midi_path: Union[str, Path]) -> pretty_midi.PrettyMIDI:
        """Load a MIDI file into a PrettyMIDI object.

        Simple wrapper around PrettyMIDI constructor for consistency.

        Args:
            midi_path: Path to the MIDI file.

        Returns:
            PrettyMIDI object for further manipulation.

        Raises:
            IOError: If the file cannot be read.

        Example:
            >>> parser = MIDIParser()
            >>> pm = parser.load("song.mid")
            >>> print(f"Duration: {pm.get_end_time():.2f}s")
        """
        return pretty_midi.PrettyMIDI(str(midi_path))

    def parse(self, midi_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse a MIDI file and extract structured data.

        Loads a MIDI file and extracts notes, tempo map, and metadata
        into a structured dictionary with a pandas DataFrame for notes.

        Args:
            midi_path: Path to the MIDI file.

        Returns:
            Dictionary containing:
            
            - ``notes_df``: DataFrame with columns:
                - pitch: MIDI note number (0-127)
                - velocity: Note velocity (0-127)
                - start_s: Note start time in seconds
                - end_s: Note end time in seconds
                - dur_s: Note duration in seconds
                - instrument_index: Index of instrument in MIDI file
                - instrument_name: Instrument name (or "Instrument_N")
                - program: MIDI program number
                - is_drum: Whether instrument is drums
            
            - ``tempo_map``: List of dicts with:
                - time_s: Time of tempo change
                - tempo_bpm: New tempo in BPM
            
            - ``duration_s``: Total MIDI duration in seconds
            
            - ``total_notes``: Total number of notes

        Example:
            >>> parser = MIDIParser()
            >>> result = parser.parse("vocals.mid")
            >>> df = result["notes_df"]
            >>> print(df[["pitch", "start_s", "dur_s"]].head())
               pitch  start_s  dur_s
            0     60    0.000  0.250
            1     62    0.250  0.250

        Note:
            Empty MIDI files return an empty DataFrame with correct columns.

        See Also:
            - :meth:`parse_midi_object` for parsing PrettyMIDI objects
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
        """Parse a PrettyMIDI object and extract structured data.

        Same as :meth:`parse` but accepts an already-loaded PrettyMIDI
        object instead of a file path.

        Args:
            midi_data: PrettyMIDI object to parse.

        Returns:
            Dictionary with same structure as :meth:`parse`:
            notes_df, tempo_map, duration_s, total_notes.

        Example:
            >>> parser = MIDIParser()
            >>> pm = pretty_midi.PrettyMIDI("song.mid")
            >>> # ... modify pm ...
            >>> result = parser.parse_midi_object(pm)

        See Also:
            - :meth:`parse` for file-based parsing
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
        """Extract all notes from MIDI data as tuples.

        Simpler extraction format compared to :meth:`parse`. Returns
        notes as tuples sorted by start time.

        Args:
            midi_data: PrettyMIDI object.

        Returns:
            List of (start_time, end_time, pitch, velocity) tuples,
            sorted by start_time.

        Example:
            >>> parser = MIDIParser()
            >>> pm = parser.load("song.mid")
            >>> notes = parser.extract_notes(pm)
            >>> for start, end, pitch, vel in notes[:5]:
            ...     print(f"Note {pitch} at {start:.2f}s")

        See Also:
            - :meth:`parse` for DataFrame-based extraction
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
        """Get piano roll representation of MIDI data.

        Returns a 2D array where rows are MIDI pitches (0-127) and
        columns are time steps at the specified sampling frequency.

        Args:
            midi_data: PrettyMIDI object.
            fs: Sampling frequency in Hz. Default 100 gives 10ms resolution.

        Returns:
            2D numpy array of shape (128, time_steps) with velocity values.
            Zero indicates note off, non-zero indicates note on.

        Example:
            >>> parser = MIDIParser()
            >>> pm = parser.load("song.mid")
            >>> roll = parser.get_piano_roll(pm, fs=100)
            >>> print(f"Shape: {roll.shape}")  # (128, num_time_steps)

        Note:
            Piano rolls can be large for long MIDI files. Consider using
            a lower sampling frequency for memory efficiency.
        """
        return midi_data.get_piano_roll(fs=fs)

    def align_to_grid(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        grid_times: np.ndarray,
        quantize: bool = True,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Align MIDI notes to a time grid (e.g., beat or bar grid).

        Groups notes into grid segments based on their start times.
        Useful for bar-based analysis.

        Args:
            midi_data: PrettyMIDI object.
            grid_times: Array of grid boundary times in seconds. Each
                consecutive pair defines a segment.
            quantize: If True, sets note start times to segment start.
                Original start preserved in 'original_start' key.

        Returns:
            Dictionary mapping segment index to list of note dicts.
            Each note dict contains:
            - start: Note start time (quantized if quantize=True)
            - end: Note end time
            - pitch: MIDI pitch number
            - velocity: Note velocity
            - instrument: Instrument name
            - original_start: (only if quantize=True) Original start time

        Example:
            >>> parser = MIDIParser()
            >>> pm = parser.load("song.mid")
            >>> grid = np.array([0.0, 2.0, 4.0, 6.0])  # Bar boundaries
            >>> aligned = parser.align_to_grid(pm, grid)
            >>> print(f"Bar 0 has {len(aligned[0])} notes")

        Note:
            Notes are assigned to segments based on overlap—a note is
            included if it overlaps with the segment at all.

        See Also:
            - :func:`didactic_engine.align.align_notes_to_beats` for
              DataFrame-based alignment
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
        """Create a MIDI object from a list of notes.

        Convenience method for creating MIDI from extracted or generated
        note data.

        Args:
            notes: List of (start_time, end_time, pitch, velocity) tuples.
                Times are in seconds.
            program: MIDI program number (0-127). Default 0 is Acoustic
                Grand Piano. See General MIDI for instrument mapping.
            tempo: Tempo in BPM. Default 120.

        Returns:
            PrettyMIDI object ready for further manipulation or saving.

        Example:
            >>> parser = MIDIParser()
            >>> notes = [
            ...     (0.0, 0.5, 60, 100),  # C4
            ...     (0.5, 1.0, 62, 100),  # D4
            ...     (1.0, 1.5, 64, 100),  # E4
            ... ]
            >>> midi = parser.create_from_notes(notes, tempo=120)
            >>> midi.write("output.mid")

        Note:
            All notes are placed on a single instrument track. For
            multi-track MIDI, use PrettyMIDI directly.

        See Also:
            - :meth:`extract_notes` for extracting notes in this format
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
