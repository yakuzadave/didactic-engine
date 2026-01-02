"""
Export MIDI to ABC notation using music21.

Provides utilities to convert MIDI files to ABC notation format,
which is a text-based music notation standard.
"""

import os
from typing import Optional

try:
    import music21
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False


def export_abc(
    midi_path: str,
    output_path: str,
    title: Optional[str] = None,
) -> bool:
    """
    Export MIDI file to ABC notation.

    Args:
        midi_path: Path to input MIDI file.
        output_path: Path to output ABC file.
        title: Optional title for the ABC notation.

    Returns:
        True if export was successful, False otherwise.
    """
    if not MUSIC21_AVAILABLE:
        print("Warning: music21 is not installed. ABC export skipped.")
        print("Install with: pip install music21")
        _write_error_file(output_path, "music21 not installed")
        return False

    try:
        # Parse MIDI file
        score = music21.converter.parse(midi_path)

        # Set title if provided
        if title:
            score.metadata = music21.metadata.Metadata()
            score.metadata.title = title

        # Create output directory
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Write to ABC format
        score.write("abc", fp=output_path)

        return True

    except Exception as e:
        print(f"Warning: ABC export failed for {midi_path}: {e}")
        _write_error_file(output_path, str(e))
        return False


def export_abc_from_notes(
    notes: list,
    output_path: str,
    title: str = "Untitled",
    tempo: float = 120.0,
    key: str = "C",
    time_signature: str = "4/4",
) -> bool:
    """
    Export a list of notes to ABC notation.

    Args:
        notes: List of tuples (start_time, end_time, pitch, velocity).
        output_path: Path to output ABC file.
        title: Title for the ABC notation.
        tempo: Tempo in BPM.
        key: Key signature (e.g., "C", "G", "Am").
        time_signature: Time signature (e.g., "4/4", "3/4").

    Returns:
        True if export was successful, False otherwise.
    """
    if not MUSIC21_AVAILABLE:
        print("Warning: music21 is not installed. ABC export skipped.")
        _write_error_file(output_path, "music21 not installed")
        return False

    try:
        # Create a music21 stream
        stream = music21.stream.Stream()

        # Set metadata
        stream.metadata = music21.metadata.Metadata()
        stream.metadata.title = title

        # Set tempo
        mm = music21.tempo.MetronomeMark(number=tempo)
        stream.append(mm)

        # Set time signature
        ts_parts = time_signature.split("/")
        ts = music21.meter.TimeSignature(f"{ts_parts[0]}/{ts_parts[1]}")
        stream.append(ts)

        # Set key
        ks = music21.key.Key(key)
        stream.append(ks)

        # Add notes
        for start, end, pitch, velocity in notes:
            duration = end - start
            n = music21.note.Note(pitch)
            n.quarterLength = duration * tempo / 60  # Convert seconds to quarter notes
            n.volume.velocity = velocity
            stream.insert(start * tempo / 60, n)

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write to ABC format
        stream.write("abc", fp=output_path)

        return True

    except Exception as e:
        print(f"Warning: ABC export from notes failed: {e}")
        _write_error_file(output_path, str(e))
        return False


def _write_error_file(output_path: str, error_message: str) -> None:
    """
    Write an error message to the output file.

    Args:
        output_path: Path to output file.
        error_message: Error message to write.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"% ABC export failed: {error_message}\n")
        f.write("% Please check the input MIDI file or install music21.\n")
