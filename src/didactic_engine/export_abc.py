"""
Export MIDI to ABC notation using music21.

This module provides functions for converting MIDI files to ABC notation,
a text-based music notation format. ABC notation is useful for sharing
music in a simple, human-readable format.

Key Functions:
    - :func:`export_abc`: Convert MIDI file to ABC notation
    - :func:`export_abc_from_notes`: Convert note list to ABC notation

Prerequisites:
    music21 must be installed for ABC export to work::
    
        pip install music21

    If music21 is not installed, functions return False and write an
    error message to the output file.

Integration:
    ABC export is typically the final step for generating human-readable
    notation from transcribed MIDI files.

Example:
    >>> success = export_abc("vocals.mid", "vocals.abc", title="Vocal Line")
    >>> if not success:
    ...     print("ABC export failed (music21 not installed?)")

Limitations:
    - Works best with monophonic or simple polyphonic content
    - Complex rhythms may not be perfectly represented
    - Drum tracks produce poor results

See Also:
    - :mod:`didactic_engine.export_md` for Markdown reports
    - :mod:`didactic_engine.transcription` for MIDI generation
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
    """Export MIDI file to ABC notation.

    Converts a MIDI file to ABC notation using music21. Creates output
    directory if needed.

    Args:
        midi_path: Path to input MIDI file.
        output_path: Path for output ABC file.
        title: Optional title for the ABC notation. If None, music21
            uses the MIDI file's embedded title or generates one.

    Returns:
        True if export was successful, False if it failed (usually
        because music21 is not installed or MIDI parsing failed).

    Example:
        >>> success = export_abc("vocals.mid", "output/vocals.abc")
        >>> if success:
        ...     print("ABC notation created!")

    Note:
        If music21 is not installed, an error message is written to the
        output file explaining how to install it. The function still
        returns False in this case.

    See Also:
        - :func:`export_abc_from_notes` for note-list input
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
    """Export a list of notes to ABC notation.

    Creates ABC notation from a list of note tuples, useful when you
    have note data but not a MIDI file.

    Args:
        notes: List of (start_time, end_time, pitch, velocity) tuples.
            Times are in seconds. Pitch is MIDI note number (0-127).
        output_path: Path for output ABC file.
        title: Title for the ABC notation.
        tempo: Tempo in BPM. Used for rhythmic conversion.
        key: Key signature (e.g., "C", "G", "Am", "F#m").
        time_signature: Time signature (e.g., "4/4", "3/4", "6/8").

    Returns:
        True if export was successful, False if it failed.

    Example:
        >>> notes = [
        ...     (0.0, 0.5, 60, 100),  # C4
        ...     (0.5, 1.0, 62, 100),  # D4
        ...     (1.0, 1.5, 64, 100),  # E4
        ... ]
        >>> success = export_abc_from_notes(
        ...     notes, "melody.abc",
        ...     title="Simple Melody",
        ...     tempo=120,
        ...     key="C"
        ... )

    Note:
        Duration conversion uses: quarter_length = duration_s * tempo / 60
        This may not perfectly represent complex rhythms.

    See Also:
        - :func:`export_abc` for MIDI file input
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
    """Write an error message to the output file.

    Internal helper for creating placeholder ABC files when export fails.

    Args:
        output_path: Path for output file.
        error_message: Error message to include in the file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"% ABC export failed: {error_message}\n")
        f.write("% Please check the input MIDI file or install music21.\n")
