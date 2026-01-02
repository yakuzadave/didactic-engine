"""
Export MIDI to ABC notation using music21.
"""

from pathlib import Path
import music21


def export_abc(midi_path: Path, output_path: Path) -> None:
    """
    Export MIDI file to ABC notation.

    Args:
        midi_path: Path to MIDI file
        output_path: Path to output ABC file
    """
    try:
        # Parse MIDI file
        score = music21.converter.parse(str(midi_path))

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to ABC format
        score.write("abc", fp=str(output_path))

    except Exception as e:
        # Log error but don't crash
        print(f"Warning: ABC export failed for {midi_path}: {e}")

        # Write error message to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"% ABC export failed: {e}\n")
