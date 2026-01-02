"""
MIDI transcription with Basic Pitch.
"""

import shutil
import subprocess
from pathlib import Path
import time


def transcribe_to_midi(wav_path: Path, midi_output_dir: Path) -> Path:
    """
    Transcribe audio to MIDI using Basic Pitch.

    Args:
        wav_path: Path to input WAV file
        midi_output_dir: Directory to save MIDI file

    Returns:
        Path to generated MIDI file

    Raises:
        RuntimeError: If basic-pitch is not found or fails
    """
    # Check if basic-pitch is available
    if shutil.which("basic-pitch") is None:
        raise RuntimeError(
            "basic-pitch command not found. Please install Basic Pitch:\n"
            "  pip install basic-pitch\n"
            "and ensure it's on your PATH."
        )

    midi_output_dir.mkdir(parents=True, exist_ok=True)

    # Run basic-pitch
    cmd = [
        "basic-pitch",
        str(midi_output_dir),
        str(wav_path),
        "--save-midi",
        "--no-onset",
        "--no-contour",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=300
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Basic Pitch failed: {e.stderr}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Basic Pitch timed out after 5 minutes") from e

    # Find newest .mid file in output directory
    midi_files = list(midi_output_dir.rglob("*.mid"))

    if not midi_files:
        raise RuntimeError(f"No MIDI files found in {midi_output_dir} after transcription")

    # Get newest file by modification time
    newest_midi = max(midi_files, key=lambda p: p.stat().st_mtime)

    # Copy to canonical location
    stem_name = wav_path.stem
    canonical_path = midi_output_dir / f"{stem_name}.mid"

    if newest_midi != canonical_path:
        shutil.copy2(newest_midi, canonical_path)

    return canonical_path
