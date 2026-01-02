"""
MIDI transcription with Basic Pitch.
"""

import shutil
import subprocess
from pathlib import Path
import time


def transcribe_to_midi(wav_path: Path, midi_output_dir: Path, timeout: int | None = None) -> Path:
    """
    Transcribe audio to MIDI using Basic Pitch.

    Args:
        wav_path: Path to input WAV file
        midi_output_dir: Directory to save MIDI file
        timeout: Timeout in seconds. If None, calculated based on file size
                 (roughly 30s per MB, minimum 180s, maximum 900s).

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

    # Calculate timeout based on file size if not provided
    if timeout is None:
        file_size_mb = wav_path.stat().st_size / (1024 * 1024)
        # Roughly 30 seconds per MB, minimum 3 minutes, maximum 15 minutes
        timeout = max(180, min(900, int(file_size_mb * 30)))

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
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Basic Pitch failed: {e.stderr}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Basic Pitch timed out after {timeout} seconds") from e

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
