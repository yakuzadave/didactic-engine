"""
MIDI transcription module using Basic Pitch.

Transcribes audio to MIDI using Spotify's Basic Pitch model.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Union


class BasicPitchTranscriber:
    """Transcribe audio to MIDI using Basic Pitch CLI."""

    def __init__(self):
        """
        Initialize the Basic Pitch transcriber.

        Checks for basic-pitch CLI availability.
        """
        self._check_available()

    def _check_available(self) -> bool:
        """
        Check if basic-pitch CLI is available.

        Returns:
            True if basic-pitch is available.

        Raises:
            RuntimeError: If basic-pitch is not installed.
        """
        if shutil.which("basic-pitch") is None:
            raise RuntimeError(
                "basic-pitch command not found. Please install Basic Pitch:\n"
                "  pip install basic-pitch\n"
                "and ensure it's on your PATH."
            )
        return True

    def transcribe(
        self,
        stem_wav: Union[str, Path],
        out_dir: Union[str, Path],
    ) -> Path:
        """
        Transcribe audio to MIDI using Basic Pitch CLI.

        Args:
            stem_wav: Path to input WAV file.
            out_dir: Output directory for MIDI file.

        Returns:
            Path to the generated MIDI file.

        Raises:
            RuntimeError: If transcription fails.
        """
        stem_wav = Path(stem_wav)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run basic-pitch CLI
        cmd = [
            "basic-pitch",
            str(out_dir),
            str(stem_wav),
            "--save-midi",
            "--no-sonify",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Basic Pitch transcription failed:\n{e.stderr}"
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "basic-pitch command not found. Please install Basic Pitch:\n"
                "  pip install basic-pitch"
            ) from e

        # Find the newest .mid file in output directory
        midi_files = list(out_dir.rglob("*.mid"))

        if not midi_files:
            raise RuntimeError(
                f"No MIDI files found in {out_dir} after transcription"
            )

        # Get newest file by modification time
        newest_midi = max(midi_files, key=lambda p: p.stat().st_mtime)

        # Copy to canonical path
        stem_name = stem_wav.stem
        canonical_path = out_dir / f"{stem_name}.mid"

        if newest_midi != canonical_path:
            shutil.copy2(newest_midi, canonical_path)

        return canonical_path


# Legacy alias for backward compatibility
MIDITranscriber = BasicPitchTranscriber
