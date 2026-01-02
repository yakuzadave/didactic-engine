"""
MIDI transcription module using Basic Pitch.

This module provides the ``BasicPitchTranscriber`` class for converting
audio to MIDI using Spotify's Basic Pitch model. Basic Pitch is a neural
network model that transcribes audio to MIDI with reasonable accuracy,
especially for monophonic and simple polyphonic content.

Key Features:
    - Audio-to-MIDI transcription via CLI
    - Automatic MIDI file discovery and canonicalization
    - Graceful error handling with installation instructions

Prerequisites:
    Basic Pitch must be installed separately::
    
        pip install basic-pitch

Integration:
    Transcription typically occurs after stem separation, converting each
    stem's audio to MIDI for further analysis and alignment.

Example:
    >>> transcriber = BasicPitchTranscriber()
    >>> midi_path = transcriber.transcribe(
    ...     Path("stems/vocals.wav"),
    ...     Path("output/midi")
    ... )
    >>> print(f"MIDI saved to: {midi_path}")

Limitations:
    - Works best on isolated instruments (use after stem separation)
    - May struggle with complex polyphonic content
    - Timing accuracy depends on input audio quality
    - Does not detect time signature or key signature

See Also:
    - :mod:`didactic_engine.midi_parser` for parsing the output MIDI
    - :mod:`didactic_engine.align` for aligning notes to beats
"""

import shutil
import subprocess
from pathlib import Path
from typing import Union


class BasicPitchTranscriber:
    """Transcribe audio to MIDI using Basic Pitch CLI.
    
    Wraps the Basic Pitch command-line tool to convert audio files to MIDI.
    The transcriber automatically handles file naming and discovery.
    
    Workflow:
        1. Check Basic Pitch availability at initialization
        2. Run CLI with --save-midi flag
        3. Find newest .mid file in output directory
        4. Copy to canonical location (stem_name.mid)
    
    Example:
        >>> transcriber = BasicPitchTranscriber()
        >>> 
        >>> # Transcribe vocals stem
        >>> midi_path = transcriber.transcribe(
        ...     "stems/vocals.wav",
        ...     "output/midi"
        ... )
        >>> print(f"Created: {midi_path}")
        Created: output/midi/vocals.mid
    
    Note:
        Basic Pitch outputs MIDI files with unpredictable names. The
        transcriber handles this by finding the newest .mid file and
        copying it to a canonical location.
    """

    def __init__(self):
        """Initialize the Basic Pitch transcriber.

        Checks for basic-pitch CLI availability at initialization. This
        provides early failure if the tool is not installed.

        Raises:
            RuntimeError: If basic-pitch command is not found on PATH.
                Error message includes installation instructions.

        Example:
            >>> try:
            ...     transcriber = BasicPitchTranscriber()
            ... except RuntimeError as e:
            ...     print(f"Setup required: {e}")
        """
        self._check_available()

    def _check_available(self) -> bool:
        """Check if basic-pitch CLI is available.

        Verifies that the basic-pitch command exists on the system PATH.

        Returns:
            True if basic-pitch is available.

        Raises:
            RuntimeError: If basic-pitch is not installed, with
                installation instructions in the error message.

        Note:
            Called automatically during initialization. You typically
            don't need to call this directly.
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
        """Transcribe audio to MIDI using Basic Pitch CLI.

        Runs Basic Pitch on the input WAV file and saves the resulting
        MIDI file to the output directory with a canonical name based
        on the input filename.

        Args:
            stem_wav: Path to input WAV file. Should be a single stem
                (e.g., vocals, bass) for best results. Full mixes may
                produce poor transcriptions.
            out_dir: Output directory for MIDI file. Created if it
                doesn't exist.

        Returns:
            Path to the generated MIDI file. The file is named after
            the input file (e.g., 'vocals.wav' -> 'vocals.mid').

        Raises:
            RuntimeError: If transcription fails (subprocess error) or
                if no MIDI file is found after transcription.

        Example:
            >>> transcriber = BasicPitchTranscriber()
            >>> midi_path = transcriber.transcribe(
            ...     "stems/bass.wav",
            ...     "output/midi"
            ... )
            >>> print(midi_path)
            output/midi/bass.mid

        Note:
            Basic Pitch is called with ``--save-midi --no-sonify`` flags.
            The ``--no-sonify`` flag prevents generation of audio files
            (we only want MIDI output).

            If Basic Pitch creates a MIDI file with a different name than
            expected, the newest .mid file in the output directory is
            copied to the canonical location.

        See Also:
            - :class:`didactic_engine.midi_parser.MIDIParser` for parsing
              the output
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
