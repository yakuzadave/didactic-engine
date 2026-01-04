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

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Union, Optional

from didactic_engine.subprocess_utils import run_checked

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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

    def __init__(
        self,
        model_serialization: str = "tf",
        timeout_s: Optional[float] = None,
        keep_runs: Optional[int] = None,
    ):
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
        self.model_serialization = model_serialization
        self.timeout_s = timeout_s
        self.keep_runs = keep_runs
        self._check_available()
        self._supports_model_serialization = self._probe_model_serialization_support()

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

    def _probe_model_serialization_support(self) -> bool:
        """Probe if basic-pitch CLI supports --model-serialization flag.

        Some older versions of basic-pitch don't accept the --model-serialization
        flag. This method checks for support by running basic-pitch --help and
        looking for the flag in the help text.

        Returns:
            True if --model-serialization is supported, False otherwise.

        Note:
            Called automatically during initialization. If the flag is not
            supported, transcribe() will skip it to maintain compatibility.
        """
        try:
            result = subprocess.run(
                ["basic-pitch", "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Check if --model-serialization appears in help text
            return "--model-serialization" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # If we can't determine support, assume it's not supported
            # to maintain compatibility with older versions
            return False

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
            Basic Pitch is called with the ``--save-midi`` flag.
            Sonification (audio file generation) is disabled by default,
            so only MIDI output is generated.

            Each run writes into an isolated subdirectory under
            ``out_dir/runs/<stem>/<run_id>/`` to avoid stale MIDI pickup.
            The newest .mid from that run is copied to the canonical
            ``out_dir/<stem>.mid`` path.

        See Also:
            - :class:`didactic_engine.midi_parser.MIDIParser` for parsing
              the output
        """
        stem_wav = Path(stem_wav)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem_name = stem_wav.stem

        # Use temporary directory for Basic Pitch output to avoid accumulating
        # orphaned run directories. The temp dir is automatically cleaned up.
        import tempfile

        # Create temp directory with descriptive prefix for debugging
        temp_prefix = f"basic_pitch_{stem_name}_"
        temp_dir_obj = tempfile.TemporaryDirectory(prefix=temp_prefix)
        run_dir = Path(temp_dir_obj.name)

        try:
            # Run basic-pitch CLI
            cmd = ["basic-pitch"]

            # Prefer explicit backend selection. This is particularly useful on WSL where
            # TensorFlow GPU libraries may not be available, but ONNX Runtime GPU can be.
            # Only add flag if the CLI version supports it (compatibility with older versions).
            if self.model_serialization and self._supports_model_serialization:
                cmd.extend(["--model-serialization", self.model_serialization])

            cmd.extend([
                str(run_dir),
                str(stem_wav),
                "--save-midi",
            ])

            # Run via shared helper for consistent timeout/error reporting.
            # If the requested backend isn't supported by the installed CLI, fall back
            # to default behavior instead of failing the entire pipeline.
            try:
                run_checked(cmd, timeout_s=self.timeout_s, tool_name="Basic Pitch")
            except RuntimeError as exc:
                msg = str(exc)

                # If we attempted to use --model-serialization and the CLI rejected it
                # (some versions differ), retry once without it.
                if "--model-serialization" in msg and any(
                    k in msg.lower() for k in ["unrecognized", "unknown option", "no such option"]
                ):
                    fallback_cmd = []
                    skip_next = False
                    for i, c in enumerate(cmd):
                        if skip_next:
                            skip_next = False
                            continue
                        if c == "--model-serialization":
                            # Drop the flag itself and, if present and matching, its value.
                            if i + 1 < len(cmd) and cmd[i + 1] == self.model_serialization:
                                skip_next = True
                            continue
                        fallback_cmd.append(c)
                    run_checked(fallback_cmd, timeout_s=self.timeout_s, tool_name="Basic Pitch")
                else:
                    raise

            # Find the newest .mid file in output directory
            midi_files = list(run_dir.rglob("*.mid"))

            if not midi_files:
                raise RuntimeError(
                    f"No MIDI files found in {run_dir} after transcription"
                )

            # Get newest file by modification time
            newest_midi = max(midi_files, key=lambda p: p.stat().st_mtime)

            # Copy to canonical path before temp directory is cleaned up
            canonical_path = out_dir / f"{stem_name}.mid"

            if newest_midi != canonical_path:
                shutil.copy2(newest_midi, canonical_path)

            return canonical_path

        finally:
            # Ensure temporary directory is cleaned up
            # TemporaryDirectory auto-cleans on context exit, but we're using manual cleanup
            temp_dir_obj.cleanup()

    def _cleanup_run_dirs(self, run_root: Path) -> None:
        """Delete old Basic Pitch run directories, keeping the newest N."""
        if self.keep_runs is None:
            return
        try:
            keep_runs = int(self.keep_runs)
        except (TypeError, ValueError):
            return
        if keep_runs <= 0:
            return
        if not run_root.exists():
            return

        try:
            run_dirs = [p for p in run_root.iterdir() if p.is_dir()]
        except OSError:
            return

        if len(run_dirs) <= keep_runs:
            return

        run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for old_dir in run_dirs[keep_runs:]:
            try:
                shutil.rmtree(old_dir)
            except Exception as exc:
                logger.warning("Failed to remove old Basic Pitch run dir %s: %s", old_dir, exc)


# Legacy alias for backward compatibility
MIDITranscriber = BasicPitchTranscriber
