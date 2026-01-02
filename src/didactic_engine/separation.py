"""
Stem separation module using Demucs.

Separates audio into individual stems (vocals, drums, bass, other).
"""

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import soundfile as sf


class StemSeparator:
    """Separate audio into stems using Demucs."""

    def __init__(self, model: str = "htdemucs", device: str = "cpu"):
        """
        Initialize the stem separator.

        Args:
            model: Demucs model to use (htdemucs, htdemucs_ft, etc.)
            device: Device to use for separation (cpu, cuda)
        """
        self.model = model
        self.device = device
        self.stem_names = ["vocals", "drums", "bass", "other"]

    def _check_demucs_available(self) -> bool:
        """
        Check if Demucs is available.

        Returns:
            True if Demucs CLI is available, False otherwise.
        """
        # Check CLI availability
        if shutil.which("demucs") is not None:
            return True

        # Check Python module availability
        try:
            import demucs
            return True
        except ImportError:
            return False

    def separate(
        self,
        audio_path: Union[str, Path],
        out_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """
        Separate audio into stems.

        Args:
            audio_path: Path to input audio file.
            out_dir: Directory to save separated stems.

        Returns:
            Dictionary mapping stem names to WAV file paths.

        Raises:
            RuntimeError: If Demucs is not installed or separation fails.
        """
        audio_path = Path(audio_path)
        out_dir = Path(out_dir)

        if not self._check_demucs_available():
            raise RuntimeError(
                "Demucs is not installed or not available on PATH.\n"
                "Please install Demucs:\n"
                "  pip install demucs\n"
                "Or for the latest version:\n"
                "  pip install -U git+https://github.com/facebookresearch/demucs\n"
                "Make sure the 'demucs' command is available in your PATH."
            )

        out_dir.mkdir(parents=True, exist_ok=True)

        # Run Demucs separation
        cmd = [
            "demucs",
            "-n", self.model,
            "--device", self.device,
            "-o", str(out_dir),
            str(audio_path),
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
                f"Demucs separation failed:\n{e.stderr}"
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "Demucs command not found. Please install Demucs:\n"
                "  pip install demucs"
            ) from e

        # Discover all WAV files in output directory using rglob
        wav_files = list(out_dir.rglob("*.wav"))

        if not wav_files:
            raise RuntimeError(
                f"No WAV files found in {out_dir} after Demucs separation"
            )

        # Build dictionary keyed by canonical stem names
        stems: Dict[str, Path] = {}
        canonical_names = set(self.stem_names)

        for wav_path in wav_files:
            stem_name = wav_path.stem.lower()

            # Check if it matches a canonical stem name
            if stem_name in canonical_names:
                stems[stem_name] = wav_path
            elif stem_name == "no_vocals":
                # Map no_vocals to accompaniment or other
                stems["accompaniment"] = wav_path
            else:
                # Use filename as stem name
                stems[stem_name] = wav_path

        return stems

    def separate_audio_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        out_dir: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """
        Separate audio array into stems.

        This is a convenience method that saves the audio to a temporary
        file, runs separation, and loads the results back.

        Args:
            audio: Input audio array (1D mono or 2D stereo).
            sample_rate: Sample rate of the audio.
            out_dir: Directory to save separated stems.

        Returns:
            Dictionary mapping stem names to audio arrays.

        Raises:
            RuntimeError: If Demucs is not installed or separation fails.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save input audio to temporary file
        input_path = out_dir / "input_temp.wav"
        sf.write(str(input_path), audio, sample_rate)

        try:
            # Run separation
            stem_paths = self.separate(input_path, out_dir)

            # Load separated stems
            stems: Dict[str, np.ndarray] = {}
            for stem_name, stem_path in stem_paths.items():
                stem_audio, _ = sf.read(str(stem_path))
                stems[stem_name] = stem_audio

            return stems

        finally:
            # Clean up temporary input file
            if input_path.exists():
                input_path.unlink()
