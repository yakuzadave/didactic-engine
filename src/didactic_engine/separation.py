"""
Stem separation module using Demucs.

This module provides the ``StemSeparator`` class for separating audio into
individual stems (vocals, drums, bass, other) using Facebook's Demucs model.
It wraps the Demucs CLI for reliable, production-ready separation.

Key Features:
    - Support for multiple Demucs models (htdemucs, htdemucs_ft, etc.)
    - Automatic stem discovery via filesystem glob
    - Graceful error handling with actionable error messages
    - Support for both file-based and array-based separation

Prerequisites:
    Demucs must be installed separately::
    
        pip install demucs
        
    Or for the latest version::
    
        pip install -U git+https://github.com/facebookresearch/demucs

Integration:
    Stem separation typically occurs early in the pipeline, after optional
    preprocessing. Each stem is then analyzed and transcribed independently.

Example:
    >>> separator = StemSeparator(model="htdemucs")
    >>> stems = separator.separate(Path("song.wav"), Path("output/stems"))
    >>> print(stems.keys())
    dict_keys(['vocals', 'drums', 'bass', 'other'])

See Also:
    - :mod:`didactic_engine.analysis` for stem analysis
    - :mod:`didactic_engine.transcription` for MIDI transcription
"""

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import soundfile as sf

from didactic_engine.subprocess_utils import run_checked


class StemSeparator:
    """Separate audio into stems using Demucs.
    
    Wraps the Demucs CLI to split a mix into individual stems. By default,
    produces four stems: vocals, drums, bass, and other.
    
    Model Options:
        - ``htdemucs``: Default hybrid transformer model (best quality)
        - ``htdemucs_ft``: Fine-tuned version (slightly better on some music)
        - ``mdx``: Alternative architecture
        - ``demucs``: Original Demucs model
    
    Device Options:
        - ``cpu``: Use CPU (slower but always available)
        - ``cuda``: Use NVIDIA GPU (much faster)
    
    Attributes:
        model: Demucs model name.
        device: Processing device ('cpu' or 'cuda').
        stem_names: Expected stem names for this model.
    
    Example:
        >>> separator = StemSeparator(model="htdemucs", device="cuda")
        >>> 
        >>> # Check if Demucs is available
        >>> if separator._check_demucs_available():
        ...     stems = separator.separate(input_path, output_dir)
        ...     for name, path in stems.items():
        ...         print(f"{name}: {path}")
    """

    def __init__(
        self,
        model: str = "htdemucs",
        device: str = "cpu",
        timeout_s: Optional[float] = None,
    ):
        """Initialize the stem separator.

        Args:
            model: Demucs model to use. Options:
                - ``htdemucs``: Hybrid transformer (default, best quality)
                - ``htdemucs_ft``: Fine-tuned hybrid transformer
                - ``mdx``: Music Demixing architecture
                - ``demucs``: Original convolutional model
            device: Processing device. Options:
                - ``cpu``: Use CPU (works everywhere, slower)
                - ``cuda``: Use NVIDIA GPU (requires CUDA, much faster)
                - ``cuda:0``, ``cuda:1``: Specific GPU device
        
        Example:
            >>> # Default CPU processing
            >>> separator = StemSeparator()
            
            >>> # GPU processing with fine-tuned model
            >>> separator = StemSeparator(model="htdemucs_ft", device="cuda")
        
        Note:
            The stem_names attribute is set to common Demucs outputs. Some
            models may produce different stems (e.g., 6-stem models).
        """
        self.model = model
        self.device = device
        self.timeout_s = timeout_s
        self.stem_names = ["vocals", "drums", "bass", "other"]

    def _check_demucs_available(self) -> bool:
        """Check if Demucs is available for use.

        Checks both CLI availability (via PATH) and Python module import.
        Either method is sufficient for separation to work.

        Returns:
            True if Demucs is available via CLI or Python import.
            False if Demucs cannot be found.

        Example:
            >>> separator = StemSeparator()
            >>> if not separator._check_demucs_available():
            ...     print("Please install Demucs: pip install demucs")

        Note:
            CLI availability is checked first since the :meth:`separate`
            method uses subprocess to call the demucs command.
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
        """Separate audio file into stems using Demucs CLI.

        Runs Demucs via subprocess and discovers the output stem WAV files.
        Stems are named according to their content (vocals, drums, bass, other).

        Args:
            audio_path: Path to input audio file. Supports WAV, MP3, FLAC,
                and other formats supported by Demucs.
            out_dir: Directory to save separated stems. Demucs creates
                subdirectories like ``<model>/<filename>/`` containing stems.

        Returns:
            Dictionary mapping stem names to their file paths. Keys are
            lowercase stem names (e.g., 'vocals', 'drums'). Values are
            pathlib.Path objects pointing to WAV files.
            
            Example return::
            
                {
                    'vocals': Path('output/htdemucs/song/vocals.wav'),
                    'drums': Path('output/htdemucs/song/drums.wav'),
                    'bass': Path('output/htdemucs/song/bass.wav'),
                    'other': Path('output/htdemucs/song/other.wav'),
                }

        Raises:
            RuntimeError: If Demucs is not installed, not on PATH, or if
                separation fails. Error message includes installation
                instructions.

        Example:
            >>> separator = StemSeparator()
            >>> try:
            ...     stems = separator.separate("song.wav", "output/stems")
            ...     print(f"Vocals: {stems['vocals']}")
            ... except RuntimeError as e:
            ...     print(f"Separation failed: {e}")

        Note:
            Stem discovery uses ``rglob("*.wav")`` to handle Demucs's nested
            output structure. Unknown stem names are preserved as-is in the
            returned dictionary.

        See Also:
            - :meth:`separate_audio_array` for numpy array input
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

        # Use centralized subprocess helper so timeouts and error messages
        # are consistent and include the invoked command.
        run_checked(cmd, timeout_s=self.timeout_s, tool_name="Demucs")

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
        """Separate a numpy audio array into stems.

        Convenience method that saves the audio to a temporary file, runs
        separation, and loads the results back as numpy arrays.

        Args:
            audio: Input audio array (1D mono or 2D stereo).
            sample_rate: Sample rate of the audio in Hz.
            out_dir: Directory to save separated stems. Also used for
                the temporary input file.

        Returns:
            Dictionary mapping stem names to numpy arrays. Each array
            has the same sample rate as the input.

        Raises:
            RuntimeError: If Demucs is not installed or separation fails.

        Example:
            >>> separator = StemSeparator()
            >>> audio, sr = ingester.load("song.wav")
            >>> stems = separator.separate_audio_array(audio, sr, "output")
            >>> vocals = stems['vocals']

        Note:
            A temporary file ``input_temp.wav`` is created in out_dir and
            deleted after separation. If separation fails, the temp file
            is still cleaned up.

        See Also:
            - :meth:`separate` for file-to-file separation
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
