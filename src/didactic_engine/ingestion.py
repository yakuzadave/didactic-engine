"""
WAV file ingestion module.

This module provides the ``WAVIngester`` class for loading, validating, and
optionally resampling WAV audio files. It serves as the entry point for audio
data into the didactic-engine pipeline.

Key Features:
    - Load WAV files with automatic stereo-to-mono conversion
    - Optional resampling to a target sample rate
    - Validation of audio data (NaN/Inf detection, float type verification)
    - Save processed audio back to disk

Example:
    >>> ingester = WAVIngester(sample_rate=22050)
    >>> audio, sr = ingester.load("path/to/audio.wav")
    >>> if ingester.validate(audio, sr):
    ...     print(f"Loaded {len(audio)} samples at {sr} Hz")

See Also:
    - :mod:`didactic_engine.preprocessing` for further audio processing
    - :mod:`didactic_engine.analysis` for feature extraction
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import soundfile as sf
import librosa


class WAVIngester:
    """Load and validate WAV audio files for pipeline processing.
    
    This class provides methods to ingest WAV files, converting them to
    mono numpy arrays suitable for downstream analysis. It handles:
    
    - Stereo to mono conversion (averaging channels)
    - Optional resampling to a target sample rate
    - Data validation (type, range, NaN/Inf checking)
    - Audio file output
    
    Attributes:
        sample_rate: Target sample rate for resampling, or None to keep original.
    
    Example:
        >>> # Load audio at original sample rate
        >>> ingester = WAVIngester()
        >>> audio, sr = ingester.load("song.wav")
        
        >>> # Load and resample to 22050 Hz
        >>> ingester = WAVIngester(sample_rate=22050)
        >>> audio, sr = ingester.load("song.wav")
        >>> assert sr == 22050
    """

    def __init__(self, sample_rate: Optional[int] = None):
        """Initialize the WAV ingester.

        Args:
            sample_rate: Target sample rate for resampling. If None, the 
                original sample rate of the audio file is preserved. Common
                values are 44100 (CD quality), 22050 (analysis), or 16000
                (speech processing).
        
        Example:
            >>> ingester = WAVIngester()  # Keep original sample rate
            >>> ingester = WAVIngester(sample_rate=22050)  # Resample to 22050 Hz
        """
        self.sample_rate = sample_rate

    def load(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load a WAV file and return mono audio as a numpy array.

        Reads an audio file using soundfile, automatically converts stereo
        to mono by averaging channels, and optionally resamples to the
        target sample rate specified during initialization.

        Args:
            file_path: Path to the WAV file. Supports both string paths and
                pathlib.Path objects. The file must exist and be readable.

        Returns:
            A tuple containing:
                - audio (np.ndarray): 1D float32 array of audio samples,
                  normalized to range [-1.0, 1.0]
                - sample_rate (int): Sample rate in Hz (either original or
                  resampled)

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is corrupt or in an unsupported audio format.
            MemoryError: If insufficient memory is available to load the file.
            PermissionError: If the file cannot be read due to permissions.
            IOError: If an I/O error occurs (disk full, network issues, etc.).

        Example:
            >>> ingester = WAVIngester(sample_rate=22050)
            >>> audio, sr = ingester.load("examples/test.wav")
            >>> print(f"Shape: {audio.shape}, SR: {sr}")
            Shape: (220500,), SR: 22050

        Note:
            The returned audio is always mono (1D). For stereo files, channels
            are averaged. For multi-channel files (>2), all channels are
            averaged together.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            import time
            import logging
            logger = logging.getLogger(__name__)

            # Load audio file using soundfile first to get native sample rate
            t0 = time.time()
            audio, native_sr = sf.read(str(file_path))
            t1 = time.time()
            logger.debug(f"soundfile.read took {t1-t0:.3f}s for {file_path.name}")

            # Convert to mono if stereo (soundfile returns (samples, channels))
            if audio.ndim == 2:
                t_mono_start = time.time()
                audio = np.mean(audio, axis=1)
                logger.debug(f"Mono conversion took {time.time()-t_mono_start:.3f}s")

            # Ensure float type
            audio = audio.astype(np.float32)

            # Resample if target sample rate is specified and differs
            if self.sample_rate is not None and self.sample_rate != native_sr:
                t_resample_start = time.time()
                audio = librosa.resample(
                    audio, orig_sr=native_sr, target_sr=self.sample_rate
                )
                t_resample_end = time.time()
                logger.debug(f"Resampling {native_sr}Hz -> {self.sample_rate}Hz took {t_resample_end-t_resample_start:.3f}s")
                return audio, self.sample_rate

            return audio, native_sr

        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is (already handled above, but be defensive)
            raise
        except sf.LibsndfileError as e:
            # Corrupt or unsupported audio format
            raise ValueError(
                f"Corrupt or invalid audio file {file_path}: {e}. "
                f"Check that the file is a valid audio format."
            ) from e
        except MemoryError as e:
            # File too large for available memory
            file_size_mb = file_path.stat().st_size / 1e6 if file_path.exists() else 0
            raise MemoryError(
                f"Insufficient memory to load {file_path} ({file_size_mb:.1f} MB). "
                f"Try reducing the sample rate or closing other applications."
            ) from e
        except PermissionError as e:
            # Permission denied on file
            raise PermissionError(
                f"Permission denied reading audio file {file_path}. "
                f"Check file permissions."
            ) from e
        except (OSError, IOError) as e:
            # I/O errors (disk full, network issues, etc.)
            raise IOError(
                f"I/O error reading audio file {file_path}: {e}. "
                f"Check disk space and file accessibility."
            ) from e
        # Let other exceptions propagate with context

    def validate(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Validate that audio data is suitable for processing.

        Performs comprehensive validation of audio data to ensure it meets
        the requirements for downstream processing. Checks include:
        
        - Non-None and non-empty array
        - Correct type (numpy ndarray)
        - Floating-point dtype
        - No NaN or Inf values
        - Positive sample rate

        Args:
            audio: Audio data array. Should be a 1D numpy array of floats
                with values in range [-1.0, 1.0] (though not strictly enforced).
            sample_rate: Sample rate in Hz. Must be positive.

        Returns:
            True if the audio data passes all validation checks, False otherwise.
            This method never raises exceptions; it returns False for invalid data.

        Example:
            >>> ingester = WAVIngester()
            >>> audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            >>> ingester.validate(audio, 44100)
            True
            
            >>> bad_audio = np.array([np.nan, 0.2])
            >>> ingester.validate(bad_audio, 44100)
            False

        Note:
            This method does not check for audio quality issues like clipping
            or DC offset—only structural validity. Use preprocessing for
            audio quality improvements.
        """
        # Check if audio is None or empty
        if audio is None:
            return False
        
        if not isinstance(audio, np.ndarray):
            return False
        
        if audio.size == 0:
            return False
        
        # Check sample rate
        if sample_rate <= 0:
            return False
        
        # Check for NaN or Inf values
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            return False
        
        # Check that audio is float type
        if not np.issubdtype(audio.dtype, np.floating):
            return False
        
        return True

    def save(self, audio: np.ndarray, sample_rate: int, output_path: Union[str, Path]) -> None:
        """Save audio data to a WAV file.

        Writes a numpy array to disk as a WAV file. Creates parent directories
        if they don't exist. Suitable for saving processed audio for later use.

        Args:
            audio: Audio data array. Can be 1D (mono) or 2D (stereo, with
                shape (samples, channels) or (channels, samples)).
            sample_rate: Sample rate in Hz for the output file.
            output_path: Destination path for the WAV file. Parent directories
                will be created if they don't exist.

        Raises:
            OSError: If the file cannot be written (permissions, disk space).
            ValueError: If audio format is invalid.

        Example:
            >>> ingester = WAVIngester()
            >>> audio = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
            >>> ingester.save(audio, 44100, "output/test_tone.wav")

        Note:
            The output format is determined by soundfile defaults (16-bit PCM
            for WAV). Audio values should be in range [-1.0, 1.0] to avoid
            clipping.

        See Also:
            - :meth:`load` for loading WAV files
        """
        output_path = Path(output_path)
        
        # Ensure output directory exists
        if output_path.parent.as_posix() != ".":
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # soundfile expects (samples,) for mono or (samples, channels) for stereo
        sf.write(str(output_path), audio, sample_rate)
