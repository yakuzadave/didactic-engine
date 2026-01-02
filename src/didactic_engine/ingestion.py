"""
WAV file ingestion module.

Handles loading and validation of WAV audio files.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import soundfile as sf
import librosa


class WAVIngester:
    """Ingest and validate WAV audio files."""

    def __init__(self, sample_rate: Optional[int] = None):
        """
        Initialize the WAV ingester.

        Args:
            sample_rate: Target sample rate for resampling. If None, uses original sample rate.
        """
        self.sample_rate = sample_rate

    def load(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load a WAV file and return mono audio.

        Args:
            file_path: Path to the WAV file.

        Returns:
            Tuple of (mono audio data as 1D numpy array, sample rate)

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a valid audio file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # Load audio file using soundfile first to get native sample rate
            audio, native_sr = sf.read(str(file_path))
            
            # Convert to mono if stereo (soundfile returns (samples, channels))
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            
            # Ensure float type
            audio = audio.astype(np.float32)
            
            # Resample if target sample rate is specified and differs
            if self.sample_rate is not None and self.sample_rate != native_sr:
                audio = librosa.resample(
                    audio, orig_sr=native_sr, target_sr=self.sample_rate
                )
                return audio, self.sample_rate
            
            return audio, native_sr
            
        except Exception as e:
            raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")

    def validate(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Validate audio data.

        Args:
            audio: Audio data array (should be 1D mono).
            sample_rate: Sample rate of the audio.

        Returns:
            True if valid, False otherwise.
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
        """
        Save audio data to a WAV file.

        Args:
            audio: Audio data array (1D mono or 2D stereo).
            sample_rate: Sample rate of the audio.
            output_path: Path to save the WAV file.
        """
        output_path = Path(output_path)
        
        # Ensure output directory exists
        if output_path.parent.as_posix() != ".":
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # soundfile expects (samples,) for mono or (samples, channels) for stereo
        sf.write(str(output_path), audio, sample_rate)
