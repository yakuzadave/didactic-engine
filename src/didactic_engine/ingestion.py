"""
WAV file ingestion module.

Handles loading and validation of WAV audio files.
"""

import os
from typing import Tuple, Optional
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

    def load(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load a WAV file.

        Args:
            file_path: Path to the WAV file.

        Returns:
            Tuple of (audio data, sample rate)

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a valid audio file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=False)
            
            # Ensure 2D array (channels, samples)
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")

    def validate(self, audio: np.ndarray, sample_rate: int) -> bool:
        """
        Validate audio data.

        Args:
            audio: Audio data array.
            sample_rate: Sample rate of the audio.

        Returns:
            True if valid, False otherwise.
        """
        if audio is None or len(audio) == 0:
            return False
        
        if sample_rate <= 0:
            return False
        
        # Check for NaN or Inf values
        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            return False
        
        return True

    def save(self, audio: np.ndarray, sample_rate: int, output_path: str) -> None:
        """
        Save audio data to a WAV file.

        Args:
            audio: Audio data array.
            sample_rate: Sample rate of the audio.
            output_path: Path to save the WAV file.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Transpose if needed (soundfile expects (samples, channels))
        if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
        
        sf.write(output_path, audio, sample_rate)
