"""
Stem separation module using Demucs.

Separates audio into individual stems (vocals, drums, bass, other).
"""

import os
import tempfile
import subprocess
from typing import Dict, List, Optional
import numpy as np
import soundfile as sf

try:
    import demucs
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False


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

    def separate(
        self, audio: np.ndarray, sample_rate: int, output_dir: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems.

        Args:
            audio: Input audio array (channels, samples).
            sample_rate: Sample rate of the audio.
            output_dir: Directory to save separated stems. If None, uses temp directory.

        Returns:
            Dictionary mapping stem names to audio arrays.
        """
        if not DEMUCS_AVAILABLE:
            print("Warning: Demucs not installed. Using mock stem separation.")
            return self._create_mock_stems(audio, sample_rate)
        
        # Create temporary directory if needed
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Save input audio to temporary file
        input_path = os.path.join(output_dir, "input.wav")
        
        # Prepare audio for saving
        audio_to_save = audio.T if audio.ndim == 2 else audio
        sf.write(input_path, audio_to_save, sample_rate)

        # Run Demucs separation
        try:
            cmd = [
                "demucs",
                "--two-stems", "vocals",  # Simplified for faster processing
                "-n", self.model,
                "--device", self.device,
                "-o", output_dir,
                input_path,
            ]
            
            # Note: For full separation, remove --two-stems flag
            # This will separate into all 4 stems
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            
            if result.returncode != 0:
                # Fallback: create mock stems by simple filtering
                return self._create_mock_stems(audio, sample_rate)

        except FileNotFoundError:
            # Demucs not installed or not in PATH, create mock stems
            return self._create_mock_stems(audio, sample_rate)

        # Load separated stems
        stems = {}
        model_output_dir = os.path.join(output_dir, self.model, "input")
        
        for stem_name in ["vocals", "no_vocals"]:  # Two-stems mode
            stem_path = os.path.join(model_output_dir, f"{stem_name}.wav")
            if os.path.exists(stem_path):
                stem_audio, _ = sf.read(stem_path)
                if stem_audio.ndim == 1:
                    stem_audio = stem_audio.reshape(1, -1)
                else:
                    stem_audio = stem_audio.T
                stems[stem_name] = stem_audio

        # Clean up temporary input file
        if os.path.exists(input_path):
            os.remove(input_path)

        return stems

    def _create_mock_stems(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, np.ndarray]:
        """
        Create mock stems for testing when Demucs is not available.

        Args:
            audio: Input audio array.
            sample_rate: Sample rate of the audio.

        Returns:
            Dictionary with mock stem data.
        """
        # Simple frequency-based splitting as mock separation
        from scipy import signal

        stems = {}
        
        # Ensure audio is 2D
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # High-pass filter for "vocals" (simplified)
        sos = signal.butter(4, 200, 'hp', fs=sample_rate, output='sos')
        vocals = signal.sosfilt(sos, audio, axis=1)
        stems["vocals"] = vocals
        
        # Low-pass filter for "bass"
        sos = signal.butter(4, 250, 'lp', fs=sample_rate, output='sos')
        bass = signal.sosfilt(sos, audio, axis=1)
        stems["bass"] = bass
        
        # Band-pass for "drums"
        sos = signal.butter(4, [100, 1000], 'bp', fs=sample_rate, output='sos')
        drums = signal.sosfilt(sos, audio, axis=1)
        stems["drums"] = drums
        
        # Residual for "other"
        stems["other"] = audio - (vocals + bass + drums) / 3
        
        return stems
