"""
Audio preprocessing module using pydub.

Provides utilities for audio preprocessing such as normalization, 
filtering, and format conversion.
"""

from typing import Optional
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import io


class AudioPreprocessor:
    """Preprocess audio stems using pydub."""

    def __init__(self):
        """Initialize the audio preprocessor."""
        pass

    def normalize(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Normalize audio to standard loudness.

        Args:
            audio: Input audio array (channels, samples).
            sample_rate: Sample rate of the audio.

        Returns:
            Normalized audio array.
        """
        # Convert to pydub AudioSegment
        audio_segment = self._numpy_to_audiosegment(audio, sample_rate)
        
        # Apply normalization
        normalized = normalize(audio_segment)
        
        # Convert back to numpy
        return self._audiosegment_to_numpy(normalized)

    def compress(
        self, audio: np.ndarray, sample_rate: int, threshold: float = -20.0
    ) -> np.ndarray:
        """
        Apply dynamic range compression.

        Args:
            audio: Input audio array (channels, samples).
            sample_rate: Sample rate of the audio.
            threshold: Compression threshold in dB.

        Returns:
            Compressed audio array.
        """
        # Convert to pydub AudioSegment
        audio_segment = self._numpy_to_audiosegment(audio, sample_rate)
        
        # Apply compression
        compressed = compress_dynamic_range(audio_segment)
        
        # Convert back to numpy
        return self._audiosegment_to_numpy(compressed)

    def trim_silence(
        self, audio: np.ndarray, sample_rate: int, silence_thresh: int = -50
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio.

        Args:
            audio: Input audio array (channels, samples).
            sample_rate: Sample rate of the audio.
            silence_thresh: Silence threshold in dB.

        Returns:
            Trimmed audio array.
        """
        from pydub.silence import detect_leading_silence
        
        # Convert to pydub AudioSegment
        audio_segment = self._numpy_to_audiosegment(audio, sample_rate)
        
        # Detect silence at start and end
        start_trim = detect_leading_silence(audio_segment, silence_threshold=silence_thresh)
        end_trim = detect_leading_silence(audio_segment.reverse(), silence_threshold=silence_thresh)
        
        # Trim
        duration = len(audio_segment)
        trimmed = audio_segment[start_trim:duration-end_trim]
        
        # Convert back to numpy
        return self._audiosegment_to_numpy(trimmed)

    def apply_fade(
        self, audio: np.ndarray, sample_rate: int, fade_in_ms: int = 10, fade_out_ms: int = 10
    ) -> np.ndarray:
        """
        Apply fade in/out to audio.

        Args:
            audio: Input audio array (channels, samples).
            sample_rate: Sample rate of the audio.
            fade_in_ms: Fade in duration in milliseconds.
            fade_out_ms: Fade out duration in milliseconds.

        Returns:
            Audio array with fades applied.
        """
        # Convert to pydub AudioSegment
        audio_segment = self._numpy_to_audiosegment(audio, sample_rate)
        
        # Apply fades
        faded = audio_segment.fade_in(fade_in_ms).fade_out(fade_out_ms)
        
        # Convert back to numpy
        return self._audiosegment_to_numpy(faded)

    def _numpy_to_audiosegment(
        self, audio: np.ndarray, sample_rate: int
    ) -> AudioSegment:
        """
        Convert numpy array to pydub AudioSegment.

        Args:
            audio: Audio array (channels, samples).
            sample_rate: Sample rate.

        Returns:
            AudioSegment object.
        """
        # Ensure 2D array
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Interleave channels if stereo
        if audio_int16.shape[0] == 2:
            audio_int16 = np.stack([audio_int16[0], audio_int16[1]], axis=1).flatten()
            channels = 2
        else:
            audio_int16 = audio_int16.flatten()
            channels = 1
        
        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=channels
        )
        
        return audio_segment

    def _audiosegment_to_numpy(self, audio_segment: AudioSegment) -> np.ndarray:
        """
        Convert pydub AudioSegment to numpy array.

        Args:
            audio_segment: AudioSegment object.

        Returns:
            Audio array (channels, samples).
        """
        # Get raw data
        samples = np.array(audio_segment.get_array_of_samples())
        
        # Convert to float
        audio = samples.astype(np.float32) / 32768.0
        
        # Reshape for channels
        if audio_segment.channels == 2:
            audio = audio.reshape(-1, 2).T
        else:
            audio = audio.reshape(1, -1)
        
        return audio
