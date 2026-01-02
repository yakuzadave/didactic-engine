"""
Audio preprocessing module using pydub.

Provides utilities for audio preprocessing such as normalization,
silence trimming, resampling, and channel conversion.
"""

from typing import Tuple, TYPE_CHECKING
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize
from pydub.silence import detect_nonsilent
import librosa

if TYPE_CHECKING:
    from didactic_engine.config import PipelineConfig


class AudioPreprocessor:
    """Preprocess audio using pydub and librosa."""

    def __init__(self):
        """Initialize the audio preprocessor."""
        pass

    def normalize(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Normalize audio to peak amplitude.

        Args:
            audio: Input audio array (1D mono).
            sample_rate: Sample rate of the audio.

        Returns:
            Normalized audio array with peak amplitude ~1.0.
        """
        # Convert to pydub AudioSegment
        audio_segment = self._numpy_to_audiosegment(audio, sample_rate)

        # Apply peak normalization
        normalized = pydub_normalize(audio_segment)

        # Convert back to numpy
        return self._audiosegment_to_numpy(normalized)

    def trim_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        thresh_db: float = -40.0,
        keep_ms: int = 80,
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio.

        Uses pydub's detect_nonsilent to find non-silent ranges and
        keeps a bit of leading/trailing silence as specified.

        Args:
            audio: Input audio array (1D mono).
            sample_rate: Sample rate of the audio.
            thresh_db: Silence threshold in dBFS.
            keep_ms: Milliseconds of silence to keep at start/end.

        Returns:
            Trimmed audio array.
        """
        # Convert to pydub AudioSegment
        audio_segment = self._numpy_to_audiosegment(audio, sample_rate)

        # Detect non-silent ranges
        nonsilent_ranges = detect_nonsilent(
            audio_segment,
            min_silence_len=100,
            silence_thresh=thresh_db,
            seek_step=10,
        )

        if not nonsilent_ranges:
            # All silence - return as is
            return audio

        # Get first and last non-silent ranges
        start_trim = max(0, nonsilent_ranges[0][0] - keep_ms)
        end_trim = min(len(audio_segment), nonsilent_ranges[-1][1] + keep_ms)

        # Trim
        trimmed = audio_segment[start_trim:end_trim]

        # Convert back to numpy
        return self._audiosegment_to_numpy(trimmed)

    def resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Input audio array (1D mono).
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio array.
        """
        if orig_sr == target_sr:
            return audio

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono.

        Args:
            audio: Input audio array (1D or 2D).

        Returns:
            Mono audio array (1D).
        """
        if audio.ndim == 1:
            return audio

        # If 2D, average channels
        if audio.ndim == 2:
            # Handle both (channels, samples) and (samples, channels)
            if audio.shape[0] <= 2:
                # (channels, samples) format
                return np.mean(audio, axis=0)
            else:
                # (samples, channels) format
                return np.mean(audio, axis=1)

        return audio

    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int,
        cfg: "PipelineConfig",
    ) -> Tuple[np.ndarray, int]:
        """
        Orchestrate full preprocessing based on config flags.

        Args:
            audio: Input audio array.
            sample_rate: Sample rate of the audio.
            cfg: Pipeline configuration with preprocessing flags.

        Returns:
            Tuple of (preprocessed audio, output sample rate).
        """
        # Convert to mono if requested
        if cfg.preprocess_mono:
            audio = self.to_mono(audio)

        # Resample if requested
        output_sr = sample_rate
        if cfg.preprocess_target_sr != sample_rate:
            audio = self.resample(audio, sample_rate, cfg.preprocess_target_sr)
            output_sr = cfg.preprocess_target_sr

        # Trim silence if requested
        if cfg.preprocess_trim_silence:
            audio = self.trim_silence(
                audio,
                output_sr,
                thresh_db=cfg.preprocess_silence_thresh_dbfs,
                keep_ms=cfg.preprocess_keep_silence_ms,
            )

        # Normalize if requested
        if cfg.preprocess_normalize:
            audio = self.normalize(audio, output_sr)

        return audio, output_sr

    def _numpy_to_audiosegment(
        self, audio: np.ndarray, sample_rate: int
    ) -> AudioSegment:
        """
        Convert numpy array to pydub AudioSegment.

        Args:
            audio: Audio array (1D mono or 2D stereo).
            sample_rate: Sample rate.

        Returns:
            AudioSegment object.
        """
        # Ensure 1D for mono
        if audio.ndim == 2:
            audio = self.to_mono(audio)

        # Clip to prevent overflow
        audio = np.clip(audio, -1.0, 1.0)

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1,
        )

        return audio_segment

    def _audiosegment_to_numpy(self, audio_segment: AudioSegment) -> np.ndarray:
        """
        Convert pydub AudioSegment to numpy array.

        Args:
            audio_segment: AudioSegment object.

        Returns:
            Audio array (1D mono).
        """
        # Get raw data
        samples = np.array(audio_segment.get_array_of_samples())

        # Convert to float
        audio = samples.astype(np.float32) / 32768.0

        # If stereo, convert to mono
        if audio_segment.channels == 2:
            audio = audio.reshape(-1, 2)
            audio = np.mean(audio, axis=1)

        return audio
