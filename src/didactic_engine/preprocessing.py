"""
Audio preprocessing module using pydub.

This module provides the ``AudioPreprocessor`` class for preparing audio
data before analysis. It handles common preprocessing tasks including:

- Peak normalization
- Silence trimming
- Resampling
- Stereo to mono conversion

The preprocessing is configurable via ``PipelineConfig`` and is an optional
step in the pipeline—useful for cleaning up recordings with excessive silence
or inconsistent levels.

Integration:
    Preprocessing occurs after ingestion and before analysis/separation.
    It modifies the audio data in-place or creates new preprocessed files.

Example:
    >>> preprocessor = AudioPreprocessor()
    >>> normalized = preprocessor.normalize(audio, 44100)
    >>> trimmed = preprocessor.trim_silence(audio, 44100, thresh_db=-40)

Dependencies:
    - pydub: Audio manipulation library (requires ffmpeg for some formats)
    - librosa: Used for resampling

See Also:
    - :mod:`didactic_engine.ingestion` for loading audio
    - :mod:`didactic_engine.config` for preprocessing configuration
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
    """Preprocess audio using pydub and librosa.
    
    Provides methods for common audio preprocessing tasks. All methods
    work with numpy arrays and handle the conversion to/from pydub's
    AudioSegment format internally.
    
    Processing Order:
        When using :meth:`preprocess`, operations are applied in this order:
        
        1. Mono conversion (if requested)
        2. Resampling (if target SR differs)
        3. Silence trimming (if requested)
        4. Normalization (if requested)
        
        This order ensures consistent behavior regardless of input format.
    
    Example:
        >>> preprocessor = AudioPreprocessor()
        >>> 
        >>> # Individual operations
        >>> normalized = preprocessor.normalize(audio, 44100)
        >>> trimmed = preprocessor.trim_silence(audio, 44100)
        >>> mono = preprocessor.to_mono(stereo_audio)
        >>> 
        >>> # Or use config-driven preprocessing
        >>> processed, new_sr = preprocessor.preprocess(audio, sr, config)
    """

    def __init__(self):
        """Initialize the audio preprocessor.
        
        The preprocessor is stateless—all configuration is passed per-method
        or via PipelineConfig to the :meth:`preprocess` method.
        """
        pass

    def normalize(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize audio to peak amplitude using pydub.

        Applies peak normalization so that the maximum absolute sample value
        approaches 1.0. This does not affect dynamic range or introduce
        compression—it simply scales the entire signal.

        Args:
            audio: Input audio array (1D mono). Should be float with values
                roughly in range [-1.0, 1.0].
            sample_rate: Sample rate in Hz. Needed for pydub conversion.

        Returns:
            Normalized audio array with peak amplitude near 1.0. Same shape
            and dtype as input.

        Example:
            >>> preprocessor = AudioPreprocessor()
            >>> quiet_audio = np.array([0.1, -0.05, 0.08])
            >>> normalized = preprocessor.normalize(quiet_audio, 44100)
            >>> print(f"Max before: {np.max(np.abs(quiet_audio)):.2f}")
            >>> print(f"Max after: {np.max(np.abs(normalized)):.2f}")
            Max before: 0.10
            Max after: 1.00

        Note:
            Uses pydub's ``normalize`` function which finds the peak and
            scales accordingly. Very quiet signals or signals with DC offset
            may not reach exactly 1.0.
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
        """Trim silence from the beginning and end of audio.

        Detects non-silent regions using pydub's silence detection and
        removes excessive silence from the start and end while preserving
        a configurable amount of lead-in/lead-out silence.

        Args:
            audio: Input audio array (1D mono). Float values in [-1, 1].
            sample_rate: Sample rate in Hz.
            thresh_db: Silence threshold in dBFS. Audio below this level
                is considered silence. Default -40 dBFS is suitable for
                most music; use -50 or lower for quiet recordings.
            keep_ms: Milliseconds of silence to preserve at the start and
                end. This prevents abrupt starts/stops. Default 80ms
                provides a natural feel.

        Returns:
            Trimmed audio array. May be shorter than input if silence was
            removed. If the entire signal is silence, returns original.

        Example:
            >>> preprocessor = AudioPreprocessor()
            >>> # Audio with 1 second of silence, then content
            >>> audio = np.concatenate([np.zeros(44100), signal])
            >>> trimmed = preprocessor.trim_silence(audio, 44100, thresh_db=-40)
            >>> print(f"Original: {len(audio)/44100:.2f}s")
            >>> print(f"Trimmed: {len(trimmed)/44100:.2f}s")

        Note:
            Uses pydub's ``detect_nonsilent`` with min_silence_len=100ms.
            Very short sounds surrounded by silence may be detected as
            separate regions.

        See Also:
            - :meth:`preprocess` for config-driven trimming
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
        """Resample audio to a different sample rate.

        Uses librosa's high-quality resampling (Kaiser best by default).

        Args:
            audio: Input audio array (1D mono).
            orig_sr: Original sample rate in Hz.
            target_sr: Target sample rate in Hz. Common values:
                - 44100: CD quality, suitable for preservation
                - 22050: Common for analysis (half CD rate)
                - 16000: Speech processing

        Returns:
            Resampled audio array. Length changes proportionally to the
            ratio of sample rates. If orig_sr == target_sr, returns
            original unchanged.

        Example:
            >>> preprocessor = AudioPreprocessor()
            >>> audio_44k, _ = ingester.load("song.wav")  # 44100 Hz
            >>> audio_22k = preprocessor.resample(audio_44k, 44100, 22050)
            >>> print(f"Original: {len(audio_44k)} samples")
            >>> print(f"Resampled: {len(audio_22k)} samples")  # ~half

        Note:
            Downsampling may cause aliasing if the source contains
            frequencies above the new Nyquist frequency. Librosa applies
            an anti-aliasing filter by default.
        """
        if orig_sr == target_sr:
            return audio

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo or multi-channel audio to mono.

        Averages channels together. For stereo (L+R)/2. For mono input,
        returns unchanged.

        Args:
            audio: Input audio array. Can be:
                - 1D: Already mono, returned as-is
                - 2D with shape (channels, samples): librosa format
                - 2D with shape (samples, channels): common format

        Returns:
            1D mono audio array. For 2D input, the result is the mean
            across channels.

        Example:
            >>> preprocessor = AudioPreprocessor()
            >>> stereo = np.random.randn(2, 44100)  # (channels, samples)
            >>> mono = preprocessor.to_mono(stereo)
            >>> print(mono.shape)
            (44100,)

        Note:
            Simple averaging may not be ideal for all content. Some
            stereo recordings have out-of-phase content that cancels
            when summed. For critical applications, consider more
            sophisticated downmix algorithms.
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
        """Orchestrate full preprocessing based on pipeline configuration.

        Applies a sequence of preprocessing steps based on flags in the
        config. This is the recommended way to preprocess audio in the
        pipeline as it ensures consistent ordering of operations.

        Processing order:
            1. Mono conversion (if cfg.preprocess_mono)
            2. Resampling (if cfg.preprocess_target_sr != sample_rate)
            3. Silence trimming (if cfg.preprocess_trim_silence)
            4. Normalization (if cfg.preprocess_normalize)

        Args:
            audio: Input audio array (1D or 2D).
            sample_rate: Original sample rate in Hz.
            cfg: Pipeline configuration object containing:
                - ``preprocess_mono``: Convert to mono if True
                - ``preprocess_target_sr``: Target sample rate
                - ``preprocess_trim_silence``: Trim silence if True
                - ``preprocess_silence_thresh_dbfs``: Silence threshold
                - ``preprocess_keep_silence_ms``: Silence to keep
                - ``preprocess_normalize``: Normalize if True

        Returns:
            Tuple of (preprocessed_audio, output_sample_rate). The sample
            rate may differ from input if resampling was applied.

        Example:
            >>> from didactic_engine.config import PipelineConfig
            >>> cfg = PipelineConfig(
            ...     song_id="test",
            ...     input_wav=Path("song.wav"),
            ...     out_dir=Path("output"),
            ...     preprocess_normalize=True,
            ...     preprocess_trim_silence=True,
            ... )
            >>> preprocessor = AudioPreprocessor()
            >>> audio, sr = preprocessor.preprocess(raw_audio, 44100, cfg)

        See Also:
            - :class:`didactic_engine.config.PipelineConfig` for config options
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
        """Convert numpy array to pydub AudioSegment.

        Internal helper for using pydub's processing functions.

        Args:
            audio: Audio array (1D mono or 2D stereo). Values should be
                in range [-1.0, 1.0].
            sample_rate: Sample rate in Hz.

        Returns:
            pydub AudioSegment ready for processing.

        Note:
            Audio is clipped to [-1, 1] and converted to 16-bit PCM
            for pydub compatibility. Some precision is lost.
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
        """Convert pydub AudioSegment back to numpy array.

        Internal helper for retrieving processed audio from pydub.

        Args:
            audio_segment: pydub AudioSegment to convert.

        Returns:
            1D mono numpy array with float32 values in range [-1, 1].
            Stereo segments are converted to mono by averaging channels.
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
