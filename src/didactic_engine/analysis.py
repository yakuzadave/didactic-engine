"""
Audio analysis module using librosa and Essentia.

Provides comprehensive audio analysis including tempo, beats, 
spectral features, and more.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import librosa


class AudioAnalyzer:
    """Analyze audio using librosa and optionally Essentia."""

    def __init__(self, use_essentia: bool = False):
        """
        Initialize the audio analyzer.

        Args:
            use_essentia: Whether to use Essentia for additional analysis.
        """
        self.use_essentia = use_essentia

    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Perform comprehensive audio analysis.

        Args:
            audio: Input audio array (channels, samples).
            sample_rate: Sample rate of the audio.

        Returns:
            Dictionary containing analysis results.
        """
        # Convert to mono for analysis
        if audio.ndim == 2:
            audio_mono = librosa.to_mono(audio)
        else:
            audio_mono = audio.flatten()

        analysis = {}
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=audio_mono, sr=sample_rate)
        analysis["tempo"] = float(tempo)
        analysis["beat_frames"] = beats.tolist()
        analysis["beat_times"] = librosa.frames_to_time(beats, sr=sample_rate).tolist()

        # Onset detection
        onset_env = librosa.onset.onset_strength(y=audio_mono, sr=sample_rate)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate)
        analysis["onset_frames"] = onsets.tolist()
        analysis["onset_times"] = librosa.frames_to_time(onsets, sr=sample_rate).tolist()

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_mono, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_mono, sr=sample_rate)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_mono, sr=sample_rate)
        
        analysis["spectral_centroids"] = spectral_centroids.tolist()
        analysis["spectral_rolloff"] = spectral_rolloff.tolist()
        analysis["spectral_contrast"] = spectral_contrast.tolist()

        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_mono, sr=sample_rate, n_mfcc=13)
        analysis["mfccs"] = mfccs.tolist()

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_mono)[0]
        analysis["zero_crossing_rate"] = zcr.tolist()

        # RMS energy
        rms = librosa.feature.rms(y=audio_mono)[0]
        analysis["rms_energy"] = rms.tolist()

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_mono, sr=sample_rate)
        analysis["chroma"] = chroma.tolist()

        # Add Essentia analysis if enabled
        if self.use_essentia:
            try:
                essentia_features = self._analyze_with_essentia(audio_mono, sample_rate)
                analysis["essentia"] = essentia_features
            except ImportError:
                analysis["essentia"] = {"error": "Essentia not available"}

        return analysis

    def extract_beat_times(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract beat times from audio.

        Args:
            audio: Input audio array.
            sample_rate: Sample rate of the audio.

        Returns:
            Array of beat times in seconds.
        """
        # Convert to mono
        if audio.ndim == 2:
            audio_mono = librosa.to_mono(audio)
        else:
            audio_mono = audio.flatten()

        # Track beats
        tempo, beats = librosa.beat.beat_track(y=audio_mono, sr=sample_rate)
        beat_times = librosa.frames_to_time(beats, sr=sample_rate)
        
        return beat_times

    def extract_bar_times(
        self, audio: np.ndarray, sample_rate: int, beats_per_bar: int = 4
    ) -> np.ndarray:
        """
        Extract bar (measure) times from audio.

        Args:
            audio: Input audio array.
            sample_rate: Sample rate of the audio.
            beats_per_bar: Number of beats per bar (default: 4 for 4/4 time).

        Returns:
            Array of bar start times in seconds.
        """
        beat_times = self.extract_beat_times(audio, sample_rate)
        
        # Select every Nth beat as bar boundary
        bar_times = beat_times[::beats_per_bar]
        
        return bar_times

    def _analyze_with_essentia(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Perform analysis using Essentia.

        Args:
            audio: Input audio array (mono).
            sample_rate: Sample rate of the audio.

        Returns:
            Dictionary containing Essentia analysis results.
        """
        try:
            import essentia
            import essentia.standard as es
        except ImportError:
            return {"error": "Essentia not installed"}

        # Ensure float32
        audio = audio.astype(np.float32)

        features = {}

        # Key detection
        try:
            key_detector = es.KeyExtractor()
            key, scale, strength = key_detector(audio)
            features["key"] = key
            features["scale"] = scale
            features["key_strength"] = float(strength)
        except Exception as e:
            features["key_error"] = str(e)

        # Rhythm descriptors
        try:
            rhythm_extractor = es.RhythmExtractor2013()
            bpm, beats, _, _, _ = rhythm_extractor(audio)
            features["bpm"] = float(bpm)
            features["essentia_beats"] = beats.tolist()
        except Exception as e:
            features["rhythm_error"] = str(e)

        # Tonal descriptors
        try:
            hpcp = es.HPCP()(
                es.SpectralPeaks()(
                    *es.Spectrum()(audio)
                )[0],
                es.SpectralPeaks()(
                    *es.Spectrum()(audio)
                )[1]
            )
            features["hpcp"] = hpcp.tolist()
        except Exception as e:
            features["tonal_error"] = str(e)

        return features
