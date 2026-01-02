"""
Audio analysis module using librosa and Essentia.

Provides comprehensive audio analysis including tempo, beats,
spectral features, MFCCs, chroma, and more.
"""

from typing import Dict, List, Any
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
            audio: Input audio array (1D mono or 2D stereo).
            sample_rate: Sample rate of the audio.

        Returns:
            Dictionary containing analysis results with keys:
            - "librosa": librosa analysis results
            - "essentia": Essentia analysis results (if enabled)
        """
        # Convert to mono for analysis
        audio_mono = self._to_mono(audio)

        # Get librosa analysis
        librosa_features = self._analyze_librosa(audio_mono, sample_rate)

        result = {
            "librosa": librosa_features,
            # Also include top-level convenience keys
            "tempo": librosa_features["tempo_bpm"],
            "beat_frames": librosa_features.get("beat_frames", []),
            "beat_times": librosa_features["beat_times"],
        }

        # Add Essentia analysis if enabled
        if self.use_essentia:
            essentia_features = self._analyze_essentia(audio_mono, sample_rate)
            result["essentia"] = essentia_features

        return result

    def _analyze_librosa(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Perform analysis using librosa.

        Args:
            audio: Input audio array (1D mono).
            sample_rate: Sample rate of the audio.

        Returns:
            Dictionary containing librosa analysis results.
        """
        features: Dict[str, Any] = {}

        # Duration
        features["duration_s"] = float(len(audio) / sample_rate)

        # Tempo and beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)
        features["tempo_bpm"] = float(tempo)
        features["beat_frames"] = beat_frames.tolist()
        features["beat_times"] = librosa.frames_to_time(
            beat_frames, sr=sample_rate
        ).tolist()

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        features["chroma_mean"] = [float(np.mean(chroma[i])) for i in range(12)]

        # MFCCs with statistics
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfcc_stats = []
        for i in range(13):
            coef = mfccs[i]
            mfcc_stats.append({
                "mean": float(np.mean(coef)),
                "std": float(np.std(coef)),
                "min": float(np.min(coef)),
                "max": float(np.max(coef)),
            })
        features["mfcc_stats"] = mfcc_stats

        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        features["spectral_centroid"] = {
            "mean": float(np.mean(spectral_centroid)),
            "std": float(np.std(spectral_centroid)),
            "min": float(np.min(spectral_centroid)),
            "max": float(np.max(spectral_centroid)),
        }

        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
        features["spectral_bandwidth"] = {
            "mean": float(np.mean(spectral_bandwidth)),
            "std": float(np.std(spectral_bandwidth)),
            "min": float(np.min(spectral_bandwidth)),
            "max": float(np.max(spectral_bandwidth)),
        }

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        features["spectral_rolloff"] = {
            "mean": float(np.mean(spectral_rolloff)),
            "std": float(np.std(spectral_rolloff)),
            "min": float(np.min(spectral_rolloff)),
            "max": float(np.max(spectral_rolloff)),
        }

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features["zcr"] = {
            "mean": float(np.mean(zcr)),
            "std": float(np.std(zcr)),
            "min": float(np.min(zcr)),
            "max": float(np.max(zcr)),
        }

        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features["rms"] = {
            "mean": float(np.mean(rms)),
            "std": float(np.std(rms)),
            "min": float(np.min(rms)),
            "max": float(np.max(rms)),
        }

        return features

    def _analyze_essentia(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Perform analysis using Essentia.

        Args:
            audio: Input audio array (1D mono).
            sample_rate: Sample rate of the audio.

        Returns:
            Dictionary containing Essentia analysis results.
        """
        try:
            import essentia.standard as es
        except ImportError:
            return {
                "available": False,
                "error": "Essentia not installed. Install with: pip install essentia",
            }

        # Ensure float32
        audio = audio.astype(np.float32)

        features: Dict[str, Any] = {"available": True}

        # Stevens Loudness
        try:
            loudness = es.Loudness()(audio)
            features["loudness"] = float(loudness)
        except Exception as e:
            features["loudness_error"] = str(e)

        # EBU R128 Loudness (scalar only)
        try:
            loudness_ebur128 = es.LoudnessEBUR128()
            integrated, _, _, _ = loudness_ebur128(audio)
            features["loudness_ebu_r128"] = float(integrated)
        except Exception as e:
            features["loudness_ebu_r128_error"] = str(e)

        # Spectral centroid
        try:
            centroids = []
            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
                windowed = es.Windowing(type="hann")(frame)
                spectrum = es.Spectrum()(windowed)
                centroid = es.Centroid()(spectrum)
                centroids.append(centroid)

            if centroids:
                arr = np.array(centroids)
                features["spectral_centroid"] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
        except Exception as e:
            features["spectral_centroid_error"] = str(e)

        # MFCCs (13 coefficients)
        try:
            mfcc_algo = es.MFCC(numberCoefficients=13)
            mfccs_list = []

            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
                windowed = es.Windowing(type="hann")(frame)
                spectrum = es.Spectrum()(windowed)
                _, mfcc_coeffs = mfcc_algo(spectrum)
                mfccs_list.append(mfcc_coeffs)

            if mfccs_list:
                mfccs_arr = np.array(mfccs_list)
                for i in range(13):
                    features[f"mfcc_{i:02d}_mean"] = float(np.mean(mfccs_arr[:, i]))
                    features[f"mfcc_{i:02d}_std"] = float(np.std(mfccs_arr[:, i]))
        except Exception as e:
            features["mfcc_error"] = str(e)

        return features

    def extract_beat_times(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract beat times from audio.

        Args:
            audio: Input audio array (1D or 2D).
            sample_rate: Sample rate of the audio.

        Returns:
            Array of beat times in seconds.
        """
        audio_mono = self._to_mono(audio)
        tempo, beat_frames = librosa.beat.beat_track(y=audio_mono, sr=sample_rate)
        return librosa.frames_to_time(beat_frames, sr=sample_rate)

    def extract_bar_times(
        self,
        audio: np.ndarray,
        sample_rate: int,
        beats_per_bar: int = 4,
    ) -> np.ndarray:
        """
        Extract bar (measure) times from audio.

        Args:
            audio: Input audio array.
            sample_rate: Sample rate of the audio.
            beats_per_bar: Number of beats per bar.

        Returns:
            Array of bar start times in seconds.
        """
        beat_times = self.extract_beat_times(audio, sample_rate)

        # Select every Nth beat as bar boundary
        bar_times = beat_times[::beats_per_bar]

        return bar_times

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mono if needed."""
        if audio.ndim == 1:
            return audio

        if audio.ndim == 2:
            # Handle both (channels, samples) and (samples, channels)
            if audio.shape[0] <= 2:
                return librosa.to_mono(audio)
            else:
                return np.mean(audio, axis=1)

        return audio.flatten()
