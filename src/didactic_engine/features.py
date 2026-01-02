"""
Feature extraction module.

Extracts evolving bar-level features from audio segments.
"""

from typing import Dict, List, Any
import numpy as np
import librosa


class FeatureExtractor:
    """Extract bar-level audio features."""

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def extract_bar_features(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Extract features from a single bar/segment of audio.

        Args:
            audio: Input audio array (channels, samples).
            sample_rate: Sample rate of the audio.

        Returns:
            Dictionary containing extracted features.
        """
        # Convert to mono for feature extraction
        if audio.ndim == 2:
            audio_mono = librosa.to_mono(audio)
        else:
            audio_mono = audio.flatten()

        features = {}

        # Time-domain features
        features["rms"] = float(np.sqrt(np.mean(audio_mono**2)))
        features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(audio_mono)))
        features["energy"] = float(np.sum(audio_mono**2))

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_mono, sr=sample_rate)
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_mono, sr=sample_rate)
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=sample_rate)
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))

        spectral_contrast = librosa.feature.spectral_contrast(y=audio_mono, sr=sample_rate)
        features["spectral_contrast_mean"] = [
            float(np.mean(spectral_contrast[i])) for i in range(spectral_contrast.shape[0])
        ]

        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_mono, sr=sample_rate, n_mfcc=13)
        features["mfcc_mean"] = [float(np.mean(mfccs[i])) for i in range(mfccs.shape[0])]
        features["mfcc_std"] = [float(np.std(mfccs[i])) for i in range(mfccs.shape[0])]

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_mono, sr=sample_rate)
        features["chroma_mean"] = [float(np.mean(chroma[i])) for i in range(chroma.shape[0])]
        features["chroma_std"] = [float(np.std(chroma[i])) for i in range(chroma.shape[0])]

        # Tempo and rhythm
        try:
            onset_env = librosa.onset.onset_strength(y=audio_mono, sr=sample_rate)
            features["onset_strength_mean"] = float(np.mean(onset_env))
            features["onset_strength_std"] = float(np.std(onset_env))
        except:
            features["onset_strength_mean"] = 0.0
            features["onset_strength_std"] = 0.0

        return features

    def extract_evolving_features(
        self, audio: np.ndarray, sample_rate: int, n_segments: int = 4
    ) -> Dict[str, Any]:
        """
        Extract evolving features by dividing bar into sub-segments.

        Args:
            audio: Input audio array (channels, samples).
            sample_rate: Sample rate of the audio.
            n_segments: Number of sub-segments to divide the bar into.

        Returns:
            Dictionary containing evolving features.
        """
        # Ensure audio is 1D
        if audio.ndim == 2:
            audio = librosa.to_mono(audio)
        else:
            audio = audio.flatten()

        segment_length = len(audio) // n_segments
        evolving_features = {
            "n_segments": n_segments,
            "segments": [],
        }

        for i in range(n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length if i < n_segments - 1 else len(audio)
            segment = audio[start:end]

            segment_features = self.extract_bar_features(
                segment.reshape(1, -1), sample_rate
            )
            segment_features["segment_index"] = i
            segment_features["segment_start_time"] = start / sample_rate
            segment_features["segment_end_time"] = end / sample_rate

            evolving_features["segments"].append(segment_features)

        # Calculate feature evolution metrics (e.g., trends)
        evolving_features["evolution"] = self._calculate_evolution_metrics(
            evolving_features["segments"]
        )

        return evolving_features

    def extract_features_from_chunks(
        self, chunk_paths: List[str], sample_rate: int
    ) -> List[Dict[str, Any]]:
        """
        Extract features from multiple audio chunks.

        Args:
            chunk_paths: List of paths to audio chunk files.
            sample_rate: Sample rate of the audio.

        Returns:
            List of feature dictionaries, one per chunk.
        """
        import soundfile as sf

        all_features = []

        for i, chunk_path in enumerate(chunk_paths):
            # Load chunk
            audio, sr = sf.read(chunk_path)
            
            # Convert to proper shape (channels, samples)
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            else:
                audio = audio.T

            # Extract features
            features = self.extract_bar_features(audio, sr)
            features["chunk_index"] = i
            features["chunk_path"] = chunk_path

            all_features.append(features)

        return all_features

    def _calculate_evolution_metrics(
        self, segment_features: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate metrics describing how features evolve across segments.

        Args:
            segment_features: List of feature dictionaries for each segment.

        Returns:
            Dictionary containing evolution metrics.
        """
        evolution = {}

        # Extract time series for scalar features
        feature_names = [
            "rms",
            "zcr",
            "spectral_centroid_mean",
            "spectral_rolloff_mean",
        ]

        for feature_name in feature_names:
            values = [seg[feature_name] for seg in segment_features]
            
            # Calculate trend (linear regression slope)
            x = np.arange(len(values))
            y = np.array(values)
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                evolution[f"{feature_name}_trend"] = float(slope)
            else:
                evolution[f"{feature_name}_trend"] = 0.0

            # Calculate variability
            evolution[f"{feature_name}_variance"] = float(np.var(values))

        return evolution
