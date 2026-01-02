"""
Essentia feature extraction module.

Provides optional audio analysis using the Essentia library.
If Essentia is not installed, functions return appropriate fallback values.
"""

from pathlib import Path
from typing import Any, Dict, Union
import numpy as np


def extract_essentia_features(
    wav_path: Union[str, Path],
    sample_rate: int = 44100,
) -> Dict[str, Any]:
    """
    Extract audio features using Essentia.

    Args:
        wav_path: Path to WAV file.
        sample_rate: Sample rate for analysis.

    Returns:
        Dictionary containing extracted features or error info if Essentia
        is not installed. Keys include:
        - available: bool indicating if Essentia is available
        - error: error message if not available
        - loudness: Stevens loudness value
        - spectral_centroid_mean/std/min/max: spectral centroid statistics
        - mfcc_XX_mean/std: MFCC statistics for coefficients 0-12
    """
    try:
        import essentia.standard as es
    except ImportError:
        return {
            "available": False,
            "error": "Essentia is not installed. Install with: pip install essentia",
        }

    wav_path = Path(wav_path)
    if not wav_path.exists():
        return {
            "available": False,
            "error": f"Audio file not found: {wav_path}",
        }

    try:
        # Load audio with MonoLoader
        loader = es.MonoLoader(filename=str(wav_path), sampleRate=sample_rate)
        audio = loader()

        features: Dict[str, Any] = {"available": True}

        # Stevens Loudness
        try:
            loudness_algo = es.Loudness()
            features["loudness"] = float(loudness_algo(audio))
        except Exception as e:
            features["loudness_error"] = str(e)

        # Spectral Centroid
        try:
            centroids = []
            frame_size = 2048
            hop_size = 512

            for frame in es.FrameGenerator(
                audio, frameSize=frame_size, hopSize=hop_size
            ):
                windowed = es.Windowing(type="hann")(frame)
                spectrum = es.Spectrum()(windowed)
                centroid = es.Centroid()(spectrum)
                centroids.append(centroid)

            if centroids:
                centroids_arr = np.array(centroids)
                features["spectral_centroid_mean"] = float(np.mean(centroids_arr))
                features["spectral_centroid_std"] = float(np.std(centroids_arr))
                features["spectral_centroid_min"] = float(np.min(centroids_arr))
                features["spectral_centroid_max"] = float(np.max(centroids_arr))
        except Exception as e:
            features["spectral_centroid_error"] = str(e)

        # MFCCs (13 coefficients)
        try:
            mfcc_algo = es.MFCC(numberCoefficients=13)
            mfccs_list = []

            for frame in es.FrameGenerator(
                audio, frameSize=2048, hopSize=512
            ):
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

        # EBU R128 Loudness (scalar only)
        try:
            loudness_ebur128 = es.LoudnessEBUR128()
            integrated, _, _, _ = loudness_ebur128(audio)
            features["loudness_ebu_r128"] = float(integrated)
        except Exception as e:
            # EBU R128 may not be available in all Essentia builds
            features["loudness_ebu_r128_error"] = str(e)

        return features

    except Exception as e:
        return {
            "available": False,
            "error": f"Essentia feature extraction failed: {str(e)}",
        }


def extract_essentia_features_from_array(
    audio: np.ndarray,
    sample_rate: int = 44100,
) -> Dict[str, Any]:
    """
    Extract Essentia features from a numpy array.

    Args:
        audio: Audio array (1D mono).
        sample_rate: Sample rate of the audio.

    Returns:
        Dictionary containing extracted features.
    """
    try:
        import essentia.standard as es
    except ImportError:
        return {
            "available": False,
            "error": "Essentia is not installed. Install with: pip install essentia",
        }

    try:
        # Ensure float32
        audio = audio.astype(np.float32)

        # Ensure 1D
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0) if audio.shape[0] <= 2 else np.mean(audio, axis=1)

        features: Dict[str, Any] = {"available": True}

        # Stevens Loudness
        try:
            loudness_algo = es.Loudness()
            features["loudness"] = float(loudness_algo(audio))
        except Exception as e:
            features["loudness_error"] = str(e)

        # Spectral Centroid
        try:
            centroids = []
            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
                windowed = es.Windowing(type="hann")(frame)
                spectrum = es.Spectrum()(windowed)
                centroid = es.Centroid()(spectrum)
                centroids.append(centroid)

            if centroids:
                centroids_arr = np.array(centroids)
                features["spectral_centroid_mean"] = float(np.mean(centroids_arr))
                features["spectral_centroid_std"] = float(np.std(centroids_arr))
                features["spectral_centroid_min"] = float(np.min(centroids_arr))
                features["spectral_centroid_max"] = float(np.max(centroids_arr))
        except Exception as e:
            features["spectral_centroid_error"] = str(e)

        # MFCCs
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

    except Exception as e:
        return {
            "available": False,
            "error": f"Essentia feature extraction failed: {str(e)}",
        }
