"""
Essentia feature extraction module.

This module provides optional audio analysis using the Essentia library.
Essentia offers additional audio features not available in librosa, including
Stevens loudness and EBU R128 loudness measurements.

Key Functions:
    - :func:`extract_essentia_features`: Extract features from WAV file
    - :func:`extract_essentia_features_from_array`: Extract from numpy array

Optional Dependency:
    Essentia is optional and must be installed separately::
    
        pip install essentia
        
    If Essentia is not installed, functions return a dict with
    ``available=False`` instead of raising an exception.

Features Extracted:
    - Stevens Loudness (perceptual loudness)
    - EBU R128 Integrated Loudness (broadcast standard)
    - Spectral Centroid statistics (mean/std/min/max)
    - MFCC coefficients (13 coefficients with mean/std)

Integration:
    Essentia features can be enabled in the pipeline via
    ``use_essentia=True`` in PipelineConfig. They're merged with
    librosa features in analysis results.

Example:
    >>> features = extract_essentia_features("vocals.wav", sample_rate=44100)
    >>> if features["available"]:
    ...     print(f"Loudness: {features['loudness']:.2f}")
    ... else:
    ...     print(f"Essentia not available: {features['error']}")

See Also:
    - :mod:`didactic_engine.analysis` for librosa features
    - :class:`didactic_engine.config.PipelineConfig` for enabling Essentia
"""

from pathlib import Path
from typing import Any, Dict, Union
import numpy as np


def extract_essentia_features(
    wav_path: Union[str, Path],
    sample_rate: int = 44100,
) -> Dict[str, Any]:
    """Extract audio features using Essentia from a WAV file.

    Loads audio with Essentia's MonoLoader and extracts loudness,
    spectral, and MFCC features.

    Args:
        wav_path: Path to WAV file.
        sample_rate: Sample rate for analysis. Default 44100.

    Returns:
        Dictionary with either:
        
        If Essentia is available and extraction succeeds:
            - ``available``: True
            - ``loudness``: Stevens loudness value
            - ``loudness_ebu_r128``: EBU R128 integrated loudness (LUFS)
            - ``spectral_centroid_mean/std/min/max``: Spectral centroid stats
            - ``mfcc_00_mean`` through ``mfcc_12_mean``: MFCC means
            - ``mfcc_00_std`` through ``mfcc_12_std``: MFCC std deviations
            
        If Essentia is unavailable:
            - ``available``: False
            - ``error``: Description of the problem

    Example:
        >>> features = extract_essentia_features("vocals.wav")
        >>> if features["available"]:
        ...     print(f"Loudness: {features['loudness']:.2f}")
        ...     print(f"Centroid mean: {features['spectral_centroid_mean']:.0f}")

    Note:
        Some features (like EBU R128) may not be available in all Essentia
        builds. Individual feature errors are captured in keys like
        ``loudness_ebu_r128_error`` rather than failing the entire call.

    See Also:
        - :func:`extract_essentia_features_from_array` for numpy input
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
    """Extract Essentia features from a numpy audio array.

    Same features as :func:`extract_essentia_features` but works
    directly with numpy arrays instead of files.

    Args:
        audio: Audio array (1D mono). Values should be float32 in
            approximate range [-1, 1].
        sample_rate: Sample rate of the audio.

    Returns:
        Dictionary with same structure as :func:`extract_essentia_features`:
        
        If successful:
            - ``available``: True
            - ``loudness``: Stevens loudness
            - ``spectral_centroid_*``: Centroid statistics
            - ``mfcc_*``: MFCC statistics
            
        If unavailable:
            - ``available``: False
            - ``error``: Error description

    Example:
        >>> audio, sr = librosa.load("song.wav", sr=44100, mono=True)
        >>> features = extract_essentia_features_from_array(audio, sr)
        >>> if features["available"]:
        ...     print(f"Loudness: {features['loudness']:.2f}")

    Note:
        Multi-channel audio is converted to mono by averaging channels.
        Audio is converted to float32 if needed.

    See Also:
        - :func:`extract_essentia_features` for file input
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
