"""
Optional Essentia feature extraction.
"""

from pathlib import Path


def extract_essentia_features(wav_path: Path, sr: int = 44100) -> dict:
    """
    Extract features using Essentia (optional).

    Args:
        wav_path: Path to WAV file
        sr: Sample rate for analysis

    Returns:
        Dictionary with extracted features or error info
    """
    try:
        import essentia.standard as es
    except ImportError:
        return {
            "available": False,
            "error": "Essentia not installed. Install with: pip install essentia",
        }

    try:
        # Load audio
        loader = es.MonoLoader(filename=str(wav_path), sampleRate=sr)
        audio = loader()

        features = {"available": True}

        # Loudness
        try:
            loudness = es.Loudness()(audio)
            features["loudness"] = float(loudness)
        except Exception as e:
            features["loudness_error"] = str(e)

        # Spectral centroid
        try:
            centroids = []
            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
                spectrum = es.Spectrum()(frame)
                centroid = es.Centroid()(spectrum)
                centroids.append(centroid)

            if centroids:
                import numpy as np
                features["spectral_centroid_mean"] = float(np.mean(centroids))
                features["spectral_centroid_std"] = float(np.std(centroids))
        except Exception as e:
            features["spectral_centroid_error"] = str(e)

        # MFCCs
        try:
            mfcc_extractor = es.MFCC(numberCoefficients=13)
            mfccs_list = []

            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
                spectrum = es.Spectrum()(frame)
                _, mfcc_coeffs = mfcc_extractor(spectrum)
                mfccs_list.append(mfcc_coeffs)

            if mfccs_list:
                import numpy as np
                mfccs_array = np.array(mfccs_list)
                for i in range(13):
                    features[f"mfcc_{i:02d}_mean"] = float(np.mean(mfccs_array[:, i]))
                    features[f"mfcc_{i:02d}_std"] = float(np.std(mfccs_array[:, i]))
        except Exception as e:
            features["mfcc_error"] = str(e)

        return features

    except Exception as e:
        return {
            "available": False,
            "error": f"Essentia feature extraction failed: {str(e)}",
        }
