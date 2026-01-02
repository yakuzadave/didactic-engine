"""
WAV feature extraction with librosa.
"""

from pathlib import Path
import numpy as np
import librosa


def _stats(arr: np.ndarray) -> dict[str, float]:
    """
    Compute statistics for an array.

    Args:
        arr: Input array

    Returns:
        Dictionary with mean, std, min, max
    """
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def extract_wav_features(
    wav_path: Path, sr: int = 22050, hop_length: int = 512
) -> dict:
    """
    Extract audio features using librosa.

    Args:
        wav_path: Path to WAV file
        sr: Sample rate for analysis
        hop_length: Hop length for STFT

    Returns:
        Dictionary containing extracted features
    """
    # Load audio
    y, actual_sr = librosa.load(wav_path, sr=sr, mono=True)

    # Duration
    duration_s = float(librosa.get_duration(y=y, sr=actual_sr))

    # Tempo and beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=actual_sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=actual_sr, hop_length=hop_length)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=actual_sr, hop_length=hop_length)
    chroma_mean = [float(np.mean(chroma[i])) for i in range(12)]

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=actual_sr, n_mfcc=13, hop_length=hop_length)
    mfcc_stats = {}
    for i in range(13):
        stats = _stats(mfccs[i])
        for key, val in stats.items():
            mfcc_stats[f"mfcc_{i:02d}_{key}"] = val

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=actual_sr, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=actual_sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=actual_sr, hop_length=hop_length)[0]

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    return {
        "tempo_bpm": float(tempo),
        "beats": beat_times.tolist(),
        "duration_s": duration_s,
        "chroma_mean": chroma_mean,
        **mfcc_stats,
        "spectral_centroid": _stats(spectral_centroid),
        "spectral_bandwidth": _stats(spectral_bandwidth),
        "spectral_rolloff": _stats(spectral_rolloff),
        "zcr": _stats(zcr),
    }
