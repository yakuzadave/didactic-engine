"""
Stem selection for melody transcription.

This module provides functions to score and select the best stem for
melody transcription using pitch salience analysis.

Key Functions:
    - :func:`score_melody_stem`: Score a stem's melody suitability
    - :func:`select_best_melody_stem`: Select best stem from multiple candidates

Algorithm:
    Uses librosa.yin() to detect pitch and compute:
    - Fraction of voiced (pitched) frames
    - Stability of fundamental frequency (f0) over time
    
    Higher scores indicate stems with clearer, more continuous melodic content.

Example:
    >>> from pathlib import Path
    >>> stem_paths = {
    ...     "vocals": Path("vocals.wav"),
    ...     "other": Path("other.wav"),
    ...     "bass": Path("bass.wav"),
    ... }
    >>> best_stem, best_path, scores = select_best_melody_stem(
    ...     stem_paths, sample_rate=22050
    ... )
    >>> print(f"Best stem: {best_stem} (score: {scores[best_stem]:.3f})")

See Also:
    - :mod:`didactic_engine.transcription` for MIDI transcription
    - :mod:`didactic_engine.separation` for stem separation
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import librosa


def score_melody_stem(audio: np.ndarray, sample_rate: int) -> float:
    """
    Score a stem's suitability for melody transcription.
    
    Uses pitch detection (librosa.yin) to compute a melody score based on:
    - Fraction of voiced frames (continuous pitch)
    - Stability of fundamental frequency (f0)
    
    Higher scores indicate clearer melodic content.
    
    Args:
        audio: Audio samples (1D mono numpy.float32 array).
        sample_rate: Sample rate in Hz.
        
    Returns:
        Melody score between 0.0 and 1.0. Higher is better.
        Returns 0.0 if audio is too short or has no voiced frames.
        
    Example:
        >>> audio = np.random.randn(44100).astype(np.float32)
        >>> score = score_melody_stem(audio, 44100)
        >>> print(f"Melody score: {score:.3f}")
        
    Note:
        - Drums typically score near 0.0 (no clear pitch)
        - Vocals/lead instruments score 0.5-0.9
        - Very short audio (<1s) may score poorly
    
    See Also:
        - :func:`select_best_melody_stem` for multi-stem comparison
    """
    # Validate input
    if len(audio) < sample_rate:  # Less than 1 second
        return 0.0
        
    # Compute f0 (fundamental frequency) with librosa.yin
    # fmin=65Hz (C2), fmax=1200Hz (D6) covers most melodic ranges
    try:
        f0 = librosa.yin(audio, fmin=65.0, fmax=1200.0, sr=sample_rate)
    except Exception:
        # If YIN fails, return 0
        return 0.0
    
    # Voiced frames: finite and positive f0
    voiced = np.isfinite(f0) & (f0 > 0)
    voiced_frac = float(np.mean(voiced))
    
    # If less than 5% voiced, not a melody stem
    if voiced_frac < 0.05:
        return 0.0
    
    # Compute f0 stability (lower std = more stable = better melody)
    f0_voiced = f0[voiced]
    mean_f0 = float(np.mean(f0_voiced))
    std_f0 = float(np.std(f0_voiced))
    
    # Normalized stability: 1.0 / (1.0 + relative_std)
    # A perfectly stable note has std=0, score=1.0
    # High variation reduces the score
    stability_score = 1.0 / (1.0 + std_f0 / (mean_f0 + 1e-9))
    
    # Weight voiced fraction strongly (70%) and stability (30%)
    # This prioritizes stems with continuous pitch over perfectly stable pitch
    melody_score = voiced_frac * 0.7 + stability_score * 0.3
    
    return melody_score


def select_best_melody_stem(
    stem_paths: Dict[str, Path],
    sample_rate: int = 22050,
    candidate_stems: Optional[Tuple[str, ...]] = None,
) -> Tuple[str, Path, Dict[str, float]]:
    """
    Select the best stem for melody transcription from available stems.
    
    Loads each candidate stem, computes melody scores, and returns the
    stem with the highest score.
    
    Args:
        stem_paths: Dictionary mapping stem names to file paths.
            Example: {"vocals": Path("vocals.wav"), "other": Path("other.wav")}
        sample_rate: Sample rate to load audio at. Default 22050.
        candidate_stems: Optional tuple of stem names to consider.
            If None, defaults to ("vocals", "other", "bass").
            "drums" is typically excluded as it has no melodic content.
            
    Returns:
        Tuple of (best_stem_name, best_stem_path, all_scores).
        - best_stem_name: Name of the best stem (e.g., "vocals")
        - best_stem_path: Path to the best stem file
        - all_scores: Dict mapping all evaluated stem names to their scores
        
    Raises:
        FileNotFoundError: If no candidate stems exist in stem_paths.
        
    Example:
        >>> stem_paths = {
        ...     "vocals": Path("output/stems/vocals.wav"),
        ...     "drums": Path("output/stems/drums.wav"),
        ...     "bass": Path("output/stems/bass.wav"),
        ...     "other": Path("output/stems/other.wav"),
        ... }
        >>> name, path, scores = select_best_melody_stem(stem_paths)
        >>> print(f"Best: {name} (score: {scores[name]:.3f})")
        >>> print(f"All scores: {scores}")
        Best: vocals (score: 0.723)
        All scores: {'vocals': 0.723, 'other': 0.612, 'bass': 0.234}
        
    Note:
        - Defaults exclude "drums" as drums rarely contain melodic information
        - For instrumental music, "other" often contains the lead instrument
        - For vocal music, "vocals" usually scores highest
        
    See Also:
        - :func:`score_melody_stem` for scoring algorithm details
    """
    if candidate_stems is None:
        candidate_stems = ("vocals", "other", "bass")
    
    best_name = ""
    best_path = Path()
    best_score = -1.0
    all_scores: Dict[str, float] = {}
    
    for name in candidate_stems:
        if name not in stem_paths:
            continue
            
        path = stem_paths[name]
        if not path.exists():
            continue
        
        # Load audio at specified sample rate
        audio, _ = librosa.load(path, sr=sample_rate, mono=True)
        audio = audio.astype(np.float32)
        
        # Score this stem
        score = score_melody_stem(audio, sample_rate)
        all_scores[name] = score
        
        if score > best_score:
            best_score = score
            best_name = name
            best_path = path
    
    # If no stems scored above 0, fall back to any available stem
    if best_score <= 0:
        # Try fallback order: vocals, other, bass, any
        for fallback_name in ["vocals", "other", "bass"]:
            if fallback_name in stem_paths and stem_paths[fallback_name].exists():
                return fallback_name, stem_paths[fallback_name], all_scores
        
        # Last resort: return any stem
        for name, path in stem_paths.items():
            if path.exists():
                return name, path, all_scores
        
        raise FileNotFoundError(f"No usable stems found in {stem_paths}")
    
    return best_name, best_path, all_scores
