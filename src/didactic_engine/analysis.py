"""
Audio analysis module using librosa and optionally Essentia.

This module provides the ``AudioAnalyzer`` class for comprehensive audio
analysis including tempo detection, beat tracking, spectral features, MFCCs,
chroma, and more. It serves as the primary feature extraction component of
the didactic-engine pipeline.

Key Features:
    - Beat and tempo detection using librosa
    - Spectral features (centroid, bandwidth, rolloff, ZCR)
    - MFCC extraction with statistics
    - Chroma feature extraction
    - Optional Essentia integration for additional features
    - Bar time detection based on time signature

Integration:
    The analyzer is typically used after ingestion/preprocessing and before
    MIDI transcription. Results are used to align MIDI events to a beat grid.

Example:
    >>> analyzer = AudioAnalyzer(use_essentia=False)
    >>> audio, sr = WAVIngester().load("song.wav")
    >>> results = analyzer.analyze(audio, sr)
    >>> print(f"Tempo: {results['tempo']:.1f} BPM")
    >>> print(f"Beats detected: {len(results['beat_times'])}")

See Also:
    - :mod:`didactic_engine.essentia_features` for Essentia-only features
    - :mod:`didactic_engine.align` for aligning notes to beats
    - :mod:`didactic_engine.segmentation` for bar-based segmentation
"""

from typing import Dict, List, Any
import numpy as np
import librosa


class AudioAnalyzer:
    """Comprehensive audio analyzer using librosa and optionally Essentia.
    
    This class extracts a wide range of audio features suitable for music
    information retrieval tasks. Features are organized into categories:
    
    **Temporal Features:**
        - Tempo (BPM) and beat times
        - Duration
    
    **Spectral Features:**
        - Spectral centroid, bandwidth, rolloff
        - Zero crossing rate (ZCR)
        - RMS energy
    
    **Timbral Features:**
        - MFCCs (Mel-frequency cepstral coefficients)
        - Chroma features
    
    **Optional Essentia Features:**
        - Stevens loudness
        - EBU R128 loudness
        - Additional spectral/MFCC statistics
    
    Attributes:
        use_essentia: Whether to include Essentia features in analysis.
    
    Example:
        >>> analyzer = AudioAnalyzer(use_essentia=False)
        >>> results = analyzer.analyze(audio, 22050)
        >>> print(results["librosa"]["tempo_bpm"])
        120.5
    """

    def __init__(self, use_essentia: bool = False):
        """Initialize the audio analyzer.

        Args:
            use_essentia: If True, includes Essentia-based features in the
                analysis output. Essentia provides additional loudness metrics
                and alternative implementations of spectral features. If
                Essentia is not installed and this is True, the analysis will
                still run but the "essentia" key will contain availability
                status and error info.
        
        Example:
            >>> # Basic analysis with librosa only
            >>> analyzer = AudioAnalyzer()
            
            >>> # Include Essentia features (requires Essentia installation)
            >>> analyzer = AudioAnalyzer(use_essentia=True)
        """
        self.use_essentia = use_essentia

    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Perform comprehensive audio analysis.

        Extracts a wide range of audio features using librosa and optionally
        Essentia. This is the main entry point for audio analysis.

        Args:
            audio: Input audio array. Can be 1D (mono) or 2D (stereo).
                Stereo audio is automatically converted to mono before
                analysis. Values should be in range [-1.0, 1.0].
            sample_rate: Sample rate of the audio in Hz. Common values are
                44100 (CD quality) or 22050 (analysis quality).

        Returns:
            Dictionary containing analysis results with keys:
            
            - ``"librosa"``: Dict of librosa-extracted features including:
                - ``tempo_bpm``: Estimated tempo in BPM
                - ``beat_times``: List of beat times in seconds
                - ``beat_frames``: List of beat frame indices
                - ``duration_s``: Audio duration in seconds
                - ``chroma_mean``: 12-element list of mean chroma values
                - ``mfcc_stats``: List of dicts with mean/std/min/max per MFCC
                - ``spectral_centroid``: Dict with mean/std/min/max
                - ``spectral_bandwidth``: Dict with mean/std/min/max
                - ``spectral_rolloff``: Dict with mean/std/min/max
                - ``zcr``: Dict with zero-crossing rate stats
                - ``rms``: Dict with RMS energy stats
            
            - ``"tempo"``: Shortcut to ``librosa["tempo_bpm"]``
            - ``"beat_frames"``: Shortcut to ``librosa["beat_frames"]``
            - ``"beat_times"``: Shortcut to ``librosa["beat_times"]``
            
            - ``"essentia"`` (if use_essentia=True): Dict of Essentia features
                or error info if Essentia is unavailable

        Example:
            >>> analyzer = AudioAnalyzer()
            >>> results = analyzer.analyze(audio, 22050)
            >>> print(f"Tempo: {results['tempo']:.1f} BPM")
            >>> print(f"Duration: {results['librosa']['duration_s']:.2f}s")
            >>> print(f"Mean spectral centroid: {results['librosa']['spectral_centroid']['mean']:.0f} Hz")

        Note:
            Beat detection assumes a relatively steady tempo. For audio with
            significant tempo changes, results may be less accurate.

        See Also:
            - :meth:`extract_beat_times` for beat-only extraction
            - :meth:`extract_bar_times` for bar boundary detection
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
        """Extract audio features using librosa.

        This private method performs the actual feature extraction using
        librosa. It's called by :meth:`analyze` after mono conversion.

        Args:
            audio: Input audio array (1D mono, float32).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            Dictionary containing:
            
            - ``duration_s``: Total duration in seconds
            - ``tempo_bpm``: Estimated tempo in beats per minute
            - ``beat_frames``: Frame indices of detected beats
            - ``beat_times``: Beat times in seconds
            - ``chroma_mean``: 12 mean values for pitch classes (C, C#, ..., B)
            - ``mfcc_stats``: List of 13 dicts, each with mean/std/min/max
            - ``spectral_centroid``: Brightness indicator (higher = brighter)
            - ``spectral_bandwidth``: Spectral spread around centroid
            - ``spectral_rolloff``: Frequency below which 85% of energy lies
            - ``zcr``: Zero crossing rate (indicator of noisiness)
            - ``rms``: Root mean square energy

        Note:
            All spectral features return statistics (mean, std, min, max)
            computed across all analysis frames, not frame-by-frame values.
            This keeps the output size manageable for JSON serialization.
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
        """Extract additional features using Essentia (if available).

        Provides complementary feature extraction using the Essentia library.
        If Essentia is not installed, returns an error dict instead of raising.

        Args:
            audio: Input audio array (1D mono, float32).
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            Dictionary containing:
            
            If Essentia is available:
                - ``available``: True
                - ``loudness``: Stevens loudness value
                - ``loudness_ebu_r128``: EBU R128 integrated loudness (LUFS)
                - ``spectral_centroid``: Dict with mean/std/min/max
                - ``mfcc_XX_mean``, ``mfcc_XX_std``: MFCC statistics (XX=00-12)
                
            If Essentia is unavailable:
                - ``available``: False
                - ``error``: Installation instructions

        Note:
            Some Essentia features (like EBU R128) may not be available in
            all Essentia builds. Individual feature errors are captured in
            keys like ``loudness_error`` rather than failing the entire call.
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
        """Extract beat times from audio.

        Convenience method for extracting only beat times without the full
        analysis. Uses librosa's beat_track function.

        Args:
            audio: Input audio array (1D or 2D). Stereo is converted to mono.
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            Numpy array of beat times in seconds, sorted chronologically.
            Empty array if no beats are detected.

        Example:
            >>> analyzer = AudioAnalyzer()
            >>> beat_times = analyzer.extract_beat_times(audio, 22050)
            >>> print(f"First beat at {beat_times[0]:.3f}s")

        Note:
            For more control over beat detection parameters, call
            :meth:`analyze` and access the full beat frame data.

        See Also:
            - :meth:`extract_bar_times` for bar boundary detection
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
        """Extract bar (measure) start times from audio.

        Detects beats and groups them into bars based on the specified
        beats per bar (typically from time signature numerator).

        Args:
            audio: Input audio array (1D or 2D). Stereo is converted to mono.
            sample_rate: Sample rate of the audio in Hz.
            beats_per_bar: Number of beats per bar. Common values:
                - 4 for 4/4 time
                - 3 for 3/4 (waltz) time
                - 6 for 6/8 time (treating eighth note as beat)

        Returns:
            Numpy array of bar start times in seconds. The array length is
            approximately (num_beats / beats_per_bar).

        Example:
            >>> analyzer = AudioAnalyzer()
            >>> bar_times = analyzer.extract_bar_times(audio, 22050, beats_per_bar=4)
            >>> print(f"Bar 1 starts at {bar_times[0]:.3f}s")
            >>> print(f"Bar 2 starts at {bar_times[1]:.3f}s")

        Note:
            This assumes a constant time signature throughout. For music
            with time signature changes, you'll need custom bar detection.

        See Also:
            - :mod:`didactic_engine.segmentation` for bar-based audio slicing
        """
        beat_times = self.extract_beat_times(audio, sample_rate)

        # Select every Nth beat as bar boundary
        bar_times = beat_times[::beats_per_bar]

        return bar_times

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mono if needed.
        
        Args:
            audio: Input audio array (1D mono or 2D stereo).
            
        Returns:
            1D mono audio array. If already mono, returns unchanged.
        """
        if audio.ndim == 1:
            return audio

        if audio.ndim == 2:
            # Handle both (channels, samples) and (samples, channels)
            if audio.shape[0] <= 2:
                return librosa.to_mono(audio)
            else:
                return np.mean(audio, axis=1)

        return audio.flatten()
