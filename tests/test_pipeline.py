"""
Basic tests for the didactic-engine audio processing pipeline.
"""

import pytest
import numpy as np
import os
import tempfile

from didactic_engine.ingestion import WAVIngester
from didactic_engine.analysis import AudioAnalyzer
from didactic_engine.preprocessing import AudioPreprocessor
from didactic_engine.features import FeatureExtractor
from didactic_engine.segmentation import StemSegmenter
from didactic_engine.midi_parser import MIDIParser


class TestWAVIngester:
    """Test WAV ingestion functionality."""

    def test_validate_audio(self):
        """Test audio validation."""
        ingester = WAVIngester()
        
        # Valid audio
        audio = np.random.randn(2, 44100)
        assert ingester.validate(audio, 44100)
        
        # Invalid audio (contains NaN)
        audio_invalid = np.array([1.0, np.nan, 3.0])
        assert not ingester.validate(audio_invalid, 44100)
        
        # Invalid sample rate
        assert not ingester.validate(audio, 0)


class TestAudioAnalyzer:
    """Test audio analysis functionality."""

    def test_analyze_basic(self):
        """Test basic audio analysis."""
        analyzer = AudioAnalyzer(use_essentia=False)
        
        # Create synthetic audio (440 Hz tone)
        duration = 2.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        audio = audio.reshape(1, -1)
        
        # Analyze
        analysis = analyzer.analyze(audio, sample_rate)
        
        # Check required fields
        assert "tempo" in analysis
        assert "beat_frames" in analysis
        assert "beat_times" in analysis
        assert "spectral_centroids" in analysis
        assert "mfccs" in analysis
        assert isinstance(analysis["tempo"], float)

    def test_extract_beat_times(self):
        """Test beat time extraction."""
        analyzer = AudioAnalyzer()
        
        # Create synthetic audio
        sample_rate = 22050
        duration = 2.0
        audio = np.random.randn(1, int(sample_rate * duration))
        
        beat_times = analyzer.extract_beat_times(audio, sample_rate)
        
        assert isinstance(beat_times, np.ndarray)
        assert len(beat_times) > 0


class TestAudioPreprocessor:
    """Test audio preprocessing functionality."""

    def test_normalize(self):
        """Test audio normalization."""
        preprocessor = AudioPreprocessor()
        
        # Create audio with varying amplitude
        audio = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
        sample_rate = 44100
        
        # Normalize
        normalized = preprocessor.normalize(audio, sample_rate)
        
        assert normalized is not None
        assert normalized.shape == audio.shape


class TestFeatureExtractor:
    """Test feature extraction functionality."""

    def test_extract_bar_features(self):
        """Test bar-level feature extraction."""
        extractor = FeatureExtractor()
        
        # Create synthetic audio
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(1, int(sample_rate * duration))
        
        features = extractor.extract_bar_features(audio, sample_rate)
        
        # Check required features
        assert "rms" in features
        assert "zcr" in features
        assert "spectral_centroid_mean" in features
        assert "mfcc_mean" in features
        assert isinstance(features["rms"], float)

    def test_extract_evolving_features(self):
        """Test evolving feature extraction."""
        extractor = FeatureExtractor()
        
        sample_rate = 22050
        duration = 2.0
        audio = np.random.randn(1, int(sample_rate * duration))
        
        evolving = extractor.extract_evolving_features(audio, sample_rate, n_segments=4)
        
        assert "n_segments" in evolving
        assert "segments" in evolving
        assert "evolution" in evolving
        assert len(evolving["segments"]) == 4


class TestStemSegmenter:
    """Test stem segmentation functionality."""

    def test_segment_by_bars(self):
        """Test bar-based segmentation."""
        segmenter = StemSegmenter()
        
        # Create synthetic audio
        sample_rate = 22050
        duration = 4.0
        audio = np.random.randn(2, int(sample_rate * duration))
        
        # Define bar times
        bar_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        # Segment
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = segmenter.segment_by_bars(
                audio, sample_rate, bar_times, tmpdir, "test"
            )
            
            assert len(chunks) == 4
            for chunk_path in chunks:
                assert os.path.exists(chunk_path)


class TestMIDIParser:
    """Test MIDI parsing functionality."""

    def test_create_from_notes(self):
        """Test creating MIDI from note list."""
        parser = MIDIParser()
        
        # Create some notes
        notes = [
            (0.0, 0.5, 60, 100),  # C4
            (0.5, 1.0, 64, 100),  # E4
            (1.0, 1.5, 67, 100),  # G4
        ]
        
        midi = parser.create_from_notes(notes, program=0, tempo=120.0)
        
        assert len(midi.instruments) == 1
        assert len(midi.instruments[0].notes) == 3

    def test_extract_notes(self):
        """Test note extraction."""
        parser = MIDIParser()
        
        # Create MIDI
        notes = [
            (0.0, 0.5, 60, 100),
            (0.5, 1.0, 64, 100),
        ]
        midi = parser.create_from_notes(notes)
        
        # Extract notes
        extracted = parser.extract_notes(midi)
        
        assert len(extracted) == 2
        assert extracted[0][2] == 60  # First note pitch

    def test_align_to_grid(self):
        """Test MIDI alignment to grid."""
        parser = MIDIParser()
        
        # Create MIDI with notes
        notes = [
            (0.1, 0.5, 60, 100),
            (1.2, 1.8, 64, 100),
            (2.3, 2.9, 67, 100),
        ]
        midi = parser.create_from_notes(notes)
        
        # Define grid (bar times)
        grid_times = np.array([0.0, 1.0, 2.0, 3.0])
        
        # Align
        aligned = parser.align_to_grid(midi, grid_times, quantize=True)
        
        assert len(aligned) == 3
        assert 0 in aligned
        assert 1 in aligned
        assert 2 in aligned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
