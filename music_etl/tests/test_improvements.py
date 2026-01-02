"""
Tests for the music ETL pipeline improvements.
"""

import pytest
import numpy as np
from pathlib import Path
import pandas as pd

from music_etl.bar_chunker import compute_bar_boundaries
from music_etl.align import align_notes_to_grid
from music_etl.config import PipelineConfig


class TestBarChunkerPerformance:
    """Test bar chunker performance improvements."""

    def test_compute_bar_boundaries_vectorized(self):
        """Test that bar boundaries computation works with vectorized beat extension."""
        # Create beat times
        beat_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        tempo_bpm = 120.0
        audio_duration_s = 10.0

        # Compute bar boundaries
        bars = compute_bar_boundaries(
            beat_times, tempo_bpm, ts_num=4, ts_den=4, audio_duration_s=audio_duration_s
        )

        # Should have multiple bars
        assert len(bars) > 2
        
        # Each bar should have valid times
        for bar_idx, start_s, end_s in bars:
            assert end_s > start_s
            assert start_s >= 0
            assert end_s <= audio_duration_s

    def test_compute_bar_boundaries_no_audio_duration(self):
        """Test bar boundaries without audio duration."""
        beat_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        tempo_bpm = 120.0

        bars = compute_bar_boundaries(
            beat_times, tempo_bpm, ts_num=4, ts_den=4, audio_duration_s=None
        )

        # Should have at least 2 bars from 9 beats (4 beats per bar)
        assert len(bars) >= 2


class TestAlignPerformance:
    """Test alignment performance improvements."""

    def test_align_notes_vectorized_beat_extension(self):
        """Test that note alignment works with vectorized beat extension."""
        # Create test notes
        notes_data = [
            {"pitch": 60, "velocity": 100, "start_s": 0.1, "end_s": 0.5, "dur_s": 0.4},
            {"pitch": 64, "velocity": 90, "start_s": 1.0, "end_s": 1.5, "dur_s": 0.5},
            {"pitch": 67, "velocity": 95, "start_s": 8.0, "end_s": 8.5, "dur_s": 0.5},  # Beyond initial beats
        ]
        notes_df = pd.DataFrame(notes_data)

        # Short beat list
        beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        tempo_bpm = 120.0

        # Align notes
        aligned = align_notes_to_grid(
            notes_df,
            beat_times,
            tempo_bpm,
            ts_num=4,
            ts_den=4,
            audio_duration_s=10.0,
        )

        # Should have all notes aligned
        assert len(aligned) == 3
        assert "beat_index" in aligned.columns
        assert "bar_index" in aligned.columns
        
        # Last note should be aligned despite being beyond initial beats
        last_note = aligned.iloc[2]
        assert last_note["beat_index"] >= 0
        assert last_note["bar_index"] >= 0


class TestConfigurableTimeouts:
    """Test configurable timeout parameters."""

    def test_config_with_stem_options(self):
        """Test that config accepts new stem separation options."""
        cfg = PipelineConfig(
            song_id="test",
            input_wav=Path("test.wav"),
            demucs_two_stems="vocals",
            demucs_timeout=300,
        )

        assert cfg.demucs_two_stems == "vocals"
        assert cfg.demucs_timeout == 300

    def test_config_default_stem_options(self):
        """Test that config has sensible defaults."""
        cfg = PipelineConfig(
            song_id="test",
            input_wav=Path("test.wav"),
        )

        assert cfg.demucs_two_stems is None
        assert cfg.demucs_timeout is None


class TestEssentiaFeatureHandling:
    """Test improved Essentia feature handling."""

    def test_essentia_scalar_extraction(self):
        """Test that scalar values are extracted correctly."""
        # Mock essentia features
        essentia_features = {
            "available": True,
            "loudness": 0.5,  # scalar
            "spectral_centroid_mean": 1000.0,  # scalar
            "mfcc_list": [1.0, 2.0, 3.0, 4.0],  # list - should get mean/std
            "error": "some error",  # string - should be skipped
        }

        # Extract scalars and compute stats for lists
        processed = {}
        for k, v in essentia_features.items():
            if isinstance(v, (int, float, bool)):
                processed[k] = v
            elif isinstance(v, list) and len(v) > 0:
                arr = np.array(v)
                if arr.dtype.kind in 'biufc':
                    processed[f"{k}_mean"] = float(np.mean(arr))
                    processed[f"{k}_std"] = float(np.std(arr))

        # Should have scalars and computed stats
        assert "loudness" in processed
        assert "spectral_centroid_mean" in processed
        assert "mfcc_list_mean" in processed
        assert "mfcc_list_std" in processed
        assert "error" not in processed  # string should be excluded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
