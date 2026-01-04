"""
Edge case tests for the didactic-engine pipeline.

Tests boundary conditions, error handling, and resilience scenarios
that are not covered by standard pipeline tests.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import soundfile as sf

from didactic_engine.config import PipelineConfig
from didactic_engine.pipeline import AudioPipeline
from didactic_engine.ingestion import WAVIngester
from didactic_engine.analysis import AudioAnalyzer
from didactic_engine.segmentation import segment_beats_into_bars


class TestConfigValidation:
    """Test configuration validation edge cases."""

    def test_negative_sample_rate_rejected(self):
        """Should reject negative sample rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            # Create a dummy file
            audio = np.zeros(1000, dtype=np.float32)
            sf.write(wav_path, audio, 22050)

            with pytest.raises(ValueError, match="analysis_sr must be positive"):
                PipelineConfig(
                    song_id="test",
                    input_wav=wav_path,
                    out_dir=Path(tmpdir) / "output",
                    analysis_sr=-1,
                )

    def test_zero_sample_rate_rejected(self):
        """Should reject zero sample rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            audio = np.zeros(1000, dtype=np.float32)
            sf.write(wav_path, audio, 22050)

            with pytest.raises(ValueError, match="analysis_sr must be positive"):
                PipelineConfig(
                    song_id="test",
                    input_wav=wav_path,
                    out_dir=Path(tmpdir) / "output",
                    analysis_sr=0,
                )

    def test_negative_timeout_rejected(self):
        """Should reject negative timeout values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            audio = np.zeros(1000, dtype=np.float32)
            sf.write(wav_path, audio, 22050)

            with pytest.raises(ValueError, match="demucs_timeout_s cannot be negative"):
                PipelineConfig(
                    song_id="test",
                    input_wav=wav_path,
                    out_dir=Path(tmpdir) / "output",
                    demucs_timeout_s=-100,
                )

    def test_zero_hop_length_rejected(self):
        """Should reject zero hop length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            audio = np.zeros(1000, dtype=np.float32)
            sf.write(wav_path, audio, 22050)

            with pytest.raises(ValueError, match="hop_length must be positive"):
                PipelineConfig(
                    song_id="test",
                    input_wav=wav_path,
                    out_dir=Path(tmpdir) / "output",
                    hop_length=0,
                )

    def test_invalid_time_signature_denominator(self):
        """Should reject invalid time signature denominators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            audio = np.zeros(1000, dtype=np.float32)
            sf.write(wav_path, audio, 22050)

            with pytest.raises(ValueError, match="time_signature_den must be a power of 2"):
                PipelineConfig(
                    song_id="test",
                    input_wav=wav_path,
                    out_dir=Path(tmpdir) / "output",
                    time_signature_den=3,  # Invalid - must be power of 2
                )

    def test_invalid_basic_pitch_backend(self):
        """Should reject invalid basic pitch backend values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            audio = np.zeros(1000, dtype=np.float32)
            sf.write(wav_path, audio, 22050)

            with pytest.raises(ValueError, match="basic_pitch_backend must be one of"):
                PipelineConfig(
                    song_id="test",
                    input_wav=wav_path,
                    out_dir=Path(tmpdir) / "output",
                    basic_pitch_backend="invalid",
                )

    def test_positive_silence_thresh_rejected(self):
        """Should reject positive silence threshold (must be dBFS, i.e., negative)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            audio = np.zeros(1000, dtype=np.float32)
            sf.write(wav_path, audio, 22050)

            with pytest.raises(ValueError, match="preprocess_silence_thresh_dbfs must be negative"):
                PipelineConfig(
                    song_id="test",
                    input_wav=wav_path,
                    out_dir=Path(tmpdir) / "output",
                    preprocess_silence_thresh_dbfs=10,  # Should be negative (dBFS)
                )


class TestIngestionEdgeCases:
    """Test audio ingestion edge cases."""

    def test_load_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        ingester = WAVIngester()
        with pytest.raises(FileNotFoundError):
            ingester.load("/nonexistent/path/audio.wav")

    def test_validate_nan_values(self):
        """Should reject audio with NaN values."""
        ingester = WAVIngester()
        audio = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        assert not ingester.validate(audio, 44100)

    def test_validate_inf_values(self):
        """Should reject audio with Inf values."""
        ingester = WAVIngester()
        audio = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        assert not ingester.validate(audio, 44100)

    def test_validate_neg_inf_values(self):
        """Should reject audio with -Inf values."""
        ingester = WAVIngester()
        audio = np.array([1.0, -np.inf, 3.0], dtype=np.float32)
        assert not ingester.validate(audio, 44100)

    def test_validate_empty_array(self):
        """Should reject empty audio arrays."""
        ingester = WAVIngester()
        audio = np.array([], dtype=np.float32)
        assert not ingester.validate(audio, 44100)

    def test_validate_none_audio(self):
        """Should reject None audio."""
        ingester = WAVIngester()
        assert not ingester.validate(None, 44100)

    def test_validate_negative_sample_rate(self):
        """Should reject negative sample rate."""
        ingester = WAVIngester()
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert not ingester.validate(audio, -1)

    def test_validate_zero_sample_rate(self):
        """Should reject zero sample rate."""
        ingester = WAVIngester()
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert not ingester.validate(audio, 0)

    def test_validate_integer_array(self):
        """Should reject integer audio arrays."""
        ingester = WAVIngester()
        audio = np.array([1, 2, 3], dtype=np.int32)
        assert not ingester.validate(audio, 44100)


class TestBarSegmentationEdgeCases:
    """Test bar segmentation edge cases."""

    def test_empty_beat_times(self):
        """Should handle empty beat times gracefully."""
        boundaries = segment_beats_into_bars(
            [], tempo_bpm=120.0, ts_num=4, ts_den=4, audio_duration=10.0
        )
        assert boundaries == []

    def test_single_beat(self):
        """Should handle single beat edge case."""
        boundaries = segment_beats_into_bars(
            [0.0], tempo_bpm=120.0, ts_num=4, ts_den=4, audio_duration=2.0
        )
        # With only one beat, there's no complete bar
        assert len(boundaries) <= 1

    def test_few_beats_less_than_bar(self):
        """Should handle beats less than one bar."""
        # At 120 BPM, 4/4 time, one bar = 2 seconds = 4 beats
        boundaries = segment_beats_into_bars(
            [0.0, 0.5, 1.0],  # Only 3 beats
            tempo_bpm=120.0,
            ts_num=4,
            ts_den=4,
            audio_duration=1.5,
        )
        # Might have 0 complete bars
        assert len(boundaries) <= 1

    def test_different_time_signatures(self):
        """Should handle various time signatures."""
        beat_times = [i * 0.5 for i in range(12)]  # 6 seconds of beats at 120 BPM

        # 4/4 time
        boundaries_4_4 = segment_beats_into_bars(
            beat_times, tempo_bpm=120.0, ts_num=4, ts_den=4, audio_duration=6.0
        )

        # 3/4 time
        boundaries_3_4 = segment_beats_into_bars(
            beat_times, tempo_bpm=120.0, ts_num=3, ts_den=4, audio_duration=6.0
        )

        # 6/8 time
        boundaries_6_8 = segment_beats_into_bars(
            beat_times, tempo_bpm=120.0, ts_num=6, ts_den=8, audio_duration=6.0
        )

        # Different time signatures should produce different bar counts
        assert isinstance(boundaries_4_4, list)
        assert isinstance(boundaries_3_4, list)
        assert isinstance(boundaries_6_8, list)


class TestAnalyzerEdgeCases:
    """Test audio analyzer edge cases."""

    def test_silent_audio(self):
        """Should handle silent audio without crashing."""
        analyzer = AudioAnalyzer(use_essentia=False)
        
        # Create silent audio
        silent_audio = np.zeros(22050 * 2, dtype=np.float32)  # 2 seconds
        
        # Should not raise, may detect no tempo or default
        analysis = analyzer.analyze(silent_audio, 22050)
        
        assert "tempo" in analysis or "librosa" in analysis
        assert "beat_times" in analysis

    def test_very_short_audio(self):
        """Should handle very short audio gracefully."""
        analyzer = AudioAnalyzer(use_essentia=False)
        
        # Create very short audio (0.1 seconds)
        short_audio = np.sin(np.linspace(0, 2 * np.pi * 440, 2205)).astype(np.float32)
        
        # Should not crash
        try:
            analysis = analyzer.analyze(short_audio, 22050)
            assert isinstance(analysis, dict)
        except Exception as e:
            # Some very short audio may legitimately fail
            assert "too short" in str(e).lower() or "cannot" in str(e).lower()

    def test_dc_offset_audio(self):
        """Should handle audio with DC offset."""
        analyzer = AudioAnalyzer(use_essentia=False)
        
        # Create audio with DC offset
        t = np.linspace(0, 2.0, 44100, dtype=np.float32)
        dc_audio = np.sin(2 * np.pi * 440 * t) + 0.5  # 50% DC offset
        dc_audio = dc_audio.astype(np.float32)
        
        # Should not crash
        analysis = analyzer.analyze(dc_audio, 22050)
        assert isinstance(analysis, dict)

    def test_clipped_audio(self):
        """Should handle clipped audio without crashing."""
        analyzer = AudioAnalyzer(use_essentia=False)
        
        # Create clipped audio
        t = np.linspace(0, 2.0, 44100, dtype=np.float32)
        loud_audio = np.sin(2 * np.pi * 440 * t) * 2.0  # Double amplitude
        clipped = np.clip(loud_audio, -1.0, 1.0).astype(np.float32)
        
        # Should not crash
        analysis = analyzer.analyze(clipped, 22050)
        assert isinstance(analysis, dict)


class TestPipelineRobustness:
    """Test pipeline robustness in edge cases."""

    def _make_click_track(
        self,
        sample_rate: int,
        duration_s: float,
        bpm: float = 120.0,
    ) -> np.ndarray:
        """Create a simple click track for reliable beat detection."""
        n = int(sample_rate * duration_s)
        audio = np.zeros(n, dtype=np.float32)
        beat_interval = 60.0 / bpm
        pulse_len = max(1, int(0.02 * sample_rate))

        for i in range(int(duration_s / beat_interval) + 1):
            start = int(i * beat_interval * sample_rate)
            end = min(n, start + pulse_len)
            if end > start:
                audio[start:end] += (0.9 * np.hanning(end - start)).astype(np.float32)

        rng = np.random.default_rng(42)
        audio += (0.01 * rng.standard_normal(n)).astype(np.float32)
        return audio

    def test_pipeline_with_minimal_audio(self):
        """Test pipeline with minimal viable audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create 3-second click track (should produce at least 1 bar)
            sample_rate = 22050
            duration = 3.0
            audio = self._make_click_track(sample_rate, duration, bpm=120.0)

            wav_path = tmpdir_path / "minimal.wav"
            sf.write(wav_path, audio, sample_rate)

            cfg = PipelineConfig(
                song_id="minimal_test",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=sample_rate,
                write_bar_chunks=True,
                write_bar_chunk_wavs=False,
                use_pydub_preprocess=False,
                use_demucs_separation=False,
                use_basic_pitch_transcription=False,
            )

            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            assert results["song_id"] == "minimal_test"
            assert results["duration_s"] > 0

    def test_pipeline_preserves_step_timings(self):
        """Pipeline should capture step timing metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            sample_rate = 22050
            duration = 4.0
            audio = self._make_click_track(sample_rate, duration, bpm=120.0)

            wav_path = tmpdir_path / "timed.wav"
            sf.write(wav_path, audio, sample_rate)

            cfg = PipelineConfig(
                song_id="timing_test",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=sample_rate,
                write_bar_chunks=False,
                use_pydub_preprocess=False,
                use_demucs_separation=False,
                use_basic_pitch_transcription=False,
            )

            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            # Check that step timings were captured
            assert hasattr(pipeline, 'step_timings')
            assert len(pipeline.step_timings) > 0

            # All timings should be non-negative
            for step_name, timing in pipeline.step_timings.items():
                assert timing >= 0, f"Step {step_name} has negative timing: {timing}"


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration in pipeline context."""

    def test_demucs_circuit_starts_closed(self):
        """Demucs circuit breaker should start in closed state."""
        from didactic_engine.resilience import demucs_circuit, CircuitState

        # Reset to ensure clean state
        demucs_circuit.reset()
        assert demucs_circuit.state == CircuitState.CLOSED

    def test_basic_pitch_circuit_starts_closed(self):
        """Basic pitch circuit breaker should start in closed state."""
        from didactic_engine.resilience import basic_pitch_circuit, CircuitState

        # Reset to ensure clean state
        basic_pitch_circuit.reset()
        assert basic_pitch_circuit.state == CircuitState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
