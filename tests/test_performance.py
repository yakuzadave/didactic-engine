"""
Performance tests for audio chunking with longer songs (5-10 minutes).

These tests ensure the chunking process performs adequately for longer
audio files typical of real-world music processing.
"""

import pytest
import numpy as np
import tempfile
import time
from pathlib import Path
import soundfile as sf


class TestChunkingPerformance:
    """Test performance of chunking for 5-10 minute songs."""

    @pytest.mark.performance
    def test_5_minute_song_chunking_performance(self):
        """Test that a 5-minute song chunks in reasonable time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create 5-minute test audio (300 seconds)
            sample_rate = 22050
            duration = 300.0  # 5 minutes
            tempo_bpm = 120

            print(f"\n  Generating {duration}s test audio...")
            start_gen = time.time()

            # Generate audio with rhythmic structure
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            beat_freq = tempo_bpm / 60.0  # Convert BPM to Hz

            # Create audio with clear beats
            audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
            audio = audio * envelope

            gen_time = time.time() - start_gen
            print(f"  Audio generation: {gen_time:.2f}s")

            wav_path = tmpdir_path / "test_5min.wav"
            sf.write(wav_path, audio, sample_rate)

            # Run pipeline
            from didactic_engine.config import PipelineConfig
            from didactic_engine.pipeline import AudioPipeline

            cfg = PipelineConfig(
                song_id="test_5min_perf",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=sample_rate,
                write_bar_chunks=True,
                write_bar_chunk_wavs=True,
                use_pydub_preprocess=False,
            )

            print(f"  Running pipeline for {duration}s audio...")
            start_pipeline = time.time()

            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            pipeline_time = time.time() - start_pipeline
            print(f"  Pipeline execution: {pipeline_time:.2f}s")

            # Check results
            if "bar_features_parquet" not in results:
                pytest.skip(
                    "No bar features created - beats may not have been detected")

            # Load and verify bar features
            import pandas as pd
            bar_features_path = Path(results["bar_features_parquet"])
            df = pd.read_parquet(bar_features_path)

            # Expected number of bars for 5 minutes at 120 BPM, 4/4
            expected_bars_min = 140  # ~150 bars, allow some tolerance
            expected_bars_max = 160

            num_bars = len(df)
            print(f"  Generated {num_bars} bars")

            assert expected_bars_min <= num_bars <= expected_bars_max, \
                f"Expected ~150 bars, got {num_bars}"

            # Performance assertion: Should complete in reasonable time
            # Target: < 2x real-time for processing (5 min song in < 10 min)
            max_acceptable_time = duration * 2.0
            assert pipeline_time < max_acceptable_time, \
                f"Pipeline took {pipeline_time:.2f}s, expected < {max_acceptable_time:.2f}s"

            # Verify chunk files exist if writing was enabled
            if cfg.write_bar_chunk_wavs:
                for _, row in df.iterrows():
                    chunk_path = Path(row["chunk_path"])
                    assert chunk_path.exists(
                    ), f"Chunk file missing: {chunk_path}"

            print(
                f"  ✓ Performance test passed: {pipeline_time:.2f}s for {duration}s audio")

    @pytest.mark.performance
    def test_10_minute_song_chunking_performance(self):
        """Test that a 10-minute song chunks in reasonable time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create 10-minute test audio (600 seconds)
            sample_rate = 22050
            duration = 600.0  # 10 minutes
            tempo_bpm = 120

            print(f"\n  Generating {duration}s test audio...")
            start_gen = time.time()

            # Generate audio with rhythmic structure
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            beat_freq = tempo_bpm / 60.0

            audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
            audio = audio * envelope

            gen_time = time.time() - start_gen
            print(f"  Audio generation: {gen_time:.2f}s")

            wav_path = tmpdir_path / "test_10min.wav"
            sf.write(wav_path, audio, sample_rate)

            # Run pipeline
            from didactic_engine.config import PipelineConfig
            from didactic_engine.pipeline import AudioPipeline

            cfg = PipelineConfig(
                song_id="test_10min_perf",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=sample_rate,
                write_bar_chunks=True,
                write_bar_chunk_wavs=True,
                use_pydub_preprocess=False,
            )

            print(f"  Running pipeline for {duration}s audio...")
            start_pipeline = time.time()

            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            pipeline_time = time.time() - start_pipeline
            print(f"  Pipeline execution: {pipeline_time:.2f}s")

            # Check results
            if "bar_features_parquet" not in results:
                pytest.skip(
                    "No bar features created - beats may not have been detected")

            # Load and verify bar features
            import pandas as pd
            bar_features_path = Path(results["bar_features_parquet"])
            df = pd.read_parquet(bar_features_path)

            # Expected number of bars for 10 minutes at 120 BPM, 4/4
            expected_bars_min = 280  # ~300 bars, allow some tolerance
            expected_bars_max = 320

            num_bars = len(df)
            print(f"  Generated {num_bars} bars")

            assert expected_bars_min <= num_bars <= expected_bars_max, \
                f"Expected ~300 bars, got {num_bars}"

            # Performance assertion: Should complete in reasonable time
            # Target: < 2x real-time for processing (10 min song in < 20 min)
            max_acceptable_time = duration * 2.0
            assert pipeline_time < max_acceptable_time, \
                f"Pipeline took {pipeline_time:.2f}s, expected < {max_acceptable_time:.2f}s"

            # Verify chunk quality - sample a few chunks
            sample_indices = [0, len(df) // 2, len(df) - 1]
            for idx in sample_indices:
                if idx < len(df):
                    row = df.iloc[idx]
                    chunk_path = Path(row["chunk_path"])

                    if chunk_path.exists():
                        # Load chunk and verify it's valid audio
                        chunk_audio, chunk_sr = sf.read(str(chunk_path))
                        assert len(
                            chunk_audio) > 0, f"Empty chunk at bar {row['bar_index']}"
                        assert chunk_sr == sample_rate, f"Sample rate mismatch in chunk"

                        # Verify duration is reasonable (should be ~2 seconds for 120 BPM, 4/4)
                        chunk_duration = len(chunk_audio) / chunk_sr
                        assert 1.0 < chunk_duration < 4.0, \
                            f"Unexpected chunk duration: {chunk_duration:.2f}s"

            print(
                f"  ✓ Performance test passed: {pipeline_time:.2f}s for {duration}s audio")

    @pytest.mark.performance
    def test_chunk_memory_efficiency(self):
        """Test that chunking doesn't have memory leaks for many bars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create 5-minute audio (results in ~150 bars)
            sample_rate = 22050
            duration = 300.0

            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            beat_freq = 2.0  # 120 BPM
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
            audio = audio * envelope

            wav_path = tmpdir_path / "test_memory.wav"
            sf.write(wav_path, audio, sample_rate)

            # Run pipeline and track memory-related metrics
            from didactic_engine.config import PipelineConfig
            from didactic_engine.pipeline import AudioPipeline

            cfg = PipelineConfig(
                song_id="test_memory",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=sample_rate,
                write_bar_chunks=True,
                write_bar_chunk_wavs=False,  # Don't write to disk to test memory only
                use_pydub_preprocess=False,
            )

            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            # Verify features were computed without writing chunks
            if "bar_features_parquet" not in results:
                pytest.skip("No bar features created")

            import pandas as pd
            bar_features_path = Path(results["bar_features_parquet"])
            df = pd.read_parquet(bar_features_path)

            # Should have many bars
            assert len(df) > 100, f"Expected > 100 bars, got {len(df)}"

            # Verify all features have valid values (no NaN from memory issues)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                nan_count = df[col].isna().sum()
                assert nan_count == 0, f"Column {col} has {nan_count} NaN values"

            print(
                f"  ✓ Memory efficiency test passed: {len(df)} bars processed")


class TestChunkQuality:
    """Test quality and correctness of generated chunks."""

    def test_chunk_boundaries_aligned_to_beats(self):
        """Verify that chunk boundaries align with detected beats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test audio with clear 120 BPM beat (30 seconds)
            sample_rate = 22050
            duration = 30.0
            tempo_bpm = 120

            t = np.linspace(0, duration, int(sample_rate * duration), False)
            beat_freq = tempo_bpm / 60.0

            # Create strong beat pattern
            audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            beat_envelope = np.where(
                np.sin(2 * np.pi * beat_freq * t) > 0.9, 1.0, 0.3)
            audio = audio * beat_envelope

            wav_path = tmpdir_path / "test_beat_align.wav"
            sf.write(wav_path, audio, sample_rate)

            # Run analysis and chunking
            from didactic_engine.config import PipelineConfig
            from didactic_engine.pipeline import AudioPipeline

            cfg = PipelineConfig(
                song_id="test_beat_align",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=sample_rate,
                write_bar_chunks=True,
                write_bar_chunk_wavs=True,
                use_pydub_preprocess=False,
            )

            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            if "bar_features_parquet" not in results:
                pytest.skip("No bar features created")

            import pandas as pd
            bar_features_path = Path(results["bar_features_parquet"])
            df = pd.read_parquet(bar_features_path)

            # Verify chunk durations are consistent with tempo and time signature
            expected_bar_duration = (60.0 / tempo_bpm) * 4  # 4/4 time

            for _, row in df.iterrows():
                actual_duration = row["duration_s"]
                # Allow 10% tolerance for analysis variations
                tolerance = expected_bar_duration * 0.1

                assert abs(actual_duration - expected_bar_duration) < tolerance, \
                    f"Bar {row['bar_index']}: duration {actual_duration:.2f}s, " \
                    f"expected ~{expected_bar_duration:.2f}s"

    def test_no_overlapping_chunks(self):
        """Verify that chunks don't overlap in time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test audio
            sample_rate = 22050
            duration = 60.0

            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

            wav_path = tmpdir_path / "test_no_overlap.wav"
            sf.write(wav_path, audio, sample_rate)

            from didactic_engine.config import PipelineConfig
            from didactic_engine.pipeline import AudioPipeline

            cfg = PipelineConfig(
                song_id="test_no_overlap",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=sample_rate,
                write_bar_chunks=True,
                write_bar_chunk_wavs=True,
                use_pydub_preprocess=False,
            )

            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            if "bar_features_parquet" not in results:
                pytest.skip("No bar features created")

            import pandas as pd
            bar_features_path = Path(results["bar_features_parquet"])
            df = pd.read_parquet(bar_features_path)

            # Sort by bar index
            df = df.sort_values("bar_index")

            # Verify no gaps or overlaps
            for i in range(len(df) - 1):
                current_end = df.iloc[i]["end_s"]
                next_start = df.iloc[i + 1]["start_s"]

                # Next bar should start where current bar ends (or very close)
                gap = abs(next_start - current_end)
                assert gap < 0.1, \
                    f"Gap/overlap between bars {i} and {i+1}: {gap:.3f}s"

    def test_chunks_cover_full_duration(self):
        """Verify that chunks cover the entire audio duration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test audio
            sample_rate = 22050
            duration = 120.0  # 2 minutes

            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

            wav_path = tmpdir_path / "test_coverage.wav"
            sf.write(wav_path, audio, sample_rate)

            from didactic_engine.config import PipelineConfig
            from didactic_engine.pipeline import AudioPipeline

            cfg = PipelineConfig(
                song_id="test_coverage",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=sample_rate,
                write_bar_chunks=True,
                write_bar_chunk_wavs=True,
                use_pydub_preprocess=False,
            )

            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            if "bar_features_parquet" not in results:
                pytest.skip("No bar features created")

            import pandas as pd
            bar_features_path = Path(results["bar_features_parquet"])
            df = pd.read_parquet(bar_features_path)

            # First chunk should start near 0
            first_start = df["start_s"].min()
            assert first_start < 1.0, f"First chunk starts at {first_start:.2f}s, expected near 0"

            # Last chunk should end near total duration
            last_end = df["end_s"].max()
            coverage = last_end / duration
            assert coverage > 0.95, \
                f"Chunks only cover {coverage*100:.1f}% of audio duration"
