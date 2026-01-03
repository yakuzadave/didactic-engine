"""
Basic tests for the didactic-engine audio processing pipeline.
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path
import soundfile as sf

from didactic_engine.ingestion import WAVIngester
from didactic_engine.analysis import AudioAnalyzer
from didactic_engine.preprocessing import AudioPreprocessor
from didactic_engine.features import FeatureExtractor
from didactic_engine.segmentation import StemSegmenter, segment_beats_into_bars
from didactic_engine.midi_parser import MIDIParser
from didactic_engine.align import align_notes_to_beats
from didactic_engine.pipeline import AudioPipeline
import pandas as pd


class TestWAVIngester:
    """Test WAV ingestion functionality."""

    def test_validate_audio(self):
        """Test audio validation."""
        ingester = WAVIngester()
        
        # Valid audio (1D mono float)
        audio = np.random.randn(44100).astype(np.float32)
        assert ingester.validate(audio, 44100)
        
        # Invalid audio (contains NaN)
        audio_invalid = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        assert not ingester.validate(audio_invalid, 44100)
        
        # Invalid sample rate
        assert not ingester.validate(audio, 0)
        
        # Invalid type (not float)
        audio_int = np.array([1, 2, 3], dtype=np.int32)
        assert not ingester.validate(audio_int, 44100)


class TestAudioAnalyzer:
    """Test audio analysis functionality."""

    def test_analyze_basic(self):
        """Test basic audio analysis."""
        analyzer = AudioAnalyzer(use_essentia=False)
        
        # Create synthetic audio (440 Hz tone)
        duration = 2.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Analyze
        analysis = analyzer.analyze(audio, sample_rate)
        
        # Check required fields
        assert "tempo" in analysis
        assert "beat_times" in analysis
        assert "librosa" in analysis
        assert isinstance(analysis["tempo"], float)

    def test_extract_beat_times(self):
        """Test beat time extraction."""
        analyzer = AudioAnalyzer()
        
        # Create synthetic audio (1D mono)
        sample_rate = 22050
        duration = 2.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        beat_times = analyzer.extract_beat_times(audio, sample_rate)
        
        assert isinstance(beat_times, np.ndarray)


class TestAudioPreprocessor:
    """Test audio preprocessing functionality."""

    def test_normalize(self):
        """Test audio normalization."""
        preprocessor = AudioPreprocessor()
        
        # Create audio with low amplitude (1D mono)
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        sample_rate = 44100
        
        # Normalize
        normalized = preprocessor.normalize(audio, sample_rate)
        
        assert normalized is not None
        # After normalization, max amplitude should be close to 1.0
        assert np.max(np.abs(normalized)) > np.max(np.abs(audio))


class TestFeatureExtractor:
    """Test feature extraction functionality."""

    def test_extract_bar_features(self):
        """Test bar-level feature extraction."""
        extractor = FeatureExtractor()
        
        # Create synthetic audio (1D mono)
        sample_rate = 22050
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
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
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        evolving = extractor.extract_evolving_features(audio, sample_rate, n_segments=4)
        
        assert "n_segments" in evolving
        assert "segments" in evolving
        assert len(evolving["segments"]) == 4

    def test_extract_beats(self):
        """Test beats extraction."""
        extractor = FeatureExtractor()
        
        beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        beats_df = extractor.extract_beats(beat_times, 120.0, "vocals", "test_song")
        
        assert len(beats_df) == 5
        assert "song_id" in beats_df.columns
        assert "stem" in beats_df.columns
        assert "beat_index" in beats_df.columns


class TestStemSegmenter:
    """Test stem segmentation functionality."""

    def test_segment_by_bars(self):
        """Test bar-based segmentation."""
        segmenter = StemSegmenter()
        
        # Create synthetic audio (1D mono)
        sample_rate = 22050
        duration = 4.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
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


class TestSegmentBeatsIntoBars:
    """Test beat-to-bar segmentation."""

    def test_segment_beats_into_bars(self):
        """Test bar boundary computation."""
        beat_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        boundaries = segment_beats_into_bars(
            beat_times, tempo_bpm=120.0, ts_num=4, ts_den=4, audio_duration=4.0
        )
        
        # With 4/4 time and 8 beats, should have 2 bars
        assert len(boundaries) == 2
        
        # Check structure
        for bar_idx, start_s, end_s in boundaries:
            assert isinstance(bar_idx, int)
            assert isinstance(start_s, float)
            assert isinstance(end_s, float)
            assert end_s > start_s


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


class TestAlignNotes:
    """Test note alignment functionality."""

    def test_align_notes_to_beats(self):
        """Test note-to-beat alignment."""
        notes_df = pd.DataFrame({
            "pitch": [60, 64, 67],
            "velocity": [100, 90, 80],
            "start_s": [0.1, 0.6, 1.2],
            "end_s": [0.5, 1.0, 1.8],
        })
        
        beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        
        aligned = align_notes_to_beats(
            notes_df, beat_times, tempo_bpm=120.0, ts_num=4, ts_den=4
        )
        
        assert "beat_index" in aligned.columns
        assert "bar_index" in aligned.columns
        assert "beat_in_bar" in aligned.columns
        assert len(aligned) == 3


class TestExportMD:
    """Test Markdown export functionality."""

    def test_export_midi_markdown(self):
        """Test MIDI Markdown export."""
        from didactic_engine.export_md import export_midi_markdown

        # Create test aligned notes
        aligned_notes = {
            0: [
                {"start": 0.1, "end": 0.5, "pitch": 60, "velocity": 100},
                {"start": 0.3, "end": 0.6, "pitch": 64, "velocity": 90},
            ],
            1: [
                {"start": 1.0, "end": 1.5, "pitch": 67, "velocity": 80},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.md")
            export_midi_markdown(aligned_notes, output_path, song_id="test_song")

            assert os.path.exists(output_path)

            # Check content
            with open(output_path, "r") as f:
                content = f.read()
                assert "MIDI Analysis Report" in content
                assert "test_song" in content
                assert "Bar 0" in content
                assert "Bar 1" in content

    def test_export_midi_markdown_with_dataframe_columns(self):
        """Test MIDI Markdown export with DataFrame-style column names (start_s/end_s)."""
        from didactic_engine.export_md import export_midi_markdown

        # Create test aligned notes using DataFrame column names
        # This simulates what the pipeline actually produces
        aligned_notes = {
            0: [
                {"start_s": 0.1, "end_s": 0.5, "pitch": 60, "velocity": 100},
                {"start_s": 0.3, "end_s": 0.6, "pitch": 64, "velocity": 90},
            ],
            1: [
                {"start_s": 1.0, "end_s": 1.5, "pitch": 67, "velocity": 80},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.md")
            export_midi_markdown(aligned_notes, output_path, song_id="test_song")

            assert os.path.exists(output_path)

            # Check content includes non-zero time and duration values
            with open(output_path, "r") as f:
                content = f.read()
                assert "MIDI Analysis Report" in content
                assert "test_song" in content
                assert "Bar 0" in content
                assert "Bar 1" in content
                # Verify time values are not all 0.000
                assert "0.100" in content  # start_s of first note
                assert "0.400" in content  # duration of first note (0.5 - 0.1)
                # Ensure we're not getting 0.000 for everything
                assert content.count("0.000") < 10  # Some 0.000 is OK but not all values

    def test_pitch_to_name(self):
        """Test pitch to note name conversion."""
        from didactic_engine.export_md import pitch_to_name

        assert pitch_to_name(60) == "C4"
        assert pitch_to_name(69) == "A4"
        assert pitch_to_name(48) == "C3"


class TestExportABC:
    """Test ABC notation export functionality."""

    def test_export_abc_available(self):
        """Test ABC export module availability."""
        from didactic_engine.export_abc import MUSIC21_AVAILABLE

        # music21 is now a core dependency
        assert MUSIC21_AVAILABLE is True


class TestEssentiaFeatures:
    """Test Essentia feature extraction."""

    def test_essentia_not_installed(self):
        """Test Essentia returns available=False when not installed."""
        from didactic_engine.essentia_features import extract_essentia_features

        # Test with a non-existent file (should return error)
        result = extract_essentia_features("/nonexistent/file.wav", 44100)
        
        # Should either say not available or file not found
        assert result.get("available") is False or "error" in result


class TestConfig:
    """Test pipeline configuration."""

    def test_pipeline_config(self):
        """Test PipelineConfig creation."""
        from didactic_engine.config import PipelineConfig
        from pathlib import Path

        cfg = PipelineConfig(
            song_id="test_song",
            input_wav=Path("/tmp/test.wav"),
            out_dir=Path("/tmp/output"),
        )

        assert cfg.song_id == "test_song"
        assert cfg.stems_dir == Path("/tmp/output/stems/test_song")
        assert cfg.midi_dir == Path("/tmp/output/midi/test_song")
        assert cfg.analysis_dir == Path("/tmp/output/analysis/test_song")


class TestUtilsFlatten:
    """Test dictionary flattening utilities."""

    def test_flatten_dict(self):
        """Test nested dict flattening."""
        from didactic_engine.utils_flatten import flatten_dict

        nested = {"a": {"b": 1, "c": 2}, "d": 3}
        flat = flatten_dict(nested)

        assert flat["a.b"] == 1
        assert flat["a.c"] == 2
        assert flat["d"] == 3


class TestONNXInference:
    """Test ONNX Runtime inference module."""

    def test_onnxruntime_availability_check(self):
        """Test ONNX Runtime availability checking."""
        from didactic_engine.onnx_inference import is_onnxruntime_available

        # Function should return a boolean
        result = is_onnxruntime_available()
        assert isinstance(result, bool)

    def test_onnxruntime_version(self):
        """Test ONNX Runtime version retrieval."""
        from didactic_engine.onnx_inference import (
            is_onnxruntime_available,
            get_onnxruntime_version,
        )

        version = get_onnxruntime_version()

        if is_onnxruntime_available():
            # Should return a version string
            assert version is not None
            assert isinstance(version, str)
            assert len(version) > 0
        else:
            # Should return None if not installed
            assert version is None

    def test_get_available_providers(self):
        """Test getting available ONNX execution providers."""
        from didactic_engine.onnx_inference import (
            is_onnxruntime_available,
            get_available_providers,
        )

        providers = get_available_providers()

        # Should always return a list
        assert isinstance(providers, list)

        if is_onnxruntime_available():
            # Should have at least CPU provider
            assert len(providers) > 0
            assert 'CPUExecutionProvider' in providers
        else:
            # Should return empty list if not installed
            assert providers == []

    def test_inference_session_without_model(self):
        """Test ONNXInferenceSession raises error for missing model."""
        from didactic_engine.onnx_inference import (
            is_onnxruntime_available,
            ONNXInferenceSession,
        )

        if not is_onnxruntime_available():
            # Should raise RuntimeError if onnxruntime not installed
            with pytest.raises(RuntimeError):
                ONNXInferenceSession("/nonexistent/model.onnx")
        else:
            # Should raise FileNotFoundError for missing model file
            with pytest.raises(FileNotFoundError):
                ONNXInferenceSession("/nonexistent/model.onnx")

    def test_create_inference_session_without_onnx(self):
        """Test create_inference_session error handling."""
        from didactic_engine.onnx_inference import (
            is_onnxruntime_available,
            create_inference_session,
        )

        if not is_onnxruntime_available():
            # Should raise RuntimeError if onnxruntime not installed
            with pytest.raises(RuntimeError):
                create_inference_session("/nonexistent/model.onnx")


class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_batch_processing_with_valid_files(self):
        """Test batch processing with multiple valid WAV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create multiple test WAV files
            sample_rate = 22050
            duration = 1.0
            
            input_files = []
            for i in range(3):
                # Create unique audio (different frequencies)
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                audio = np.sin(2 * np.pi * (440 + i * 50) * t).astype(np.float32)
                
                wav_path = tmpdir_path / f"test_audio_{i}.wav"
                sf.write(wav_path, audio, sample_rate)
                input_files.append(wav_path)
            
            # Process batch
            output_dir = tmpdir_path / "output"
            results = AudioPipeline.process_batch(
                input_files,
                output_dir,
                analysis_sr=22050,
                use_essentia_features=False,
                write_bar_chunks=False,  # Disable to speed up test
            )
            
            # Verify results structure
            assert "successful" in results
            assert "failed" in results
            assert "total" in results
            assert "success_count" in results
            assert "failure_count" in results
            
            # All should succeed
            assert results["total"] == 3
            assert results["success_count"] == 3
            assert results["failure_count"] == 0
            assert len(results["successful"]) == 3
            assert len(results["failed"]) == 0

    def test_batch_processing_with_custom_song_ids(self):
        """Test batch processing with custom song IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create test WAV files
            sample_rate = 22050
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            input_files = []
            for i in range(2):
                wav_path = tmpdir_path / f"file_{i}.wav"
                sf.write(wav_path, audio, sample_rate)
                input_files.append(wav_path)
            
            # Custom song IDs
            custom_ids = ["song_alpha", "song_beta"]
            
            # Process batch with custom IDs
            output_dir = tmpdir_path / "output"
            results = AudioPipeline.process_batch(
                input_files,
                output_dir,
                song_ids=custom_ids,
                analysis_sr=22050,
                write_bar_chunks=False,
            )
            
            # Verify custom IDs were used
            assert results["success_count"] == 2
            song_ids_used = [item[0] for item in results["successful"]]
            assert "song_alpha" in song_ids_used
            assert "song_beta" in song_ids_used
            
            # Verify output directories are created with custom song IDs
            assert (output_dir / "input" / "song_alpha").exists()
            assert (output_dir / "input" / "song_beta").exists()
            assert (output_dir / "preprocessed" / "song_alpha").exists()
            assert (output_dir / "preprocessed" / "song_beta").exists()
            assert (output_dir / "analysis" / "song_alpha").exists()
            assert (output_dir / "analysis" / "song_beta").exists()

    def test_batch_processing_with_missing_file(self):
        """Test batch processing handles missing files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create one valid file
            sample_rate = 22050
            duration = 0.5
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            valid_file = tmpdir_path / "valid.wav"
            sf.write(valid_file, audio, sample_rate)
            
            # Reference a non-existent file
            missing_file = tmpdir_path / "missing.wav"
            
            input_files = [valid_file, missing_file]
            
            # Process batch
            output_dir = tmpdir_path / "output"
            results = AudioPipeline.process_batch(
                input_files,
                output_dir,
                analysis_sr=22050,
                write_bar_chunks=False,
            )
            
            # Should have 1 success and 1 failure
            assert results["total"] == 2
            assert results["success_count"] == 1
            assert results["failure_count"] == 1
            assert len(results["successful"]) == 1
            assert len(results["failed"]) == 1
            
            # Verify error information for missing file
            failed_song_id, failed_path, error_msg = results["failed"][0]
            assert failed_song_id == "missing"
            assert "missing.wav" in failed_path
            assert "File not found" in error_msg or "not found" in error_msg.lower()

    def test_batch_processing_song_ids_mismatch(self):
        """Test that mismatched song_ids count raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            input_files = [
                tmpdir_path / "file1.wav",
                tmpdir_path / "file2.wav",
            ]
            
            # Wrong number of song IDs
            wrong_ids = ["id1"]  # Only 1 ID for 2 files
            
            output_dir = tmpdir_path / "output"
            
            with pytest.raises(ValueError, match="must match"):
                AudioPipeline.process_batch(
                    input_files,
                    output_dir,
                    song_ids=wrong_ids,
                )

    def test_batch_processing_empty_input_files(self):
        """Test that empty input_files list raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_dir = tmpdir_path / "output"
            
            with pytest.raises(ValueError, match="cannot be empty"):
                AudioPipeline.process_batch(
                    [],
                    output_dir,
                )


class TestChunkPathHandling:
    """Test chunk_path None handling in bar features."""

    def test_chunk_path_is_none_when_wavs_not_written(self):
        """Test that chunk_path is None when write_bar_chunk_wavs=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create longer test audio with clear beat (10 seconds to ensure multiple bars)
            sample_rate = 22050
            duration = 10.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Create a rhythmic beat pattern (120 BPM = 2 Hz)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            # Add some rhythmic structure
            beat_freq = 2.0  # 120 BPM
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
            audio = audio * envelope

            wav_path = tmpdir_path / "test.wav"
            sf.write(wav_path, audio, sample_rate)

            # Create pipeline config with write_bar_chunk_wavs=False
            from didactic_engine.config import PipelineConfig
            cfg = PipelineConfig(
                song_id="test_chunk_path",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=sample_rate,
                write_bar_chunks=True,
                write_bar_chunk_wavs=False,  # Don't write chunk files
                use_pydub_preprocess=False,
            )

            # Run pipeline
            from didactic_engine.pipeline import AudioPipeline
            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            # Check if bar_features was created
            if "bar_features_parquet" not in results:
                pytest.skip("No bar features created - audio may be too short or beats not detected")

            # Load bar features parquet
            bar_features_path = Path(results["bar_features_parquet"])
            assert bar_features_path.exists(), "Bar features parquet should exist"

            import pandas as pd
            df = pd.read_parquet(bar_features_path)

            # Verify chunk_path is None for all rows
            assert not df.empty, "Should have bar features"
            assert "chunk_path" in df.columns, "chunk_path column should exist"
            assert df["chunk_path"].isna().all(), "All chunk_path values should be None/NaN"

    def test_chunk_path_is_string_when_wavs_written(self):
        """Test that chunk_path is a valid path string when write_bar_chunk_wavs=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create longer test audio with clear beat (10 seconds to ensure multiple bars)
            sample_rate = 22050
            duration = 10.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Create a rhythmic beat pattern (120 BPM = 2 Hz)
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            # Add some rhythmic structure
            beat_freq = 2.0  # 120 BPM
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
            audio = audio * envelope

            wav_path = tmpdir_path / "test.wav"
            sf.write(wav_path, audio, sample_rate)

            # Create pipeline config with write_bar_chunk_wavs=True
            from didactic_engine.config import PipelineConfig
            cfg = PipelineConfig(
                song_id="test_chunk_path_written",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=sample_rate,
                write_bar_chunks=True,
                write_bar_chunk_wavs=True,  # Write chunk files
                use_pydub_preprocess=False,
            )

            # Run pipeline
            from didactic_engine.pipeline import AudioPipeline
            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            # Check if bar_features was created
            if "bar_features_parquet" not in results:
                pytest.skip("No bar features created - audio may be too short or beats not detected")

            # Load bar features parquet
            bar_features_path = Path(results["bar_features_parquet"])
            assert bar_features_path.exists(), "Bar features parquet should exist"

            import pandas as pd
            df = pd.read_parquet(bar_features_path)

            # Verify chunk_path is a valid string for all rows
            assert not df.empty, "Should have bar features"
            assert "chunk_path" in df.columns, "chunk_path column should exist"
            assert df["chunk_path"].notna().all(), "All chunk_path values should be non-null"

            # Verify the files actually exist
            for chunk_path_str in df["chunk_path"]:
                chunk_path = Path(chunk_path_str)
                assert chunk_path.exists(), f"Chunk file should exist: {chunk_path}"
                assert chunk_path.suffix == ".wav", "Chunk should be a WAV file"


class TestPreserveChunkAudio:
    """Test preserve_chunk_audio feature."""

    def test_preserve_chunk_audio_disabled(self):
        """Test that chunks are mono at analysis_sr when preserve_chunk_audio=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create stereo test audio (10 seconds with rhythmic beat)
            sample_rate = 44100
            duration = 10.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Create rhythmic structure
            beat_freq = 2.0  # 120 BPM
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
            left = (np.sin(2 * np.pi * 440 * t) * envelope).astype(np.float32)
            right = (np.sin(2 * np.pi * 880 * t) * envelope).astype(np.float32)
            stereo_audio = np.column_stack((left, right))

            wav_path = tmpdir_path / "test_stereo.wav"
            sf.write(wav_path, stereo_audio, sample_rate)

            # Create pipeline config with preserve_chunk_audio=False
            from didactic_engine.config import PipelineConfig
            analysis_sr = 22050
            cfg = PipelineConfig(
                song_id="test_preserve_disabled",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=analysis_sr,
                write_bar_chunks=True,
                write_bar_chunk_wavs=True,
                preserve_chunk_audio=False,  # Use analysis audio
                use_pydub_preprocess=False,
            )

            # Run pipeline
            from didactic_engine.pipeline import AudioPipeline
            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            # Check chunk files
            chunks_dir = cfg.chunks_dir / "full_mix"
            if not chunks_dir.exists() or not list(chunks_dir.glob("*.wav")):
                pytest.skip("No chunk files created - beats may not have been detected")

            chunk_files = list(chunks_dir.glob("*.wav"))
            assert len(chunk_files) > 0, "Should have chunk files"

            # Verify first chunk is mono at analysis_sr
            first_chunk = chunk_files[0]
            chunk_audio, chunk_sr = sf.read(str(first_chunk))

            assert chunk_sr == analysis_sr, f"Chunk SR should be {analysis_sr}, got {chunk_sr}"
            assert chunk_audio.ndim == 1, "Chunk should be mono (1D array)"

    def test_preserve_chunk_audio_enabled_stereo(self):
        """Test that chunks preserve stereo when preserve_chunk_audio=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create stereo test audio (10 seconds with rhythmic beat)
            sample_rate = 44100
            duration = 10.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Create rhythmic structure
            beat_freq = 2.0  # 120 BPM
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
            left = (np.sin(2 * np.pi * 440 * t) * envelope).astype(np.float32)
            right = (np.sin(2 * np.pi * 880 * t) * envelope).astype(np.float32)
            stereo_audio = np.column_stack((left, right))

            wav_path = tmpdir_path / "test_stereo.wav"
            sf.write(wav_path, stereo_audio, sample_rate)

            # Create pipeline config with preserve_chunk_audio=True
            from didactic_engine.config import PipelineConfig
            analysis_sr = 22050
            cfg = PipelineConfig(
                song_id="test_preserve_enabled",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=analysis_sr,
                write_bar_chunks=True,
                write_bar_chunk_wavs=True,
                preserve_chunk_audio=True,  # Preserve original
                use_pydub_preprocess=False,
            )

            # Run pipeline
            from didactic_engine.pipeline import AudioPipeline
            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            # Check chunk files
            chunks_dir = cfg.chunks_dir / "full_mix"
            if not chunks_dir.exists() or not list(chunks_dir.glob("*.wav")):
                pytest.skip("No chunk files created - beats may not have been detected")

            chunk_files = list(chunks_dir.glob("*.wav"))
            assert len(chunk_files) > 0, "Should have chunk files"

            # Verify first chunk is stereo at original SR
            first_chunk = chunk_files[0]
            chunk_audio, chunk_sr = sf.read(str(first_chunk))

            assert chunk_sr == sample_rate, f"Chunk SR should be {sample_rate}, got {chunk_sr}"
            assert chunk_audio.ndim == 2, "Chunk should be stereo (2D array)"
            assert chunk_audio.shape[1] == 2, "Chunk should have 2 channels"

    def test_preserve_chunk_audio_enabled_mono(self):
        """Test that mono files stay mono when preserve_chunk_audio=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create mono test audio at high sample rate (10 seconds with rhythmic beat)
            sample_rate = 48000
            duration = 10.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Create rhythmic structure
            beat_freq = 2.0  # 120 BPM
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
            mono_audio = (np.sin(2 * np.pi * 440 * t) * envelope).astype(np.float32)

            wav_path = tmpdir_path / "test_mono.wav"
            sf.write(wav_path, mono_audio, sample_rate)

            # Create pipeline config with preserve_chunk_audio=True
            from didactic_engine.config import PipelineConfig
            analysis_sr = 22050
            cfg = PipelineConfig(
                song_id="test_preserve_mono",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=analysis_sr,
                write_bar_chunks=True,
                write_bar_chunk_wavs=True,
                preserve_chunk_audio=True,  # Preserve original
                use_pydub_preprocess=False,
            )

            # Run pipeline
            from didactic_engine.pipeline import AudioPipeline
            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            # Check chunk files
            chunks_dir = cfg.chunks_dir / "full_mix"
            if not chunks_dir.exists() or not list(chunks_dir.glob("*.wav")):
                pytest.skip("No chunk files created - beats may not have been detected")

            chunk_files = list(chunks_dir.glob("*.wav"))
            assert len(chunk_files) > 0, "Should have chunk files"

            # Verify first chunk is mono at original SR
            first_chunk = chunk_files[0]
            chunk_audio, chunk_sr = sf.read(str(first_chunk))

            assert chunk_sr == sample_rate, f"Chunk SR should be {sample_rate}, got {chunk_sr}"
            assert chunk_audio.ndim == 1, "Chunk should be mono (1D array)"

    def test_features_still_use_analysis_sr(self):
        """Test that features still use analysis_sr even when preserve_chunk_audio=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test audio at high sample rate (10 seconds with rhythmic beat)
            sample_rate = 48000
            duration = 10.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Create rhythmic structure
            beat_freq = 2.0  # 120 BPM
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_freq * t)
            audio = (np.sin(2 * np.pi * 440 * t) * envelope).astype(np.float32)

            wav_path = tmpdir_path / "test.wav"
            sf.write(wav_path, audio, sample_rate)

            # Create pipeline config with preserve_chunk_audio=True
            from didactic_engine.config import PipelineConfig
            analysis_sr = 22050
            cfg = PipelineConfig(
                song_id="test_features_sr",
                input_wav=wav_path,
                out_dir=tmpdir_path / "output",
                analysis_sr=analysis_sr,
                write_bar_chunks=True,
                write_bar_chunk_wavs=True,
                preserve_chunk_audio=True,
                use_pydub_preprocess=False,
            )

            # Run pipeline
            from didactic_engine.pipeline import AudioPipeline
            pipeline = AudioPipeline(cfg)
            results = pipeline.run()

            # Check if bar_features was created
            if "bar_features_parquet" not in results:
                pytest.skip("No bar features created - audio may be too short or beats not detected")

            # Load bar features parquet
            bar_features_path = Path(results["bar_features_parquet"])
            assert bar_features_path.exists(), "Bar features parquet should exist"

            import pandas as pd
            df = pd.read_parquet(bar_features_path)

            # Features should be computed - verify we have feature columns
            assert not df.empty, "Should have bar features"
            assert "rms_energy" in df.columns, "Should have RMS energy feature"

            # The features should be valid (not NaN) - features are computed from analysis_sr audio
            assert df["rms_energy"].notna().all(), "RMS energy should be computed"


class TestBasicPitchTranscriber:
    """Test BasicPitchTranscriber functionality."""

    def test_probe_model_serialization_support(self):
        """Test that the model serialization support probe works correctly."""
        try:
            from didactic_engine.transcription import BasicPitchTranscriber
        except ImportError:
            pytest.skip("basic-pitch not installed")

        # This should not raise an error even if basic-pitch is not installed
        # or doesn't support --model-serialization
        try:
            transcriber = BasicPitchTranscriber()
            # Probe should return a boolean
            assert isinstance(transcriber._supports_model_serialization, bool)
        except RuntimeError as e:
            # If basic-pitch is not installed, we expect a RuntimeError
            if "not found" not in str(e):
                raise

    def test_transcriber_initialization_without_basic_pitch(self, monkeypatch):
        """Test that initialization fails gracefully when basic-pitch is not available."""
        from didactic_engine.transcription import BasicPitchTranscriber
        import shutil

        # Mock shutil.which to simulate basic-pitch not being available
        def mock_which(cmd):
            if cmd == "basic-pitch":
                return None
            return shutil.which(cmd)

        monkeypatch.setattr(shutil, "which", mock_which)

        with pytest.raises(RuntimeError, match="basic-pitch command not found"):
            BasicPitchTranscriber()

    def test_transcriber_handles_unsupported_flag(self, monkeypatch):
        """Test that transcriber handles CLI versions without --model-serialization."""
        try:
            from didactic_engine.transcription import BasicPitchTranscriber
        except ImportError:
            pytest.skip("basic-pitch not installed")

        import subprocess
        import shutil

        # Mock shutil.which to simulate basic-pitch being available
        def mock_which(cmd):
            if cmd == "basic-pitch":
                return "/usr/bin/basic-pitch"  # Fake path
            return shutil.which(cmd)

        # Mock the help output to not include --model-serialization
        def mock_run(cmd, *args, **kwargs):
            if cmd == ["basic-pitch", "--help"]:
                # Return help text without --model-serialization flag
                class MockResult:
                    stdout = "Usage: basic-pitch [OPTIONS] OUTPUT INPUT\n\nOptions:\n  --save-midi"
                    returncode = 0
                return MockResult()
            # Default for other commands
            class MockResult:
                stdout = ""
                returncode = 0
            return MockResult()

        monkeypatch.setattr(shutil, "which", mock_which)
        monkeypatch.setattr(subprocess, "run", mock_run)

        # Recreate transcriber with mocked help output
        transcriber = BasicPitchTranscriber(model_serialization="tf")

        # Should detect that --model-serialization is not supported
        assert transcriber._supports_model_serialization is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
