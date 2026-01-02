"""
Tests for example scripts to ensure they remain valid and executable.

These tests validate that:
- Example scripts can be imported without errors
- Example code structure is correct
- API calls used in examples are valid
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

# Import example functions
from examples.example_usage import (
    example_basic_usage,
    example_advanced_usage,
    example_batch_processing,
    example_custom_configuration,
    example_accessing_components,
)

from didactic_engine.pipeline import AudioPipeline
from didactic_engine.config import PipelineConfig
from didactic_engine.ingestion import WAVIngester
from didactic_engine.analysis import AudioAnalyzer
from didactic_engine.features import FeatureExtractor


class TestExampleImports:
    """Test that example modules can be imported."""

    def test_import_example_usage(self):
        """Test that example_usage module can be imported."""
        import examples.example_usage
        assert hasattr(examples.example_usage, 'example_basic_usage')
        assert hasattr(examples.example_usage, 'example_advanced_usage')
        assert hasattr(examples.example_usage, 'example_batch_processing')
        assert hasattr(examples.example_usage, 'example_custom_configuration')
        assert hasattr(examples.example_usage, 'example_accessing_components')


class TestExampleFunctions:
    """Test that example functions execute without errors when files don't exist."""

    def test_example_basic_usage_no_file(self):
        """Test basic usage example handles missing file gracefully."""
        # Should print message and return without error
        example_basic_usage()

    def test_example_advanced_usage_no_file(self):
        """Test advanced usage example handles missing file gracefully."""
        # Should print message and return without error
        example_advanced_usage()

    def test_example_batch_processing_no_files(self):
        """Test batch processing example handles missing files gracefully."""
        # Should print message and return without error
        example_batch_processing()

    def test_example_custom_configuration_no_file(self):
        """Test custom configuration example runs without file."""
        # Should only print configuration, no file required
        example_custom_configuration()

    def test_example_accessing_components_no_file(self):
        """Test component access example handles missing file gracefully."""
        # Should print message and return without error
        example_accessing_components()


class TestExampleAPIStructure:
    """Test that example code uses correct API structure."""

    def test_pipeline_config_creation(self):
        """Test PipelineConfig can be created as shown in examples."""
        cfg = PipelineConfig(
            song_id="test_song",
            input_wav=Path("test.wav"),
            out_dir=Path("output/test"),
            analysis_sr=22050,
            use_essentia_features=False,
            write_bar_chunks=True,
        )
        assert cfg.song_id == "test_song"
        assert cfg.analysis_sr == 22050
        assert cfg.use_essentia_features is False

    def test_pipeline_config_with_time_signature(self):
        """Test PipelineConfig with custom time signature as in examples."""
        cfg = PipelineConfig(
            song_id="waltz",
            input_wav=Path("test.wav"),
            out_dir=Path("output/test"),
            time_signature_num=3,
            time_signature_den=4,
        )
        assert cfg.time_signature_num == 3
        assert cfg.time_signature_den == 4

    def test_individual_components_instantiation(self):
        """Test that individual components can be created as in examples."""
        ingester = WAVIngester(sample_rate=22050)
        analyzer = AudioAnalyzer(use_essentia=False)
        feature_extractor = FeatureExtractor()
        
        assert ingester is not None
        assert analyzer is not None
        assert feature_extractor is not None


class TestExampleWorkflow:
    """Test example workflows with actual audio data."""

    def test_individual_components_workflow(self):
        """Test the individual components workflow from examples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test audio file
            sample_rate = 22050
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            test_wav = Path(tmpdir) / "test_audio.wav"
            sf.write(test_wav, audio_data, sample_rate)
            
            # Use components as shown in examples
            ingester = WAVIngester(sample_rate=22050)
            analyzer = AudioAnalyzer(use_essentia=False)
            feature_extractor = FeatureExtractor()
            
            # Load audio
            audio, sr = ingester.load(test_wav)
            assert audio is not None
            assert sr == 22050
            
            # Analyze
            analysis = analyzer.analyze(audio, sr)
            assert 'tempo' in analysis
            # Tempo detection may return 0 for simple sine waves, which is valid
            assert analysis['tempo'] >= 0
            
            # Extract features
            features = feature_extractor.extract_bar_features(audio, sr)
            assert len(features) > 0
            assert 'rms' in features

    def test_basic_pipeline_workflow(self):
        """Test basic pipeline workflow as shown in examples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test audio file
            sample_rate = 22050
            duration = 2.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            test_wav = Path(tmpdir) / "test_audio.wav"
            sf.write(test_wav, audio_data, sample_rate)
            
            output_dir = Path(tmpdir) / "output"
            
            # Create config as in examples
            cfg = PipelineConfig(
                song_id="test_song",
                input_wav=test_wav,
                out_dir=output_dir,
                analysis_sr=22050,
                use_essentia_features=False,
                write_bar_chunks=False,  # Disable to speed up test
            )
            
            # Run pipeline
            pipeline = AudioPipeline(cfg)
            results = pipeline.run()
            
            # Verify results structure matches what examples expect
            assert 'analysis' in results
            assert 'tempo_bpm' in results['analysis']
            assert 'num_bars' in results
            # Tempo detection may return 0 for simple sine waves, which is valid
            assert results['analysis']['tempo_bpm'] >= 0

    def test_batch_processing_workflow(self):
        """Test batch processing workflow as shown in examples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test audio files
            sample_rate = 22050
            duration = 1.0
            
            input_files = []
            for i in range(2):
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                freq = 440 + i * 50
                audio_data = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
                
                test_wav = Path(tmpdir) / f"test_{i}.wav"
                sf.write(test_wav, audio_data, sample_rate)
                input_files.append(test_wav)
            
            output_dir = Path(tmpdir) / "output"
            
            # Process batch as in examples
            results = AudioPipeline.process_batch(
                input_files,
                output_dir,
                analysis_sr=22050,
                use_essentia_features=False,
                write_bar_chunks=False,
            )
            
            # Verify results structure matches what examples expect
            assert 'success_count' in results
            assert 'failure_count' in results
            assert 'failed' in results
            assert 'successful' in results
            assert results['success_count'] == 2
            assert results['failure_count'] == 0


class TestExampleNotebookCompatibility:
    """Test that code patterns used in notebooks are valid."""

    def test_notebook_imports(self):
        """Test that imports used in tutorial notebook are valid."""
        from pathlib import Path
        from didactic_engine import AudioPipeline, PipelineConfig
        from didactic_engine.ingestion import WAVIngester
        from didactic_engine.analysis import AudioAnalyzer
        from didactic_engine.preprocessing import AudioPreprocessor
        
        # All imports should succeed
        assert AudioPipeline is not None
        assert PipelineConfig is not None
        assert WAVIngester is not None
        assert AudioAnalyzer is not None
        assert AudioPreprocessor is not None

    def test_notebook_config_pattern(self):
        """Test config pattern used in notebook."""
        config = PipelineConfig(
            song_id="sample_song",
            input_wav=Path("sample_audio/test_audio.wav"),
            out_dir=Path("output"),
            analysis_sr=22050,
            time_signature_num=4,
            time_signature_den=4,
            use_essentia_features=False,
            write_bar_chunks=True,
        )
        
        assert config.song_id == "sample_song"
        assert config.time_signature_num == 4
        assert config.analysis_sr == 22050

    def test_notebook_analyzer_pattern(self):
        """Test analyzer usage pattern from notebook."""
        analyzer = AudioAnalyzer(use_essentia=False)
        
        # Create synthetic audio as in notebook
        duration = 2.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Analyze as in notebook
        analysis = analyzer.analyze(audio, sample_rate)
        
        # Check expected fields from notebook
        assert 'tempo' in analysis
        assert 'beat_times' in analysis
        assert 'librosa' in analysis
        assert isinstance(analysis['tempo'], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
