"""
Example script demonstrating the didactic-engine audio processing pipeline.
"""

import os
from pathlib import Path
from didactic_engine.pipeline import AudioPipeline
from didactic_engine.config import PipelineConfig


def example_basic_usage():
    """Basic usage example."""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Define paths
    input_wav = Path("path/to/your/audio.wav")
    output_dir = Path("output/basic_example")
    
    # Check if example file exists
    if not input_wav.exists():
        print(f"Note: Create or specify a WAV file at: {input_wav}")
        print("Skipping execution - this is just an example structure.")
        return
    
    # Create configuration
    cfg = PipelineConfig(
        song_id="my_song",
        input_wav=input_wav,
        out_dir=output_dir,
        analysis_sr=22050,  # Sample rate for analysis
        use_essentia_features=False,
        write_bar_chunks=True,
    )
    
    # Initialize and run pipeline
    pipeline = AudioPipeline(cfg)
    results = pipeline.run()
    
    print("\nResults:")
    print(f"  - Tempo: {results['analysis']['tempo_bpm']:.2f} BPM")
    print(f"  - Number of bars: {results['num_bars']}")
    print(f"  - Stems: {results.get('stems', [])}")


def example_advanced_usage():
    """Advanced usage example with Essentia."""
    print("\n" + "="*60)
    print("Example 2: Advanced Usage with Essentia")
    print("="*60)
    
    input_wav = Path("path/to/your/audio.wav")
    output_dir = Path("output/advanced_example")
    
    if not input_wav.exists():
        print("Note: Specify a valid WAV file path")
        return
    
    # Create configuration with Essentia enabled
    cfg = PipelineConfig(
        song_id="my_song_advanced",
        input_wav=input_wav,
        out_dir=output_dir,
        analysis_sr=22050,
        use_essentia_features=True,  # Enable advanced analysis
        write_bar_chunks=True,
        time_signature_num=4,
        time_signature_den=4,
    )
    
    # Run pipeline
    pipeline = AudioPipeline(cfg)
    results = pipeline.run()
    
    print(f"\nProcessed: {results['song_id']}")
    print(f"Duration: {results['duration_s']:.2f}s")
    print(f"Tempo: {results['analysis']['tempo_bpm']:.2f} BPM")


def example_batch_processing():
    """Batch processing example."""
    print("\n" + "="*60)
    print("Example 3: Batch Processing")
    print("="*60)
    
    # Process multiple files
    input_files = [
        Path("path/to/audio1.wav"),
        Path("path/to/audio2.wav"),
        Path("path/to/audio3.wav"),
    ]
    output_base_dir = Path("output/batch_example")
    
    # Check if files exist
    existing_files = [f for f in input_files if f.exists()]
    if not existing_files:
        print("Note: Add valid WAV file paths to process")
        return
    
    # Process batch using static method
    results = AudioPipeline.process_batch(
        existing_files,
        output_base_dir,
        analysis_sr=22050,
        use_essentia_features=False,
    )
    
    print(f"\nProcessed {results['success_count']} files successfully")
    print(f"Failed: {results['failure_count']} files")
    
    if results['failed']:
        print("\nFailed files:")
        for song_id, path, error in results['failed']:
            print(f"  - {song_id}: {error}")


def example_custom_configuration():
    """Example with custom configuration."""
    print("\n" + "="*60)
    print("Example 4: Custom Configuration")
    print("="*60)
    
    # Custom pipeline configuration
    cfg = PipelineConfig(
        song_id="waltz_song",
        input_wav=Path("path/to/waltz.wav"),
        out_dir=Path("output/custom_example"),
        analysis_sr=22050,
        use_essentia_features=True,
        write_bar_chunks=True,
        time_signature_num=3,  # 3/4 time signature (waltz)
        time_signature_den=4,
        use_pydub_preprocess=True,
        preprocess_target_sr=44100,
    )
    
    print("Pipeline configured with:")
    print(f"  - Sample rate: {cfg.analysis_sr} Hz")
    print(f"  - Preprocess target SR: {cfg.preprocess_target_sr} Hz")
    print(f"  - Essentia enabled: {cfg.use_essentia_features}")
    print(f"  - Time signature: {cfg.time_signature_num}/{cfg.time_signature_den}")


def example_accessing_components():
    """Example of accessing individual pipeline components."""
    print("\n" + "="*60)
    print("Example 5: Accessing Individual Components")
    print("="*60)
    
    from didactic_engine.ingestion import WAVIngester
    from didactic_engine.analysis import AudioAnalyzer
    from didactic_engine.features import FeatureExtractor
    
    # Use individual components
    ingester = WAVIngester(sample_rate=22050)
    analyzer = AudioAnalyzer(use_essentia=False)
    feature_extractor = FeatureExtractor()
    
    input_wav = Path("path/to/audio.wav")
    
    if not input_wav.exists():
        print("Note: Specify a valid WAV file to run this example")
        return
    
    # Load audio
    audio, sr = ingester.load(input_wav)
    print(f"Loaded audio: shape={audio.shape}, sr={sr}")
    
    # Analyze
    analysis = analyzer.analyze(audio, sr)
    print(f"Detected tempo: {analysis['tempo']:.2f} BPM")
    
    # Extract features from the full audio
    features = feature_extractor.extract_bar_features(audio, sr)
    print(f"Extracted {len(features)} features")


if __name__ == "__main__":
    print("Didactic Engine - Audio Processing Pipeline Examples")
    print("="*60)
    
    # Run examples (these are demonstrations - they won't execute without valid audio files)
    try:
        example_basic_usage()
        example_advanced_usage()
        example_batch_processing()
        example_custom_configuration()
        example_accessing_components()
    except Exception as e:
        print(f"\nNote: Examples require valid audio files to execute.")
        print(f"This script demonstrates the API usage structure.")
        print(f"\nError encountered: {e}")
    
    print("\n" + "="*60)
    print("Examples completed")
    print("="*60)
