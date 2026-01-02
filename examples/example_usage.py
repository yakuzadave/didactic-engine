"""
Example script demonstrating the didactic-engine audio processing pipeline.
"""

import os
from didactic_engine.pipeline import AudioPipeline


def example_basic_usage():
    """Basic usage example."""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Initialize pipeline with default settings
    pipeline = AudioPipeline(
        sample_rate=44100,
        use_essentia=False,
        preprocess_stems=True,
        beats_per_bar=4,
    )
    
    # Process a WAV file
    input_wav = "path/to/your/audio.wav"
    output_dir = "output/basic_example"
    
    # Check if example file exists
    if not os.path.exists(input_wav):
        print(f"Note: Create or specify a WAV file at: {input_wav}")
        print("Skipping execution - this is just an example structure.")
        return
    
    results = pipeline.process(input_wav, output_dir)
    
    print("\nResults:")
    print(f"  - Tempo: {results['analysis']['tempo']:.2f} BPM")
    print(f"  - Number of bars: {len(results['bar_times'])}")
    print(f"  - Stems separated: {results['stem_names']}")
    print(f"  - Total notes in MIDI: {results.get('midi_info', {}).get('total_notes', 0)}")


def example_advanced_usage():
    """Advanced usage example with Essentia."""
    print("\n" + "="*60)
    print("Example 2: Advanced Usage with Essentia")
    print("="*60)
    
    # Initialize pipeline with Essentia enabled
    pipeline = AudioPipeline(
        sample_rate=48000,
        use_essentia=True,  # Enable advanced analysis
        preprocess_stems=True,
        beats_per_bar=4,
    )
    
    input_wav = "path/to/your/audio.wav"
    output_dir = "output/advanced_example"
    
    if not os.path.exists(input_wav):
        print("Note: Specify a valid WAV file path")
        return
    
    results = pipeline.process(input_wav, output_dir)
    
    # Access Essentia-specific features
    if "essentia" in results["analysis"]:
        essentia_results = results["analysis"]["essentia"]
        print(f"\nEssentia Analysis:")
        print(f"  - Key: {essentia_results.get('key', 'N/A')}")
        print(f"  - Scale: {essentia_results.get('scale', 'N/A')}")
        print(f"  - BPM: {essentia_results.get('bpm', 'N/A')}")


def example_batch_processing():
    """Batch processing example."""
    print("\n" + "="*60)
    print("Example 3: Batch Processing")
    print("="*60)
    
    pipeline = AudioPipeline()
    
    # Process multiple files
    input_files = [
        "path/to/audio1.wav",
        "path/to/audio2.wav",
        "path/to/audio3.wav",
    ]
    output_base_dir = "output/batch_example"
    
    # Check if files exist
    existing_files = [f for f in input_files if os.path.exists(f)]
    if not existing_files:
        print("Note: Add valid WAV file paths to process")
        return
    
    results = pipeline.process_batch(existing_files, output_base_dir)
    
    print(f"\nProcessed {len(results)} files successfully")


def example_custom_configuration():
    """Example with custom configuration."""
    print("\n" + "="*60)
    print("Example 4: Custom Configuration")
    print("="*60)
    
    # Custom pipeline configuration
    pipeline = AudioPipeline(
        sample_rate=48000,        # Higher sample rate
        use_essentia=True,         # Advanced analysis
        preprocess_stems=True,     # Apply normalization
        beats_per_bar=3,          # 3/4 time signature (waltz)
    )
    
    print("Pipeline configured with:")
    print(f"  - Sample rate: {pipeline.sample_rate} Hz")
    print(f"  - Essentia enabled: {pipeline.use_essentia}")
    print(f"  - Preprocessing: {pipeline.preprocess_stems}")
    print(f"  - Beats per bar: {pipeline.beats_per_bar}")


def example_accessing_components():
    """Example of accessing individual pipeline components."""
    print("\n" + "="*60)
    print("Example 5: Accessing Individual Components")
    print("="*60)
    
    from didactic_engine.ingestion import WAVIngester
    from didactic_engine.analysis import AudioAnalyzer
    from didactic_engine.features import FeatureExtractor
    
    # Use individual components
    ingester = WAVIngester(sample_rate=44100)
    analyzer = AudioAnalyzer(use_essentia=False)
    feature_extractor = FeatureExtractor()
    
    input_wav = "path/to/audio.wav"
    
    if not os.path.exists(input_wav):
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
