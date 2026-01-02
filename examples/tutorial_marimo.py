# /// script
# dependencies = ["git+https://github.com/yakuzadave/didactic-engine.git"]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import subprocess
    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Didactic Engine Tutorial

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yakuzadave/didactic-engine/blob/main/examples/tutorial.ipynb)

    This notebook demonstrates how to install and use **Didactic Engine**, a comprehensive Python audio processing toolkit for music analysis, stem separation, MIDI transcription, and dataset generation.

    ## What is Didactic Engine?

    Didactic Engine is an end-to-end audio processing pipeline that:

    1. **Ingests** WAV files and validates their format
    2. **Preprocesses** audio (resampling, mono conversion, normalization, silence trimming)
    3. **Separates stems** using Demucs (vocals, drums, bass, other)
    4. **Extracts features** (tempo, beats, spectral features, MFCCs, chroma)
    5. **Transcribes to MIDI** using Spotify's Basic Pitch
    6. **Aligns notes** to a beat/bar grid based on a configurable time signature
    7. **Segments audio** into per-bar WAV chunks
    8. **Generates datasets** in Parquet format (events, beats, bars, bar features)
    9. **Exports reports** in Markdown and ABC notation

    ## Requirements

    - Python 3.11 or higher
    - FFmpeg (for audio processing)
    - Optional: Demucs and Basic Pitch (for advanced features)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Installation

    ### Detect Environment (Local vs Google Colab)
    """)
    return


@app.cell
def _():
    # Detect if running in Google Colab
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("Running in local environment")
    return (IN_COLAB,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Install System Dependencies

    FFmpeg is required for audio processing with pydub.
    """)
    return


@app.cell
def _(IN_COLAB, subprocess):
    # Install FFmpeg (required for pydub)
    if IN_COLAB:
        #! apt-get update -qq
        subprocess.call(['apt-get', 'update', '-qq'])
        #! apt-get install -y -qq ffmpeg
        subprocess.call(['apt-get', 'install', '-y', '-qq', 'ffmpeg'])
    else:
        print("Please ensure FFmpeg is installed on your system:")
        print("  - Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  - macOS: brew install ffmpeg")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Install Didactic Engine

    Install the core package and optionally install ML extras (Demucs, Basic Pitch).
    """)
    return


@app.cell
def _(IN_COLAB, subprocess):
    # Install didactic-engine from GitHub
    if IN_COLAB:
        subprocess.call(['pip', 'install', '-q', 'git+https://github.com/yakuzadave/didactic-engine.git'])
    else:
        print("To install locally, run:")
        print("  git clone https://github.com/yakuzadave/didactic-engine.git")
        print("  cd didactic-engine")
        print("  pip install -e .")
        print("\nOr install directly from GitHub:")
        print("  pip install git+https://github.com/yakuzadave/didactic-engine.git")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Install Optional Dependencies

    For advanced features, you can install:
    - **ML extras**: Demucs (stem separation), Basic Pitch (MIDI transcription)
    - **Essentia**: Advanced audio features
    """)
    return


@app.cell
def _():
    # Optional: Install ML features (Demucs, Basic Pitch)
    # Note: Basic Pitch requires Python 3.11 (not available on Python 3.12+)
    # Uncomment the following line to install:
    # !pip install -q demucs basic-pitch

    # Optional: Install Essentia for advanced audio features
    # Uncomment the following line to install:
    # !pip install -q essentia

    print("Optional dependencies can be installed as needed.")
    print("For full functionality: pip install demucs basic-pitch essentia")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Import Libraries

    Let's import the necessary libraries and verify the installation.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import numpy as np

    # Import didactic-engine components
    from didactic_engine import AudioPipeline, PipelineConfig
    from didactic_engine.ingestion import WAVIngester
    from didactic_engine.analysis import AudioAnalyzer
    from didactic_engine.preprocessing import AudioPreprocessor

    print("✓ Successfully imported didactic_engine")

    # Check for optional dependencies
    try:
        import librosa
        print("✓ librosa available")
    except ImportError:
        print("✗ librosa not available")

    try:
        import soundfile
        print("✓ soundfile available")
    except ImportError:
        print("✗ soundfile not available")

    try:
        import pydub
        print("✓ pydub available")
    except ImportError:
        print("✗ pydub not available")
    return (
        AudioAnalyzer,
        AudioPipeline,
        AudioPreprocessor,
        Path,
        PipelineConfig,
        WAVIngester,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Generate Sample Audio (For Testing)

    Since we may not have a real audio file, let's generate a simple synthetic audio signal for demonstration purposes.
    """)
    return


@app.cell
def _(Path, np):
    import soundfile as sf

    # Create a sample audio file (2 seconds, 440 Hz sine wave - A4 note)
    sample_rate = 22050
    duration = 2.0  # seconds
    frequency = 440.0  # Hz (A4)

    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Add some variation (simple envelope)
    envelope = np.exp(-t / duration)
    audio_data = audio_data * envelope

    # Create output directory
    sample_dir = Path("sample_audio")
    sample_dir.mkdir(exist_ok=True)

    # Save as WAV file
    sample_wav_path = sample_dir / "test_audio.wav"
    sf.write(sample_wav_path, audio_data, sample_rate)

    print(f"✓ Generated sample audio: {sample_wav_path}")
    print(f"  Duration: {duration} seconds")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Frequency: {frequency} Hz (A4 note)")
    return (sample_wav_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Basic Usage: Load and Analyze Audio

    Let's use individual components to load and analyze the audio file.
    """)
    return


@app.cell
def _(WAVIngester, sample_wav_path):
    # Load audio using WAVIngester
    ingester = WAVIngester(sample_rate=22050)
    audio, sr = ingester.load(str(sample_wav_path))

    print(f"Audio loaded:")
    print(f"  Shape: {audio.shape}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio) / sr:.2f} seconds")
    print(f"  Min value: {audio.min():.4f}")
    print(f"  Max value: {audio.max():.4f}")
    return audio, sr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analyze Audio Features

    Extract tempo, beats, and audio features using the AudioAnalyzer.
    """)
    return


@app.cell
def _(AudioAnalyzer, audio, sr):
    # Analyze audio (without Essentia)
    analyzer = AudioAnalyzer(use_essentia=False)
    analysis_result = analyzer.analyze(audio, sr)

    print("Audio Analysis Results:")
    print(f"  Tempo: {analysis_result['tempo']:.2f} BPM")
    print(f"  Number of beats: {len(analysis_result['beat_times'])}")
    print(f"\nSpectral Features:")
    print(f"  Spectral centroid (mean): {analysis_result['spectral_centroid_mean']:.2f} Hz")
    print(f"  Spectral bandwidth (mean): {analysis_result['spectral_bandwidth_mean']:.2f} Hz")
    print(f"  Zero crossing rate (mean): {analysis_result['zero_crossing_rate_mean']:.4f}")
    print(f"\nEnergy:")
    print(f"  RMS energy (mean): {analysis_result['rms_mean']:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preprocess Audio

    Apply normalization and other preprocessing steps.
    """)
    return


@app.cell
def _(AudioPreprocessor, audio, np, sr):
    # Preprocess audio
    preprocessor = AudioPreprocessor()
    normalized_audio = preprocessor.normalize(audio, sr)

    print("Audio Preprocessing:")
    print(f"  Original max amplitude: {np.abs(audio).max():.4f}")
    print(f"  Normalized max amplitude: {np.abs(normalized_audio).max():.4f}")

    # You can also trim silence
    trimmed_audio = preprocessor.trim_silence(audio, sr)
    print(f"  Original length: {len(audio)} samples")
    print(f"  Trimmed length: {len(trimmed_audio)} samples")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Full Pipeline Usage

    Now let's use the full AudioPipeline to process an audio file end-to-end.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Configure the Pipeline

    Create a PipelineConfig with all the settings for processing.
    """)
    return


@app.cell
def _(Path, PipelineConfig, sample_wav_path):
    # Configure the pipeline
    config = PipelineConfig(
        song_id="sample_song",
        input_wav=sample_wav_path,
        out_dir=Path("output"),
        analysis_sr=22050,
        time_signature_num=4,
        time_signature_den=4,
        use_essentia_features=False,  # Set to True if Essentia is installed
        write_bar_chunks=True,
    )

    print("Pipeline Configuration:")
    print(f"  Song ID: {config.song_id}")
    print(f"  Input WAV: {config.input_wav}")
    print(f"  Output directory: {config.out_dir}")
    print(f"  Sample rate: {config.analysis_sr} Hz")
    print(f"  Time signature: {config.time_signature_num}/{config.time_signature_den}")
    print(f"  Essentia features: {config.use_essentia_features}")
    print(f"  Write bar chunks: {config.write_bar_chunks}")
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run the Pipeline

    **Note**: The full pipeline requires Demucs and Basic Pitch to be installed.
    For this demo, we'll show the API but expect it might fail if those tools aren't available.
    """)
    return


@app.cell
def _(AudioPipeline, config):
    # Create the pipeline
    pipeline = AudioPipeline(config)

    # Run the pipeline (this may fail if Demucs/Basic Pitch are not installed)
    try:
        print("Running full pipeline...")
        print("This will:")
        print("  1. Copy input file")
        print("  2. Separate stems (Demucs)")
        print("  3. Preprocess audio")
        print("  4. Analyze audio features")
        print("  5. Transcribe to MIDI (Basic Pitch)")
        print("  6. Align notes to beat grid")
        print("  7. Create bar chunks")
        print("  8. Generate datasets")
        print("  9. Export reports\n")

        # Execute the pipeline for its side effects (writing outputs to disk).
        pipeline.run()

        print("✓ Pipeline completed successfully!")
        print(f"\nResults saved to: {config.out_dir}")
    except Exception as e:
        print(f"⚠ Pipeline execution encountered an issue: {e}")
        print("\nThis is expected if Demucs or Basic Pitch are not installed.")
        print("To use the full pipeline, install:")
        print("  pip install demucs basic-pitch")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Understanding the Output Structure

    When the full pipeline runs successfully, it creates the following directory structure:
    """)
    return


@app.cell
def _(Path):
    # Display the output directory structure
    import os

    def show_directory_tree(path, prefix="", max_depth=3, current_depth=0):
        """Display directory tree structure."""
        if current_depth >= max_depth:
            return
    
        path = Path(path)
        if not path.exists():
            print(f"{prefix}(Directory not yet created)")
            return
    
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
        
            if item.is_dir():
                extension = "    " if is_last else "│   "
                show_directory_tree(item, prefix + extension, max_depth, current_depth + 1)

    print("\nExpected Output Directory Structure:")
    print("\noutput/")
    print("├── input/<song_id>/          # Original input file")
    print("├── preprocessed/<song_id>/   # Preprocessed audio")
    print("├── stems/<song_id>/          # Separated stems (vocals, drums, bass, other)")
    print("├── chunks/<song_id>/<stem>/  # Per-bar audio chunks")
    print("├── midi/<song_id>/           # MIDI transcriptions")
    print("├── analysis/<song_id>/       # Analysis results (combined.json)")
    print("├── reports/<song_id>/        # Markdown and ABC notation reports")
    print("└── datasets/<song_id>/       # Parquet datasets")
    print("    ├── events.parquet        # Individual note events")
    print("    ├── beats.parquet         # Beat grid information")
    print("    ├── bars.parquet          # Bar-level aggregations")
    print("    └── bar_features.parquet  # Bar-level audio features")

    print("\n\nActual output directory (if created):")
    if Path("output").exists():
        show_directory_tree("output")
    else:
        print("(Not yet created - run the full pipeline to generate)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Working with Pipeline Outputs

    If the pipeline runs successfully, you can load and inspect the generated datasets.
    """)
    return


@app.cell
def _(config):
    import pandas as pd

    # Check if datasets were generated
    datasets_dir = config.datasets_dir

    if datasets_dir.exists():
        print("Loading generated datasets...\n")
    
        # Load events dataset
        events_path = datasets_dir / "events.parquet"
        if events_path.exists():
            events_df = pd.read_parquet(events_path)
            print(f"Events Dataset: {len(events_df)} rows")
            print(events_df.head())
            print()
    
        # Load bars dataset
        bars_path = datasets_dir / "bars.parquet"
        if bars_path.exists():
            bars_df = pd.read_parquet(bars_path)
            print(f"\nBars Dataset: {len(bars_df)} rows")
            print(bars_df.head())
            print()
    
        # Load bar features dataset
        bar_features_path = datasets_dir / "bar_features.parquet"
        if bar_features_path.exists():
            bar_features_df = pd.read_parquet(bar_features_path)
            print(f"\nBar Features Dataset: {len(bar_features_df)} rows")
            print(bar_features_df.head())
    else:
        print("Datasets not yet generated.")
        print("Run the full pipeline with Demucs and Basic Pitch installed to generate datasets.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Advanced: Custom Configuration

    You can customize various aspects of the pipeline.
    """)
    return


@app.cell
def _(Path, PipelineConfig, sample_wav_path):
    # Example: 3/4 time signature (waltz)
    custom_config = PipelineConfig(
        song_id="waltz_example",
        input_wav=sample_wav_path,
        out_dir=Path("output_custom"),
        analysis_sr=44100,  # Higher sample rate
        time_signature_num=3,  # 3/4 time
        time_signature_den=4,
        use_pydub_preprocess=True,
        preprocess_normalize=True,
        preprocess_trim_silence=True,
        use_essentia_features=False,
        write_bar_chunks=True,
    )

    print("Custom Configuration:")
    print(f"  Time signature: {custom_config.time_signature_num}/{custom_config.time_signature_den}")
    print(f"  Sample rate: {custom_config.analysis_sr} Hz")
    print(f"  Preprocessing: {custom_config.use_pydub_preprocess}")
    print(f"  Normalize: {custom_config.preprocess_normalize}")
    print(f"  Trim silence: {custom_config.preprocess_trim_silence}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Troubleshooting

    ### Common Issues

    1. **FFmpeg not found**: Install FFmpeg for your system
       - Ubuntu/Debian: `sudo apt-get install ffmpeg`
       - macOS: `brew install ffmpeg`
       - Windows: Download from https://ffmpeg.org/download.html

    2. **Demucs command not found**: Install Demucs
       ```bash
       pip install demucs
       ```

    3. **Basic Pitch not available**: Install Basic Pitch (Python 3.11 only)
       ```bash
       pip install basic-pitch
       ```

    4. **Essentia import error**: Install Essentia (optional)
       ```bash
       pip install essentia
       ```

    ### Python Version Compatibility

    - **Python 3.11**: Full support including Basic Pitch (TensorFlow backend)
    - **Python 3.12+**: Core features work, but Basic Pitch is not available due to TensorFlow compatibility

    ### Memory Issues

    Processing large audio files may require significant memory. Consider:
    - Using a lower sample rate (e.g., 22050 Hz instead of 44100 Hz)
    - Processing shorter audio clips
    - Running on a machine with more RAM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Next Steps

    Now that you've seen the basics, here are some ideas for further exploration:

    1. **Try with real music**: Use your own WAV files
    2. **Experiment with different time signatures**: 3/4, 6/8, etc.
    3. **Analyze the Parquet datasets**: Use pandas to explore the generated data
    4. **Enable Essentia features**: Get advanced audio analysis metrics
    5. **Build a music dataset**: Process multiple songs and aggregate the results
    6. **Integrate with ML models**: Use the extracted features for machine learning tasks

    ### Additional Resources

    - GitHub Repository: https://github.com/yakuzadave/didactic-engine
    - Documentation: Check the README.md and docs/ folder
    - Examples: See examples/example_usage.py for more code samples
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Cleanup (Optional)

    Remove generated files and directories.
    """)
    return


@app.cell
def _():
    import shutil

    # Uncomment to clean up generated files
    # if Path("sample_audio").exists():
    #     shutil.rmtree("sample_audio")
    #     print("✓ Removed sample_audio/")

    # if Path("output").exists():
    #     shutil.rmtree("output")
    #     print("✓ Removed output/")

    # if Path("output_custom").exists():
    #     shutil.rmtree("output_custom")
    #     print("✓ Removed output_custom/")

    print("To clean up, uncomment the lines above and run this cell.")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
