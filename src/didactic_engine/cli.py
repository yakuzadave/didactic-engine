"""
Command-line interface for the didactic-engine audio processing pipeline.
"""

import argparse
import sys
from pathlib import Path

from didactic_engine.config import PipelineConfig
from didactic_engine.pipeline import run_all


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="didactic-engine",
        description="Audio processing pipeline for stem separation, analysis, MIDI transcription, and feature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single WAV file
  didactic-engine --wav input.wav --song-id my_song --out output/

  # Process with custom sample rate
  didactic-engine --wav input.wav --song-id my_song --out output/ --sr 48000

  # Enable Essentia for advanced analysis
  didactic-engine --wav input.wav --song-id my_song --out output/ --use-essentia

  # Disable bar chunking
  didactic-engine --wav input.wav --song-id my_song --out output/ --no-bar-chunks
        """,
    )

    parser.add_argument(
        "--wav",
        required=True,
        type=Path,
        help="Input WAV file to process",
    )
    parser.add_argument(
        "--song-id",
        required=True,
        help="Unique identifier for the song",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("data"),
        help="Output directory for results (default: data)",
    )
    parser.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="Demucs model name (default: htdemucs)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Sample rate for analysis (default: 22050)",
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=512,
        help="Hop length for STFT (default: 512)",
    )
    parser.add_argument(
        "--ts-num",
        type=int,
        default=4,
        help="Time signature numerator (default: 4)",
    )
    parser.add_argument(
        "--ts-den",
        type=int,
        default=4,
        help="Time signature denominator (default: 4)",
    )
    parser.add_argument(
        "--use-essentia",
        action="store_true",
        help="Use Essentia for additional audio analysis",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip audio preprocessing",
    )
    parser.add_argument(
        "--no-bar-chunks",
        action="store_true",
        help="Skip per-bar audio chunking",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="didactic-engine 0.1.0",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.wav.exists():
        print(f"Error: Input file not found: {args.wav}", file=sys.stderr)
        return 1

    # Build PipelineConfig
    cfg = PipelineConfig(
        song_id=args.song_id,
        input_wav=args.wav,
        out_dir=args.out,
        demucs_model=args.demucs_model,
        analysis_sr=args.sr,
        hop_length=args.hop,
        time_signature_num=args.ts_num,
        time_signature_den=args.ts_den,
        use_pydub_preprocess=not args.no_preprocess,
        use_essentia_features=args.use_essentia,
        write_bar_chunks=not args.no_bar_chunks,
    )

    # Run pipeline
    print("=" * 60)
    print("Didactic Engine - Audio Processing Pipeline")
    print("=" * 60)
    print(f"Input:    {args.wav}")
    print(f"Song ID:  {args.song_id}")
    print(f"Output:   {args.out}")
    print("=" * 60)

    try:
        run_all(cfg)
        print("\n" + "=" * 60)
        print("Processing completed successfully!")
        print(f"Results saved to: {args.out}")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nError during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
