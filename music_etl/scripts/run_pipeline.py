#!/usr/bin/env python3
"""
Command-line entry point for the music ETL pipeline.
"""

import argparse
import sys
from pathlib import Path

from music_etl.config import PipelineConfig
from music_etl.pipeline import run_all


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Music ETL Pipeline - Process audio for stem separation, analysis, and feature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/run_pipeline.py --wav data/input/song.wav --song-id my_song

  # With custom output directory
  python scripts/run_pipeline.py --wav input.wav --song-id song1 --out output/

  # With custom time signature (3/4 waltz)
  python scripts/run_pipeline.py --wav waltz.wav --song-id waltz --ts-num 3

  # Disable preprocessing
  python scripts/run_pipeline.py --wav song.wav --song-id song1 --no-preprocess
        """,
    )

    # Required arguments
    parser.add_argument(
        "--wav",
        type=Path,
        required=True,
        help="Path to input WAV file",
    )
    parser.add_argument(
        "--song-id",
        type=str,
        required=True,
        help="Unique identifier for this song",
    )

    # Optional arguments
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="Output base directory (default: data)",
    )
    parser.add_argument(
        "--demucs-model",
        type=str,
        default="htdemucs",
        help="Demucs model to use (default: htdemucs)",
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
        help="Hop length for analysis (default: 512)",
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
        "--no-preprocess",
        action="store_true",
        help="Skip pydub preprocessing",
    )
    parser.add_argument(
        "--no-essentia",
        action="store_true",
        help="Skip Essentia feature extraction",
    )
    parser.add_argument(
        "--no-chunks",
        action="store_true",
        help="Skip writing per-bar audio chunks",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.wav.exists():
        print(f"Error: Input WAV file not found: {args.wav}", file=sys.stderr)
        return 1

    # Build configuration
    cfg = PipelineConfig(
        song_id=args.song_id,
        input_wav=args.wav,
        out_dir=args.out,
        analysis_sr=args.sr,
        hop_length=args.hop,
        time_signature_num=args.ts_num,
        time_signature_den=args.ts_den,
        demucs_model=args.demucs_model,
        use_pydub_preprocess=not args.no_preprocess,
        use_essentia_features=not args.no_essentia,
        write_bar_chunks=not args.no_chunks,
    )

    # Run pipeline
    try:
        run_all(cfg)
        return 0
    except Exception as e:
        print(f"\nError: Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
