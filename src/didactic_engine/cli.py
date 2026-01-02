"""
Command-line interface for the didactic-engine audio processing pipeline.
"""

import argparse
import sys
import os
from pathlib import Path

from didactic_engine.pipeline import AudioPipeline


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Audio processing pipeline for stem separation, analysis, and feature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single WAV file
  didactic-engine input.wav -o output/

  # Process with custom sample rate
  didactic-engine input.wav -o output/ --sample-rate 48000

  # Enable Essentia for advanced analysis
  didactic-engine input.wav -o output/ --use-essentia

  # Process multiple files
  didactic-engine file1.wav file2.wav file3.wav -o output/
        """,
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input WAV file(s) to process",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "-sr",
        "--sample-rate",
        type=int,
        default=44100,
        help="Target sample rate for processing (default: 44100)",
    )
    parser.add_argument(
        "--use-essentia",
        action="store_true",
        help="Use Essentia for additional audio analysis",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip stem preprocessing",
    )
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        default=4,
        help="Number of beats per bar (default: 4)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="didactic-engine 0.1.0",
    )

    args = parser.parse_args()

    # Validate input files
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}", file=sys.stderr)
            return 1

    # Create pipeline
    print("Initializing audio processing pipeline...")
    pipeline = AudioPipeline(
        sample_rate=args.sample_rate,
        use_essentia=args.use_essentia,
        preprocess_stems=not args.no_preprocess,
        beats_per_bar=args.beats_per_bar,
    )

    # Process files
    try:
        if len(args.input_files) == 1:
            # Single file processing
            results = pipeline.process(args.input_files[0], args.output_dir)
            print("\n" + "="*60)
            print("Processing completed successfully!")
            print(f"Results saved to: {args.output_dir}")
            print("="*60)
        else:
            # Batch processing
            results = pipeline.process_batch(args.input_files, args.output_dir)
            print("\n" + "="*60)
            print(f"Batch processing completed!")
            print(f"Processed {len(results)} files")
            print(f"Results saved to: {args.output_dir}")
            print("="*60)
        
        return 0

    except Exception as e:
        print(f"\nError during processing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
