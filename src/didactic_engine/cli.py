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

  # Process multiple WAV files (batch mode with auto-generated song IDs)
  didactic-engine --wav song1.wav song2.wav song3.wav --out output/

  # Process all WAV files in a directory
  didactic-engine --wav *.wav --out output/

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
        nargs="+",
        required=True,
        type=Path,
        help="Input WAV file(s) to process. Can specify multiple files or use wildcards.",
    )
    parser.add_argument(
        "--song-id",
        help="Unique identifier for the song (required for single file, auto-generated for batch)",
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

    # Normalize wav to list for consistent handling
    wav_files = args.wav if isinstance(args.wav, list) else [args.wav]
    
    # Validate input files
    missing_files = [f for f in wav_files if not f.exists()]
    if missing_files:
        print(f"Error: Input file(s) not found:", file=sys.stderr)
        for f in missing_files:
            print(f"  - {f}", file=sys.stderr)
        return 1
    
    # Check if song_id is required (single file without song_id)
    if len(wav_files) == 1 and not args.song_id:
        print("Error: --song-id is required when processing a single file", file=sys.stderr)
        return 1
    
    # Batch processing
    if len(wav_files) > 1:
        # Warn if song_id is provided in batch mode
        if args.song_id:
            print("Warning: --song-id is ignored in batch mode. Song IDs will be auto-generated from filenames.", file=sys.stderr)
        
        print("=" * 60)
        print("Didactic Engine - Batch Processing Mode")
        print("=" * 60)
        print(f"Files to process: {len(wav_files)}")
        print(f"Output directory: {args.out}")
        print("=" * 60)
        
        results = []
        errors = []
        
        for idx, wav_file in enumerate(wav_files, 1):
            # Auto-generate song_id from filename
            song_id = wav_file.stem
            
            print(f"\n[{idx}/{len(wav_files)}] Processing: {wav_file.name}")
            print(f"  Song ID: {song_id}")
            
            # Build PipelineConfig for this file
            cfg = PipelineConfig(
                song_id=song_id,
                input_wav=wav_file,
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
            
            try:
                result = run_all(cfg)
                results.append((song_id, wav_file, "SUCCESS"))
                print(f"  ✓ Completed successfully")
            except Exception as e:
                errors.append((song_id, wav_file, str(e)))
                print(f"  ✗ Failed: {e}", file=sys.stderr)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Batch Processing Summary")
        print("=" * 60)
        print(f"Total files: {len(wav_files)}")
        print(f"Successful: {len(results)}")
        print(f"Failed: {len(errors)}")
        
        if errors:
            print("\nFailed files:")
            for song_id, wav_file, error in errors:
                print(f"  - {wav_file.name} ({song_id}): {error}")
        
        print(f"\nResults saved to: {args.out}")
        print("=" * 60)
        
        return 1 if errors else 0
    
    # Single file processing
    else:
        wav_file = wav_files[0]
        song_id = args.song_id
        
        # Build PipelineConfig
        cfg = PipelineConfig(
            song_id=song_id,
            input_wav=wav_file,
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
        print(f"Input:    {wav_file}")
        print(f"Song ID:  {song_id}")
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
