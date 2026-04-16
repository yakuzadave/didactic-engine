#!/usr/bin/env python3
"""
ABC Notation Dataset Generation Example

This script demonstrates how to use the didactic-engine pipeline to generate
a dataset suitable for audio generation models conditioned on ABC notation.

The pipeline will:
1. Separate audio into stems (vocals, drums, bass, other)
2. Transcribe stems to MIDI using Basic Pitch
3. Quantize MIDI notes for cleaner ABC notation
4. Export ABC notation from quantized MIDI
5. Generate metadata.jsonl for HuggingFace AudioFolder

Usage:
    python abc_dataset_example.py input.wav output_dir/
    python abc_dataset_example.py --help

Requirements:
    - didactic-engine with [ml] extras (demucs, basic-pitch)
    - FFmpeg for audio processing
"""

import argparse
import logging
from pathlib import Path

from didactic_engine.config import PipelineConfig
from didactic_engine.pipeline import AudioPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate ABC notation dataset from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file with default settings
  python abc_dataset_example.py song.wav output/

  # Enable MIDI quantization (cleaner ABC notation)
  python abc_dataset_example.py song.wav output/ --quantize

  # Customize quantization division (16=sixteenth notes, 8=eighth notes)
  python abc_dataset_example.py song.wav output/ --quantize --division 8

  # Export metadata.jsonl for HuggingFace AudioFolder
  python abc_dataset_example.py song.wav output/ --metadata

  # Full dataset generation pipeline
  python abc_dataset_example.py song.wav output/ --quantize --metadata
        """
    )
    parser.add_argument(
        "input_wav",
        type=Path,
        help="Input WAV file to process"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for generated datasets"
    )
    parser.add_argument(
        "--song-id",
        type=str,
        help="Song identifier (default: input filename stem)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable MIDI quantization for cleaner ABC notation"
    )
    parser.add_argument(
        "--division",
        type=int,
        default=16,
        choices=[4, 8, 12, 16, 24],
        help="Quantization division (16=sixteenth notes, 8=eighth notes, etc.)"
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Export metadata.jsonl for HuggingFace AudioFolder"
    )
    parser.add_argument(
        "--trigger-token",
        type=str,
        default="abcstyle",
        help="Trigger token for ABC prompts (default: 'abcstyle')"
    )
    parser.add_argument(
        "--time-sig",
        type=str,
        default="4/4",
        help="Time signature (default: 4/4)"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip audio preprocessing"
    )
    parser.add_argument(
        "--analysis-sr",
        type=int,
        default=22050,
        choices=[16000, 22050, 44100],
        help="Sample rate for audio analysis (default: 22050)"
    )
    parser.add_argument(
        "--no-chunk-wavs",
        action="store_true",
        help="Skip writing per-bar chunk WAV files (faster, smaller)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_wav.exists():
        print(f"Error: Input file not found: {args.input_wav}")
        return 1
    
    # Determine song ID
    song_id = args.song_id if args.song_id else args.input_wav.stem
    
    # Parse time signature
    try:
        ts_parts = args.time_sig.split("/")
        time_sig_num = int(ts_parts[0])
        time_sig_den = int(ts_parts[1])
    except (ValueError, IndexError):
        print(f"Error: Invalid time signature: {args.time_sig}")
        print("Expected format: 4/4, 3/4, 6/8, etc.")
        return 1
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    # Create pipeline configuration
    logger.info("Configuring pipeline for ABC dataset generation")
    logger.info("  Input: %s", args.input_wav)
    logger.info("  Output: %s", args.output_dir)
    logger.info("  Song ID: %s", song_id)
    logger.info("  MIDI quantization: %s", "enabled" if args.quantize else "disabled")
    if args.quantize:
        logger.info("  Quantization division: 1/%d notes", args.division)
    logger.info("  Metadata export: %s", "enabled" if args.metadata else "disabled")
    
    config = PipelineConfig(
        song_id=song_id,
        input_wav=args.input_wav,
        out_dir=args.output_dir,
        # Audio processing
        use_pydub_preprocess=not args.no_preprocess,
        analysis_sr=args.analysis_sr,
        time_signature_num=time_sig_num,
        time_signature_den=time_sig_den,
        # Chunk generation
        write_bar_chunks=True,
        write_bar_chunk_wavs=not args.no_chunk_wavs,
        # ABC notation features
        quantize_midi=args.quantize,
        quantize_division=args.division,
        export_metadata_jsonl=args.metadata,
        abc_trigger_token=args.trigger_token,
    )
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    logger.info("Starting pipeline...")
    try:
        pipeline = AudioPipeline(config, logger=logger)
        results = pipeline.run()
        
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info("Output Summary:")
        logger.info("=" * 80)
        
        # Report ABC notation files
        abc_files = [k for k in results.keys() if k.startswith("abc_")]
        if abc_files:
            logger.info("\nABC Notation Files:")
            for key in abc_files:
                if not key.endswith("_text"):
                    logger.info("  - %s", results[key])
        
        # Report quantized MIDI files
        quantized_midi = [k for k in results.keys() if k.startswith("midi_quantized_")]
        if quantized_midi:
            logger.info("\nQuantized MIDI Files:")
            for key in quantized_midi:
                logger.info("  - %s", results[key])
        
        # Report metadata
        if "metadata_jsonl" in results:
            logger.info("\nMetadata Export:")
            logger.info("  - File: %s", results["metadata_jsonl"])
            logger.info("  - Entries: %d", results.get("metadata_entry_count", 0))
            logger.info("\nYou can now load this dataset with HuggingFace datasets:")
            logger.info("  from datasets import load_dataset")
            logger.info("  ds = load_dataset('audiofolder', data_dir='%s')", 
                       Path(results["metadata_jsonl"]).parent)
        
        # Report bar features
        if "bar_features_parquet" in results:
            logger.info("\nBar Features Dataset:")
            logger.info("  - %s", results["bar_features_parquet"])
        
        logger.info("\n" + "=" * 80)
        logger.info("Next Steps:")
        logger.info("=" * 80)
        logger.info("1. Review ABC notation files in: %s", config.reports_dir)
        logger.info("2. Check bar chunks in: %s", config.chunks_dir)
        if args.metadata:
            logger.info("3. Load the dataset for training:")
            logger.info("   from datasets import load_dataset")
            logger.info("   ds = load_dataset('audiofolder', data_dir='%s')", 
                       config.datasets_dir)
        logger.info("\n")
        
        return 0
        
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
