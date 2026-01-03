"""
Command-line interface for the didactic-engine audio processing pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

from didactic_engine.config import PipelineConfig
from didactic_engine.pipeline import run_all

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="didactic-engine",
        description=(
            "Audio processing pipeline for stem separation, analysis, MIDI "
            "transcription, and feature extraction"
        ),
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
        "--demucs-device",
        default="cpu",
        help=(
            "Demucs device to use: cpu, cuda, cuda:0, etc. "
            "(default: cpu)"
        ),
    )
    parser.add_argument(
        "--demucs-timeout",
        type=float,
        default=None,
        help=(
            "Optional timeout (seconds) for Demucs separation. "
            "If not set, Demucs is allowed to run without a timeout."
        ),
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
        "--no-chunk-wavs",
        action="store_true",
        help=(
            "Do not write bar chunk WAV files to disk (faster / less I/O). "
            "Bar features are still computed unless --no-bar-chunks is set."
        ),
    )
    parser.add_argument(
        "--preserve-chunk-audio",
        action="store_true",
        help=(
            "When writing chunk WAVs, preserve original sample rate and channels. "
            "Features are still computed from mono audio at --sr."
        ),
    )
    parser.add_argument(
        "--basic-pitch-backend",
        default="tf",
        choices=["tf", "onnx", "tflite", "coreml"],
        help=(
            "Basic Pitch inference backend (default: tf). "
            "On WSL, 'onnx' can be faster (and GPU-capable) when paired with onnxruntime-gpu."
        ),
    )
    parser.add_argument(
        "--basic-pitch-timeout",
        type=float,
        default=None,
        help=(
            "Optional timeout (seconds) for Basic Pitch transcription per stem. "
            "If not set, Basic Pitch is allowed to run without a timeout."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="didactic-engine 0.1.0",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Normalize wav to list for consistent handling
    wav_files = args.wav if isinstance(args.wav, list) else [args.wav]
    
    # Validate input files
    missing_files = [f for f in wav_files if not f.exists()]
    if missing_files:
        logger.error("Input file(s) not found:")
        for f in missing_files:
            logger.error("  - %s", f)
        return 1
    
    # Check if song_id is required (single file without song_id)
    if len(wav_files) == 1 and not args.song_id:
        logger.error("--song-id is required when processing a single file")
        return 1
    
    # Batch processing
    if len(wav_files) > 1:
        # Warn if song_id is provided in batch mode
        if args.song_id:
            logger.warning(
                "--song-id is ignored in batch mode. Song IDs are auto-generated from filenames."
            )
        
        logger.info("Didactic Engine - Batch Processing Mode")
        logger.info("Files to process: %d", len(wav_files))
        logger.info("Output directory: %s", args.out)
        
        # Use AudioPipeline.process_batch() to handle batch processing
        from didactic_engine.pipeline import AudioPipeline
        
        batch_results = AudioPipeline.process_batch(
            wav_files,
            args.out,
            demucs_model=args.demucs_model,
            demucs_device=args.demucs_device,
            demucs_timeout_s=args.demucs_timeout,
            analysis_sr=args.sr,
            hop_length=args.hop,
            time_signature_num=args.ts_num,
            time_signature_den=args.ts_den,
            use_pydub_preprocess=not args.no_preprocess,
            use_essentia_features=args.use_essentia,
            write_bar_chunks=not args.no_bar_chunks,
            write_bar_chunk_wavs=(not args.no_chunk_wavs) and (
                not args.no_bar_chunks),
            preserve_chunk_audio=args.preserve_chunk_audio,
            basic_pitch_backend=args.basic_pitch_backend,
            basic_pitch_timeout_s=args.basic_pitch_timeout,
            logger=logger,
        )
        
        logger.info(
            "Batch results: total=%d successful=%d failed=%d",
            batch_results["total"],
            batch_results["success_count"],
            batch_results["failure_count"],
        )
        
        if batch_results["failed"]:
            logger.warning("Failed files:")
            for song_id, input_path, error in batch_results["failed"]:
                filename = Path(input_path).name
                logger.warning("  - %s (%s): %s", filename, song_id, error)
        
        logger.info("Results saved to: %s", args.out)
        
        return 1 if batch_results["failed"] else 0
    
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
            demucs_device=args.demucs_device,
            demucs_timeout_s=args.demucs_timeout,
            analysis_sr=args.sr,
            hop_length=args.hop,
            time_signature_num=args.ts_num,
            time_signature_den=args.ts_den,
            use_pydub_preprocess=not args.no_preprocess,
            use_essentia_features=args.use_essentia,
            write_bar_chunks=not args.no_bar_chunks,
            write_bar_chunk_wavs=(not args.no_chunk_wavs) and (
                not args.no_bar_chunks),
            preserve_chunk_audio=args.preserve_chunk_audio,
            basic_pitch_backend=args.basic_pitch_backend,
            basic_pitch_timeout_s=args.basic_pitch_timeout,
        )

        # Run pipeline
        logger.info("Didactic Engine - Audio Processing Pipeline")
        logger.info("Input: %s", wav_file)
        logger.info("Song ID: %s", song_id)
        logger.info("Output: %s", args.out)

        try:
            run_all(cfg, logger=logger)
            logger.info(
                "Processing completed successfully; results saved to: %s", args.out)
            return 0

        except Exception as e:
            logger.exception("Error during processing: %s", e)
            return 1


if __name__ == "__main__":
    sys.exit(main())
