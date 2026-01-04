"""
Benchmark script for didactic-engine pipeline.
Tracks timing, memory usage, and issues encountered.
"""

import time
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark_results.log')
    ]
)

logger = logging.getLogger(__name__)

def benchmark_pipeline(
    input_wav: Path,
    output_dir: Path,
    preserve_chunk_audio: bool = False,
    run_name: str = "default"
):
    """Run pipeline and collect benchmark metrics."""
    from didactic_engine import PipelineConfig, AudioPipeline

    # Track metrics
    metrics = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_wav),
        "preserve_chunk_audio": preserve_chunk_audio,
        "steps": {},
        "warnings": [],
        "errors": [],
    }

    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARK RUN: {run_name}")
    logger.info(f"Input: {input_wav}")
    logger.info(f"Preserve chunk audio: {preserve_chunk_audio}")
    logger.info(f"{'='*80}\n")

    # Create config
    cfg = PipelineConfig(
        song_id=f"benchmark_{run_name}",
        input_wav=input_wav,
        out_dir=output_dir,
        analysis_sr=22050,
        write_bar_chunks=True,
        write_bar_chunk_wavs=True,
        preserve_chunk_audio=preserve_chunk_audio,
        use_pydub_preprocess=False,  # Disable to speed up
        time_signature_num=4,
        time_signature_den=4,
    )

    # Run pipeline with timing
    start_time = time.time()

    try:
        pipeline = AudioPipeline(cfg)

        # Capture warnings
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            results = pipeline.run()

            # Record warnings
            for warning in w:
                metrics["warnings"].append({
                    "category": warning.category.__name__,
                    "message": str(warning.message),
                    "filename": warning.filename,
                    "lineno": warning.lineno,
                })

        end_time = time.time()
        total_time = end_time - start_time

        # Collect metrics
        metrics["total_time_seconds"] = total_time
        metrics["success"] = True
        metrics["results"] = {k: str(v) if not isinstance(v, (int, float, bool)) else v
                             for k, v in results.items()}

        # Check output files
        output_stats = {}

        # Bar features
        if "bar_features_parquet" in results:
            bf_path = Path(results["bar_features_parquet"])
            if bf_path.exists():
                import pandas as pd
                df = pd.read_parquet(bf_path)
                output_stats["bar_features_count"] = len(df)
                output_stats["bar_features_size_mb"] = bf_path.stat().st_size / 1024 / 1024

        # Chunk files
        if cfg.chunks_dir.exists():
            chunk_files = list(cfg.chunks_dir.rglob("*.wav"))
            output_stats["chunk_files_count"] = len(chunk_files)
            total_chunk_size = sum(f.stat().st_size for f in chunk_files)
            output_stats["chunk_files_size_mb"] = total_chunk_size / 1024 / 1024

        metrics["output_stats"] = output_stats

        logger.info(f"\n{'='*80}")
        logger.info(f"BENCHMARK COMPLETE: {run_name}")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        logger.info(f"Bar features: {output_stats.get('bar_features_count', 0)} rows")
        logger.info(f"Chunk files: {output_stats.get('chunk_files_count', 0)} files " +
                   f"({output_stats.get('chunk_files_size_mb', 0):.2f} MB)")
        logger.info(f"Warnings: {len(metrics['warnings'])}")
        logger.info(f"{'='*80}\n")

    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time

        metrics["total_time_seconds"] = total_time
        metrics["success"] = False
        metrics["errors"].append({
            "type": type(e).__name__,
            "message": str(e),
            "traceback": __import__('traceback').format_exc()
        })

        logger.error(f"BENCHMARK FAILED: {run_name}")
        logger.error(f"Error: {e}")
        logger.error(__import__('traceback').format_exc())

    return metrics


def main():
    """Run benchmark scenarios."""
    input_wav = Path("data/sample_run/input/as_above_so_below_1_0_0.wav")
    output_base = Path("benchmark_output")

    if not input_wav.exists():
        logger.error(f"Input file not found: {input_wav}")
        return

    # Get file info
    import soundfile as sf
    info = sf.info(str(input_wav))
    logger.info(f"\nInput File Info:")
    logger.info(f"  Duration: {info.duration:.2f}s ({info.duration/60:.2f} min)")
    logger.info(f"  Sample rate: {info.samplerate} Hz")
    logger.info(f"  Channels: {info.channels}")
    logger.info(f"  Format: {info.format}, {info.subtype}")

    all_metrics = []

    # Scenario 1: Default settings (mono chunks at analysis_sr)
    logger.info("\n" + "="*80)
    logger.info("SCENARIO 1: Default Settings (preserve_chunk_audio=False)")
    logger.info("="*80)
    metrics1 = benchmark_pipeline(
        input_wav,
        output_base / "scenario1_default",
        preserve_chunk_audio=False,
        run_name="default_mono_chunks"
    )
    all_metrics.append(metrics1)

    # Scenario 2: Preserve original audio (stereo chunks at 48kHz)
    logger.info("\n" + "="*80)
    logger.info("SCENARIO 2: Preserve Original Audio (preserve_chunk_audio=True)")
    logger.info("="*80)
    metrics2 = benchmark_pipeline(
        input_wav,
        output_base / "scenario2_preserve",
        preserve_chunk_audio=True,
        run_name="preserve_stereo_chunks"
    )
    all_metrics.append(metrics2)

    # Save all metrics
    metrics_file = output_base / "benchmark_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\nMetrics saved to: {metrics_file}")

    # Print comparison
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)

    for m in all_metrics:
        if m["success"]:
            logger.info(f"\n{m['run_name']}:")
            logger.info(f"  Time: {m['total_time_seconds']:.2f}s ({m['total_time_seconds']/60:.2f} min)")
            logger.info(f"  Bar features: {m['output_stats'].get('bar_features_count', 0)} rows")
            logger.info(f"  Chunk files: {m['output_stats'].get('chunk_files_count', 0)} files")
            logger.info(f"  Chunk size: {m['output_stats'].get('chunk_files_size_mb', 0):.2f} MB")
            logger.info(f"  Warnings: {len(m['warnings'])}")
        else:
            logger.info(f"\n{m['run_name']}: FAILED")
            logger.info(f"  Time: {m['total_time_seconds']:.2f}s")
            logger.info(f"  Errors: {len(m['errors'])}")


if __name__ == "__main__":
    main()
