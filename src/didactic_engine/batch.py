"""
Enhanced batch processing module with resilience features.

This module provides robust batch processing capabilities including:
- Parallel processing with configurable worker pools
- Automatic retry with exponential backoff
- Progress tracking and reporting
- Graceful error handling and recovery
- Resource management and cleanup

The batch processor is designed for production workloads where
reliability and visibility are critical.

Example:
    >>> from didactic_engine.batch import BatchProcessor, BatchConfig
    >>>
    >>> config = BatchConfig(
    ...     max_workers=4,
    ...     retry_config=RetryConfig(max_retries=3),
    ...     progress_enabled=True,
    ... )
    >>> processor = BatchProcessor(config)
    >>> 
    >>> results = processor.process(
    ...     input_files=[Path("song1.wav"), Path("song2.wav")],
    ...     output_dir=Path("output"),
    ... )
    >>> print(f"Success rate: {results.success_rate:.1%}")
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from didactic_engine.resilience import RetryConfig, compute_delay

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ProcessingStatus(Enum):
    """Status of a file processing result."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class FileResult:
    """Result of processing a single file.
    
    Attributes:
        song_id: Song identifier.
        input_path: Path to input file.
        status: Processing status.
        output_dir: Output directory (if successful).
        tempo_bpm: Detected tempo (if successful).
        duration_s: Audio duration in seconds.
        num_bars: Number of bars detected.
        error: Error message (if failed).
        attempts: Number of processing attempts.
        processing_time_s: Time taken to process.
    """
    song_id: str
    input_path: Path
    status: ProcessingStatus
    output_dir: Optional[Path] = None
    tempo_bpm: Optional[float] = None
    duration_s: Optional[float] = None
    num_bars: Optional[int] = None
    error: Optional[str] = None
    attempts: int = 1
    processing_time_s: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "song_id": self.song_id,
            "input_path": str(self.input_path),
            "status": self.status.value,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "tempo_bpm": self.tempo_bpm,
            "duration_s": self.duration_s,
            "num_bars": self.num_bars,
            "error": self.error,
            "attempts": self.attempts,
            "processing_time_s": round(self.processing_time_s, 3),
        }


@dataclass
class BatchResult:
    """Aggregate result of batch processing.
    
    Attributes:
        total: Total number of files.
        successful: Number of successful files.
        failed: Number of failed files.
        skipped: Number of skipped files.
        results: Individual file results.
        start_time: When batch started.
        end_time: When batch completed.
        total_time_s: Total processing time.
    """
    total: int
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[FileResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    total_time_s: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Percentage of successful files."""
        if self.total == 0:
            return 0.0
        return self.successful / self.total
    
    @property
    def failure_rate(self) -> float:
        """Percentage of failed files."""
        if self.total == 0:
            return 0.0
        return self.failed / self.total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": round(self.success_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "total_time_s": round(self.total_time_s, 3),
            "avg_time_per_file_s": round(
                self.total_time_s / self.total if self.total > 0 else 0, 3
            ),
            "results": [r.to_dict() for r in self.results],
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved batch results to %s", path)
    
    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        print("\n" + "=" * 60)
        print("Batch Processing Summary")
        print("=" * 60)
        print(f"Total files:    {self.total}")
        print(f"Successful:     {self.successful} ({self.success_rate:.1%})")
        print(f"Failed:         {self.failed} ({self.failure_rate:.1%})")
        print(f"Skipped:        {self.skipped}")
        print(f"Total time:     {self.total_time_s:.1f}s")
        if self.total > 0:
            print(f"Avg per file:   {self.total_time_s / self.total:.1f}s")
        
        if self.failed > 0:
            print("\nFailed files:")
            for result in self.results:
                if result.status == ProcessingStatus.FAILED:
                    print(f"  - {result.song_id}: {result.error}")
        
        print("=" * 60)


@dataclass
class BatchConfig:
    """Configuration for batch processing.
    
    Attributes:
        max_workers: Maximum parallel workers (0 = sequential).
        use_processes: Use processes instead of threads.
        retry_config: Configuration for retry behavior.
        progress_enabled: Whether to show progress bars.
        stop_on_error: Stop batch on first error.
        skip_existing: Skip files with existing output.
        cleanup_on_error: Clean up partial outputs on error.
        save_results: Automatically save results to JSON.
    """
    max_workers: int = 4
    use_processes: bool = True
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    progress_enabled: bool = True
    stop_on_error: bool = False
    skip_existing: bool = False
    cleanup_on_error: bool = True
    save_results: bool = True


def _process_single_file(
    input_path: Path,
    output_dir: Path,
    song_id: Optional[str],
    retry_config: RetryConfig,
    pipeline_kwargs: Dict[str, Any],
) -> FileResult:
    """Process a single file with retry logic.
    
    This is the worker function called by the batch processor.
    It handles retries, timing, and error capture.
    """
    # Import here to avoid issues with multiprocessing
    from didactic_engine.pipeline import AudioPipeline
    from didactic_engine.config import PipelineConfig
    
    if song_id is None:
        song_id = input_path.stem
    
    start_time = time.time()
    last_error: Optional[Exception] = None
    attempts = 0
    
    for attempt in range(retry_config.max_retries + 1):
        attempts = attempt + 1
        
        try:
            # Create configuration
            cfg = PipelineConfig(
                song_id=song_id,
                input_wav=input_path,
                out_dir=output_dir,
                **pipeline_kwargs,
            )
            
            # Run pipeline
            pipeline = AudioPipeline(cfg)
            result = pipeline.run()
            
            # Success
            processing_time = time.time() - start_time
            return FileResult(
                song_id=song_id,
                input_path=input_path,
                status=ProcessingStatus.SUCCESS,
                output_dir=cfg.datasets_dir,
                tempo_bpm=result.get("analysis", {}).get("tempo_bpm"),
                duration_s=result.get("duration_s"),
                num_bars=result.get("num_bars"),
                attempts=attempts,
                processing_time_s=processing_time,
            )
        
        except FileNotFoundError as e:
            # Don't retry for missing files
            processing_time = time.time() - start_time
            return FileResult(
                song_id=song_id,
                input_path=input_path,
                status=ProcessingStatus.FAILED,
                error=f"File not found: {e}",
                attempts=attempts,
                processing_time_s=processing_time,
            )
        
        except Exception as e:
            last_error = e
            
            if attempt < retry_config.max_retries:
                # Calculate delay and wait
                delay = compute_delay(
                    attempt,
                    retry_config.base_delay,
                    retry_config.max_delay,
                    retry_config.exponential_base,
                    retry_config.jitter,
                )
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %.2fs...",
                    attempt + 1,
                    retry_config.max_retries + 1,
                    song_id,
                    str(e),
                    delay,
                )
                time.sleep(delay)
    
    # All retries exhausted
    processing_time = time.time() - start_time
    return FileResult(
        song_id=song_id,
        input_path=input_path,
        status=ProcessingStatus.FAILED,
        error=str(last_error) if last_error else "Unknown error",
        attempts=attempts,
        processing_time_s=processing_time,
    )


class BatchProcessor:
    """Enhanced batch processor with resilience features.
    
    Provides parallel processing with automatic retry, progress
    tracking, and comprehensive error handling.
    
    Example:
        >>> processor = BatchProcessor(BatchConfig(max_workers=4))
        >>> 
        >>> results = processor.process(
        ...     input_files=[Path("song1.wav"), Path("song2.wav")],
        ...     output_dir=Path("output"),
        ...     analysis_sr=22050,
        ... )
        >>> 
        >>> results.print_summary()
    """
    
    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the batch processor.
        
        Args:
            config: Batch processing configuration.
            logger: Optional logger for messages.
        """
        self.config = config or BatchConfig()
        self.logger = logger or logging.getLogger(__name__)
    
    def process(
        self,
        input_files: Sequence[Union[str, Path]],
        output_dir: Union[str, Path],
        song_ids: Optional[Sequence[str]] = None,
        **pipeline_kwargs: Any,
    ) -> BatchResult:
        """Process multiple audio files.
        
        Args:
            input_files: Paths to input WAV files.
            output_dir: Base output directory.
            song_ids: Optional song IDs (defaults to filenames).
            **pipeline_kwargs: Additional PipelineConfig parameters.
        
        Returns:
            BatchResult with individual and aggregate results.
        """
        input_files = [Path(p) for p in input_files]
        output_dir = Path(output_dir)
        
        if not input_files:
            return BatchResult(total=0)
        
        if song_ids is None:
            song_ids = [f.stem for f in input_files]
        
        if len(song_ids) != len(input_files):
            raise ValueError(
                f"song_ids length ({len(song_ids)}) must match "
                f"input_files length ({len(input_files)})"
            )
        
        result = BatchResult(total=len(input_files))
        result.start_time = time.time()
        
        # Setup progress bar if available
        try:
            from tqdm import tqdm
            progress_enabled = self.config.progress_enabled
        except ImportError:
            tqdm = None
            progress_enabled = False
        
        self.logger.info("Starting batch processing of %d files...", len(input_files))
        
        if self.config.max_workers <= 1:
            # Sequential processing
            result = self._process_sequential(
                input_files, output_dir, song_ids, pipeline_kwargs,
                tqdm if progress_enabled else None,
            )
        else:
            # Parallel processing
            result = self._process_parallel(
                input_files, output_dir, song_ids, pipeline_kwargs,
                tqdm if progress_enabled else None,
            )
        
        result.end_time = time.time()
        result.total_time_s = result.end_time - result.start_time
        
        self.logger.info(
            "Batch complete: %d/%d successful (%.1f%%) in %.1fs",
            result.successful,
            result.total,
            result.success_rate * 100,
            result.total_time_s,
        )
        
        # Auto-save results
        if self.config.save_results:
            results_path = output_dir / "batch_results.json"
            result.save(results_path)
        
        return result
    
    def _process_sequential(
        self,
        input_files: List[Path],
        output_dir: Path,
        song_ids: Sequence[str],
        pipeline_kwargs: Dict[str, Any],
        tqdm_class: Optional[type],
    ) -> BatchResult:
        """Process files sequentially."""
        result = BatchResult(total=len(input_files))
        
        iterator = zip(input_files, song_ids)
        if tqdm_class:
            iterator = tqdm_class(
                list(iterator),
                desc="Processing",
                unit="file",
            )
        
        for input_path, song_id in iterator:
            file_result = self._process_with_skip_check(
                input_path, output_dir, song_id, pipeline_kwargs
            )
            result.results.append(file_result)
            
            if file_result.status == ProcessingStatus.SUCCESS:
                result.successful += 1
            elif file_result.status == ProcessingStatus.FAILED:
                result.failed += 1
                if self.config.stop_on_error:
                    self.logger.error("Stopping batch due to error")
                    break
            elif file_result.status == ProcessingStatus.SKIPPED:
                result.skipped += 1
        
        return result
    
    def _process_parallel(
        self,
        input_files: List[Path],
        output_dir: Path,
        song_ids: Sequence[str],
        pipeline_kwargs: Dict[str, Any],
        tqdm_class: Optional[type],
    ) -> BatchResult:
        """Process files in parallel."""
        result = BatchResult(total=len(input_files))
        
        # Choose executor type
        Executor = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor
        
        with Executor(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(
                    _process_single_file,
                    input_path,
                    output_dir,
                    song_id,
                    self.config.retry_config,
                    pipeline_kwargs,
                ): (input_path, song_id)
                for input_path, song_id in zip(input_files, song_ids)
            }
            
            # Setup progress bar
            completed_futures = as_completed(futures)
            if tqdm_class:
                completed_futures = tqdm_class(
                    completed_futures,
                    total=len(futures),
                    desc="Processing",
                    unit="file",
                )
            
            # Collect results
            for future in completed_futures:
                input_path, song_id = futures[future]
                
                try:
                    file_result = future.result()
                except Exception as e:
                    file_result = FileResult(
                        song_id=song_id,
                        input_path=input_path,
                        status=ProcessingStatus.FAILED,
                        error=f"Executor error: {e}",
                    )
                
                result.results.append(file_result)
                
                if file_result.status == ProcessingStatus.SUCCESS:
                    result.successful += 1
                    self.logger.info(
                        "✓ %s: %.1f BPM, %.1fs",
                        file_result.song_id,
                        file_result.tempo_bpm or 0,
                        file_result.duration_s or 0,
                    )
                elif file_result.status == ProcessingStatus.FAILED:
                    result.failed += 1
                    self.logger.error(
                        "✗ %s: %s",
                        file_result.song_id,
                        file_result.error,
                    )
                elif file_result.status == ProcessingStatus.SKIPPED:
                    result.skipped += 1
        
        return result
    
    def _process_with_skip_check(
        self,
        input_path: Path,
        output_dir: Path,
        song_id: str,
        pipeline_kwargs: Dict[str, Any],
    ) -> FileResult:
        """Process a file with optional skip check for existing outputs."""
        # Check if should skip
        if self.config.skip_existing:
            expected_output = output_dir / "datasets" / song_id / "events.parquet"
            if expected_output.exists():
                self.logger.info("Skipping %s (output exists)", song_id)
                return FileResult(
                    song_id=song_id,
                    input_path=input_path,
                    status=ProcessingStatus.SKIPPED,
                    output_dir=expected_output.parent,
                )
        
        return _process_single_file(
            input_path,
            output_dir,
            song_id,
            self.config.retry_config,
            pipeline_kwargs,
        )


def process_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    pattern: str = "**/*.wav",
    config: Optional[BatchConfig] = None,
    **pipeline_kwargs: Any,
) -> BatchResult:
    """Process all WAV files in a directory tree.
    
    Convenience function for processing entire directories.
    
    Args:
        input_dir: Root directory to search.
        output_dir: Output directory.
        pattern: Glob pattern for finding files.
        config: Batch configuration.
        **pipeline_kwargs: Additional pipeline parameters.
    
    Returns:
        BatchResult with processing results.
    
    Example:
        >>> results = process_directory(
        ...     "music_library/",
        ...     "processed/",
        ...     pattern="**/*.wav",
        ...     max_workers=8,
        ... )
    """
    input_dir = Path(input_dir)
    input_files = list(input_dir.glob(pattern))
    
    if not input_files:
        logger.warning("No files found matching %s in %s", pattern, input_dir)
        return BatchResult(total=0)
    
    logger.info("Found %d files matching %s", len(input_files), pattern)
    
    processor = BatchProcessor(config or BatchConfig())
    return processor.process(input_files, output_dir, **pipeline_kwargs)


__all__ = [
    "ProcessingStatus",
    "FileResult",
    "BatchResult",
    "BatchConfig",
    "BatchProcessor",
    "process_directory",
]
