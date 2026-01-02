"""
Batch processing examples for Didactic Engine with progress tracking and error handling.

This module demonstrates different patterns for processing multiple WAV files:
1. Simple batch processing with default settings
2. Parallel processing with multiprocessing
3. Batch processing with progress bars and logging
4. Error recovery and retry patterns
5. Processing from a directory structure
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json

from didactic_engine.pipeline import AudioPipeline
from didactic_engine.config import PipelineConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BatchResults:
    """Container for batch processing results."""
    total: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    errors: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total': self.total,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': self.successful / self.total if self.total > 0 else 0,
            'results': self.results,
            'errors': self.errors
        }


def example_simple_batch():
    """Simple batch processing using AudioPipeline.process_batch()."""
    print("="*60)
    print("Example 1: Simple Batch Processing")
    print("="*60)
    
    # Define input files
    input_files = [
        Path("sample_audio/song1.wav"),
        Path("sample_audio/song2.wav"),
        Path("sample_audio/song3.wav"),
    ]
    
    # Check which files exist
    existing_files = [f for f in input_files if f.exists()]
    if not existing_files:
        logger.warning("No input files found. This is a demonstration.")
        logger.info("To run, place WAV files in sample_audio/ directory")
        return
    
    output_dir = Path("output/batch_simple")
    
    # Process all files at once
    results = AudioPipeline.process_batch(
        wav_files=existing_files,
        out_dir=output_dir,
        analysis_sr=22050,
        use_essentia_features=False,
        write_bar_chunks=True,
    )
    
    # Display results
    print(f"\nProcessed {results['success_count']}/{len(existing_files)} files successfully")
    
    if results['failed']:
        print("\nFailed files:")
        for song_id, path, error in results['failed']:
            print(f"  - {song_id}: {error}")
    
    return results


def process_single_file(wav_path: Path, output_dir: Path, **kwargs) -> Dict[str, Any]:
    """
    Process a single WAV file with error handling.
    
    Args:
        wav_path: Path to input WAV file
        output_dir: Base output directory
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary with processing results or error information
    """
    song_id = wav_path.stem
    
    try:
        cfg = PipelineConfig(
            song_id=song_id,
            input_wav=wav_path,
            out_dir=output_dir,
            **kwargs
        )
        
        pipeline = AudioPipeline(cfg)
        result = pipeline.run()
        
        return {
            'song_id': song_id,
            'path': str(wav_path),
            'status': 'success',
            'tempo': result['analysis']['tempo_bpm'],
            'duration': result['duration_s'],
            'num_bars': result['num_bars'],
        }
        
    except Exception as e:
        logger.error(f"Failed to process {wav_path}: {e}")
        return {
            'song_id': song_id,
            'path': str(wav_path),
            'status': 'error',
            'error': str(e),
        }


def example_parallel_batch():
    """Parallel batch processing using ProcessPoolExecutor."""
    print("\n" + "="*60)
    print("Example 2: Parallel Batch Processing")
    print("="*60)
    
    # Define input files
    input_dir = Path("sample_audio")
    input_files = list(input_dir.glob("*.wav"))
    
    if not input_files:
        logger.warning(f"No WAV files found in {input_dir}")
        return
    
    output_dir = Path("output/batch_parallel")
    max_workers = 4  # Number of parallel processes
    
    results = []
    errors = {}
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(
                process_single_file,
                wav_path,
                output_dir,
                analysis_sr=22050,
                use_essentia_features=False,
                write_bar_chunks=True,
            ): wav_path
            for wav_path in input_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            wav_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    logger.info(f"✓ Processed {result['song_id']}: "
                              f"{result['tempo']:.1f} BPM, {result['duration']:.1f}s")
                else:
                    errors[result['song_id']] = result['error']
                    logger.error(f"✗ Failed {result['song_id']}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Exception processing {wav_path}: {e}")
                errors[wav_path.stem] = str(e)
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\nProcessed {successful}/{len(input_files)} files successfully")
    print(f"Failed: {len(errors)} files")
    
    return BatchResults(
        total=len(input_files),
        successful=successful,
        failed=len(errors),
        results=results,
        errors=errors
    )


def example_batch_with_progress():
    """Batch processing with tqdm progress bar (requires tqdm package)."""
    print("\n" + "="*60)
    print("Example 3: Batch Processing with Progress Bar")
    print("="*60)
    
    try:
        from tqdm import tqdm
    except ImportError:
        logger.warning("tqdm not installed. Install with: pip install tqdm")
        logger.info("Falling back to basic progress logging")
        tqdm = None
    
    input_dir = Path("sample_audio")
    input_files = list(input_dir.glob("*.wav"))
    
    if not input_files:
        logger.warning(f"No WAV files found in {input_dir}")
        return
    
    output_dir = Path("output/batch_progress")
    results = []
    errors = {}
    
    # Create progress bar
    iterator = tqdm(input_files, desc="Processing audio") if tqdm else input_files
    
    for wav_path in iterator:
        result = process_single_file(
            wav_path,
            output_dir,
            analysis_sr=22050,
            use_essentia_features=False,
            write_bar_chunks=True,
        )
        results.append(result)
        
        if result['status'] == 'error':
            errors[result['song_id']] = result['error']
        
        # Update progress bar description
        if tqdm:
            successful = sum(1 for r in results if r['status'] == 'success')
            iterator.set_postfix({
                'success': successful,
                'failed': len(errors)
            })
    
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\n✓ Successfully processed: {successful}/{len(input_files)}")
    
    return BatchResults(
        total=len(input_files),
        successful=successful,
        failed=len(errors),
        results=results,
        errors=errors
    )


def example_batch_from_directory_tree():
    """Process WAV files from a nested directory structure."""
    print("\n" + "="*60)
    print("Example 4: Processing from Directory Tree")
    print("="*60)
    
    # Process all WAV files recursively
    input_root = Path("sample_audio")
    output_root = Path("output/batch_tree")
    
    # Find all WAV files recursively
    input_files = list(input_root.rglob("*.wav"))
    
    if not input_files:
        logger.warning(f"No WAV files found under {input_root}")
        return
    
    logger.info(f"Found {len(input_files)} WAV files")
    
    results = []
    errors = {}
    
    for wav_path in input_files:
        # Preserve directory structure in output
        relative_path = wav_path.relative_to(input_root)
        song_id = str(relative_path.with_suffix('')).replace('/', '_')
        
        try:
            cfg = PipelineConfig(
                song_id=song_id,
                input_wav=wav_path,
                out_dir=output_root,
                analysis_sr=22050,
                write_bar_chunks=True,
            )
            
            pipeline = AudioPipeline(cfg)
            result = pipeline.run()
            
            results.append({
                'song_id': song_id,
                'path': str(wav_path),
                'status': 'success',
                'tempo': result['analysis']['tempo_bpm'],
                'duration': result['duration_s'],
            })
            logger.info(f"✓ {song_id}")
            
        except Exception as e:
            logger.error(f"✗ {song_id}: {e}")
            errors[song_id] = str(e)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\nProcessed {successful}/{len(input_files)} files")
    
    return BatchResults(
        total=len(input_files),
        successful=successful,
        failed=len(errors),
        results=results,
        errors=errors
    )


def example_batch_with_retry():
    """Batch processing with retry logic for failed files."""
    print("\n" + "="*60)
    print("Example 5: Batch Processing with Retry")
    print("="*60)
    
    input_dir = Path("sample_audio")
    input_files = list(input_dir.glob("*.wav"))
    
    if not input_files:
        logger.warning(f"No WAV files found in {input_dir}")
        return
    
    output_dir = Path("output/batch_retry")
    max_retries = 2
    
    results = []
    errors = {}
    
    for wav_path in input_files:
        song_id = wav_path.stem
        
        for attempt in range(max_retries + 1):
            try:
                cfg = PipelineConfig(
                    song_id=song_id,
                    input_wav=wav_path,
                    out_dir=output_dir,
                    analysis_sr=22050,
                    write_bar_chunks=True,
                )
                
                pipeline = AudioPipeline(cfg)
                result = pipeline.run()
                
                results.append({
                    'song_id': song_id,
                    'status': 'success',
                    'attempts': attempt + 1,
                })
                logger.info(f"✓ {song_id} (attempt {attempt + 1})")
                break
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {song_id}")
                else:
                    logger.error(f"✗ {song_id} failed after {max_retries + 1} attempts")
                    errors[song_id] = str(e)
                    results.append({
                        'song_id': song_id,
                        'status': 'error',
                        'attempts': attempt + 1,
                        'error': str(e),
                    })
    
    successful = sum(1 for r in results if r.get('status') == 'success')
    print(f"\nSuccessful: {successful}/{len(input_files)}")
    
    return BatchResults(
        total=len(input_files),
        successful=successful,
        failed=len(errors),
        results=results,
        errors=errors
    )


def save_batch_results(results: BatchResults, output_path: Path):
    """Save batch processing results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """Run all batch processing examples."""
    print("Didactic Engine - Batch Processing Examples")
    print("="*60)
    
    # Run examples
    try:
        # Simple batch
        results1 = example_simple_batch()
        
        # Parallel processing
        results2 = example_parallel_batch()
        if results2:
            save_batch_results(results2, Path("output/batch_parallel/results.json"))
        
        # With progress bar
        results3 = example_batch_with_progress()
        if results3:
            save_batch_results(results3, Path("output/batch_progress/results.json"))
        
        # Directory tree
        results4 = example_batch_from_directory_tree()
        if results4:
            save_batch_results(results4, Path("output/batch_tree/results.json"))
        
        # With retry
        results5 = example_batch_with_retry()
        if results5:
            save_batch_results(results5, Path("output/batch_retry/results.json"))
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
    
    print("\n" + "="*60)
    print("Batch processing examples completed")
    print("="*60)


if __name__ == "__main__":
    main()
