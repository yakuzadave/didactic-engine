#!/usr/bin/env python3
"""
Complete workflow: Batch process → Aggregate → Visualize

This script demonstrates a production-ready workflow:
1. Batch process multiple WAV files in parallel
2. Aggregate results across all songs
3. Generate comparative visualizations
4. Export summary reports

Usage:
    python workflow_batch_viz.py sample_audio/ --output results/
    python workflow_batch_viz.py sample_audio/*.wav --workers 4
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

try:
    from didactic_engine import AudioPipeline, PipelineConfig
except ImportError:
    print("Error: didactic-engine not installed")
    print("Install with: pip install -e .")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_single_file(wav_path: Path, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single WAV file with the pipeline.
    
    Args:
        wav_path: Path to WAV file
        output_dir: Output directory
        config: Pipeline configuration parameters
        
    Returns:
        Dictionary with processing results
    """
    song_id = wav_path.stem
    
    try:
        cfg = PipelineConfig(
            song_id=song_id,
            input_wav=wav_path,
            out_dir=output_dir,
            **config
        )
        
        logger.info(f"Processing: {song_id}")
        pipeline = AudioPipeline(cfg)
        result = pipeline.run()
        
        return {
            'song_id': song_id,
            'path': str(wav_path),
            'status': 'success',
            'duration': result['duration_s'],
            'tempo': result['analysis']['tempo_bpm'],
            'num_bars': result['num_bars'],
            'num_beats': len(result['analysis']['beat_times']),
            'stems': result.get('stems', []),
        }
        
    except Exception as e:
        logger.error(f"Failed to process {song_id}: {e}")
        return {
            'song_id': song_id,
            'path': str(wav_path),
            'status': 'error',
            'error': str(e),
        }


def batch_process(
    input_files: List[Path],
    output_dir: Path,
    max_workers: int = 4,
    analysis_sr: int = 22050,
    write_bar_chunks: bool = True,
) -> Dict[str, Any]:
    """
    Process multiple files in parallel.
    
    Args:
        input_files: List of WAV file paths
        output_dir: Output directory
        max_workers: Number of parallel workers
        analysis_sr: Sample rate for analysis
        write_bar_chunks: Whether to write per-bar chunks
        
    Returns:
        Dictionary with batch results
    """
    config = {
        'analysis_sr': analysis_sr,
        'write_bar_chunks': write_bar_chunks,
        'use_essentia_features': False,
    }
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, wav_path, output_dir, config): wav_path
            for wav_path in input_files
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result['status'] == 'success':
                logger.info(f"✓ {result['song_id']}: {result['tempo']:.1f} BPM, "
                          f"{result['duration']:.1f}s, {result['num_bars']} bars")
            else:
                logger.error(f"✗ {result['song_id']}: {result.get('error', 'Unknown error')}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    return {
        'total': len(input_files),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / len(input_files) if input_files else 0,
        'results': results,
    }


def _read_bar_features_polars(output_dir: Path, song_ids: List[str]) -> Optional[pd.DataFrame]:
    """Aggregate bar_features using Polars (lazy scan for speed)."""
    if not POLARS_AVAILABLE or pl is None:
        return None

    lazy_frames = []
    for song_id in song_ids:
        dataset_path = output_dir / "datasets" / song_id / "bar_features.parquet"
        if dataset_path.exists():
            lazy_frames.append(
                pl.scan_parquet(dataset_path).with_columns(pl.lit(song_id).alias("song_id"))
            )
        else:
            logger.warning("Dataset not found for %s", song_id)

    if not lazy_frames:
        return None

    # Use 'diagonal' concat because bar_features schemas may evolve or differ slightly
    # between songs (e.g., additional feature columns). Diagonal concat safely unions
    # columns and fills missing values, whereas 'vertical' would assume identical schemas.
    combined = pl.concat(lazy_frames, how="diagonal")
    collected = combined.collect()
    logger.info(
        "Aggregated %d bars from %d songs (polars, lazy scan)",
        collected.height,
        len(lazy_frames),
    )
    return collected.to_pandas()


def aggregate_datasets(output_dir: Path, song_ids: List[str]) -> pd.DataFrame:
    """
    Aggregate bar_features datasets from multiple songs.
    
    Args:
        output_dir: Base output directory
        song_ids: List of song identifiers
        
    Returns:
        Combined DataFrame with all bar features
    """
    polars_result = _read_bar_features_polars(output_dir, song_ids)
    if polars_result is not None:
        return polars_result

    dfs = []
    for song_id in song_ids:
        dataset_path = output_dir / "datasets" / song_id / "bar_features.parquet"

        if dataset_path.exists():
            df = pd.read_parquet(dataset_path)
            df["song_id"] = song_id
            dfs.append(df)
        else:
            logger.warning(f"Dataset not found for {song_id}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Aggregated {len(combined)} bars from {len(dfs)} songs (pandas)")
    return combined


def create_visualizations(
    batch_results: Dict[str, Any],
    aggregated_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Create comprehensive visualizations.
    
    Args:
        batch_results: Results from batch processing
        aggregated_df: Combined features DataFrame
        output_dir: Output directory for visualizations
    """
    try:
        import importlib

        go = importlib.import_module("plotly.graph_objects")
        make_subplots = importlib.import_module("plotly.subplots").make_subplots
    except ImportError:
        logger.error("Plotly not installed. Install with: pip install plotly")
        return

    template = "plotly_dark"
    paper_bg = "#0b1021"
    plot_bg = "#0f1629"
    accent = "#4dd0e1"
    accent_warn = "#ff7043"
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Batch Summary
    logger.info("Creating batch summary visualization...")
    
    successful_results = [r for r in batch_results['results'] if r['status'] == 'success']
    
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Processing Success Rate',
            'Tempo Distribution',
            'Duration Distribution',
            'Bars per Song'
        ),
        specs=[
            [{"type": "pie"}, {"type": "histogram"}],
            [{"type": "histogram"}, {"type": "bar"}]
        ],
    )
    
    # Success rate
    fig1.add_trace(
        go.Pie(
            labels=['Success', 'Failed'],
            values=[batch_results['successful'], batch_results['failed']],
            marker_colors=['#1abc9c', accent_warn],
        ),
        row=1, col=1
    )
    
    # Tempo distribution
    tempos = [r['tempo'] for r in successful_results if 'tempo' in r]
    if tempos:
        fig1.add_trace(
            go.Histogram(
                x=tempos,
                name='Tempo',
                nbinsx=20,
                marker_color=accent,
                opacity=0.75,
                showlegend=False,
            ),
            row=1, col=2
        )
        fig1.update_xaxes(title_text="BPM", row=1, col=2)
    
    # Duration distribution
    durations = [r['duration'] for r in successful_results if 'duration' in r]
    if durations:
        fig1.add_trace(
            go.Histogram(
                x=durations,
                name='Duration',
                nbinsx=20,
                marker_color='#fbc02d',
                opacity=0.75,
                showlegend=False,
            ),
            row=2, col=1
        )
        fig1.update_xaxes(title_text="Seconds", row=2, col=1)
    
    # Bars per song
    song_ids = [r['song_id'] for r in successful_results]
    num_bars = [r['num_bars'] for r in successful_results if 'num_bars' in r]
    if song_ids and num_bars:
        fig1.add_trace(
            go.Bar(
                x=song_ids,
                y=num_bars,
                marker_color='#9ccc65',
                showlegend=False,
            ),
            row=2, col=2
        )
        fig1.update_xaxes(title_text="Song", row=2, col=2)
        fig1.update_yaxes(title_text="Bars", row=2, col=2)
    
    fig1.update_layout(
        title="Batch Processing Summary",
        height=800,
        template=template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
    )
    fig1.write_html(viz_dir / "batch_summary.html")
    logger.info(f"✓ Batch summary: {viz_dir / 'batch_summary.html'}")
    
    # 2. Feature Comparisons (if aggregated data available)
    if not aggregated_df.empty:
        logger.info("Creating feature comparison visualizations...")
        
        # Feature distributions across all songs
        numeric_features = aggregated_df.select_dtypes(include=['float64', 'int64']).columns
        numeric_features = [f for f in numeric_features if f not in ['bar_idx', 'bar_start', 'bar_end']]
        
        if len(numeric_features) > 0:
            # Select top features by variance
            variances = aggregated_df[numeric_features].var().sort_values(ascending=False)
            top_features = variances.head(6).index.tolist()
            
            # Create violin plots
            fig2 = make_subplots(
                rows=3, cols=2,
                subplot_titles=top_features,
                vertical_spacing=0.12,
            )
            
            for idx, feature in enumerate(top_features):
                row = (idx // 2) + 1
                col = (idx % 2) + 1
                
                fig2.add_trace(
                    go.Violin(
                        y=aggregated_df[feature],
                        x=aggregated_df['song_id'],
                        name=feature,
                        showlegend=False,
                    ),
                    row=row, col=col
                )
                
                fig2.update_yaxes(title_text=feature, row=row, col=col)
            
            fig2.update_layout(
                title="Feature Distributions Across Songs",
                height=900,
                template=template,
                paper_bgcolor=paper_bg,
                plot_bgcolor=plot_bg,
            )
            fig2.write_html(viz_dir / "feature_comparisons.html")
            logger.info(f"✓ Feature comparisons: {viz_dir / 'feature_comparisons.html'}")
            
            # Create correlation heatmap
            if len(top_features) > 1:
                corr_matrix = aggregated_df[top_features].corr()
                
                fig3 = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                ))
                
                fig3.update_layout(
                    title="Feature Correlation Matrix",
                    height=600,
                    template=template,
                    paper_bgcolor=paper_bg,
                    plot_bgcolor=plot_bg,
                )
                fig3.write_html(viz_dir / "feature_correlations.html")
                logger.info(f"✓ Feature correlations: {viz_dir / 'feature_correlations.html'}")
    
    logger.info(f"\n✓ All visualizations saved to: {viz_dir}")


def export_summary_report(
    batch_results: Dict[str, Any],
    aggregated_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Export summary report as JSON and CSV.
    
    Args:
        batch_results: Results from batch processing
        aggregated_df: Combined features DataFrame
        output_dir: Output directory
    """
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Save batch results as JSON
    with open(report_dir / "batch_results.json", 'w') as f:
        json.dump(batch_results, f, indent=2)
    logger.info(f"✓ Batch results: {report_dir / 'batch_results.json'}")
    
    # Save aggregated features as CSV
    if not aggregated_df.empty:
        if POLARS_AVAILABLE:
            assert pl is not None
            pl.from_pandas(aggregated_df).write_csv(report_dir / "aggregated_features.csv")
            logger.info(
                f"✓ Aggregated features: {report_dir / 'aggregated_features.csv'} (polars)"
            )

            # Polars describe() is fast and works well for wide tables.
            pl.from_pandas(aggregated_df).describe().write_csv(
                report_dir / "feature_statistics.csv"
            )
            logger.info(f"✓ Feature statistics: {report_dir / 'feature_statistics.csv'} (polars)")
        else:
            aggregated_df.to_csv(report_dir / "aggregated_features.csv", index=False)
            logger.info(f"✓ Aggregated features: {report_dir / 'aggregated_features.csv'}")

            # Save summary statistics
            summary_stats = aggregated_df.describe().T
            summary_stats.to_csv(report_dir / "feature_statistics.csv")
            logger.info(f"✓ Feature statistics: {report_dir / 'feature_statistics.csv'}")


def main():
    """Main workflow execution."""
    parser = argparse.ArgumentParser(
        description="Batch process and visualize multiple audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="WAV files or directory to process"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output/batch_workflow"),
        help="Output directory (default: output/batch_workflow)"
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Analysis sample rate (default: 22050)"
    )
    
    parser.add_argument(
        "--no-chunks",
        action="store_true",
        help="Skip writing per-bar chunks (faster)"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    # Collect input files
    input_files = []
    for path in args.input_files:
        if path.is_dir():
            input_files.extend(path.glob("*.wav"))
        elif path.suffix.lower() == ".wav":
            input_files.append(path)
    
    input_files = [f for f in input_files if f.exists()]
    
    if not input_files:
        logger.error("No valid WAV files found")
        return
    
    logger.info(f"Found {len(input_files)} WAV files to process")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Parallel workers: {args.workers}")
    
    # Step 1: Batch process
    logger.info("\n" + "="*60)
    logger.info("Step 1: Batch Processing")
    logger.info("="*60)
    
    batch_results = batch_process(
        input_files=input_files,
        output_dir=args.output,
        max_workers=args.workers,
        analysis_sr=args.sr,
        write_bar_chunks=not args.no_chunks,
    )
    
    logger.info(f"\n✓ Processed {batch_results['successful']}/{batch_results['total']} files")
    logger.info(f"  Success rate: {batch_results['success_rate']*100:.1f}%")
    
    # Step 2: Aggregate datasets
    logger.info("\n" + "="*60)
    logger.info("Step 2: Aggregating Datasets")
    logger.info("="*60)
    
    successful_ids = [
        r['song_id'] for r in batch_results['results']
        if r['status'] == 'success'
    ]
    
    aggregated_df = aggregate_datasets(args.output, successful_ids)
    
    # Step 3: Create visualizations
    if not args.no_viz:
        logger.info("\n" + "="*60)
        logger.info("Step 3: Creating Visualizations")
        logger.info("="*60)
        
        create_visualizations(batch_results, aggregated_df, args.output)
    
    # Step 4: Export reports
    logger.info("\n" + "="*60)
    logger.info("Step 4: Exporting Reports")
    logger.info("="*60)
    
    export_summary_report(batch_results, aggregated_df, args.output)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Workflow Complete!")
    logger.info("="*60)
    logger.info(f"Output directory: {args.output}")
    logger.info(f"  - Analysis results: {args.output / 'analysis'}")
    logger.info(f"  - Datasets: {args.output / 'datasets'}")
    logger.info(f"  - Reports: {args.output / 'reports'}")
    
    if not args.no_viz:
        logger.info(f"  - Visualizations: {args.output / 'visualizations'}")
    
    logger.info("\n✓ All done!")


if __name__ == "__main__":
    main()
