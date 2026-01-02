#!/usr/bin/env python3
"""
Quickstart demo: Process audio → Generate visualizations

This script demonstrates the complete workflow:
1. Process a single WAV file
2. Generate interactive visualizations
3. Display summary statistics

Usage:
    python quickstart_demo.py path/to/audio.wav
    python quickstart_demo.py --help
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from didactic_engine import AudioPipeline, PipelineConfig
except ImportError:
    print("Error: didactic-engine not installed")
    print("Install with: pip install -e .")
    sys.exit(1)


def process_and_visualize(
    input_wav: Path,
    output_dir: Path,
    song_id: Optional[str] = None,
    create_visualizations: bool = True,
) -> None:
    """
    Process audio file and create visualizations.
    
    Args:
        input_wav: Path to input WAV file
        output_dir: Output directory for results
        song_id: Optional song identifier (defaults to filename stem)
        create_visualizations: Whether to generate Plotly visualizations
    """
    # Validate input
    if not input_wav.exists():
        print(f"Error: File not found: {input_wav}")
        sys.exit(1)
    
    if song_id is None:
        song_id = input_wav.stem
    
    print("="*60)
    print(f"Didactic Engine Quickstart Demo")
    print("="*60)
    print(f"Input: {input_wav}")
    print(f"Song ID: {song_id}")
    print(f"Output: {output_dir}")
    print()
    
    # Step 1: Configure pipeline
    print("[1/4] Configuring pipeline...")
    cfg = PipelineConfig(
        song_id=song_id,
        input_wav=input_wav,
        out_dir=output_dir,
        analysis_sr=22050,
        use_essentia_features=False,
        write_bar_chunks=True,
        time_signature_num=4,
        time_signature_den=4,
    )
    print(f"✓ Configuration created")
    
    # Step 2: Run pipeline
    print("\n[2/4] Running audio processing pipeline...")
    print("  This may take a few minutes depending on file size...")
    
    try:
        pipeline = AudioPipeline(cfg)
        results = pipeline.run()
        print("✓ Pipeline completed successfully")
        
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        sys.exit(1)
    
    # Step 3: Display summary
    print("\n[3/4] Processing Summary:")
    print(f"  Duration: {results['duration_s']:.2f} seconds")
    print(f"  Tempo: {results['analysis']['tempo_bpm']:.2f} BPM")
    print(f"  Number of bars: {results['num_bars']}")
    print(f"  Number of beats: {len(results['analysis']['beat_times'])}")
    
    if 'stems' in results and results['stems']:
        print(f"  Stems generated: {', '.join(results['stems'])}")
    
    if 'midi_files' in results and results['midi_files']:
        print(f"  MIDI files: {len(results['midi_files'])}")
    
    # Step 4: Create visualizations
    if create_visualizations:
        print("\n[4/4] Creating visualizations...")
        
        try:
            import pandas as pd
            from visualization_plotly import (
                plot_tempo_and_beats,
                plot_feature_timeline,
                plot_feature_distributions,
                plot_bar_features_heatmap,
                load_analysis_results,
            )
            
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Load analysis
            analysis = load_analysis_results(song_id, output_dir)
            
            # Tempo and beats
            fig1 = plot_tempo_and_beats(analysis)
            fig1.write_html(viz_dir / f"{song_id}_tempo_beats.html")
            print(f"  ✓ Tempo and beats: {viz_dir / f'{song_id}_tempo_beats.html'}")
            
            # Feature visualizations
            dataset_path = output_dir / "datasets" / song_id / "bar_features.parquet"
            if dataset_path.exists():
                df = pd.read_parquet(dataset_path)
                
                # Feature timeline
                features = ['rms', 'spectral_centroid', 'zero_crossing_rate', 'tempo']
                available = [f for f in features if f in df.columns]
                
                if available:
                    fig2 = plot_feature_timeline(df, available)
                    fig2.write_html(viz_dir / f"{song_id}_feature_timeline.html")
                    print(f"  ✓ Feature timeline: {viz_dir / f'{song_id}_feature_timeline.html'}")
                    
                    fig3 = plot_feature_distributions(df, available)
                    fig3.write_html(viz_dir / f"{song_id}_feature_distributions.html")
                    print(f"  ✓ Feature distributions: {viz_dir / f'{song_id}_feature_distributions.html'}")
                    
                    # Heatmap (limit to first 10 features)
                    all_features = [col for col in df.columns if col not in ['bar_idx', 'bar_start', 'bar_end']]
                    heatmap_features = all_features[:10]
                    
                    if heatmap_features:
                        fig4 = plot_bar_features_heatmap(df, heatmap_features)
                        fig4.write_html(viz_dir / f"{song_id}_feature_heatmap.html")
                        print(f"  ✓ Feature heatmap: {viz_dir / f'{song_id}_feature_heatmap.html'}")
            
            # MIDI piano roll
            midi_path = output_dir / "midi" / song_id / f"{song_id}_transcribed.mid"
            if midi_path.exists():
                try:
                    from visualization_plotly import plot_midi_piano_roll
                    fig5 = plot_midi_piano_roll(midi_path, end_time=30.0)
                    fig5.write_html(viz_dir / f"{song_id}_piano_roll.html")
                    print(f"  ✓ Piano roll: {viz_dir / f'{song_id}_piano_roll.html'}")
                except Exception as e:
                    print(f"  ⚠ Piano roll skipped: {e}")
            
            print(f"\n✓ Visualizations saved to: {viz_dir}")
            print(f"  Open HTML files in your browser to explore interactive charts")
            
        except ImportError:
            print("  ⚠ Plotly not installed - skipping visualizations")
            print("  Install with: pip install -e \".[viz]\"")
        
        except Exception as e:
            print(f"  ⚠ Visualization error: {e}")
    
    # Final output locations
    print("\n" + "="*60)
    print("Output Structure:")
    print("="*60)
    print(f"Analysis:       {output_dir / 'analysis' / song_id}")
    print(f"Datasets:       {output_dir / 'datasets' / song_id}")
    print(f"Preprocessed:   {output_dir / 'preprocessed' / song_id}")
    
    if (output_dir / 'stems' / song_id).exists():
        print(f"Stems:          {output_dir / 'stems' / song_id}")
    
    if (output_dir / 'midi' / song_id).exists():
        print(f"MIDI:           {output_dir / 'midi' / song_id}")
    
    if (output_dir / 'reports' / song_id).exists():
        print(f"Reports:        {output_dir / 'reports' / song_id}")
    
    if create_visualizations:
        print(f"Visualizations: {output_dir / 'visualizations'}")
    
    print("\n✓ Processing complete!")


def main():
    """Parse arguments and run demo."""
    parser = argparse.ArgumentParser(
        description="Quickstart demo for Didactic Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file with visualizations
  python quickstart_demo.py sample_audio/song.wav
  
  # Process with custom output directory
  python quickstart_demo.py song.wav --output my_output/
  
  # Process without visualizations
  python quickstart_demo.py song.wav --no-viz
  
  # Process with custom song ID
  python quickstart_demo.py song.wav --song-id my_track
        """
    )
    
    parser.add_argument(
        "input_wav",
        type=Path,
        help="Path to input WAV file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output/quickstart"),
        help="Output directory (default: output/quickstart)"
    )
    
    parser.add_argument(
        "-s", "--song-id",
        type=str,
        default=None,
        help="Song identifier (default: input filename)"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    process_and_visualize(
        input_wav=args.input_wav,
        output_dir=args.output,
        song_id=args.song_id,
        create_visualizations=not args.no_viz,
    )


if __name__ == "__main__":
    main()
