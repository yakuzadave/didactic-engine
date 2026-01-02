"""
Plotly visualization examples for Didactic Engine results.

This module demonstrates how to create interactive visualizations of:
1. Audio waveforms and spectrograms
2. Tempo and beat detection results
3. Feature distributions across bars
4. Stem comparison charts
5. MIDI note visualizations
6. Batch processing results and statistics

Requires: plotly, pandas, numpy
Install with: pip install plotly pandas numpy
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def load_analysis_results(song_id: str, output_dir: Path) -> Dict[str, Any]:
    """
    Load analysis results from JSON file.
    
    Args:
        song_id: Song identifier
        output_dir: Base output directory
        
    Returns:
        Dictionary containing analysis results
    """
    analysis_file = output_dir / "analysis" / song_id / "combined.json"
    
    with open(analysis_file, 'r') as f:
        return json.load(f)


def plot_waveform(audio: np.ndarray, sr: int, title: str = "Audio Waveform") -> go.Figure:
    """
    Create interactive waveform plot.
    
    Args:
        audio: Audio signal as 1D numpy array
        sr: Sample rate
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    # Create time axis
    time = np.arange(len(audio)) / sr
    
    # Downsample for large files
    if len(audio) > 100000:
        factor = len(audio) // 100000
        audio = audio[::factor]
        time = time[::factor]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=audio,
        mode='lines',
        name='Amplitude',
        line=dict(width=0.5, color='steelblue'),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        hovermode='x unified',
        template='plotly_white',
        height=400,
    )
    
    return fig


def plot_spectrogram(audio: np.ndarray, sr: int, title: str = "Spectrogram") -> go.Figure:
    """
    Create interactive spectrogram plot.
    
    Args:
        audio: Audio signal as 1D numpy array
        sr: Sample rate
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for spectrogram visualization")
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    
    # Create time and frequency axes
    times = librosa.times_like(D, sr=sr)
    freqs = librosa.fft_frequencies(sr=sr)
    
    fig = go.Figure(data=go.Heatmap(
        z=D,
        x=times,
        y=freqs,
        colorscale='Viridis',
        colorbar=dict(title='dB'),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Frequency (Hz)',
        template='plotly_white',
        height=500,
    )
    
    return fig


def plot_tempo_and_beats(analysis: Dict[str, Any]) -> go.Figure:
    """
    Plot tempo and beat detection results.
    
    Args:
        analysis: Analysis dictionary containing tempo and beat information
        
    Returns:
        Plotly Figure object
    """
    beat_times = analysis.get('beat_times', [])
    tempo = analysis.get('tempo_bpm', 0)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Beat Detection', 'Tempo Information'),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.15,
    )
    
    # Beat markers
    if beat_times:
        fig.add_trace(
            go.Scatter(
                x=beat_times,
                y=[1] * len(beat_times),
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='line-ns-open',
                    line=dict(width=2),
                ),
                name='Beats',
            ),
            row=1, col=1
        )
    
    # Tempo display
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=tempo,
            title={'text': "Tempo (BPM)"},
            gauge={
                'axis': {'range': [0, 200]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "gray"},
                    {'range': [80, 120], 'color': "lightgreen"},
                    {'range': [120, 160], 'color': "yellow"},
                    {'range': [160, 200], 'color': "orange"},
                ],
            },
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    
    fig.update_layout(
        title="Tempo and Beat Analysis",
        showlegend=True,
        height=600,
        template='plotly_white',
    )
    
    return fig


def plot_feature_distributions(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """
    Plot distributions of multiple features across bars.
    
    Args:
        df: DataFrame with bar-level features
        features: List of feature column names to plot
        
    Returns:
        Plotly Figure object
    """
    n_features = len(features)
    rows = (n_features + 1) // 2
    
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=features,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )
    
    for idx, feature in enumerate(features):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        if feature in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df[feature],
                    name=feature,
                    nbinsx=30,
                    showlegend=False,
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text=feature, row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)
    
    fig.update_layout(
        title="Feature Distributions Across Bars",
        height=300 * rows,
        template='plotly_white',
    )
    
    return fig


def plot_feature_timeline(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """
    Plot feature values over time (per bar).
    
    Args:
        df: DataFrame with bar-level features
        features: List of feature column names to plot
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    if 'bar_idx' not in df.columns:
        df['bar_idx'] = range(len(df))
    
    for feature in features:
        if feature in df.columns:
            fig.add_trace(go.Scatter(
                x=df['bar_idx'],
                y=df[feature],
                mode='lines+markers',
                name=feature,
                line=dict(width=2),
                marker=dict(size=5),
            ))
    
    fig.update_layout(
        title="Feature Timeline Across Bars",
        xaxis_title="Bar Index",
        yaxis_title="Feature Value",
        hovermode='x unified',
        template='plotly_white',
        height=500,
    )
    
    return fig


def plot_stem_comparison(stem_paths: Dict[str, Path], duration: float = 10.0) -> go.Figure:
    """
    Compare waveforms of different stems side-by-side.
    
    Args:
        stem_paths: Dictionary mapping stem names to file paths
        duration: Duration to plot (in seconds)
        
    Returns:
        Plotly Figure object
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for stem comparison")
    
    fig = make_subplots(
        rows=len(stem_paths),
        cols=1,
        subplot_titles=list(stem_paths.keys()),
        vertical_spacing=0.05,
    )
    
    for idx, (stem_name, stem_path) in enumerate(stem_paths.items(), 1):
        if stem_path.exists():
            audio, sr = librosa.load(stem_path, duration=duration, mono=True)
            time = np.arange(len(audio)) / sr
            
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=audio,
                    mode='lines',
                    name=stem_name,
                    line=dict(width=0.5),
                    showlegend=False,
                ),
                row=idx, col=1
            )
            
            fig.update_xaxes(title_text="Time (s)", row=idx, col=1)
            fig.update_yaxes(title_text="Amplitude", row=idx, col=1)
    
    fig.update_layout(
        title="Stem Comparison",
        height=200 * len(stem_paths),
        template='plotly_white',
    )
    
    return fig


def plot_midi_piano_roll(midi_path: Path, end_time: Optional[float] = None) -> go.Figure:
    """
    Create piano roll visualization from MIDI file.
    
    Args:
        midi_path: Path to MIDI file
        end_time: End time for visualization (None = full duration)
        
    Returns:
        Plotly Figure object
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi is required for MIDI visualization")
    
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    
    fig = go.Figure()
    
    for instrument in midi.instruments:
        for note in instrument.notes:
            if end_time and note.start > end_time:
                break
            
            fig.add_trace(go.Scatter(
                x=[note.start, note.end, note.end, note.start, note.start],
                y=[note.pitch, note.pitch, note.pitch + 0.8, note.pitch + 0.8, note.pitch],
                fill='toself',
                fillcolor=f'rgba(100, 150, 200, {note.velocity/127})',
                line=dict(width=0),
                showlegend=False,
                hovertemplate=f'Pitch: {note.pitch}<br>Start: {note.start:.2f}s<br>Velocity: {note.velocity}',
            ))
    
    fig.update_layout(
        title="MIDI Piano Roll",
        xaxis_title="Time (seconds)",
        yaxis_title="MIDI Note Number",
        template='plotly_white',
        height=600,
        hovermode='closest',
    )
    
    return fig


def plot_batch_results_summary(results_json: Path) -> go.Figure:
    """
    Create summary visualizations for batch processing results.
    
    Args:
        results_json: Path to batch results JSON file
        
    Returns:
        Plotly Figure object
    """
    with open(results_json, 'r') as f:
        data = json.load(f)
    
    results_df = pd.DataFrame(data['results'])
    
    # Filter successful results
    success_df = results_df[results_df['status'] == 'success']
    
    if success_df.empty:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No successful results to visualize",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Success Rate',
            'Tempo Distribution',
            'Duration Distribution',
            'Bars per Song'
        ),
        specs=[
            [{"type": "pie"}, {"type": "histogram"}],
            [{"type": "histogram"}, {"type": "bar"}]
        ],
    )
    
    # Success rate pie chart
    fig.add_trace(
        go.Pie(
            labels=['Success', 'Failed'],
            values=[data['successful'], data['failed']],
            marker_colors=['#2ecc71', '#e74c3c'],
        ),
        row=1, col=1
    )
    
    # Tempo distribution
    if 'tempo' in success_df.columns:
        fig.add_trace(
            go.Histogram(
                x=success_df['tempo'],
                name='Tempo',
                nbinsx=20,
                marker_color='steelblue',
                showlegend=False,
            ),
            row=1, col=2
        )
        fig.update_xaxes(title_text="BPM", row=1, col=2)
    
    # Duration distribution
    if 'duration' in success_df.columns:
        fig.add_trace(
            go.Histogram(
                x=success_df['duration'],
                name='Duration',
                nbinsx=20,
                marker_color='orange',
                showlegend=False,
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text="Seconds", row=2, col=1)
    
    # Bars per song
    if 'num_bars' in success_df.columns:
        fig.add_trace(
            go.Bar(
                x=success_df['song_id'],
                y=success_df['num_bars'],
                marker_color='lightgreen',
                showlegend=False,
            ),
            row=2, col=2
        )
        fig.update_xaxes(title_text="Song", row=2, col=2)
        fig.update_yaxes(title_text="Number of Bars", row=2, col=2)
    
    fig.update_layout(
        title="Batch Processing Results Summary",
        height=800,
        template='plotly_white',
    )
    
    return fig


def plot_bar_features_heatmap(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """
    Create heatmap of feature values across bars.
    
    Args:
        df: DataFrame with bar-level features
        features: List of feature column names
        
    Returns:
        Plotly Figure object
    """
    # Select and normalize features
    feature_data = df[features].copy()
    
    # Normalize each feature to 0-1 range
    for col in feature_data.columns:
        min_val = feature_data[col].min()
        max_val = feature_data[col].max()
        if max_val > min_val:
            feature_data[col] = (feature_data[col] - min_val) / (max_val - min_val)
    
    fig = go.Figure(data=go.Heatmap(
        z=feature_data.T.values,
        x=df.index if 'bar_idx' not in df.columns else df['bar_idx'],
        y=features,
        colorscale='Viridis',
        colorbar=dict(title='Normalized Value'),
    ))
    
    fig.update_layout(
        title="Bar Features Heatmap",
        xaxis_title="Bar Index",
        yaxis_title="Feature",
        template='plotly_white',
        height=max(400, len(features) * 30),
    )
    
    return fig


def example_visualizations():
    """Run example visualizations on sample data."""
    print("="*60)
    print("Plotly Visualization Examples")
    print("="*60)
    
    # Example 1: Load and visualize analysis results
    output_dir = Path("output")
    song_id = "sample_song"
    
    try:
        analysis = load_analysis_results(song_id, output_dir)
        
        # Plot tempo and beats
        fig1 = plot_tempo_and_beats(analysis)
        fig1.write_html(f"output/visualizations/{song_id}_tempo_beats.html")
        print(f"✓ Created tempo and beats visualization")
        
        # Load feature dataset
        dataset_path = output_dir / "datasets" / song_id / "bar_features.parquet"
        if dataset_path.exists():
            df = pd.read_parquet(dataset_path)
            
            # Feature distributions
            features_to_plot = ['rms', 'spectral_centroid', 'zero_crossing_rate', 'tempo']
            available_features = [f for f in features_to_plot if f in df.columns]
            
            if available_features:
                fig2 = plot_feature_distributions(df, available_features)
                fig2.write_html(f"output/visualizations/{song_id}_feature_distributions.html")
                print(f"✓ Created feature distributions visualization")
                
                fig3 = plot_feature_timeline(df, available_features)
                fig3.write_html(f"output/visualizations/{song_id}_feature_timeline.html")
                print(f"✓ Created feature timeline visualization")
                
                fig4 = plot_bar_features_heatmap(df, available_features[:10])
                fig4.write_html(f"output/visualizations/{song_id}_feature_heatmap.html")
                print(f"✓ Created feature heatmap visualization")
        
        # MIDI piano roll
        midi_path = output_dir / "midi" / song_id / f"{song_id}_transcribed.mid"
        if midi_path.exists():
            fig5 = plot_midi_piano_roll(midi_path, end_time=30.0)
            fig5.write_html(f"output/visualizations/{song_id}_piano_roll.html")
            print(f"✓ Created piano roll visualization")
        
    except FileNotFoundError as e:
        print(f"Note: Sample data not found. Run pipeline first to generate data.")
        print(f"Error: {e}")
    
    # Example 2: Batch results summary
    batch_results_path = Path("output/batch_parallel/results.json")
    if batch_results_path.exists():
        fig6 = plot_batch_results_summary(batch_results_path)
        fig6.write_html("output/visualizations/batch_results_summary.html")
        print(f"✓ Created batch results summary")
    
    print("\n" + "="*60)
    print("Visualizations saved to output/visualizations/")
    print("="*60)


if __name__ == "__main__":
    # Create output directory
    Path("output/visualizations").mkdir(parents=True, exist_ok=True)
    
    example_visualizations()
