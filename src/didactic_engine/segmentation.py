"""
Stem segmentation module.

Segments audio stems into per-bar WAV chunks aligned to beat/bar grid.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import soundfile as sf


class StemSegmenter:
    """Segment audio stems into per-bar chunks."""

    def __init__(self):
        """Initialize the stem segmenter."""
        pass

    def segment_by_bars(
        self,
        audio: np.ndarray,
        sample_rate: int,
        bar_times: np.ndarray,
        output_dir: str,
        stem_name: str = "audio",
    ) -> List[str]:
        """
        Segment audio into per-bar chunks.

        Args:
            audio: Input audio array (channels, samples).
            sample_rate: Sample rate of the audio.
            bar_times: Array of bar start times in seconds.
            output_dir: Directory to save segmented chunks.
            stem_name: Name of the stem for output filenames.

        Returns:
            List of paths to saved chunk files.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        chunk_paths = []
        
        # Ensure audio is 2D
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # Segment audio by bars
        for i in range(len(bar_times) - 1):
            start_time = bar_times[i]
            end_time = bar_times[i + 1]
            
            # Convert time to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Extract chunk
            chunk = audio[:, start_sample:end_sample]
            
            # Save chunk
            chunk_path = os.path.join(output_dir, f"{stem_name}_bar_{i:04d}.wav")
            
            # Transpose for soundfile (expects samples, channels)
            chunk_to_save = chunk.T if chunk.shape[0] < chunk.shape[1] else chunk
            sf.write(chunk_path, chunk_to_save, sample_rate)
            
            chunk_paths.append(chunk_path)
        
        return chunk_paths

    def segment_stems_by_bars(
        self,
        stems: Dict[str, np.ndarray],
        sample_rate: int,
        bar_times: np.ndarray,
        output_dir: str,
    ) -> Dict[str, List[str]]:
        """
        Segment multiple stems into per-bar chunks.

        Args:
            stems: Dictionary mapping stem names to audio arrays.
            sample_rate: Sample rate of the audio.
            bar_times: Array of bar start times in seconds.
            output_dir: Directory to save segmented chunks.

        Returns:
            Dictionary mapping stem names to lists of chunk paths.
        """
        segmented_stems = {}
        
        for stem_name, stem_audio in stems.items():
            stem_output_dir = os.path.join(output_dir, stem_name)
            chunk_paths = self.segment_by_bars(
                stem_audio, sample_rate, bar_times, stem_output_dir, stem_name
            )
            segmented_stems[stem_name] = chunk_paths
        
        return segmented_stems

    def segment_by_time_intervals(
        self,
        audio: np.ndarray,
        sample_rate: int,
        intervals: List[Tuple[float, float]],
        output_dir: str,
        stem_name: str = "audio",
    ) -> List[str]:
        """
        Segment audio by arbitrary time intervals.

        Args:
            audio: Input audio array (channels, samples).
            sample_rate: Sample rate of the audio.
            intervals: List of (start_time, end_time) tuples in seconds.
            output_dir: Directory to save segmented chunks.
            stem_name: Name of the stem for output filenames.

        Returns:
            List of paths to saved chunk files.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        chunk_paths = []
        
        # Ensure audio is 2D
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        for i, (start_time, end_time) in enumerate(intervals):
            # Convert time to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Extract chunk
            chunk = audio[:, start_sample:end_sample]
            
            # Save chunk
            chunk_path = os.path.join(
                output_dir, f"{stem_name}_segment_{i:04d}.wav"
            )
            
            # Transpose for soundfile (expects samples, channels)
            chunk_to_save = chunk.T if chunk.shape[0] < chunk.shape[1] else chunk
            sf.write(chunk_path, chunk_to_save, sample_rate)
            
            chunk_paths.append(chunk_path)
        
        return chunk_paths
