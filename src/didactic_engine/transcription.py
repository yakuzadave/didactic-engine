"""
MIDI transcription module using Basic Pitch.

Transcribes audio to MIDI using Spotify's Basic Pitch model.
"""

import os
from typing import Optional
import numpy as np

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    ICASSP_2022_MODEL_PATH = None


class MIDITranscriber:
    """Transcribe audio to MIDI using Basic Pitch."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the MIDI transcriber.

        Args:
            model_path: Path to Basic Pitch model. If None, uses default.
        
        Raises:
            ImportError: If basic_pitch is not installed.
        """
        if not BASIC_PITCH_AVAILABLE:
            raise ImportError(
                "basic-pitch is not installed. Install it with: pip install basic-pitch"
            )
        
        self.model_path = model_path or ICASSP_2022_MODEL_PATH

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length: float = 127.70,
        minimum_frequency: Optional[float] = None,
        maximum_frequency: Optional[float] = None,
        multiple_pitch_bends: bool = False,
    ) -> tuple:
        """
        Transcribe audio to MIDI.

        Args:
            audio: Input audio array (channels, samples) or (samples,).
            sample_rate: Sample rate of the audio.
            onset_threshold: Threshold for note onset detection.
            frame_threshold: Threshold for frame-wise note detection.
            minimum_note_length: Minimum note length in milliseconds.
            minimum_frequency: Minimum frequency to transcribe (Hz).
            maximum_frequency: Maximum frequency to transcribe (Hz).
            multiple_pitch_bends: Whether to allow multiple pitch bends per note.

        Returns:
            Tuple of (model_output, midi_data, note_events)
        """
        # Ensure mono audio
        if audio.ndim == 2:
            if audio.shape[0] <= 2:  # (channels, samples)
                audio = np.mean(audio, axis=0)
            else:  # (samples, channels)
                audio = np.mean(audio, axis=1)
        
        # Basic Pitch expects audio as 1D array
        audio = audio.flatten()

        # Run prediction
        model_output, midi_data, note_events = predict(
            audio,
            sample_rate,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            multiple_pitch_bends=multiple_pitch_bends,
        )

        return model_output, midi_data, note_events

    def save_midi(self, midi_data, output_path: str) -> None:
        """
        Save MIDI data to file.

        Args:
            midi_data: MIDI data from transcription.
            output_path: Path to save MIDI file.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save MIDI file
        midi_data.write(output_path)
