"""
Configuration for the didactic-engine audio processing pipeline.

Provides a frozen dataclass for pipeline configuration with deterministic
output directory paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration for the audio processing pipeline.

    This is a frozen dataclass to ensure configuration immutability
    throughout the pipeline execution.

    Attributes:
        song_id: Unique identifier for the song being processed.
        input_wav: Path to the input WAV file.
        out_dir: Base output directory for all generated files.
        demucs_model: Demucs model name for stem separation.
        demucs_device: Device for Demucs processing ('cpu', 'cuda', 'cuda:0', etc.).
        analysis_sr: Sample rate for audio analysis.
        hop_length: Hop length for STFT operations.
        time_signature_num: Time signature numerator (beats per bar).
        time_signature_den: Time signature denominator (beat unit).
        use_pydub_preprocess: Whether to preprocess audio with pydub.
        preprocess_target_sr: Target sample rate for preprocessing.
        preprocess_mono: Whether to convert to mono during preprocessing.
        preprocess_normalize: Whether to normalize audio during preprocessing.
        preprocess_trim_silence: Whether to trim silence during preprocessing.
        preprocess_silence_thresh_dbfs: Silence threshold in dBFS.
        preprocess_keep_silence_ms: Milliseconds of silence to keep.
        use_essentia_features: Whether to extract Essentia features.
        write_bar_chunks: Whether to write per-bar audio chunks.
        write_bar_chunk_wavs: Whether to persist bar chunk WAVs to disk.
        preserve_chunk_audio: Whether to preserve original sample rate and channels
            for chunk WAVs. When True, chunks are written at native SR/channels instead
            of analysis_sr/mono. Features still use analysis_sr/mono. Default False.
        basic_pitch_backend: Basic Pitch inference backend ('tf', 'onnx', 'tflite', 'coreml').
        demucs_timeout_s: Optional timeout (seconds) for Demucs separation.
        basic_pitch_timeout_s: Optional timeout (seconds) for Basic Pitch transcription.
        quantize_midi: Whether to quantize MIDI notes before ABC export.
        quantize_division: Rhythmic division for quantization (16=sixteenth notes, 8=eighth notes).
        quantize_min_duration: Optional minimum note duration in seconds after quantization.
        use_stem_selector: Whether to use automatic stem selection for melody transcription.
        stem_selector_candidates: Tuple of stem names to consider for melody selection.
        export_metadata_jsonl: Whether to export metadata.jsonl for HuggingFace AudioFolder.
        abc_trigger_token: Trigger token for ABC-based prompts (default "abcstyle").
        abc_max_chars: Maximum characters for ABC text in prompts (default 2500).
    """

    song_id: str
    input_wav: Path
    out_dir: Path
    demucs_model: str = "htdemucs"
    demucs_device: str = "cpu"
    analysis_sr: int = 22050
    hop_length: int = 512
    time_signature_num: int = 4
    time_signature_den: int = 4
    use_pydub_preprocess: bool = True
    preprocess_target_sr: int = 44100
    preprocess_mono: bool = True
    preprocess_normalize: bool = True
    preprocess_trim_silence: bool = True
    preprocess_silence_thresh_dbfs: float = -40.0
    preprocess_keep_silence_ms: int = 80
    use_essentia_features: bool = False
    write_bar_chunks: bool = True
    write_bar_chunk_wavs: bool = True
    preserve_chunk_audio: bool = False
    basic_pitch_backend: str = "tf"
    demucs_timeout_s: Optional[float] = None
    basic_pitch_timeout_s: Optional[float] = None
    quantize_midi: bool = False
    quantize_division: int = 16
    quantize_min_duration: Optional[float] = None
    use_stem_selector: bool = False
    stem_selector_candidates: Tuple[str, ...] = ("vocals", "other", "bass")
    export_metadata_jsonl: bool = False
    abc_trigger_token: str = "abcstyle"
    abc_max_chars: int = 2500

    def __post_init__(self) -> None:
        """Convert string paths to Path objects if needed."""
        # Since frozen=True, we use object.__setattr__ for initialization
        if isinstance(self.input_wav, str):
            object.__setattr__(self, "input_wav", Path(self.input_wav))
        if isinstance(self.out_dir, str):
            object.__setattr__(self, "out_dir", Path(self.out_dir))

    @property
    def stems_dir(self) -> Path:
        """Directory for separated stems."""
        return self.out_dir / "stems" / self.song_id

    @property
    def preprocess_dir(self) -> Path:
        """Directory for preprocessed audio."""
        return self.out_dir / "preprocessed" / self.song_id

    @property
    def chunks_dir(self) -> Path:
        """Directory for per-bar audio chunks."""
        return self.out_dir / "chunks" / self.song_id

    @property
    def midi_dir(self) -> Path:
        """Directory for MIDI files."""
        return self.out_dir / "midi" / self.song_id

    @property
    def analysis_dir(self) -> Path:
        """Directory for analysis results."""
        return self.out_dir / "analysis" / self.song_id

    @property
    def reports_dir(self) -> Path:
        """Directory for generated reports."""
        return self.out_dir / "reports" / self.song_id

    @property
    def datasets_dir(self) -> Path:
        """Directory for Parquet datasets."""
        return self.out_dir / "datasets" / self.song_id

    @property
    def input_dir(self) -> Path:
        """Directory for input file copies."""
        return self.out_dir / "input" / self.song_id

    def create_directories(self) -> None:
        """Create all output directories."""
        for dir_path in [
            self.stems_dir,
            self.preprocess_dir,
            self.chunks_dir,
            self.midi_dir,
            self.analysis_dir,
            self.reports_dir,
            self.datasets_dir,
            self.input_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
