"""
Configuration for the music ETL pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the music ETL pipeline."""

    # Core identifiers
    song_id: str
    input_wav: Path
    out_dir: Path = Path("data")

    # Analysis configuration
    analysis_sr: int = 22050
    hop_length: int = 512
    time_signature_num: int = 4
    time_signature_den: int = 4

    # Feature toggles
    use_pydub_preprocess: bool = True
    use_essentia_features: bool = True
    write_bar_chunks: bool = True

    # Preprocessing parameters
    preprocess_target_sr: int = 44100
    preprocess_mono: bool = True
    preprocess_normalize: bool = True
    preprocess_trim_silence: bool = True
    silence_thresh_dbfs: int = -50
    keep_silence_ms: int = 100

    # Demucs model
    demucs_model: str = "htdemucs"

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

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if not isinstance(self.input_wav, Path):
            self.input_wav = Path(self.input_wav)
        if not isinstance(self.out_dir, Path):
            self.out_dir = Path(self.out_dir)
