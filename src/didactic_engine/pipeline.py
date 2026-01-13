"""
Main audio processing pipeline.

Orchestrates the complete audio processing workflow from ingestion to feature extraction.
"""

import json
import logging
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Tuple

import pandas as pd

from didactic_engine.ingestion import WAVIngester
from didactic_engine.preprocessing import AudioPreprocessor
from didactic_engine.analysis import AudioAnalyzer
from didactic_engine.midi_parser import MIDIParser
from didactic_engine.segmentation import segment_beats_into_bars, StemSegmenter
from didactic_engine.features import FeatureExtractor
from didactic_engine.align import align_notes_to_beats
from didactic_engine.export_md import export_midi_markdown
from didactic_engine.export_abc import export_abc, export_abc_text
from didactic_engine.midi_quantizer import quantize_midi_file
from didactic_engine.metadata_export import create_metadata_entry, export_metadata_jsonl
from didactic_engine.resilience import (
    retry_with_backoff,
    demucs_circuit,
    basic_pitch_circuit,
)

# Optional tqdm import for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Define a no-op tqdm fallback
    def tqdm(iterable, **kwargs):
        return iterable


def _resolve_progress_enabled(explicit: Optional[bool]) -> bool:
    if not TQDM_AVAILABLE:
        return False
    if explicit is not None:
        return explicit
    try:
        return sys.stderr.isatty()
    except Exception:
        return False

if TYPE_CHECKING:
    from didactic_engine.config import PipelineConfig


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _PipelineStep:
    """Lightweight context manager to log step start/completion with duration."""

    def __init__(self, log: logging.Logger, name: str, timings_dict: Optional[Dict[str, float]] = None):
        self.log = log
        self.name = name
        self.start_time = 0.0
        self.timings_dict = timings_dict

    def __enter__(self) -> "_PipelineStep":
        self.start_time = time.time()
        self.log.info(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.time() - self.start_time

        # Store timing in the dict if provided
        if self.timings_dict is not None:
            # Convert step name to a dict key (e.g., "Step 2: Ingest WAV file" -> "step_2_ingest_wav")
            key = self.name.lower().replace(":", "").replace(" ", "_")
            # Remove duplicate underscores
            while "__" in key:
                key = key.replace("__", "_")
            self.timings_dict[key] = round(duration, 3)

        if exc_type is None:
            self.log.info("%s completed in %.2fs", self.name, duration)
        else:
            self.log.exception("%s failed after %.2fs", self.name, duration)
        return False


class AudioPipeline:
    """
    Complete audio processing pipeline.

    Ingests WAV audio, separates stems (optional), analyzes audio,
    transcribes to MIDI, aligns to beat/bar grid, segments stems,
    and extracts features.
    """

    def __init__(
        self,
        cfg: "PipelineConfig",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the audio pipeline.

        Args:
            cfg: Pipeline configuration.
            logger: Optional logger to use for pipeline messages. Falls back to
                the module logger when not provided.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.ingester = WAVIngester(sample_rate=cfg.analysis_sr)
        self.preprocessor = AudioPreprocessor()
        self.analyzer = AudioAnalyzer(use_essentia=cfg.use_essentia_features)
        self.midi_parser = MIDIParser()
        self.segmenter = StemSegmenter()
        self.feature_extractor = FeatureExtractor()

        # Dictionary to store step timings for performance analysis
        self.step_timings: Dict[str, float] = {}

    @retry_with_backoff(
        max_retries=3,
        base_delay=1.0,
        retryable_exceptions=(IOError, TimeoutError, OSError)
    )
    def _load_stem_with_retry(self, stem_path: Path) -> tuple[Any, int]:
        """Load a stem file with automatic retry on transient failures.

        Uses exponential backoff to handle temporary I/O issues, network
        failures, or disk contention.

        Args:
            stem_path: Path to the stem audio file.

        Returns:
            Tuple of (audio_array, sample_rate).

        Raises:
            RetryError: If all retry attempts are exhausted.
            Other exceptions: If non-retryable errors occur.
        """
        return self.ingester.load(stem_path)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete audio processing pipeline.

        Returns:
            Dictionary containing all processing results and output paths.
        """
        cfg = self.cfg
        self.step_timings.clear()
        self.logger.info(
            "Starting audio processing pipeline for %s (song_id=%s)",
            cfg.input_wav,
            cfg.song_id,
        )

        cfg.create_directories()

        # Operational warnings for common slow / brittle environments.
        # These are warnings only (no behavioral changes).
        self._emit_environment_warnings()

        results: Dict[str, Any] = {
            "song_id": cfg.song_id,
            "input_path": str(cfg.input_wav),
            "output_dir": str(cfg.out_dir),
        }

        with self._step("Step 1: Copy input file"):
            input_copy_path = self._copy_input()
        results["input_copy"] = str(input_copy_path)

        with self._step("Step 2: Ingest WAV file"):
            audio, sr = self._ingest_audio()
        results["sample_rate"] = sr
        results["duration_s"] = float(len(audio) / sr)

        with self._step("Step 3: Preprocess audio"):
            audio, sr, preprocessed_path = self._maybe_preprocess(audio, sr)
        if preprocessed_path:
            results["preprocessed_path"] = str(preprocessed_path)

        with self._step("Step 4: Analyze audio"):
            analysis, tempo_bpm, beat_times = self._analyze_audio(
                audio, sr, results["duration_s"]
            )
        results["analysis"] = {
            "tempo_bpm": tempo_bpm,
            "num_beats": len(beat_times),
            "duration_s": results["duration_s"],
            "essentia_available": analysis.get("essentia", {}).get("available", False),
        }

        with self._step("Step 5: Separate stems (Demucs)"):
            stem_paths, separation_error = self._separate_stems()
        results["stems"] = list(stem_paths.keys())
        if separation_error:
            results["stem_separation_error"] = separation_error

        with self._step("Step 6: Compute bar boundaries"):
            bar_boundaries = self._compute_bar_boundaries(
                beat_times, tempo_bpm, results["duration_s"]
            )
        results["num_bars"] = len(bar_boundaries)

        with self._step("Step 7: Process stems (chunks/transcription/align)"):
            all_notes, all_bar_features, midi_results = self._process_stems(
                stem_paths,
                bar_boundaries,
                beat_times,
                tempo_bpm,
                audio,
                sr,
            )
        results.update(midi_results)

        with self._step("Step 8: Build datasets"):
            events_df, beats_df, bars_df, bar_features_df = self._build_datasets(
                all_notes, beat_times, tempo_bpm, results.get(
                    "stems", []), all_bar_features
            )

        with self._step("Step 9: Write Parquet datasets"):
            self._write_parquet_outputs(
                events_df, beats_df, bars_df, bar_features_df, results
            )

        with self._step("Step 10: Export reports"):
            self._export_reports(all_notes, results)

        with self._step("Step 11: Write summary JSON"):
            summary_path = self._write_summary(
                results, beat_times, bar_boundaries, all_notes
            )
        results["summary_json"] = str(summary_path)

        # Optional: Export metadata.jsonl for HuggingFace AudioFolder
        if self.cfg.export_metadata_jsonl:
            with self._step("Step 12: Export metadata.jsonl"):
                self._export_metadata_jsonl(results, bar_boundaries, tempo_bpm)

        self.logger.info("Pipeline completed successfully for %s", cfg.song_id)

        return results

    def _step(self, name: str) -> _PipelineStep:
        """Return a context manager that logs the lifecycle of a pipeline step."""
        return _PipelineStep(self.logger, name, self.step_timings)

    def _progress_enabled(self) -> bool:
        return _resolve_progress_enabled(self.cfg.enable_progress)

    def _emit_environment_warnings(self) -> None:
        """Emit warnings for common environment/performance footguns.

        This is intentionally best-effort and avoids importing heavy optional
        dependencies.
        """
        cfg = self.cfg

        # WSL detection: kernel release commonly contains 'microsoft'.
        is_wsl = False
        if platform.system().lower() == "linux":
            kernel_release_path = Path("/proc/sys/kernel/osrelease")
            try:
                release = kernel_release_path.read_text(encoding="utf-8").lower()
                is_wsl = "microsoft" in release
            except Exception as exc:
                # Best-effort detection only; log at DEBUG so issues are diagnosable
                self.logger.debug(
                    "WSL detection failed when reading %s; assuming non-WSL environment. "
                    "Error: %s",
                    kernel_release_path,
                    exc,
                    exc_info=True,
                )
                is_wsl = False

        if is_wsl:
            out_str = str(cfg.out_dir)
            if out_str.startswith("/mnt/") and cfg.write_bar_chunks and cfg.write_bar_chunk_wavs:
                self.logger.warning(
                    "WSL detected and output directory is on a Windows-mounted path (%s). "
                    "Writing many chunk WAVs can be very slow here. Consider using --no-chunk-wavs "
                    "or writing outputs under the WSL filesystem (e.g., /home/<user>/...).",
                    out_str,
                )

    def _copy_input(self) -> Path:
        """Copy the original input file into the output directory."""
        input_copy_path = self.cfg.input_dir / self.cfg.input_wav.name
        shutil.copy2(self.cfg.input_wav, input_copy_path)
        self.logger.debug("Copied input to %s", input_copy_path)
        return input_copy_path

    def _ingest_audio(self) -> Tuple[Any, int]:
        """Load and validate the input WAV file."""
        audio, sr = self.ingester.load(self.cfg.input_wav)

        if not self.ingester.validate(audio, sr):
            raise ValueError("Invalid audio data")

        self.logger.info(
            "Loaded audio: duration=%.2fs, sr=%s",
            len(audio) / sr,
            sr,
        )
        return audio, sr

    def _maybe_preprocess(self, audio: Any, sr: int) -> Tuple[Any, int, Optional[Path]]:
        """Optionally preprocess audio and persist the preprocessed file."""
        if not self.cfg.use_pydub_preprocess:
            self.logger.info("Skipping preprocessing step per configuration")
            return audio, sr, None

        audio, sr = self.preprocessor.preprocess(audio, sr, self.cfg)
        preprocessed_path = self.cfg.preprocess_dir / f"{self.cfg.song_id}.wav"
        self.ingester.save(audio, sr, preprocessed_path)
        self.logger.info("Saved preprocessed audio to %s", preprocessed_path)
        return audio, sr, preprocessed_path

    def _analyze_audio(
        self,
        audio: Any,
        sr: int,
        duration_s: float,
    ) -> Tuple[Dict[str, Any], float, List[float]]:
        """Run audio analysis and return analysis results."""
        analysis = self.analyzer.analyze(audio, sr)
        tempo_bpm = analysis.get("tempo", analysis.get(
            "librosa", {}).get("tempo_bpm", 120.0))
        beat_times = analysis.get("beat_times", [])
        self.logger.info(
            "Detected tempo %.2f BPM with %d beats (duration %.2fs)",
            tempo_bpm,
            len(beat_times),
            duration_s,
        )

        if analysis.get("essentia", {}).get("available", False):
            self.logger.info("Essentia features available for this run")

        return analysis, tempo_bpm, beat_times

    def _separate_stems(self) -> tuple[Dict[str, Path], Optional[str]]:
        """Attempt Demucs stem separation with circuit breaker protection.

        Uses circuit breaker pattern to prevent cascading failures when
        Demucs is unavailable or repeatedly failing.
        """
        if not self.cfg.use_demucs_separation:
            self.logger.info("Stem separation disabled by configuration; using full_mix")
            return {"full_mix": self.cfg.input_wav}, "disabled by configuration"

        try:
            from didactic_engine.separation import StemSeparator

            # Use circuit breaker to protect against repeated Demucs failures
            with demucs_circuit:
                separator = StemSeparator(
                    model=self.cfg.demucs_model,
                    device=self.cfg.demucs_device,
                    timeout_s=self.cfg.demucs_timeout_s,
                )
                stem_paths = separator.separate(
                    self.cfg.input_wav, self.cfg.stems_dir)
                self.logger.info(
                    "Separated into %d stems: %s",
                    len(stem_paths),
                    list(stem_paths.keys()),
                )
                return stem_paths, None
        except RuntimeError as exc:
            self.logger.warning(
                "Stem separation skipped (circuit state: %s): %s",
                demucs_circuit.state.value,
                exc
            )
            return {"full_mix": self.cfg.input_wav}, str(exc)

    def _compute_bar_boundaries(
        self,
        beat_times: List[float],
        tempo_bpm: float,
        duration_s: float,
    ) -> List[Tuple[int, float, float]]:
        """Convert beat positions into bar boundaries."""
        bar_boundaries = segment_beats_into_bars(
            beat_times,
            tempo_bpm,
            self.cfg.time_signature_num,
            self.cfg.time_signature_den,
            duration_s,
        )
        if self.cfg.time_signature_num <= 0 or self.cfg.time_signature_den <= 0:
            beats_per_bar = 4.0
        else:
            beats_per_bar = self.cfg.time_signature_num * \
                (4.0 / self.cfg.time_signature_den)
        self.logger.info(
            "Computed %d bars (beats_per_bar=%.2f)",
            len(bar_boundaries),
            beats_per_bar,
        )
        return bar_boundaries

    def _process_stems(
        self,
        stem_paths: Dict[str, Path],
        bar_boundaries: List[Tuple[int, float, float]],
        beat_times: List[float],
        tempo_bpm: float,
        analysis_audio: Any,
        analysis_sr: int,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
        """Process each stem: optional chunking, MIDI transcription, and alignment."""
        all_notes_dfs: List[pd.DataFrame] = []
        all_bar_features: List[Dict[str, Any]] = []
        results_updates: Dict[str, Any] = {}

        for stem_name, stem_path in stem_paths.items():
            self.logger.info("Processing stem '%s'", stem_name)
            stem_audio = None
            stem_sr = None
            stem_audio_native = None  # May be stereo if preserve_chunk_audio=True
            stem_sr_native = None
            stem_audio_source = "stem"

            try:
                # Load once at native SR to avoid redundant I/O
                import soundfile as sf
                import librosa
                import numpy as np

                raw_audio, stem_sr_native = sf.read(str(stem_path))

                # Ensure float32 for consistency
                raw_audio = raw_audio.astype(np.float32)

                # Keep original for chunk preservation BEFORE mono conversion
                # This ensures stereo files stay stereo when preserve_chunk_audio=True
                stem_audio_native = raw_audio

                # Convert to mono for analysis (always needed for feature extraction)
                if raw_audio.ndim == 2:
                    mono_audio = np.mean(raw_audio, axis=1)
                else:
                    mono_audio = raw_audio

                # Resample to analysis SR if needed
                if stem_sr_native != self.cfg.analysis_sr:
                    stem_audio = librosa.resample(
                        mono_audio,
                        orig_sr=stem_sr_native,
                        target_sr=self.cfg.analysis_sr
                    )
                    stem_sr = self.cfg.analysis_sr
                else:
                    stem_audio = mono_audio
                    stem_sr = stem_sr_native

                num_samples = len(stem_audio_native) if stem_audio_native.ndim == 1 else stem_audio_native.shape[0]
                channels_str = "mono" if stem_audio_native.ndim == 1 else f"stereo ({stem_audio_native.shape[1]}ch)"
                self.logger.debug(
                    "Loaded stem '%s' (%d samples @ %d Hz native %s, resampled to %d Hz mono for analysis)",
                    stem_name,
                    num_samples,
                    stem_sr_native,
                    channels_str,
                    stem_sr,
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to load stem '%s' from %s: %s. Attempting fallback to analysis audio.",
                    stem_name,
                    stem_path,
                    exc,
                )

                # Validate that analysis_audio is suitable as a fallback
                if analysis_audio is None or len(analysis_audio) == 0:
                    raise RuntimeError(
                        f"Cannot use analysis audio as fallback for stem '{stem_name}': "
                        f"analysis audio is empty or None"
                    ) from exc

                # Calculate expected stem length based on analysis audio
                # Demucs should produce stems with the same length as input
                expected_duration_s = len(analysis_audio) / analysis_sr
                min_acceptable_samples = int(expected_duration_s * 0.95 * analysis_sr)

                if len(analysis_audio) < min_acceptable_samples:
                    raise RuntimeError(
                        f"Cannot use analysis audio as fallback for stem '{stem_name}': "
                        f"length mismatch detected. Expected ~{expected_duration_s:.1f}s of audio "
                        f"but analysis audio is only {len(analysis_audio)/analysis_sr:.1f}s. "
                        f"This could indicate preprocessing or truncation issues."
                    ) from exc

                # Fallback is validated as safe
                stem_audio = analysis_audio
                stem_sr = analysis_sr
                stem_audio_source = "analysis_fallback"
                results_updates[f"stem_audio_fallback_{stem_name}"] = str(exc)
                self.logger.info(
                    "Using validated analysis audio as fallback for stem '%s' "
                    "(%d samples @ %d Hz)",
                    stem_name,
                    len(stem_audio),
                    stem_sr,
                )

            # Use already-loaded native audio for chunk preservation
            # This eliminates redundant file I/O
            stem_audio_original = None
            stem_sr_original = None
            if (
                stem_audio_source == "stem"
                and self.cfg.write_bar_chunks
                and self.cfg.write_bar_chunk_wavs
                and self.cfg.preserve_chunk_audio
                and stem_audio_native is not None
            ):
                # Reuse the native audio we already loaded (no redundant I/O!)
                stem_audio_original = stem_audio_native
                stem_sr_original = stem_sr_native
                self.logger.debug(
                    "Using native-resolution stem audio for chunk preservation "
                    "(%d samples @ %d Hz)",
                    len(stem_audio_original) if stem_audio_original.ndim == 1 else stem_audio_original.shape[0],
                    stem_sr_original,
                )

            if self.cfg.write_bar_chunks:
                if stem_audio is None or stem_sr is None:
                    self.logger.warning(
                        "Skipping bar chunking for '%s' due to missing audio",
                        stem_name,
                    )
                else:
                    stem_chunks_dir = self.cfg.chunks_dir / stem_name
                    stem_chunks_dir.mkdir(parents=True, exist_ok=True)
                    write_chunk_wavs = self.cfg.write_bar_chunk_wavs and stem_audio_source == "stem"
                    if write_chunk_wavs:
                        mode_str = "original audio" if self.cfg.preserve_chunk_audio else "analysis audio (mono @ {}Hz)".format(stem_sr)
                        self.logger.info(
                            "Writing bar chunks to %s (%s)", stem_chunks_dir, mode_str)
                    else:
                        if self.cfg.write_bar_chunk_wavs and stem_audio_source != "stem":
                            self.logger.info(
                                "Skipping chunk WAV writes for '%s' (fallback audio source: %s)",
                                stem_name,
                                stem_audio_source,
                            )
                        else:
                            self.logger.info(
                                "Computing bar features without writing chunk WAVs (write_bar_chunk_wavs=False)"
                            )

                    precomputed_bar_features = None
                    if self.cfg.bar_feature_precompute:
                        try:
                            precomputed_bar_features = self.feature_extractor.precompute_bar_features(
                                stem_audio,
                                stem_sr,
                                bar_boundaries,
                                hop_length=self.cfg.hop_length,
                            )
                            if precomputed_bar_features:
                                self.logger.debug(
                                    "Precomputed bar features for '%s' (%d bars)",
                                    stem_name,
                                    len(precomputed_bar_features),
                                )
                        except Exception as exc:
                            self.logger.warning(
                                "Precomputing bar features failed for '%s': %s",
                                stem_name,
                                exc,
                            )
                            precomputed_bar_features = None

                    # Performance note:
                    # We already have `stem_audio` in memory resampled to `stem_sr` (typically
                    # cfg.analysis_sr via WAVIngester). Avoid per-chunk decode + resample by:
                    #   1) slicing `stem_audio` directly
                    #   2) computing features from the slice
                    #   3) optionally writing the slice to disk for inspection/training
                    # When preserve_chunk_audio=True, we slice from stem_audio_original for WAV writing
                    # and from stem_audio for feature extraction.
                    import soundfile as sf  # type: ignore[import-not-found]

                    # Add progress bar for bar-by-bar processing if tqdm is available
                    progress_enabled = self._progress_enabled()
                    bar_iter = tqdm(
                        bar_boundaries,
                        desc=f"Processing bars ({stem_name})",
                        unit="bar",
                        disable=not progress_enabled,
                        leave=False,
                    )

                    for bar_idx, start_s, end_s in bar_iter:
                        try:
                            start_sample = max(0, int(round(start_s * stem_sr)))
                            end_sample = min(len(stem_audio), int(round(end_s * stem_sr)))

                            # Skip empty/invalid segments (can occur due to rounding/clamping)
                            if end_sample <= start_sample:
                                continue

                            chunk_audio = stem_audio[start_sample:end_sample]

                            # Keep storage small and consistent
                            try:
                                import numpy as np

                                if (
                                    isinstance(chunk_audio, np.ndarray)
                                    and chunk_audio.dtype != np.float32
                                ):
                                    chunk_audio = chunk_audio.astype(np.float32, copy=False)
                            except Exception:
                                # If numpy isn't available or casting fails, proceed with original.
                                pass

                            chunk_path: Optional[Path] = None
                            if write_chunk_wavs:
                                chunk_path = stem_chunks_dir / f"bar_{bar_idx:04d}.wav"

                                # Use original audio for chunks if preserve_chunk_audio is enabled.
                                if (
                                    self.cfg.preserve_chunk_audio
                                    and stem_audio_original is not None
                                    and stem_sr_original is not None
                                ):
                                    start_sample_orig = max(
                                        0, int(round(start_s * stem_sr_original))
                                    )
                                    end_sample_orig = min(
                                        (
                                            len(stem_audio_original)
                                            if stem_audio_original.ndim == 1
                                            else stem_audio_original.shape[0]
                                        ),
                                        int(round(end_s * stem_sr_original)),
                                    )

                                    if stem_audio_original.ndim == 1:
                                        chunk_audio_to_write = stem_audio_original[
                                            start_sample_orig:end_sample_orig
                                        ]
                                    else:
                                        chunk_audio_to_write = stem_audio_original[
                                            start_sample_orig:end_sample_orig, :
                                        ]

                                    sf.write(
                                        str(chunk_path),
                                        chunk_audio_to_write,
                                        stem_sr_original,
                                    )
                                else:
                                    sf.write(str(chunk_path), chunk_audio, stem_sr)

                            # Feature extraction directly from the chunk audio avoids expensive
                            # per-file resampling (librosa/soxr) during large runs.
                            if precomputed_bar_features and bar_idx in precomputed_bar_features:
                                features = dict(precomputed_bar_features[bar_idx])
                            else:
                                features = self.feature_extractor.extract_bar_features_from_audio(
                                    chunk_audio,
                                    stem_sr,
                                )

                            features.update(
                                {
                                    "song_id": self.cfg.song_id,
                                    "stem": stem_name,
                                    "bar_index": bar_idx,
                                    "start_s": start_s,
                                    "end_s": end_s,
                                    "duration_s": end_s - start_s,
                                    "tempo_bpm": tempo_bpm,
                                    # Keep schema stable for Parquet.
                                    "chunk_path": str(chunk_path) if chunk_path is not None else "",
                                    "audio_source": stem_audio_source,
                                }
                            )
                            all_bar_features.append(features)
                        except Exception as exc:
                            self.logger.warning(
                                "Failed to process bar %d for stem '%s': %s",
                                bar_idx,
                                stem_name,
                                exc,
                            )
            else:
                self.logger.info(
                    "Bar chunking disabled; skipping chunk export for '%s'", stem_name)

            if not self.cfg.use_basic_pitch_transcription:
                self.logger.info(
                    "MIDI transcription disabled by configuration; skipping for stem '%s'",
                    stem_name,
                )
            else:
                try:
                    from didactic_engine.transcription import BasicPitchTranscriber

                    # Use circuit breaker to protect against repeated Basic Pitch failures
                    with basic_pitch_circuit:
                        transcriber = BasicPitchTranscriber(
                            model_serialization=self.cfg.basic_pitch_backend,
                            timeout_s=self.cfg.basic_pitch_timeout_s,
                            keep_runs=self.cfg.basic_pitch_keep_runs,
                        )
                        midi_path = transcriber.transcribe(stem_path, self.cfg.midi_dir)
                        results_updates[f"midi_{stem_name}"] = str(midi_path)
                        self.logger.info("Transcribed %s to MIDI at %s", stem_name, midi_path)

                    # Optionally quantize MIDI for cleaner ABC notation
                    if self.cfg.quantize_midi:
                        quantized_midi_path = self.cfg.midi_dir / f"{stem_name}_quantized.mid"
                        success = quantize_midi_file(
                            midi_path,
                            quantized_midi_path,
                            tempo_bpm=tempo_bpm,
                            division=self.cfg.quantize_division,
                        )
                        if success:
                            # Use quantized MIDI for ABC export
                            results_updates[f"midi_quantized_{stem_name}"] = str(quantized_midi_path)
                            self.logger.info(
                                "Quantized MIDI for %s (division=%d) at %s",
                                stem_name,
                                self.cfg.quantize_division,
                                quantized_midi_path,
                            )
                        else:
                            self.logger.warning(
                                "MIDI quantization failed for %s, using original", stem_name
                            )

                    midi_data = self.midi_parser.parse(midi_path)
                    notes_df = midi_data["notes_df"]

                    if not notes_df.empty:
                        aligned_df = align_notes_to_beats(
                            notes_df,
                            beat_times,
                            tempo_bpm,
                            self.cfg.time_signature_num,
                            self.cfg.time_signature_den,
                        )
                        aligned_df["song_id"] = self.cfg.song_id
                        aligned_df["stem"] = stem_name
                        all_notes_dfs.append(aligned_df)
                        self.logger.info(
                            "Aligned %d notes for stem '%s'",
                            len(aligned_df),
                            stem_name,
                        )
                    else:
                        self.logger.info("No notes detected for stem '%s'", stem_name)

                except Exception as exc:
                    message = str(exc)
                    results_updates[f"transcription_error_{stem_name}"] = message
                    self.logger.warning(
                        "MIDI transcription skipped for stem '%s' (circuit state: %s): %s",
                        stem_name,
                        basic_pitch_circuit.state.value,
                        message,
                    )

        all_notes = pd.concat(
            all_notes_dfs, ignore_index=True) if all_notes_dfs else pd.DataFrame()
        return all_notes, all_bar_features, results_updates

    def _build_datasets(
        self,
        all_notes: pd.DataFrame,
        beat_times: List[float],
        tempo_bpm: float,
        stems: List[str],
        all_bar_features: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create pandas datasets from aligned notes, beats, and bar features."""
        events_df = self.feature_extractor.extract_events(all_notes)

        beats_rows = []
        for stem in stems:
            beats_df = self.feature_extractor.extract_beats(
                beat_times, tempo_bpm, stem, self.cfg.song_id
            )
            beats_rows.append(beats_df)
        beats_df = pd.concat(beats_rows, ignore_index=True) if beats_rows else pd.DataFrame()

        bars_df = self.feature_extractor.extract_bars(
            all_notes, self.cfg.song_id)
        bar_features_df = pd.DataFrame(all_bar_features) if all_bar_features else pd.DataFrame()

        self.logger.info(
            "Dataset shapes - events: %d, beats: %d, bars: %d, bar_features: %d",
            len(events_df),
            len(beats_df),
            len(bars_df),
            len(bar_features_df),
        )

        return events_df, beats_df, bars_df, bar_features_df

    def _write_parquet_outputs(
        self,
        events_df: pd.DataFrame,
        beats_df: pd.DataFrame,
        bars_df: pd.DataFrame,
        bar_features_df: pd.DataFrame,
        results: Dict[str, Any],
    ) -> None:
        """Persist generated datasets to Parquet files."""
        cfg = self.cfg
        cfg.datasets_dir.mkdir(parents=True, exist_ok=True)

        if not events_df.empty:
            events_path = cfg.datasets_dir / "events.parquet"
            events_df.to_parquet(events_path, index=False)
            results["events_parquet"] = str(events_path)
            self.logger.info(
                "Wrote events dataset to %s (%d rows)", events_path, len(events_df))

        if not beats_df.empty:
            beats_path = cfg.datasets_dir / "beats.parquet"
            beats_df.to_parquet(beats_path, index=False)
            results["beats_parquet"] = str(beats_path)
            self.logger.info(
                "Wrote beats dataset to %s (%d rows)", beats_path, len(beats_df))

        if not bars_df.empty:
            bars_path = cfg.datasets_dir / "bars.parquet"
            bars_df.to_parquet(bars_path, index=False)
            results["bars_parquet"] = str(bars_path)
            self.logger.info(
                "Wrote bars dataset to %s (%d rows)", bars_path, len(bars_df))

        if not bar_features_df.empty:
            bar_features_path = cfg.datasets_dir / "bar_features.parquet"
            bar_features_df.to_parquet(bar_features_path, index=False)
            results["bar_features_parquet"] = str(bar_features_path)
            self.logger.info(
                "Wrote bar_features dataset to %s (%d rows)",
                bar_features_path,
                len(bar_features_df),
            )

    def _export_reports(self, all_notes: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Export human-readable reports based on aligned notes and MIDI files."""
        cfg = self.cfg
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)

        if not all_notes.empty:
            md_path = cfg.reports_dir / "midi_markdown.md"
            aligned_dict = {}
            for _, row in all_notes.iterrows():
                bar_idx = int(row.get("bar_index", 0))
                if bar_idx not in aligned_dict:
                    aligned_dict[bar_idx] = []
                aligned_dict[bar_idx].append(row.to_dict())
            export_midi_markdown(aligned_dict, str(md_path), cfg.song_id)
            results["markdown_report"] = str(md_path)
            self.logger.info("Wrote Markdown report to %s", md_path)

        for stem_name in results.get("stems", []):
            # Prefer quantized MIDI if available for ABC export
            midi_quantized_key = f"midi_quantized_{stem_name}"
            midi_key = f"midi_{stem_name}"
            # Prefer quantized MIDI if available for ABC export
            if midi_quantized_key in results:
                midi_path_for_abc = results[midi_quantized_key]
                self.logger.debug("Using quantized MIDI for ABC export: %s", stem_name)
            elif midi_key in results:
                midi_path_for_abc = results[midi_key]
            else:
                continue
            
            abc_path = cfg.reports_dir / f"{stem_name}.abc"
            success = export_abc(midi_path_for_abc, str(abc_path))
            if success:
                results[f"abc_{stem_name}"] = str(abc_path)
                
                # Also export ABC as text for metadata
                abc_text = export_abc_text(midi_path_for_abc)
                if abc_text:
                    results[f"abc_text_{stem_name}"] = abc_text
                
                self.logger.info(
                    "Wrote ABC notation for '%s' to %s", stem_name, abc_path)
            else:
                self.logger.warning(
                    "ABC export failed for '%s' (music21 may not be installed or MIDI parsing failed)",
                    stem_name
                )

    def _write_summary(
        self,
        results: Dict[str, Any],
        beat_times: List[float],
        bar_boundaries: List[Tuple[int, float, float]],
        all_notes: pd.DataFrame,
    ) -> Path:
        """Write a JSON summary capturing core run metadata."""
        cfg = self.cfg
        cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cfg.analysis_dir / "combined.json"

        summary = {
            "song_id": cfg.song_id,
            "duration_s": results.get("duration_s"),
            "tempo_bpm": results["analysis"]["tempo_bpm"],
            "num_beats": len(beat_times),
            "num_bars": len(bar_boundaries),
            "stems": results.get("stems", []),
            "num_notes_per_stem": {},
            "essentia_requested": cfg.use_essentia_features,
            "essentia_available": results.get("analysis", {}).get("essentia_available", False),
            "files_generated": [],
            "step_timings": self.step_timings,
            "config": {
                "analysis_sr": cfg.analysis_sr,
                "hop_length": cfg.hop_length,
                "time_signature_num": cfg.time_signature_num,
                "time_signature_den": cfg.time_signature_den,
                "use_pydub_preprocess": cfg.use_pydub_preprocess,
                "use_essentia_features": cfg.use_essentia_features,
                "write_bar_chunks": cfg.write_bar_chunks,
                "write_bar_chunk_wavs": cfg.write_bar_chunk_wavs,
                "preserve_chunk_audio": cfg.preserve_chunk_audio,
                "bar_feature_precompute": cfg.bar_feature_precompute,
                "use_demucs_separation": cfg.use_demucs_separation,
                "demucs_model": cfg.demucs_model,
                "demucs_device": cfg.demucs_device,
                "demucs_timeout_s": cfg.demucs_timeout_s,
                "use_basic_pitch_transcription": cfg.use_basic_pitch_transcription,
                "basic_pitch_backend": cfg.basic_pitch_backend,
                "basic_pitch_timeout_s": cfg.basic_pitch_timeout_s,
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        }

        # Add lightweight capability hints (best-effort).
        # We do not import heavy modules here; use PATH checks and lightweight helpers.
        try:
            import shutil

            summary["capabilities"] = {
                "demucs_on_path": shutil.which("demucs") is not None,
                "basic_pitch_on_path": shutil.which("basic-pitch") is not None,
            }
        except Exception:
            summary["capabilities"] = {}

        if not all_notes.empty:
            for stem in all_notes["stem"].unique():
                summary["num_notes_per_stem"][stem] = int(
                    len(all_notes[all_notes["stem"] == stem])
                )

        for key, value in results.items():
            if (
                key.endswith(("_parquet", "_report"))
                or key.startswith("abc_")
                or key == "markdown_report"
            ):
                summary["files_generated"].append(value)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Wrote summary to %s", summary_path)
        return summary_path

    def _export_metadata_jsonl(
        self,
        results: Dict[str, Any],
        bar_boundaries: List[Tuple[int, float, float]],
        tempo_bpm: float,
    ) -> None:
        """Export metadata.jsonl for HuggingFace AudioFolder dataset."""
        cfg = self.cfg
        
        # Create metadata directory
        metadata_dir = cfg.datasets_dir
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / "metadata.jsonl"
        
        # Collect metadata entries
        entries: List[Dict[str, Any]] = []
        
        # Get ABC texts for each stem
        abc_texts = {}
        for stem_name in results.get("stems", []):
            abc_text_key = f"abc_text_{stem_name}"
            if abc_text_key in results:
                abc_texts[stem_name] = results[abc_text_key]
        
        # If we have bar chunks with features, create entries for each
        bar_features_path = results.get("bar_features_parquet")
        if bar_features_path and Path(bar_features_path).exists():
            bar_features_df = pd.read_parquet(bar_features_path)
            
            for _, row in bar_features_df.iterrows():
                stem_name = row.get("stem", "")
                bar_index = int(row.get("bar_index", 0))
                chunk_path = row.get("chunk_path", "")
                
                # Skip if no chunk path (write_bar_chunk_wavs=False)
                if not chunk_path or not Path(chunk_path).exists():
                    continue
                
                # Get ABC text for this stem
                abc_text = abc_texts.get(stem_name, "NA")
                
                # Make chunk path relative to metadata file
                chunk_path_abs = Path(chunk_path)
                try:
                    # Try to make it relative to datasets_dir
                    file_name = chunk_path_abs.relative_to(cfg.out_dir).as_posix()
                except ValueError:
                    # If not relative, use absolute path
                    file_name = chunk_path_abs.as_posix()
                
                # Create metadata entry
                entry = create_metadata_entry(
                    file_name=file_name,
                    abc_text=abc_text,
                    source_track=cfg.input_wav.name,
                    stem_used=stem_name,
                    start_sec=float(row.get("start_s", 0.0)),
                    end_sec=float(row.get("end_s", 0.0)),
                    tempo_bpm=tempo_bpm,
                    sample_rate=cfg.analysis_sr,
                    trigger_token=cfg.abc_trigger_token,
                    extra_fields={
                        "bar_index": bar_index,
                        "rms_energy": float(row.get("rms_energy", 0.0)),
                    },
                )
                entries.append(entry)
        
        # If no bar features, create one entry per stem
        if not entries:
            for stem_name in results.get("stems", []):
                abc_text = abc_texts.get(stem_name, "NA")
                stem_path_key = f"midi_{stem_name}"
                
                if stem_path_key in results:
                    # Use stem audio file as the audio reference
                    stem_path = Path(results[stem_path_key]).parent.parent / "stems" / cfg.song_id / f"{stem_name}.wav"
                    if stem_path.exists():
                        try:
                            file_name = stem_path.relative_to(cfg.out_dir).as_posix()
                        except ValueError:
                            file_name = stem_path.as_posix()
                        
                        entry = create_metadata_entry(
                            file_name=file_name,
                            abc_text=abc_text,
                            source_track=cfg.input_wav.name,
                            stem_used=stem_name,
                            tempo_bpm=tempo_bpm,
                            sample_rate=cfg.analysis_sr,
                            trigger_token=cfg.abc_trigger_token,
                        )
                        entries.append(entry)
        
        # Export to JSONL
        if entries:
            count = export_metadata_jsonl(entries, metadata_path)
            results["metadata_jsonl"] = str(metadata_path)
            results["metadata_entry_count"] = count
            self.logger.info(
                "Wrote %d metadata entries to %s", count, metadata_path
            )
        else:
            self.logger.warning(
                "No metadata entries to export (no bar chunks or stems found)"
            )

    @staticmethod
    def process_batch(
        input_files: List[Path],
        out_dir: Path,
        song_ids: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multiple WAV files in batch mode.

        Args:
            input_files: List of input WAV file paths.
            out_dir: Base output directory for all results.
            song_ids: Optional list of song IDs (one per file). If not provided,
                     filenames (without extension) are used as song IDs.
            logger: Optional logger for emitting progress messages.
            **kwargs: Additional configuration parameters passed to PipelineConfig
                     (e.g., analysis_sr, use_essentia_features, etc.)

        Returns:
            Dictionary containing:
                - 'successful': List of (song_id, input_path, result_dict) tuples
                - 'failed': List of (song_id, input_path, error_message) tuples
                - 'total': Total number of files processed
                - 'success_count': Number of successful processings
                - 'failure_count': Number of failed processings

        Example:
            >>> from pathlib import Path
            >>> from didactic_engine.pipeline import AudioPipeline
            >>> 
            >>> files = [Path("song1.wav"), Path("song2.wav")]
            >>> results = AudioPipeline.process_batch(
            ...     files,
            ...     Path("output"),
            ...     analysis_sr=22050,
            ...     use_essentia_features=False
            ... )
            >>> print(f"Processed {results['success_count']} files successfully")
        """
        # Import here to avoid circular imports at module load time.
        # config.py doesn't import pipeline.py, but keeping this pattern
        # ensures flexibility if the dependency structure changes in the future.
        from didactic_engine.config import PipelineConfig
        
        # Validate input_files is not empty
        if not input_files:
            raise ValueError("input_files list cannot be empty")
        
        if song_ids is None:
            song_ids = [f.stem for f in input_files]
        
        if len(song_ids) != len(input_files):
            raise ValueError(
                f"Number of song_ids ({len(song_ids)}) must match "
                f"number of input_files ({len(input_files)})"
            )
        
        log = logger or logging.getLogger(__name__)
        successful = []
        failed = []

        log.info("Batch processing %d files...", len(input_files))

        # Add progress bar for batch processing if tqdm is available
        progress_enabled = _resolve_progress_enabled(kwargs.get("enable_progress"))
        file_iter = tqdm(
            enumerate(zip(input_files, song_ids), 1),
            total=len(input_files),
            desc="Processing audio files",
            unit="file",
            disable=not progress_enabled,
        )

        for idx, (input_file, song_id) in file_iter:
            log.info(
                "[%d/%d] Processing: %s (song_id=%s)",
                idx,
                len(input_files),
                input_file.name,
                song_id,
            )
            
            try:
                # Validate file exists
                if not input_file.exists():
                    raise FileNotFoundError(f"Input file not found: {input_file}")
                
                # Create configuration for this file
                cfg = PipelineConfig(
                    song_id=song_id,
                    input_wav=input_file,
                    out_dir=out_dir,
                    **kwargs
                )
                
                # Process the file
                pipeline = AudioPipeline(cfg, logger=log)
                result = pipeline.run()
                
                successful.append((song_id, str(input_file), result))
                log.info("  ✓ Completed successfully")
            
            except FileNotFoundError as e:
                # Handle expected missing-file errors explicitly so tests and callers
                # can distinguish them from other processing failures.
                error_msg = f"File not found: {e}"
                failed.append((song_id, str(input_file), error_msg))
                log.warning("  ✗ Failed (missing file): %s", error_msg)
            
            except Exception as e:
                # Preserve batch robustness while keeping the exception type visible
                # in the recorded error message for easier debugging and validation.
                error_msg = f"{type(e).__name__}: {e}"
                failed.append((song_id, str(input_file), error_msg))
                log.error("  ✗ Failed: %s", error_msg)
        
        # Return summary
        summary = {
            "successful": successful,
            "failed": failed,
            "total": len(input_files),
            "success_count": len(successful),
            "failure_count": len(failed),
        }

        log.info(
            "Batch processing complete: %d/%d successful",
            summary["success_count"],
            summary["total"],
        )
        
        return summary


def run_all(cfg: "PipelineConfig", logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Run the complete audio processing pipeline.

    Args:
        cfg: Pipeline configuration.
        logger: Optional logger to use for the pipeline run.

    Returns:
        Dictionary containing all processing results.
    """
    pipeline = AudioPipeline(cfg, logger=logger)
    return pipeline.run()
