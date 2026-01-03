"""
Main audio processing pipeline.

Orchestrates the complete audio processing workflow from ingestion to feature extraction.
"""

import json
import logging
import platform
import shutil
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
from didactic_engine.export_abc import export_abc

if TYPE_CHECKING:
    from didactic_engine.config import PipelineConfig


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _PipelineStep:
    """Lightweight context manager to log step start/completion with duration."""

    def __init__(self, log: logging.Logger, name: str):
        self.log = log
        self.name = name
        self.start_time = 0.0

    def __enter__(self) -> "_PipelineStep":
        self.start_time = time.time()
        self.log.info(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.time() - self.start_time
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

    def run(self) -> Dict[str, Any]:
        """
        Run the complete audio processing pipeline.

        Returns:
            Dictionary containing all processing results and output paths.
        """
        cfg = self.cfg
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
                stem_paths, bar_boundaries, beat_times, tempo_bpm
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

        self.logger.info("Pipeline completed successfully for %s", cfg.song_id)

        return results

    def _step(self, name: str) -> _PipelineStep:
        """Return a context manager that logs the lifecycle of a pipeline step."""
        return _PipelineStep(self.logger, name)

    def _emit_environment_warnings(self) -> None:
        """Emit warnings for common environment/performance footguns.

        This is intentionally best-effort and avoids importing heavy optional
        dependencies.
        """
        cfg = self.cfg

        # WSL detection: kernel release commonly contains 'microsoft'.
        is_wsl = False
        if platform.system().lower() == "linux":
            try:
                release = Path("/proc/sys/kernel/osrelease").read_text(
                    encoding="utf-8"
                ).lower()
                is_wsl = "microsoft" in release
            except Exception:
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

    def _separate_stems(self) -> Tuple[Dict[str, Path], Optional[str]]:
        """Attempt Demucs stem separation, falling back gracefully when unavailable."""
        try:
            from didactic_engine.separation import StemSeparator

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
            self.logger.warning("Stem separation skipped: %s", exc)
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
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
        """Process each stem: optional chunking, MIDI transcription, and alignment."""
        all_notes_dfs: List[pd.DataFrame] = []
        all_bar_features: List[Dict[str, Any]] = []
        results_updates: Dict[str, Any] = {}

        for stem_name, stem_path in stem_paths.items():
            self.logger.info("Processing stem '%s'", stem_name)
            stem_audio, stem_sr = self.ingester.load(stem_path)
            self.logger.debug(
                "Loaded stem '%s' (%d samples @ %d Hz)",
                stem_name,
                len(stem_audio),
                stem_sr,
            )

            # Load original audio if preserving chunk characteristics
            stem_audio_original = None
            stem_sr_original = None
            if self.cfg.write_bar_chunks and self.cfg.write_bar_chunk_wavs and self.cfg.preserve_chunk_audio:
                import soundfile as sf
                # Load at native SR without downmixing or resampling
                stem_audio_original, stem_sr_original = sf.read(str(stem_path))
                # Ensure float32 for consistency
                import numpy as np
                stem_audio_original = stem_audio_original.astype(np.float32)
                self.logger.debug(
                    "Loaded original stem '%s' for chunk preservation (%d samples @ %d Hz, %d channels)",
                    stem_name,
                    len(stem_audio_original) if stem_audio_original.ndim == 1 else stem_audio_original.shape[0],
                    stem_sr_original,
                    1 if stem_audio_original.ndim == 1 else stem_audio_original.shape[1],
                )

            if self.cfg.write_bar_chunks:
                stem_chunks_dir = self.cfg.chunks_dir / stem_name
                stem_chunks_dir.mkdir(parents=True, exist_ok=True)
                if self.cfg.write_bar_chunk_wavs:
                    mode_str = "original audio" if self.cfg.preserve_chunk_audio else "analysis audio (mono @ {}Hz)".format(stem_sr)
                    self.logger.info(
                        "Writing bar chunks to %s (%s)", stem_chunks_dir, mode_str)
                else:
                    self.logger.info(
                        "Computing bar features without writing chunk WAVs (write_bar_chunk_wavs=False)"
                    )

                # Performance note:
                # We already have `stem_audio` in memory resampled to `stem_sr` (typically
                # cfg.analysis_sr via WAVIngester). Avoid per-chunk decode + resample by:
                #   1) slicing `stem_audio` directly
                #   2) computing features from the slice
                #   3) optionally writing the slice to disk for inspection/training
                # When preserve_chunk_audio=True, we slice from stem_audio_original for WAV writing
                # and from stem_audio for feature extraction.
                import soundfile as sf

                for bar_idx, start_s, end_s in bar_boundaries:
                    start_sample = max(0, int(round(start_s * stem_sr)))
                    end_sample = min(len(stem_audio), int(
                        round(end_s * stem_sr)))

                    # Skip empty/invalid segments (can occur due to rounding/clamping)
                    if end_sample <= start_sample:
                        continue

                    chunk_audio = stem_audio[start_sample:end_sample]
                    # Keep storage small and consistent
                    try:
                        import numpy as np

                        if isinstance(chunk_audio, np.ndarray) and chunk_audio.dtype != np.float32:
                            chunk_audio = chunk_audio.astype(
                                np.float32, copy=False)
                    except Exception:
                        # If numpy isn't available or casting fails, proceed with original.
                        pass

                    chunk_path: Optional[Path] = None
                    if self.cfg.write_bar_chunk_wavs:
                        chunk_path = stem_chunks_dir / f"bar_{bar_idx:04d}.wav"

                        # Use original audio for chunks if preserve_chunk_audio is enabled
                        if self.cfg.preserve_chunk_audio and stem_audio_original is not None:
                            start_sample_orig = max(0, int(round(start_s * stem_sr_original)))
                            end_sample_orig = min(
                                len(stem_audio_original) if stem_audio_original.ndim == 1 else stem_audio_original.shape[0],
                                int(round(end_s * stem_sr_original))
                            )
                            if stem_audio_original.ndim == 1:
                                chunk_audio_to_write = stem_audio_original[start_sample_orig:end_sample_orig]
                            else:
                                chunk_audio_to_write = stem_audio_original[start_sample_orig:end_sample_orig, :]
                            sf.write(str(chunk_path), chunk_audio_to_write, stem_sr_original)
                        else:
                            sf.write(str(chunk_path), chunk_audio, stem_sr)

                    # Feature extraction directly from the chunk audio avoids expensive
                    # per-file resampling (librosa/soxr) during large runs.
                    features = self.feature_extractor.extract_bar_features_from_audio(
                        chunk_audio, stem_sr
                    )
                    features.update({
                        "song_id": self.cfg.song_id,
                        "stem": stem_name,
                        "bar_index": bar_idx,
                        "start_s": start_s,
                        "end_s": end_s,
                        "duration_s": end_s - start_s,
                        "tempo_bpm": tempo_bpm,
                        "chunk_path": str(chunk_path) if chunk_path is not None else "",
                    })
                    all_bar_features.append(features)
            else:
                self.logger.info(
                    "Bar chunking disabled; skipping chunk export for '%s'", stem_name)

            try:
                from didactic_engine.transcription import BasicPitchTranscriber

                transcriber = BasicPitchTranscriber(
                    model_serialization=self.cfg.basic_pitch_backend,
                    timeout_s=self.cfg.basic_pitch_timeout_s,
                )
                midi_path = transcriber.transcribe(
                    stem_path, self.cfg.midi_dir
                )
                results_updates[f"midi_{stem_name}"] = str(midi_path)
                self.logger.info(
                    "Transcribed %s to MIDI at %s", stem_name, midi_path
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
                    self.logger.info(
                        "No notes detected for stem '%s'", stem_name)

            except RuntimeError as exc:
                message = str(exc)
                results_updates[f"transcription_error_{stem_name}"] = message
                self.logger.warning(
                    "MIDI transcription skipped for stem '%s': %s",
                    stem_name,
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
            midi_key = f"midi_{stem_name}"
            if midi_key in results:
                abc_path = cfg.reports_dir / f"{stem_name}.abc"
                export_abc(results[midi_key], str(abc_path))
                results[f"abc_{stem_name}"] = str(abc_path)
                self.logger.info(
                    "Wrote ABC notation for '%s' to %s", stem_name, abc_path)

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
            "essentia_used": cfg.use_essentia_features,
            "files_generated": [],
        }

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
        
        for idx, (input_file, song_id) in enumerate(zip(input_files, song_ids), 1):
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
