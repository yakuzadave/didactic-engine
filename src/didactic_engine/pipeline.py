"""
Main audio processing pipeline.

Orchestrates the complete audio processing workflow from ingestion to feature extraction.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from didactic_engine.ingestion import WAVIngester
from didactic_engine.preprocessing import AudioPreprocessor
from didactic_engine.analysis import AudioAnalyzer
from didactic_engine.midi_parser import MIDIParser
from didactic_engine.segmentation import segment_beats_into_bars, StemSegmenter
from didactic_engine.features import FeatureExtractor
from didactic_engine.align import align_notes_to_beats
from didactic_engine.export_md import export_midi_markdown, export_full_report
from didactic_engine.export_abc import export_abc
from didactic_engine.utils_flatten import flatten_dict_for_parquet

if TYPE_CHECKING:
    from didactic_engine.config import PipelineConfig


class AudioPipeline:
    """
    Complete audio processing pipeline.

    Ingests WAV audio, separates stems (optional), analyzes audio,
    transcribes to MIDI, aligns to beat/bar grid, segments stems,
    and extracts features.
    """

    def __init__(self, cfg: "PipelineConfig"):
        """
        Initialize the audio pipeline.

        Args:
            cfg: Pipeline configuration.
        """
        self.cfg = cfg

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
        print(f"Starting audio processing pipeline for: {cfg.input_wav}")

        # Create all output directories
        cfg.create_directories()

        results: Dict[str, Any] = {
            "song_id": cfg.song_id,
            "input_path": str(cfg.input_wav),
            "output_dir": str(cfg.out_dir),
        }

        # Step 1: Copy original input for traceability
        print("Step 1: Copying input file...")
        input_copy_path = cfg.input_dir / cfg.input_wav.name
        shutil.copy2(cfg.input_wav, input_copy_path)
        results["input_copy"] = str(input_copy_path)

        # Step 2: Ingest WAV
        print("Step 2: Ingesting WAV file...")
        audio, sr = self.ingester.load(cfg.input_wav)

        if not self.ingester.validate(audio, sr):
            raise ValueError("Invalid audio data")

        results["sample_rate"] = sr
        results["duration_s"] = float(len(audio) / sr)
        print(f"  Loaded audio: duration={results['duration_s']:.2f}s, sr={sr}")

        # Step 3: Preprocess audio (optional)
        if cfg.use_pydub_preprocess:
            print("Step 3: Preprocessing audio...")
            audio, sr = self.preprocessor.preprocess(audio, sr, cfg)
            preprocessed_path = cfg.preprocess_dir / f"{cfg.song_id}.wav"
            self.ingester.save(audio, sr, preprocessed_path)
            results["preprocessed_path"] = str(preprocessed_path)
            print(f"  Saved preprocessed audio to: {preprocessed_path}")
        else:
            print("Step 3: Skipping preprocessing...")

        # Step 4: Analyze audio
        print("Step 4: Analyzing audio...")
        analysis = self.analyzer.analyze(audio, sr)
        results["analysis"] = {
            "tempo_bpm": analysis.get("tempo", analysis.get("librosa", {}).get("tempo_bpm", 120.0)),
            "num_beats": len(analysis.get("beat_times", [])),
            "duration_s": results["duration_s"],
            "essentia_available": analysis.get("essentia", {}).get("available", False),
        }
        tempo_bpm = results["analysis"]["tempo_bpm"]
        beat_times = analysis.get("beat_times", [])
        print(f"  Detected tempo: {tempo_bpm:.2f} BPM")
        print(f"  Found {len(beat_times)} beats")

        # Step 5: Separate stems (check if Demucs available)
        print("Step 5: Checking stem separation...")
        stem_paths: Dict[str, Path] = {}
        try:
            from didactic_engine.separation import StemSeparator
            separator = StemSeparator(model=cfg.demucs_model)
            stem_paths = separator.separate(cfg.input_wav, cfg.stems_dir)
            results["stems"] = list(stem_paths.keys())
            print(f"  Separated into {len(stem_paths)} stems: {list(stem_paths.keys())}")
        except RuntimeError as e:
            print(f"  Warning: Stem separation skipped: {e}")
            # Use original audio as single "stem"
            stem_paths = {"full_mix": cfg.input_wav}
            results["stems"] = ["full_mix"]
            results["stem_separation_error"] = str(e)

        # Step 6: Compute bar boundaries
        print("Step 6: Computing bar boundaries...")
        beats_per_bar = cfg.time_signature_num * (4.0 / cfg.time_signature_den)
        bar_boundaries = segment_beats_into_bars(
            beat_times, tempo_bpm,
            cfg.time_signature_num, cfg.time_signature_den,
            results["duration_s"]
        )
        results["num_bars"] = len(bar_boundaries)
        print(f"  Computed {len(bar_boundaries)} bars")

        # Step 7: Write per-bar chunks (optional)
        all_notes_dfs: List[pd.DataFrame] = []
        all_bar_features: List[Dict[str, Any]] = []

        for stem_name, stem_path in stem_paths.items():
            print(f"\nProcessing stem: {stem_name}")

            # Load stem audio
            stem_audio, stem_sr = self.ingester.load(stem_path)

            if cfg.write_bar_chunks:
                print(f"  Writing bar chunks...")
                stem_chunks_dir = cfg.chunks_dir / stem_name
                stem_chunks_dir.mkdir(parents=True, exist_ok=True)

                # Write chunks using pydub for millisecond precision
                from pydub import AudioSegment
                stem_segment = AudioSegment.from_file(str(stem_path))

                for bar_idx, start_s, end_s in bar_boundaries:
                    start_ms = int(start_s * 1000)
                    end_ms = int(end_s * 1000)
                    chunk = stem_segment[start_ms:end_ms]
                    chunk_path = stem_chunks_dir / f"bar_{bar_idx:04d}.wav"
                    chunk.export(str(chunk_path), format="wav")

                    # Extract features for this bar
                    features = self.feature_extractor.extract_bar_features_from_file(
                        chunk_path, cfg.analysis_sr
                    )
                    features.update({
                        "song_id": cfg.song_id,
                        "stem": stem_name,
                        "bar_index": bar_idx,
                        "start_s": start_s,
                        "end_s": end_s,
                        "duration_s": end_s - start_s,
                        "tempo_bpm": tempo_bpm,
                        "chunk_path": str(chunk_path),
                    })
                    all_bar_features.append(features)

            # Step 8: Transcribe to MIDI (check if Basic Pitch available)
            print(f"  Transcribing to MIDI...")
            try:
                from didactic_engine.transcription import BasicPitchTranscriber
                transcriber = BasicPitchTranscriber()
                midi_path = transcriber.transcribe(stem_path, cfg.midi_dir)
                results[f"midi_{stem_name}"] = str(midi_path)
                print(f"    Saved MIDI to: {midi_path}")

                # Step 9: Parse MIDI
                print(f"  Parsing MIDI...")
                midi_data = self.midi_parser.parse(midi_path)
                notes_df = midi_data["notes_df"]

                if not notes_df.empty:
                    # Step 10: Align notes to beats
                    print(f"  Aligning notes to beats...")
                    aligned_df = align_notes_to_beats(
                        notes_df, beat_times, tempo_bpm,
                        cfg.time_signature_num, cfg.time_signature_den
                    )
                    aligned_df["song_id"] = cfg.song_id
                    aligned_df["stem"] = stem_name
                    all_notes_dfs.append(aligned_df)
                    print(f"    Aligned {len(aligned_df)} notes")

            except RuntimeError as e:
                print(f"    Warning: MIDI transcription skipped: {e}")
                results[f"transcription_error_{stem_name}"] = str(e)

        # Concatenate all aligned notes
        if all_notes_dfs:
            all_notes = pd.concat(all_notes_dfs, ignore_index=True)
        else:
            all_notes = pd.DataFrame()

        # Step 11: Build datasets
        print("\nStep 11: Building datasets...")

        # Events dataset
        events_df = self.feature_extractor.extract_events(all_notes)

        # Beats dataset
        beats_rows = []
        for stem in results.get("stems", []):
            beats_df = self.feature_extractor.extract_beats(
                beat_times, tempo_bpm, stem, cfg.song_id
            )
            beats_rows.append(beats_df)
        beats_df = pd.concat(beats_rows, ignore_index=True) if beats_rows else pd.DataFrame()

        # Bars dataset
        bars_df = self.feature_extractor.extract_bars(all_notes, cfg.song_id)

        # Bar features dataset
        bar_features_df = pd.DataFrame(all_bar_features) if all_bar_features else pd.DataFrame()

        # Step 12: Write Parquet datasets
        print("Step 12: Writing Parquet datasets...")
        cfg.datasets_dir.mkdir(parents=True, exist_ok=True)

        if not events_df.empty:
            events_path = cfg.datasets_dir / "events.parquet"
            events_df.to_parquet(events_path, index=False)
            results["events_parquet"] = str(events_path)
            print(f"  Wrote events.parquet ({len(events_df)} rows)")

        if not beats_df.empty:
            beats_path = cfg.datasets_dir / "beats.parquet"
            beats_df.to_parquet(beats_path, index=False)
            results["beats_parquet"] = str(beats_path)
            print(f"  Wrote beats.parquet ({len(beats_df)} rows)")

        if not bars_df.empty:
            bars_path = cfg.datasets_dir / "bars.parquet"
            bars_df.to_parquet(bars_path, index=False)
            results["bars_parquet"] = str(bars_path)
            print(f"  Wrote bars.parquet ({len(bars_df)} rows)")

        if not bar_features_df.empty:
            bar_features_path = cfg.datasets_dir / "bar_features.parquet"
            bar_features_df.to_parquet(bar_features_path, index=False)
            results["bar_features_parquet"] = str(bar_features_path)
            print(f"  Wrote bar_features.parquet ({len(bar_features_df)} rows)")

        # Step 13: Export reports
        print("Step 13: Exporting reports...")
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)

        # Markdown report
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
            print(f"  Wrote {md_path}")

        # ABC notation per stem
        for stem_name in results.get("stems", []):
            midi_key = f"midi_{stem_name}"
            if midi_key in results:
                abc_path = cfg.reports_dir / f"{stem_name}.abc"
                export_abc(results[midi_key], str(abc_path))
                results[f"abc_{stem_name}"] = str(abc_path)
                print(f"  Wrote {abc_path}")

        # Step 14: Write JSON summary
        print("Step 14: Writing JSON summary...")
        cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cfg.analysis_dir / "combined.json"

        summary = {
            "song_id": cfg.song_id,
            "duration_s": results["duration_s"],
            "tempo_bpm": tempo_bpm,
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

        # List generated files
        for key, value in results.items():
            if key.endswith("_parquet") or key.endswith("_report") or key.startswith("abc_"):
                summary["files_generated"].append(value)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        results["summary_json"] = str(summary_path)
        print(f"  Wrote {summary_path}")

        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)

        return results

    @staticmethod
    def process_batch(
        input_files: List[Path],
        out_dir: Path,
        song_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multiple WAV files in batch mode.

        Args:
            input_files: List of input WAV file paths.
            out_dir: Base output directory for all results.
            song_ids: Optional list of song IDs (one per file). If not provided,
                     filenames (without extension) are used as song IDs.
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
        
        successful = []
        failed = []
        
        print(f"Batch processing {len(input_files)} files...")
        
        for idx, (input_file, song_id) in enumerate(zip(input_files, song_ids), 1):
            print(f"\n[{idx}/{len(input_files)}] Processing: {input_file.name} ({song_id})")
            
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
                pipeline = AudioPipeline(cfg)
                result = pipeline.run()
                
                successful.append((song_id, str(input_file), result))
                print(f"  ✓ Completed successfully")
            
            except FileNotFoundError as e:
                # Handle expected missing-file errors explicitly so tests and callers
                # can distinguish them from other processing failures.
                error_msg = f"File not found: {e}"
                failed.append((song_id, str(input_file), error_msg))
                print(f"  ✗ Failed (missing file): {error_msg}")
            
            except Exception as e:
                # Preserve batch robustness while keeping the exception type visible
                # in the recorded error message for easier debugging and validation.
                error_msg = f"{type(e).__name__}: {e}"
                failed.append((song_id, str(input_file), error_msg))
                print(f"  ✗ Failed: {error_msg}")
        
        # Return summary
        summary = {
            "successful": successful,
            "failed": failed,
            "total": len(input_files),
            "success_count": len(successful),
            "failure_count": len(failed),
        }
        
        print("\n" + "=" * 60)
        print(f"Batch processing complete: {summary['success_count']}/{summary['total']} successful")
        print("=" * 60)
        
        return summary


def run_all(cfg: "PipelineConfig") -> Dict[str, Any]:
    """
    Run the complete audio processing pipeline.

    Args:
        cfg: Pipeline configuration.

    Returns:
        Dictionary containing all processing results.
    """
    pipeline = AudioPipeline(cfg)
    return pipeline.run()
