"""
Main pipeline orchestrator.
"""

from pathlib import Path
import json
import pandas as pd

from music_etl.config import PipelineConfig
from music_etl.separate import run_demucs
from music_etl.audio_preprocess import preprocess_wav
from music_etl.wav_features import extract_wav_features
from music_etl.essentia_features import extract_essentia_features
from music_etl.transcribe import transcribe_to_midi
from music_etl.midi_features import parse_midi
from music_etl.align import align_notes_to_grid
from music_etl.bar_chunker import write_bar_chunks
from music_etl.export_md import export_midi_markdown
from music_etl.export_abc import export_abc
from music_etl.datasets import write_datasets
from music_etl.utils_flatten import flatten_dict


def run_all(cfg: PipelineConfig) -> None:
    """
    Run the complete music ETL pipeline.

    Args:
        cfg: Pipeline configuration
    """
    print(f"\n{'='*60}")
    print(f"Music ETL Pipeline - {cfg.song_id}")
    print(f"{'='*60}\n")

    # Create output directories
    for dir_path in [
        cfg.stems_dir,
        cfg.preprocess_dir,
        cfg.chunks_dir,
        cfg.midi_dir,
        cfg.analysis_dir,
        cfg.reports_dir,
        cfg.datasets_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Separate stems with Demucs
    print("Step 1: Separating stems with Demucs...")
    stems_map = run_demucs(
        cfg.input_wav,
        cfg.stems_dir,
        cfg.demucs_model,
        two_stems=cfg.demucs_two_stems,
        timeout=cfg.demucs_timeout,
    )
    print(f"  Found {len(stems_map)} stems: {list(stems_map.keys())}")

    # Step 2: Optional preprocessing
    if cfg.use_pydub_preprocess:
        print("\nStep 2: Preprocessing stems...")
        preprocessed_map = {}
        for stem_name, stem_path in stems_map.items():
            out_path = cfg.preprocess_dir / f"{stem_name}.wav"
            preprocess_wav(
                stem_path,
                out_path,
                target_sr=cfg.preprocess_target_sr,
                mono=cfg.preprocess_mono,
                do_normalize=cfg.preprocess_normalize,
                trim_silence=cfg.preprocess_trim_silence,
                silence_thresh_dbfs=cfg.silence_thresh_dbfs,
                keep_silence_ms=cfg.keep_silence_ms,
            )
            preprocessed_map[stem_name] = out_path
            print(f"  Preprocessed: {stem_name}")
        stems_map = preprocessed_map

    # Step 3: Analyze each stem
    print("\nStep 3: Analyzing stems...")
    stem_analyses = {}
    for stem_name, stem_path in stems_map.items():
        print(f"  Analyzing {stem_name}...")
        features = extract_wav_features(stem_path, cfg.analysis_sr, cfg.hop_length)

        # Optional Essentia enrichment
        if cfg.use_essentia_features:
            essentia_feats = extract_essentia_features(stem_path, cfg.analysis_sr)
            features["essentia"] = essentia_feats

        stem_analyses[stem_name] = features

    # Step 4: Bar chunking
    print("\nStep 4: Writing per-bar chunks...")
    all_chunk_metadata = {}
    bar_features_rows = []

    if cfg.write_bar_chunks:
        for stem_name, stem_path in stems_map.items():
            analysis = stem_analyses[stem_name]
            chunk_dir = cfg.chunks_dir / stem_name

            chunks = write_bar_chunks(
                stem_path,
                chunk_dir,
                analysis["beats"],
                analysis["tempo_bpm"],
                cfg.time_signature_num,
                cfg.time_signature_den,
            )
            all_chunk_metadata[stem_name] = chunks
            print(f"  Wrote {len(chunks)} chunks for {stem_name}")

            # Analyze each bar chunk
            for chunk_meta in chunks:
                chunk_path = Path(chunk_meta["chunk_path"])
                try:
                    chunk_features = extract_wav_features(
                        chunk_path, cfg.analysis_sr, cfg.hop_length
                    )

                    # Optional Essentia for chunks
                    if cfg.use_essentia_features:
                        chunk_essentia = extract_essentia_features(chunk_path, cfg.analysis_sr)
                        if chunk_essentia.get("available"):
                            # Flatten essentia features, keeping scalar values and select statistics
                            essentia_to_flatten = {}
                            for k, v in chunk_essentia.items():
                                if isinstance(v, (int, float, bool)):
                                    # Keep scalar values
                                    essentia_to_flatten[k] = v
                                elif isinstance(v, list) and len(v) > 0:
                                    # For lists, compute basic statistics
                                    import numpy as np
                                    try:
                                        arr = np.array(v)
                                        if arr.dtype.kind in 'biufc':  # numeric types
                                            essentia_to_flatten[f"{k}_mean"] = float(np.mean(arr))
                                            essentia_to_flatten[f"{k}_std"] = float(np.std(arr))
                                    except (ValueError, TypeError):
                                        # Skip non-numeric lists
                                        pass
                            
                            flat_essentia = flatten_dict(essentia_to_flatten, parent_key="essentia")
                            chunk_features.update(flat_essentia)

                    # Build row
                    row = {
                        "stem": stem_name,
                        "bar_index": chunk_meta["bar_index"],
                        "start_s": chunk_meta["start_s"],
                        "end_s": chunk_meta["end_s"],
                        "duration_s": chunk_meta["duration_s"],
                        "tempo_bpm": chunk_features["tempo_bpm"],
                        "chunk_path": str(chunk_path),
                    }

                    # Add chroma
                    for i, val in enumerate(chunk_features["chroma_mean"]):
                        row[f"chroma_mean_{i:02d}"] = val

                    # Add MFCC stats
                    for key, val in chunk_features.items():
                        if key.startswith("mfcc_"):
                            row[key] = val

                    # Add spectral features
                    for feat in ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]:
                        if feat in chunk_features and isinstance(chunk_features[feat], dict):
                            for stat, val in chunk_features[feat].items():
                                row[f"{feat}_{stat}"] = val

                    # Add ZCR
                    if "zcr" in chunk_features and isinstance(chunk_features["zcr"], dict):
                        for stat, val in chunk_features["zcr"].items():
                            row[f"zcr_{stat}"] = val

                    # Add flattened essentia features
                    for key, val in chunk_features.items():
                        if key.startswith("essentia") and not isinstance(val, (list, dict)):
                            row[key] = val

                    bar_features_rows.append(row)

                except Exception as e:
                    print(f"    Warning: Failed to analyze chunk {chunk_path}: {e}")

    bar_features_df = pd.DataFrame(bar_features_rows)

    # Step 5: Transcribe stems to MIDI
    print("\nStep 5: Transcribing stems to MIDI...")
    midi_files = {}
    for stem_name, stem_path in stems_map.items():
        try:
            midi_path = transcribe_to_midi(
                stem_path, cfg.midi_dir, timeout=cfg.transcribe_timeout
            )
            midi_files[stem_name] = midi_path
            print(f"  Transcribed: {stem_name}")
        except Exception as e:
            print(f"  Warning: Failed to transcribe {stem_name}: {e}")

    # Step 6: Parse MIDI files
    print("\nStep 6: Parsing MIDI files...")
    midi_data = {}
    for stem_name, midi_path in midi_files.items():
        try:
            parsed = parse_midi(midi_path)
            midi_data[stem_name] = parsed
            print(f"  Parsed {stem_name}: {len(parsed['notes_df'])} notes")
        except Exception as e:
            print(f"  Warning: Failed to parse {stem_name}: {e}")

    # Step 7: Align notes to beat/bar grid
    print("\nStep 7: Aligning notes to beat/bar grid...")
    all_notes_list = []
    for stem_name, data in midi_data.items():
        notes_df = data["notes_df"]
        if notes_df.empty:
            continue

        analysis = stem_analyses[stem_name]
        aligned = align_notes_to_grid(
            notes_df,
            analysis["beats"],
            analysis["tempo_bpm"],
            cfg.time_signature_num,
            cfg.time_signature_den,
            analysis["duration_s"],
        )
        aligned["stem"] = stem_name
        all_notes_list.append(aligned)
        print(f"  Aligned {stem_name}: {len(aligned)} notes")

    all_notes_df = pd.concat(all_notes_list, ignore_index=True) if all_notes_list else pd.DataFrame()

    # Step 8: Write combined JSON summary
    print("\nStep 8: Writing analysis summary...")
    summary = {
        "song_id": cfg.song_id,
        "input_wav": str(cfg.input_wav),
        "stems": {},
    }

    for stem_name, analysis in stem_analyses.items():
        summary["stems"][stem_name] = {
            "tempo_bpm": analysis["tempo_bpm"],
            "duration_s": analysis["duration_s"],
            "num_beats": len(analysis["beats"]),
            "num_notes": len(midi_data.get(stem_name, {}).get("notes_df", [])),
            "chroma_mean": analysis["chroma_mean"],
        }

    if not bar_features_df.empty:
        summary["num_bars"] = int(bar_features_df["bar_index"].max() + 1)

    summary_path = cfg.analysis_dir / "combined.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote summary: {summary_path}")

    # Step 9: Export Markdown report
    print("\nStep 9: Exporting Markdown report...")
    if not all_notes_df.empty:
        md_path = cfg.reports_dir / "midi_markdown.md"
        export_midi_markdown(all_notes_df, md_path, cfg.song_id)
        print(f"  Wrote report: {md_path}")

    # Step 10: Export ABC notation
    print("\nStep 10: Exporting ABC notation...")
    for stem_name, midi_path in midi_files.items():
        abc_path = cfg.reports_dir / f"{stem_name}.abc"
        export_abc(midi_path, abc_path)
        print(f"  Exported ABC: {stem_name}")

    # Step 11: Write Parquet datasets
    print("\nStep 11: Writing Parquet datasets...")
    beat_data = []
    for stem_name, analysis in stem_analyses.items():
        for idx, beat_time in enumerate(analysis["beats"]):
            beat_data.append({
                "stem": stem_name,
                "beat_index": idx,
                "time_s": beat_time,
                "tempo_bpm": analysis["tempo_bpm"],
            })

    write_datasets(
        cfg.song_id,
        all_notes_df,
        beat_data,
        bar_features_df,
        cfg.datasets_dir,
    )

    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}\n")
