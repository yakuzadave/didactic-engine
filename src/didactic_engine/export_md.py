"""
Export MIDI analysis to Markdown reports.

Generates human-readable Markdown reports summarizing MIDI events by bar and stem.
"""

from typing import Dict, List, Any
import os


def pitch_to_name(pitch: int) -> str:
    """
    Convert MIDI pitch number to note name.

    Args:
        pitch: MIDI pitch number (0-127).

    Returns:
        Note name with octave (e.g., "C4", "F#5").
    """
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (pitch // 12) - 1
    note = notes[pitch % 12]
    return f"{note}{octave}"


def export_midi_markdown(
    aligned_notes: Dict[int, List[Dict[str, Any]]],
    output_path: str,
    song_id: str = "audio",
    stem_name: str = "all",
) -> None:
    """
    Export MIDI analysis as Markdown report.

    Args:
        aligned_notes: Dictionary mapping bar index to list of notes.
        output_path: Path to output Markdown file.
        song_id: Song identifier for report title.
        stem_name: Name of the stem being exported.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"# MIDI Analysis Report: {song_id}\n\n")
        f.write(f"## Stem: {stem_name}\n\n")

        if not aligned_notes:
            f.write("No notes found.\n")
            return

        # Sort bars by index
        sorted_bars = sorted(aligned_notes.keys())

        for bar_idx in sorted_bars:
            bar_notes = aligned_notes[bar_idx]

            f.write(f"### Bar {bar_idx}\n\n")

            if not bar_notes:
                f.write("*No notes in this bar*\n\n")
                continue

            f.write("| Time (s) | Pitch | Note | Velocity | Duration (s) |\n")
            f.write("|----------|-------|------|----------|---------------|\n")

            for note in bar_notes:
                start_time = note.get("start", note.get("original_start", 0))
                end_time = note.get("end", start_time)
                pitch = note.get("pitch", 0)
                velocity = note.get("velocity", 0)
                duration = end_time - start_time

                pitch_name = pitch_to_name(pitch)

                f.write(
                    f"| {start_time:.3f} | {pitch} | {pitch_name} | "
                    f"{velocity} | {duration:.3f} |\n"
                )

            f.write("\n")


def export_full_report(
    results: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Export a comprehensive Markdown report from pipeline results.

    Args:
        results: Full pipeline results dictionary.
        output_path: Path to output Markdown file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Audio Processing Pipeline Report\n\n")
        f.write(f"**Input:** {results.get('input_path', 'Unknown')}\n\n")

        # Audio info
        f.write("## Audio Information\n\n")
        f.write(f"- **Sample Rate:** {results.get('sample_rate', 'N/A')} Hz\n")
        f.write(f"- **Shape:** {results.get('audio_shape', 'N/A')}\n\n")

        # Analysis results
        if "analysis" in results:
            analysis = results["analysis"]
            f.write("## Analysis Results\n\n")
            f.write(f"- **Tempo:** {analysis.get('tempo', 'N/A'):.2f} BPM\n")
            f.write(f"- **Beats detected:** {len(analysis.get('beat_times', []))}\n")
            f.write(f"- **Onsets detected:** {len(analysis.get('onset_times', []))}\n\n")

        # Stems
        if "stem_names" in results:
            f.write("## Separated Stems\n\n")
            for stem in results["stem_names"]:
                f.write(f"- {stem}\n")
            f.write("\n")

        # MIDI info
        if "midi_info" in results:
            midi_info = results["midi_info"]
            f.write("## MIDI Transcription\n\n")
            f.write(f"- **Total notes:** {midi_info.get('total_notes', 'N/A')}\n")
            f.write(f"- **Duration:** {midi_info.get('duration', 'N/A'):.2f} s\n")
            f.write(f"- **MIDI file:** {results.get('midi_path', 'N/A')}\n\n")

        # Segmentation
        if "segmented_stems" in results:
            f.write("## Segmentation\n\n")
            for stem, count in results["segmented_stems"].items():
                f.write(f"- **{stem}:** {count} bar chunks\n")
            f.write("\n")

        # Features
        if "features" in results:
            f.write("## Feature Extraction\n\n")
            for stem, count in results["features"].items():
                f.write(f"- **{stem}:** {count} feature vectors\n")
            f.write("\n")

        # Aligned notes summary
        if "aligned_notes" in results:
            f.write("## Aligned Notes by Bar\n\n")
            aligned = results["aligned_notes"]
            for bar_idx in sorted(aligned.keys()):
                notes = aligned[bar_idx]
                f.write(f"- **Bar {bar_idx}:** {len(notes)} notes\n")
            f.write("\n")

        f.write("---\n")
        f.write(f"*Report generated by didactic-engine*\n")
