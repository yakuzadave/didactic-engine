"""
Export MIDI analysis to Markdown reports.
"""

from pathlib import Path
import pandas as pd


def _pitch_to_name(pitch: int) -> str:
    """Convert MIDI pitch number to note name."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (pitch // 12) - 1
    note = notes[pitch % 12]
    return f"{note}{octave}"


def export_midi_markdown(
    aligned_notes_df: pd.DataFrame, output_path: Path, song_id: str
) -> None:
    """
    Export MIDI analysis as Markdown report.

    Args:
        aligned_notes_df: DataFrame with aligned notes
        output_path: Path to output Markdown file
        song_id: Song identifier
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"# MIDI Analysis Report: {song_id}\n\n")

        if aligned_notes_df.empty:
            f.write("No notes found.\n")
            return

        # Group by stem (if present) and bar
        if "stem" in aligned_notes_df.columns:
            stems = aligned_notes_df["stem"].unique()
        else:
            stems = ["all"]
            aligned_notes_df["stem"] = "all"

        for stem in sorted(stems):
            f.write(f"## Stem: {stem}\n\n")

            stem_notes = aligned_notes_df[aligned_notes_df["stem"] == stem]
            bars = sorted(stem_notes["bar_index"].unique())

            for bar_idx in bars:
                bar_notes = stem_notes[stem_notes["bar_index"] == bar_idx]

                f.write(f"### Bar {bar_idx}\n\n")
                f.write("| Beat | Pitch | Note | Velocity | Start (s) | Duration (s) |\n")
                f.write("|------|-------|------|----------|-----------|---------------|\n")

                for _, note in bar_notes.iterrows():
                    beat_pos = f"{note.get('beat_in_bar', 0):.2f}"
                    pitch_name = _pitch_to_name(note["pitch"])

                    f.write(
                        f"| {beat_pos} | {note['pitch']} | {pitch_name} | "
                        f"{note['velocity']} | {note['start_s']:.3f} | "
                        f"{note['dur_s']:.3f} |\n"
                    )

                f.write("\n")

            f.write("\n")
