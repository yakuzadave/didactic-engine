"""
Export MIDI analysis to Markdown reports.

This module provides functions for generating human-readable Markdown reports
from MIDI analysis data. Reports summarize MIDI events organized by bar and stem.

Key Functions:
    - :func:`pitch_to_name`: Convert MIDI pitch to note name (e.g., "C4")
    - :func:`export_midi_markdown`: Export bar-grouped MIDI events
    - :func:`export_full_report`: Export comprehensive pipeline report

Integration:
    Markdown export is typically the final step after analysis, transcription,
    and alignment. Reports provide human-readable summaries of the pipeline
    results.

Example:
    >>> export_midi_markdown(
    ...     aligned_notes={0: notes_bar_0, 1: notes_bar_1},
    ...     output_path="reports/midi_report.md",
    ...     song_id="my_song",
    ...     stem_name="vocals"
    ... )

See Also:
    - :mod:`didactic_engine.export_abc` for ABC notation export
    - :mod:`didactic_engine.align` for note alignment
"""

from typing import Dict, List, Any
import os


def pitch_to_name(pitch: int) -> str:
    """Convert MIDI pitch number to note name with octave.

    Converts a MIDI note number (0-127) to its musical notation
    (e.g., "C4" for middle C, "F#5" for F-sharp in octave 5).

    Args:
        pitch: MIDI pitch number (0-127). Middle C (C4) is 60.

    Returns:
        Note name with octave, using sharps for accidentals.
        Format: "{note}{octave}" e.g., "C4", "F#5", "Bb3".

    Example:
        >>> pitch_to_name(60)
        'C4'
        >>> pitch_to_name(69)
        'A4'
        >>> pitch_to_name(61)
        'C#4'

    Note:
        Uses sharps (not flats) for all accidentals. Octave numbering
        follows MIDI convention where octave -1 starts at pitch 0.
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
    """Export MIDI analysis as a Markdown report.

    Creates a formatted Markdown file with MIDI events organized by bar.
    Each bar section contains a table of notes with timing and pitch info.

    Args:
        aligned_notes: Dictionary mapping bar index to list of note dicts.
            Each note dict should have keys: start, end, pitch, velocity.
            Typically from :meth:`MIDIParser.align_to_grid`.
        output_path: Path for output Markdown file. Parent directories
            are created if needed.
        song_id: Song identifier for the report title.
        stem_name: Stem name for the report subtitle.

    Example:
        >>> aligned = {
        ...     0: [{"start": 0.0, "end": 0.5, "pitch": 60, "velocity": 100}],
        ...     1: [{"start": 2.0, "end": 2.5, "pitch": 62, "velocity": 90}],
        ... }
        >>> export_midi_markdown(aligned, "report.md", "song1", "vocals")

    Note:
        Output includes a table for each bar with columns:
        Time (s), Pitch, Note, Velocity, Duration (s)

    See Also:
        - :func:`export_full_report` for comprehensive pipeline reports
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
    """Export a comprehensive Markdown report from pipeline results.

    Creates a full report summarizing all pipeline outputs including
    audio info, analysis results, stems, MIDI info, and segmentation.

    Args:
        results: Pipeline results dictionary. Expected keys:
            - input_path: Path to input audio file
            - sample_rate: Audio sample rate
            - audio_shape: Shape of audio array
            - analysis: Dict with tempo, beat_times, onset_times
            - stem_names: List of separated stem names
            - midi_info: Dict with total_notes, duration
            - midi_path: Path to MIDI file
            - segmented_stems: Dict mapping stem to chunk count
            - features: Dict mapping stem to feature count
            - aligned_notes: Dict of bar index to notes
        output_path: Path for output Markdown file.

    Example:
        >>> results = {
        ...     "input_path": "song.wav",
        ...     "sample_rate": 44100,
        ...     "analysis": {"tempo": 120.0, "beat_times": [...]},
        ...     "stem_names": ["vocals", "drums", "bass", "other"],
        ...     ...
        ... }
        >>> export_full_report(results, "full_report.md")

    Note:
        Missing keys in results are handled gracefully with "N/A" values.

    See Also:
        - :func:`export_midi_markdown` for MIDI-only reports
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
