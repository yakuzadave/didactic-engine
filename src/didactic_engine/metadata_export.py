"""
Export metadata.jsonl for HuggingFace AudioFolder datasets.

This module provides functions to generate metadata.jsonl files that are
compatible with HuggingFace's AudioFolder dataset loader.

Key Functions:
    - :func:`create_metadata_entry`: Create a single metadata entry
    - :func:`export_metadata_jsonl`: Write metadata.jsonl file
    - :func:`build_abc_prompt`: Build ABC-based text prompt

Metadata Format:
    Each line is a JSON object with at least:
    - file_name: Relative path to audio file (e.g., "audio/clip_0001.wav")
    - text: Text prompt/caption for the audio
    
    Additional fields can include:
    - abc_raw: Raw ABC notation
    - source_track: Original track name
    - stem_used: Stem used for transcription (e.g., "vocals")
    - start_sec, end_sec: Temporal location in source
    - tempo_bpm: Detected tempo
    - sr: Sample rate

Example:
    >>> from pathlib import Path
    >>> entries = [
    ...     {
    ...         "file_name": "audio/clip_0001.wav",
    ...         "text": "abcstyle music matching this ABC notation: <abc> X:1\\nT:Melody\\nM:4/4\\n... </abc>",
    ...         "abc_raw": "X:1\\nT:Melody\\n...",
    ...         "source_track": "song.wav",
    ...         "stem_used": "vocals",
    ...     }
    ... ]
    >>> export_metadata_jsonl(entries, Path("dataset/metadata.jsonl"))

See Also:
    - :mod:`didactic_engine.export_abc` for ABC notation generation
    - HuggingFace AudioFolder: https://huggingface.co/docs/datasets/audio_dataset
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def build_abc_prompt(
    abc_text: str,
    trigger_token: str = "abcstyle",
    max_chars: int = 2500,
    include_tags: bool = True,
) -> str:
    """
    Build a text prompt from ABC notation for conditioning.
    
    Creates a consistent prompt format for ABC-based audio generation:
    "{trigger_token} music matching this ABC notation: <abc> {abc_text} </abc>"
    
    Args:
        abc_text: ABC notation text. Can be full ABC with headers or just notes.
        trigger_token: Token to trigger ABC-based generation. Default "abcstyle".
        max_chars: Maximum characters for ABC text. Longer text is truncated
            with " ..." suffix. Default 2500 (prevents token budget issues).
        include_tags: Whether to include <abc></abc> tags. Default True.
            
    Returns:
        Formatted prompt string ready for conditioning.
        
    Example:
        >>> abc = "X:1\\nT:Test\\nM:4/4\\nL:1/4\\nK:C\\nCDEF|"
        >>> prompt = build_abc_prompt(abc)
        >>> print(prompt)
        abcstyle music matching this ABC notation: <abc> X:1
        T:Test
        M:4/4
        L:1/4
        K:C
        CDEF| </abc>
        
    Note:
        - Keep trigger_token and format consistent across your dataset
        - Very long ABC (>2500 chars) may overwhelm text encoders
        - If transcription fails, use a fallback prompt (see Example below)
        
    Example (fallback for failed transcription):
        >>> prompt = build_abc_prompt("NA", include_tags=False)
        >>> # Better: build a custom fallback
        >>> prompt = "abcstyle music with unclear ABC transcription"
        
    See Also:
        - :func:`create_metadata_entry` for complete entry creation
    """
    # Trim whitespace
    abc_text = abc_text.strip()
    
    # Truncate if too long
    if len(abc_text) > max_chars:
        abc_text = abc_text[:max_chars].rstrip() + " ..."
    
    # Build prompt
    if include_tags:
        prompt = f"{trigger_token} music matching this ABC notation: <abc> {abc_text} </abc>"
    else:
        prompt = f"{trigger_token} music matching this ABC notation: {abc_text}"
    
    return prompt


def create_metadata_entry(
    file_name: str,
    abc_text: str,
    source_track: Optional[str] = None,
    stem_used: Optional[str] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    tempo_bpm: Optional[float] = None,
    sample_rate: Optional[int] = None,
    trigger_token: str = "abcstyle",
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a metadata entry for a single audio file.
    
    Builds a dictionary suitable for writing to metadata.jsonl with all
    required fields for HuggingFace AudioFolder.
    
    Args:
        file_name: Relative path to audio file (e.g., "audio/clip_0001.wav").
            Must be relative to the metadata.jsonl location.
        abc_text: ABC notation for the audio. Use "NA" if transcription failed.
        source_track: Optional original track filename.
        stem_used: Optional stem used for transcription (e.g., "vocals", "other").
        start_sec: Optional start time in source audio (seconds).
        end_sec: Optional end time in source audio (seconds).
        tempo_bpm: Optional detected tempo.
        sample_rate: Optional sample rate of the audio file.
        trigger_token: Token for conditioning. Default "abcstyle".
        extra_fields: Optional dict of additional fields to include.
        
    Returns:
        Dictionary with all metadata fields. Guaranteed to have:
        - file_name: Audio file path
        - text: Formatted prompt
        - abc_raw: Raw ABC notation (or "NA")
        
    Example:
        >>> entry = create_metadata_entry(
        ...     file_name="audio/track1_chunk0000.wav",
        ...     abc_text="X:1\\nM:4/4\\nK:C\\nCDEF|",
        ...     source_track="track1.wav",
        ...     stem_used="vocals",
        ...     start_sec=0.0,
        ...     end_sec=15.0,
        ...     tempo_bpm=120.0,
        ...     sample_rate=22050,
        ... )
        >>> print(entry["file_name"])
        audio/track1_chunk0000.wav
        >>> print(entry["text"][:50])
        abcstyle music matching this ABC notation: <abc>
        
    Note:
        - file_name should use forward slashes on all platforms
        - For failed transcription, pass abc_text="NA"
        
    See Also:
        - :func:`export_metadata_jsonl` for writing entries to file
        - :func:`build_abc_prompt` for prompt formatting details
    """
    # Build the text prompt
    if abc_text == "NA":
        # Fallback prompt for failed transcription
        text = f"{trigger_token} music with unclear ABC transcription"
    else:
        text = build_abc_prompt(abc_text, trigger_token=trigger_token)
    
    # Build entry
    entry: Dict[str, Any] = {
        "file_name": file_name,
        "text": text,
        "abc_raw": abc_text,
    }
    
    # Add optional fields
    if source_track is not None:
        entry["source_track"] = source_track
    if stem_used is not None:
        entry["stem_used"] = stem_used
    if start_sec is not None:
        entry["start_sec"] = start_sec
    if end_sec is not None:
        entry["end_sec"] = end_sec
    if tempo_bpm is not None:
        entry["tempo_bpm"] = tempo_bpm
    if sample_rate is not None:
        entry["sr"] = sample_rate
    
    # Add extra fields
    if extra_fields:
        entry.update(extra_fields)
    
    return entry


def export_metadata_jsonl(
    entries: List[Dict[str, Any]],
    output_path: Path,
) -> int:
    """
    Write metadata entries to a JSONL file.
    
    Creates a metadata.jsonl file compatible with HuggingFace AudioFolder
    dataset loader. Each line is a JSON object.
    
    Args:
        entries: List of metadata dictionaries. Each must have at least
            "file_name" and "text" fields.
        output_path: Path for the output metadata.jsonl file.
            Parent directory is created if it doesn't exist.
            
    Returns:
        Number of entries written.
        
    Example:
        >>> from pathlib import Path
        >>> entries = [
        ...     {
        ...         "file_name": "audio/clip_0001.wav",
        ...         "text": "abcstyle music...",
        ...         "abc_raw": "X:1\\n...",
        ...     },
        ...     {
        ...         "file_name": "audio/clip_0002.wav",
        ...         "text": "abcstyle music...",
        ...         "abc_raw": "X:1\\n...",
        ...     },
        ... ]
        >>> count = export_metadata_jsonl(entries, Path("data/metadata.jsonl"))
        >>> print(f"Wrote {count} entries")
        Wrote 2 entries
        
    Note:
        - Each entry is written as a single line (JSONL format)
        - File is overwritten if it exists
        - UTF-8 encoding is used for proper ABC character support
        
    See Also:
        - :func:`create_metadata_entry` for creating entries
        - HuggingFace AudioFolder: https://huggingface.co/docs/datasets/audio_dataset
    """
    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSONL (one JSON per line)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    
    return len(entries)


def load_metadata_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load metadata entries from a JSONL file.
    
    Reads a metadata.jsonl file and returns a list of metadata dictionaries.
    
    Args:
        input_path: Path to the metadata.jsonl file.
        
    Returns:
        List of metadata dictionaries.
        
    Example:
        >>> from pathlib import Path
        >>> entries = load_metadata_jsonl(Path("data/metadata.jsonl"))
        >>> print(f"Loaded {len(entries)} entries")
        >>> print(entries[0]["file_name"])
        audio/clip_0001.wav
        
    See Also:
        - :func:`export_metadata_jsonl` for writing entries
    """
    entries = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
