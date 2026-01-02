"""
Demucs stem separation wrapper.
"""

import shutil
import subprocess
from pathlib import Path


def run_demucs(input_wav: Path, out_dir: Path, model: str = "htdemucs") -> dict[str, Path]:
    """
    Run Demucs to separate audio into stems.

    Args:
        input_wav: Path to input WAV file
        out_dir: Output directory for stems
        model: Demucs model name

    Returns:
        Dictionary mapping stem names to WAV paths

    Raises:
        RuntimeError: If demucs is not found or fails
    """
    # Check if demucs is available
    if shutil.which("demucs") is None:
        raise RuntimeError(
            "demucs command not found. Please install Demucs:\n"
            "  pip install demucs\n"
            "and ensure it's on your PATH."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Run demucs
    cmd = ["demucs", "-n", model, "-o", str(out_dir), str(input_wav)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=600
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Demucs failed: {e.stderr}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("Demucs timed out after 10 minutes") from e

    # Discover all WAV files in output directory
    wav_files = list(out_dir.rglob("*.wav"))

    if not wav_files:
        raise RuntimeError(f"No WAV files found in {out_dir} after Demucs separation")

    # Map stem names to paths
    stems = {}
    known_stems = {"vocals", "accompaniment", "drums", "bass", "other", "guitar", "piano"}

    for wav_path in wav_files:
        stem_name = wav_path.stem.lower()
        
        # Check if it's a known stem name
        if stem_name in known_stems:
            stems[stem_name] = wav_path
        else:
            # Use the filename as the stem name
            stems[stem_name] = wav_path

    return stems
