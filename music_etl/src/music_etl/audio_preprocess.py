"""
Audio preprocessing with pydub.
"""

from pathlib import Path
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent


def preprocess_wav(
    in_path: Path,
    out_path: Path,
    target_sr: int = 44100,
    mono: bool = True,
    do_normalize: bool = True,
    trim_silence: bool = True,
    silence_thresh_dbfs: int = -50,
    keep_silence_ms: int = 100,
) -> Path:
    """
    Preprocess audio with pydub.

    Args:
        in_path: Input WAV path
        out_path: Output WAV path
        target_sr: Target sample rate
        mono: Convert to mono
        do_normalize: Apply normalization
        trim_silence: Trim silence from start/end
        silence_thresh_dbfs: Silence threshold in dBFS
        keep_silence_ms: Milliseconds of silence to keep at start/end

    Returns:
        Path to output WAV file
    """
    # Load audio
    audio = AudioSegment.from_file(str(in_path))

    # Convert to mono if requested
    if mono and audio.channels > 1:
        audio = audio.set_channels(1)

    # Resample if needed
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)

    # Trim silence
    if trim_silence:
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=100,
            silence_thresh=silence_thresh_dbfs,
            seek_step=10,
        )

        if nonsilent_ranges:
            # Get first and last nonsilent ranges
            start_trim = max(0, nonsilent_ranges[0][0] - keep_silence_ms)
            end_trim = min(len(audio), nonsilent_ranges[-1][1] + keep_silence_ms)
            audio = audio[start_trim:end_trim]

    # Normalize
    if do_normalize:
        audio = normalize(audio)

    # Export
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(str(out_path), format="wav")

    return out_path
