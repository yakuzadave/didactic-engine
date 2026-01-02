"""
Dataset export to Parquet format.
"""

from pathlib import Path
import pandas as pd


def write_datasets(
    song_id: str,
    all_notes_df: pd.DataFrame,
    beat_data: list[dict],
    bar_features_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Write Parquet datasets.

    Args:
        song_id: Song identifier
        all_notes_df: DataFrame with all aligned notes
        beat_data: List of beat dictionaries
        bar_features_df: DataFrame with bar-level features
        output_dir: Output directory for datasets
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Events dataset
    if not all_notes_df.empty:
        events_df = all_notes_df.copy()
        events_df["song_id"] = song_id
        events_path = output_dir / "events.parquet"
        events_df.to_parquet(events_path, index=False)
        print(f"  Wrote events dataset: {events_path}")
    else:
        print("  Skipped events dataset (no notes)")

    # 2. Beats dataset
    if beat_data:
        beats_df = pd.DataFrame(beat_data)
        beats_df["song_id"] = song_id
        beats_path = output_dir / "beats.parquet"
        beats_df.to_parquet(beats_path, index=False)
        print(f"  Wrote beats dataset: {beats_path}")
    else:
        print("  Skipped beats dataset (no beats)")

    # 3. Bars dataset (aggregated from events)
    if not all_notes_df.empty:
        bars_list = []
        for (stem, bar_idx), group in all_notes_df.groupby(["stem", "bar_index"]):
            bars_list.append({
                "song_id": song_id,
                "stem": stem,
                "bar_index": bar_idx,
                "num_notes": len(group),
                "mean_velocity": group["velocity"].mean(),
                "pitch_min": group["pitch"].min(),
                "pitch_max": group["pitch"].max(),
                "start_s": group["start_s"].min(),
                "end_s": group["end_s"].max(),
            })

        if bars_list:
            bars_df = pd.DataFrame(bars_list)
            bars_path = output_dir / "bars.parquet"
            bars_df.to_parquet(bars_path, index=False)
            print(f"  Wrote bars dataset: {bars_path}")
    else:
        print("  Skipped bars dataset (no notes)")

    # 4. Bar features dataset
    if not bar_features_df.empty:
        bar_features_df["song_id"] = song_id
        features_path = output_dir / "bar_features.parquet"
        bar_features_df.to_parquet(features_path, index=False)
        print(f"  Wrote bar_features dataset: {features_path}")
    else:
        print("  Skipped bar_features dataset (no features)")
