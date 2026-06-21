"""Analyze basic tracklet statistics from single-camera tracking output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tracklet_repair.src.utils.io import load_tracking_file


def compute_tracklet_statistics(
    df: pd.DataFrame,
    short_tracklet_threshold: int = 10,
) -> dict:
    """Compute basic statistics for tracklets grouped by track ID."""
    total_detections = int(len(df))

    if df.empty:
        return {
            "total_detections": 0,
            "num_tracklets": 0,
            "min_tracklet_length": 0,
            "max_tracklet_length": 0,
            "mean_tracklet_length": 0.0,
            "median_tracklet_length": 0.0,
            "num_short_tracklets": 0,
            "percent_short_tracklets": 0.0,
            "first_frame": None,
            "last_frame": None,
            "num_frames": 0,
            "num_tracklets_with_gaps": 0,
            "total_internal_gaps": 0,
            "per_class_tracklet_counts": {},
        }

    tracklet_lengths = df.groupby("track_id").size()
    num_tracklets = int(tracklet_lengths.size)
    num_short_tracklets = int((tracklet_lengths <= short_tracklet_threshold).sum())

    gap_counts = df.groupby("track_id")["frame_id"].apply(_count_internal_gaps)
    num_tracklets_with_gaps = int((gap_counts > 0).sum())
    total_internal_gaps = int(gap_counts.sum())

    stats = {
        "total_detections": total_detections,
        "num_tracklets": num_tracklets,
        "min_tracklet_length": int(tracklet_lengths.min()),
        "max_tracklet_length": int(tracklet_lengths.max()),
        "mean_tracklet_length": float(tracklet_lengths.mean()),
        "median_tracklet_length": float(tracklet_lengths.median()),
        "num_short_tracklets": num_short_tracklets,
        "percent_short_tracklets": float(num_short_tracklets / num_tracklets * 100),
        "first_frame": int(df["frame_id"].min()),
        "last_frame": int(df["frame_id"].max()),
        "num_frames": int(df["frame_id"].nunique()),
        "num_tracklets_with_gaps": num_tracklets_with_gaps,
        "total_internal_gaps": total_internal_gaps,
    }

    if "class_id" in df.columns:
        per_class_counts = (
            df.groupby("class_id")["track_id"]
            .nunique()
            .sort_index()
            .astype(int)
            .to_dict()
        )
        stats["per_class_tracklet_counts"] = {
            str(class_id): count for class_id, count in per_class_counts.items()
        }

    return stats


def _count_internal_gaps(frame_ids: pd.Series) -> int:
    """Count non-consecutive frame jumps inside one tracklet."""
    unique_frames = sorted(frame_ids.unique())
    if len(unique_frames) < 2:
        return 0
    gaps = 0
    for previous_frame, current_frame in zip(unique_frames, unique_frames[1:]):
        if current_frame - previous_frame > 1:
            gaps += 1
    return gaps


def run_analysis(input_path: Path, output_path: Path, short_threshold: int) -> dict:
    """Load tracks, compute statistics, and write them to JSON."""
    tracks = load_tracking_file(str(input_path))
    stats = compute_tracklet_statistics(tracks, short_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
        file.write("\n")

    print(f"Wrote tracklet statistics to {output_path}")
    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze tracklet statistics.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a comma-separated tracking text file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the JSON statistics file should be saved.",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=10,
        help="Tracklets with this many detections or fewer count as short.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the analysis command."""
    args = parse_args()
    run_analysis(args.input, args.output, args.short_threshold)


if __name__ == "__main__":
    main()
