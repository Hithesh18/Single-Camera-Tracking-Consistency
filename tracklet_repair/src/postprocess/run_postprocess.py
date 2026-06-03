"""Command-line entry point for tracklet post-processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracklet_repair.src.postprocess.merge import conservative_merge_tracklets
from tracklet_repair.src.postprocess.repair import interpolate_track_gaps
from tracklet_repair.src.utils.io import load_tracking_file, save_tracking_file


def run_postprocess(
    input_path: Path,
    output_path: Path,
    max_gap: int,
    enable_merge: bool,
    max_merge_gap: int,
    max_center_distance: float,
    max_size_ratio: float,
) -> None:
    """Run conservative merging and short-gap interpolation."""
    tracks = load_tracking_file(str(input_path))
    processed_tracks = tracks
    merge_map = {}

    if enable_merge:
        processed_tracks, merge_map = conservative_merge_tracklets(
            processed_tracks,
            max_merge_gap=max_merge_gap,
            max_center_distance=max_center_distance,
            max_size_ratio=max_size_ratio,
        )

    repaired_tracks = interpolate_track_gaps(processed_tracks, max_gap=max_gap)

    save_tracking_file(repaired_tracks, str(output_path))

    num_interpolated = int(repaired_tracks["is_interpolated"].sum())
    print(f"original detections: {len(tracks)}")
    print(f"repaired detections: {len(repaired_tracks)}")
    print(f"interpolated detections: {num_interpolated}")
    print(f"merged tracklets: {len(merge_map)}")
    if merge_map:
        print(f"merge map: {merge_map}")
    print(f"saved repaired tracks to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run tracklet post-processing.")
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
        help="Path where the repaired tracking file should be saved.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=5,
        help="Maximum number of missing frames to interpolate.",
    )
    parser.add_argument(
        "--enable-merge",
        action="store_true",
        help="Enable conservative tracklet merging before interpolation.",
    )
    parser.add_argument(
        "--max-merge-gap",
        type=int,
        default=10,
        help="Maximum missing-frame gap for merging two tracklets.",
    )
    parser.add_argument(
        "--max-center-distance",
        type=float,
        default=80.0,
        help="Maximum center distance between merge candidate boxes.",
    )
    parser.add_argument(
        "--max-size-ratio",
        type=float,
        default=1.5,
        help="Maximum width or height ratio for merge candidate boxes.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the post-processing command."""
    args = parse_args()
    run_postprocess(
        args.input,
        args.output,
        args.max_gap,
        args.enable_merge,
        args.max_merge_gap,
        args.max_center_distance,
        args.max_size_ratio,
    )


if __name__ == "__main__":
    main()
