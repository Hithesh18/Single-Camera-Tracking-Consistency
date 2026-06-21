"""Command-line entry point for tracklet post-processing."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracklet_repair.src.postprocess.merge import MERGE_MODES, merge_tracklets
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
    merge_mode: str = "conservative",
    velocity_window: int = 3,
    ambiguity_margin: float = 0.10,
    max_speed: float = 80.0,
    **merge_options,
) -> None:
    """Run the selected merging strategy and short-gap interpolation."""
    tracks = load_tracking_file(str(input_path))
    processed_tracks = tracks
    merge_map = {}

    if enable_merge:
        processed_tracks, merge_map = merge_tracklets(
            processed_tracks,
            merge_mode=merge_mode,
            max_merge_gap=max_merge_gap,
            max_center_distance=max_center_distance,
            max_size_ratio=max_size_ratio,
            velocity_window=velocity_window,
            ambiguity_margin=ambiguity_margin,
            max_speed=max_speed,
            **merge_options,
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
    parser.add_argument(
        "--merge-mode",
        choices=MERGE_MODES,
        default="conservative",
        help="Tracklet merge strategy. Existing conservative behavior is the default.",
    )
    parser.add_argument("--velocity-window", type=int, default=3)
    parser.add_argument("--ambiguity-margin", type=float, default=0.10)
    parser.add_argument("--max-speed", type=float, default=80.0)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--frame-pattern", default="{frame_id:06d}.jpg")
    parser.add_argument("--appearance-window", type=int, default=3)
    parser.add_argument("--appearance-threshold", type=float, default=0.65)
    parser.add_argument(
        "--appearance-backend", choices=("hsv", "rgb", "combined"), default="combined"
    )
    parser.add_argument("--allow-geometry-fallback", action="store_true")
    parser.add_argument("--max-global-merge-gap", type=int)
    parser.add_argument("--appearance-weight", type=float, default=0.40)
    parser.add_argument("--motion-weight", type=float, default=0.25)
    parser.add_argument("--geometry-weight", type=float, default=0.15)
    parser.add_argument("--temporal-weight", type=float, default=0.10)
    parser.add_argument("--size-weight", type=float, default=0.10)
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
        args.merge_mode,
        args.velocity_window,
        args.ambiguity_margin,
        args.max_speed,
        frames_dir=args.frames_dir,
        frame_pattern=args.frame_pattern,
        appearance_window=args.appearance_window,
        appearance_threshold=args.appearance_threshold,
        appearance_backend=args.appearance_backend,
        allow_geometry_fallback=args.allow_geometry_fallback,
        max_global_merge_gap=args.max_global_merge_gap,
        appearance_weight=args.appearance_weight,
        motion_weight=args.motion_weight,
        geometry_weight=args.geometry_weight,
        temporal_weight=args.temporal_weight,
        size_weight=args.size_weight,
    )


if __name__ == "__main__":
    main()
