"""Run the helper pipeline on one BoT-SORT single-camera JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracklet_repair.src.analysis.analyze_tracklets import compute_tracklet_statistics
from tracklet_repair.src.evaluation.evaluate_tracking import (
    compare_tracking_outputs,
    save_markdown_table,
)
from tracklet_repair.src.postprocess.merge import MERGE_MODES, merge_tracklets
from tracklet_repair.src.postprocess.repair import interpolate_track_gaps
from tracklet_repair.src.utils.io import save_tracking_file
from tracklet_repair.src.utils.json_adapter import load_single_camera_json_as_dataframe


def run_pipeline(
    input_json: Path,
    output_dir: Path,
    max_gap: int,
    enable_merge: bool,
    max_merge_gap: int,
    max_center_distance: float,
    max_size_ratio: float,
    short_threshold: int,
    merge_mode: str = "conservative",
    velocity_window: int = 3,
    ambiguity_margin: float = 0.10,
    max_speed: float = 80.0,
    **merge_options,
) -> dict:
    """Run conversion, repair, and comparison for one single-camera JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_tracks_path = output_dir / "baseline_tracks.txt"
    repaired_tracks_path = output_dir / "repaired_tracks.txt"
    baseline_stats_path = output_dir / "baseline_stats.json"
    repaired_stats_path = output_dir / "repaired_stats.json"
    comparison_json_path = output_dir / "comparison.json"
    comparison_md_path = output_dir / "comparison.md"

    baseline_tracks = load_single_camera_json_as_dataframe(str(input_json))
    save_tracking_file(baseline_tracks, str(baseline_tracks_path))

    baseline_stats = compute_tracklet_statistics(baseline_tracks, short_threshold)
    _save_json(baseline_stats, baseline_stats_path)

    processed_tracks = baseline_tracks
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
    interpolated_count = int(repaired_tracks["is_interpolated"].sum())
    save_tracking_file(repaired_tracks, str(repaired_tracks_path))

    repaired_stats = compute_tracklet_statistics(repaired_tracks, short_threshold)
    _save_json(repaired_stats, repaired_stats_path)

    comparison = compare_tracking_outputs(
        baseline_tracks,
        repaired_tracks,
        short_tracklet_threshold=short_threshold,
    )
    _save_json(comparison, comparison_json_path)
    save_markdown_table(comparison, comparison_md_path)

    summary = {
        "baseline_stats": baseline_stats,
        "repaired_stats": repaired_stats,
        "interpolated_detections": interpolated_count,
        "merged_tracklets": len(merge_map),
        "merge_map": merge_map,
        "merge_mode": merge_mode,
        "output_dir": str(output_dir),
    }
    _print_summary(summary)
    return summary


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def _print_summary(summary: dict) -> None:
    baseline = summary["baseline_stats"]
    repaired = summary["repaired_stats"]

    print("JSON tracklet helper summary")
    print(
        "total detections: "
        f"{baseline['total_detections']} -> {repaired['total_detections']}"
    )
    print(f"tracklets: {baseline['num_tracklets']} -> {repaired['num_tracklets']}")
    print(
        "mean tracklet length: "
        f"{baseline['mean_tracklet_length']:.2f} -> "
        f"{repaired['mean_tracklet_length']:.2f}"
    )
    print(f"interpolated detections: {summary['interpolated_detections']}")
    print(f"merged tracklets: {summary['merged_tracklets']}")
    if summary["merge_map"]:
        print(f"merge map: {summary['merge_map']}")
    print(f"saved outputs to {summary['output_dir']}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run tracklet repair helper pipeline on one JSON file."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to a BoT-SORT single-camera JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for converted, repaired, and comparison outputs.",
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
        "--short-threshold",
        type=int,
        default=10,
        help="Tracklets with this many detections or fewer count as short.",
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
    """Run the JSON tracklet helper pipeline."""
    args = parse_args()
    run_pipeline(
        input_json=args.input_json,
        output_dir=args.output_dir,
        max_gap=args.max_gap,
        enable_merge=args.enable_merge,
        max_merge_gap=args.max_merge_gap,
        max_center_distance=args.max_center_distance,
        max_size_ratio=args.max_size_ratio,
        short_threshold=args.short_threshold,
        merge_mode=args.merge_mode,
        velocity_window=args.velocity_window,
        ambiguity_margin=args.ambiguity_margin,
        max_speed=args.max_speed,
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
