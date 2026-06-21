"""Compare baseline, interpolation, merging, and full tracklet repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tracklet_repair.src.analysis.analyze_tracklets import compute_tracklet_statistics
from tracklet_repair.src.evaluation.evaluate_tracking import KEY_METRICS
from tracklet_repair.src.postprocess.merge import MERGE_MODES, merge_tracklets
from tracklet_repair.src.postprocess.repair import interpolate_track_gaps
from tracklet_repair.src.utils.io import save_tracking_file
from tracklet_repair.src.utils.json_adapter import load_single_camera_json_as_dataframe


VARIANT_NAMES = (
    "baseline",
    "interpolation_only",
    "merge_only",
    "full_repair",
)


def build_ablation_variants(
    baseline_tracks: pd.DataFrame,
    max_gap: int = 5,
    max_merge_gap: int = 10,
    max_center_distance: float = 80.0,
    max_size_ratio: float = 1.5,
    merge_mode: str = "conservative",
    velocity_window: int = 3,
    ambiguity_margin: float = 0.10,
    max_speed: float = 80.0,
    frames_dir: Path | None = None,
    frame_pattern: str = "{frame_id:06d}.jpg",
    appearance_window: int = 3,
    appearance_threshold: float = 0.65,
    appearance_backend: str = "combined",
    allow_geometry_fallback: bool = False,
    max_global_merge_gap: int | None = None,
    appearance_weight: float = 0.40,
    motion_weight: float = 0.25,
    geometry_weight: float = 0.15,
    temporal_weight: float = 0.10,
    size_weight: float = 0.10,
) -> dict[str, dict]:
    """Build all ablation variants from the same baseline tracks."""
    baseline = baseline_tracks.copy(deep=True)

    interpolation_only = interpolate_track_gaps(baseline, max_gap=max_gap)

    merge_options = {
        "merge_mode": merge_mode,
        "max_merge_gap": max_merge_gap,
        "max_center_distance": max_center_distance,
        "max_size_ratio": max_size_ratio,
        "velocity_window": velocity_window,
        "ambiguity_margin": ambiguity_margin,
        "max_speed": max_speed,
        "frames_dir": frames_dir,
        "frame_pattern": frame_pattern,
        "appearance_window": appearance_window,
        "appearance_threshold": appearance_threshold,
        "appearance_backend": appearance_backend,
        "allow_geometry_fallback": allow_geometry_fallback,
        "max_global_merge_gap": max_global_merge_gap,
        "appearance_weight": appearance_weight,
        "motion_weight": motion_weight,
        "geometry_weight": geometry_weight,
        "temporal_weight": temporal_weight,
        "size_weight": size_weight,
        "return_diagnostics": True,
    }
    merge_only, merge_only_map, merge_only_diagnostics = merge_tracklets(
        baseline,
        **merge_options,
    )

    full_merged, full_merge_map, full_diagnostics = merge_tracklets(
        baseline,
        **merge_options,
    )
    full_repair = interpolate_track_gaps(full_merged, max_gap=max_gap)

    return {
        "baseline": {
            "tracks": baseline,
            "interpolated_detections": 0,
            "merge_map": {},
            "merge_diagnostics": {},
        },
        "interpolation_only": {
            "tracks": interpolation_only,
            "interpolated_detections": _count_interpolated(interpolation_only),
            "merge_map": {},
            "merge_diagnostics": {},
        },
        "merge_only": {
            "tracks": merge_only,
            "interpolated_detections": 0,
            "merge_map": merge_only_map,
            "merge_diagnostics": merge_only_diagnostics,
        },
        "full_repair": {
            "tracks": full_repair,
            "interpolated_detections": _count_interpolated(full_repair),
            "merge_map": full_merge_map,
            "merge_diagnostics": full_diagnostics,
        },
    }


def run_ablation(
    input_json: Path,
    output_dir: Path,
    max_gap: int = 5,
    max_merge_gap: int = 10,
    max_center_distance: float = 80.0,
    max_size_ratio: float = 1.5,
    short_threshold: int = 10,
    merge_mode: str = "conservative",
    velocity_window: int = 3,
    ambiguity_margin: float = 0.10,
    max_speed: float = 80.0,
    frames_dir: Path | None = None,
    frame_pattern: str = "{frame_id:06d}.jpg",
    appearance_window: int = 3,
    appearance_threshold: float = 0.65,
    appearance_backend: str = "combined",
    allow_geometry_fallback: bool = False,
    max_global_merge_gap: int | None = None,
    appearance_weight: float = 0.40,
    motion_weight: float = 0.25,
    geometry_weight: float = 0.15,
    temporal_weight: float = 0.10,
    size_weight: float = 0.10,
) -> dict:
    """Run the four-way ablation on one single-camera JSON file."""
    baseline_tracks = load_single_camera_json_as_dataframe(str(input_json))
    variants = build_ablation_variants(
        baseline_tracks,
        max_gap=max_gap,
        max_merge_gap=max_merge_gap,
        max_center_distance=max_center_distance,
        max_size_ratio=max_size_ratio,
        merge_mode=merge_mode,
        velocity_window=velocity_window,
        ambiguity_margin=ambiguity_margin,
        max_speed=max_speed,
        frames_dir=frames_dir,
        frame_pattern=frame_pattern,
        appearance_window=appearance_window,
        appearance_threshold=appearance_threshold,
        appearance_backend=appearance_backend,
        allow_geometry_fallback=allow_geometry_fallback,
        max_global_merge_gap=max_global_merge_gap,
        appearance_weight=appearance_weight,
        motion_weight=motion_weight,
        geometry_weight=geometry_weight,
        temporal_weight=temporal_weight,
        size_weight=size_weight,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "configuration": {
            "input_json": str(input_json),
            "max_gap": max_gap,
            "max_merge_gap": max_merge_gap,
            "max_center_distance": max_center_distance,
            "max_size_ratio": max_size_ratio,
            "merge_mode": merge_mode,
            "velocity_window": velocity_window,
            "ambiguity_margin": ambiguity_margin,
            "max_speed": max_speed,
            "frames_dir": str(frames_dir) if frames_dir else None,
            "frame_pattern": frame_pattern,
            "appearance_window": appearance_window,
            "appearance_threshold": appearance_threshold,
            "appearance_backend": appearance_backend,
            "allow_geometry_fallback": allow_geometry_fallback,
            "max_global_merge_gap": max_global_merge_gap,
            "appearance_weight": appearance_weight,
            "motion_weight": motion_weight,
            "geometry_weight": geometry_weight,
            "temporal_weight": temporal_weight,
            "size_weight": size_weight,
            "short_tracklet_threshold": short_threshold,
            "full_repair_order": [
                f"{merge_mode}_merge",
                "interpolate_gaps",
            ],
        },
        "variants": {},
    }

    for variant_name in VARIANT_NAMES:
        variant = variants[variant_name]
        tracks = variant["tracks"]
        save_tracking_file(tracks, str(output_dir / f"{variant_name}_tracks.txt"))

        merge_map = variant["merge_map"]
        report["variants"][variant_name] = {
            "statistics": compute_tracklet_statistics(tracks, short_threshold),
            "interpolated_detections": variant["interpolated_detections"],
            "merged_tracklets": len(merge_map),
            "merge_map": merge_map,
            "merge_diagnostics": variant["merge_diagnostics"],
        }

    _save_json(report, output_dir / "ablation.json")
    save_ablation_markdown(report, output_dir / "ablation.md")
    _print_summary(report, output_dir)
    return report


def save_ablation_markdown(report: dict, output_path: Path) -> None:
    """Save a compact tracklet-level ablation table."""
    lines = [
        "# Tracklet Repair Ablation",
        "",
        "Tracklet-level comparison of the existing repair components.",
        "",
        "| metric | baseline | interpolation_only | merge_only | full_repair |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for metric in KEY_METRICS:
        values = [
            _format_value(report["variants"][name]["statistics"][metric])
            for name in VARIANT_NAMES
        ]
        lines.append(f"| {metric} | {' | '.join(values)} |")

    lines.extend(
        [
            f"| interpolated_detections | "
            f"{' | '.join(str(report['variants'][name]['interpolated_detections']) for name in VARIANT_NAMES)} |",
            f"| merged_tracklets | "
            f"{' | '.join(str(report['variants'][name]['merged_tracklets']) for name in VARIANT_NAMES)} |",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_interpolated(tracks: pd.DataFrame) -> int:
    if "is_interpolated" not in tracks.columns:
        return 0
    return int(tracks["is_interpolated"].sum())


def _save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_summary(report: dict, output_dir: Path) -> None:
    print("Tracklet repair ablation summary")
    for variant_name in VARIANT_NAMES:
        variant = report["variants"][variant_name]
        stats = variant["statistics"]
        print(
            f"- {variant_name}: detections {stats['total_detections']}, "
            f"tracklets {stats['num_tracklets']}, "
            f"internal gaps {stats['total_internal_gaps']}, "
            f"interpolated {variant['interpolated_detections']}, "
            f"merged {variant['merged_tracklets']}"
        )
    print(f"saved ablation outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a four-way tracklet repair ablation on one JSON file."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--max-merge-gap", type=int, default=10)
    parser.add_argument("--max-center-distance", type=float, default=80.0)
    parser.add_argument("--max-size-ratio", type=float, default=1.5)
    parser.add_argument("--short-threshold", type=int, default=10)
    parser.add_argument("--merge-mode", choices=MERGE_MODES, default="conservative")
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
    """Run the ablation command."""
    args = parse_args()
    run_ablation(
        input_json=args.input_json,
        output_dir=args.output_dir,
        max_gap=args.max_gap,
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
