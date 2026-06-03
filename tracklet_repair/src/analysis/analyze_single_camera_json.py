"""Analyze BoT-SORT single-camera JSON tracking output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracklet_repair.src.analysis.analyze_tracklets import compute_tracklet_statistics
from tracklet_repair.src.utils.json_adapter import load_single_camera_json_as_dataframe


def run_json_analysis(input_path: Path, output_path: Path, short_threshold: int) -> dict:
    """Load project JSON, compute tracklet statistics, and save them."""
    tracks = load_single_camera_json_as_dataframe(str(input_path))
    stats = compute_tracklet_statistics(tracks, short_threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
        file.write("\n")

    print(f"Loaded {len(tracks)} detections from {input_path}")
    print(f"Found {stats['num_tracklets']} tracklets")
    print(f"Wrote tracklet statistics to {output_path}")
    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze BoT-SORT single-camera JSON tracklets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a BoT-SORT single-camera JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where JSON statistics should be saved.",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=10,
        help="Tracklets with this many detections or fewer count as short.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the JSON analysis command."""
    args = parse_args()
    run_json_analysis(args.input, args.output, args.short_threshold)


if __name__ == "__main__":
    main()
