"""Convert BoT-SORT single-camera JSON output to MOT-style tracking text."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracklet_repair.src.utils.io import save_tracking_file
from tracklet_repair.src.utils.json_adapter import load_single_camera_json_as_dataframe


def convert_json_to_tracks(input_path: Path, output_path: Path) -> None:
    """Convert a single-camera JSON file and save standard tracking rows."""
    tracks = load_single_camera_json_as_dataframe(str(input_path))
    save_tracking_file(tracks, str(output_path))

    num_tracklets = int(tracks["track_id"].nunique())
    print(f"Loaded detections: {len(tracks)}")
    print(f"Number of tracklets: {num_tracklets}")
    print(f"Output path: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert BoT-SORT single-camera JSON to tracking text."
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
        help="Path where converted tracking text should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the converter command."""
    args = parse_args()
    convert_json_to_tracks(args.input, args.output)


if __name__ == "__main__":
    main()
