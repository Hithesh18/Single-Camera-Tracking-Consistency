"""Compare baseline and repaired tracking outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tracklet_repair.src.analysis.analyze_tracklets import compute_tracklet_statistics
from tracklet_repair.src.utils.io import load_tracking_file


KEY_METRICS = [
    "total_detections",
    "num_tracklets",
    "mean_tracklet_length",
    "median_tracklet_length",
    "num_short_tracklets",
    "percent_short_tracklets",
    "num_tracklets_with_gaps",
    "total_internal_gaps",
]


def compare_tracking_outputs(
    baseline_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
    short_tracklet_threshold: int = 10,
) -> dict:
    """Compare tracklet statistics before and after post-processing."""
    baseline_stats = compute_tracklet_statistics(baseline_df, short_tracklet_threshold)
    repaired_stats = compute_tracklet_statistics(repaired_df, short_tracklet_threshold)

    differences = {}
    relative_changes = {}
    for metric in KEY_METRICS:
        baseline_value = baseline_stats[metric]
        repaired_value = repaired_stats[metric]
        differences[f"{metric}_diff"] = repaired_value - baseline_value
        relative_changes[f"{metric}_relative_change_percent"] = _relative_change(
            baseline_value,
            repaired_value,
        )

    return {
        "baseline": baseline_stats,
        "repaired": repaired_stats,
        "differences": differences,
        "relative_changes": relative_changes,
    }


def save_markdown_table(comparison: dict, output_path: Path) -> None:
    """Save a compact Markdown before/after table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Tracking Output Comparison",
        "",
        "Tracklet-level comparison between the baseline and repaired outputs.",
        "",
        "| metric | baseline | repaired | diff |",
        "| --- | ---: | ---: | ---: |",
    ]

    for metric in KEY_METRICS:
        baseline_value = comparison["baseline"][metric]
        repaired_value = comparison["repaired"][metric]
        diff_value = comparison["differences"][f"{metric}_diff"]
        lines.append(
            f"| {metric} | {_format_value(baseline_value)} | "
            f"{_format_value(repaired_value)} | {_format_value(diff_value)} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    baseline_path: Path,
    repaired_path: Path,
    output_json_path: Path,
    output_md_path: Path,
    short_threshold: int,
) -> dict:
    """Load two tracking outputs and write comparison summaries."""
    baseline = load_tracking_file(str(baseline_path))
    repaired = load_tracking_file(str(repaired_path))
    comparison = compare_tracking_outputs(baseline, repaired, short_threshold)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)
        file.write("\n")

    save_markdown_table(comparison, output_md_path)

    print("Tracking comparison summary")
    for metric in KEY_METRICS:
        baseline_value = comparison["baseline"][metric]
        repaired_value = comparison["repaired"][metric]
        diff_value = comparison["differences"][f"{metric}_diff"]
        print(
            f"- {metric}: baseline {_format_value(baseline_value)}, "
            f"repaired {_format_value(repaired_value)}, diff {_format_value(diff_value)}"
        )
    print(f"saved JSON to {output_json_path}")
    print(f"saved Markdown to {output_md_path}")

    return comparison


def _relative_change(baseline_value: float, repaired_value: float) -> float | None:
    """Return percent change, or None when the baseline is zero."""
    if baseline_value == 0:
        return None
    return float((repaired_value - baseline_value) / baseline_value * 100)


def _format_value(value: object) -> str:
    """Format numbers for readable console and Markdown output."""
    if isinstance(value, float):
        return f"{value:.2f}"
    if value is None:
        return "n/a"
    return str(value)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare tracking outputs.")
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to the original tracking file.",
    )
    parser.add_argument(
        "--repaired",
        type=Path,
        required=True,
        help="Path to the repaired tracking file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path where the JSON comparison should be saved.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        required=True,
        help="Path where the Markdown comparison should be saved.",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=10,
        help="Tracklets with this many detections or fewer count as short.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the evaluation command."""
    args = parse_args()
    run_evaluation(
        args.baseline,
        args.repaired,
        args.output_json,
        args.output_md,
        args.short_threshold,
    )


if __name__ == "__main__":
    main()
