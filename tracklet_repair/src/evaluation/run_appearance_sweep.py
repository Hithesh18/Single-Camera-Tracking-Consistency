"""Sensitivity sweep for appearance-supported global tracklet association."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

from tracklet_repair.src.analysis.analyze_tracklets import compute_tracklet_statistics
from tracklet_repair.src.evaluation.run_ablation import build_ablation_variants
from tracklet_repair.src.utils.json_adapter import load_single_camera_json_as_dataframe


def run_sweep(
    input_json: Path,
    frames_dir: Path,
    output_dir: Path,
    *,
    frame_pattern: str = "{frame_id:06d}.jpg",
    max_gap: int = 5,
    max_center_distance: float = 80.0,
    max_size_ratio: float = 1.5,
    max_speed: float = 80.0,
    short_threshold: int = 10,
) -> list[dict]:
    """Run the fixed small sweep and save proxy-level safety summaries."""
    baseline = load_single_camera_json_as_dataframe(str(input_json))
    baseline_stats = compute_tracklet_statistics(baseline, short_threshold)
    rows = []
    for threshold, weight, margin, merge_gap in product(
        (0.55, 0.65, 0.75), (0.25, 0.40, 0.55),
        (0.05, 0.10, 0.15), (5, 8, 10),
    ):
        variants = build_ablation_variants(
            baseline,
            max_gap=max_gap,
            max_merge_gap=merge_gap,
            max_center_distance=max_center_distance,
            max_size_ratio=max_size_ratio,
            merge_mode="appearance_global",
            frames_dir=frames_dir,
            frame_pattern=frame_pattern,
            appearance_threshold=threshold,
            appearance_weight=weight,
            ambiguity_margin=margin,
            max_global_merge_gap=merge_gap,
            max_speed=max_speed,
        )
        full = variants["full_repair"]
        stats = compute_tracklet_statistics(full["tracks"], short_threshold)
        diagnostics = full["merge_diagnostics"]
        similarities = diagnostics["accepted_appearance_similarities"]
        row = {
            "appearance_threshold": threshold,
            "appearance_weight": weight,
            "ambiguity_margin": margin,
            "max_global_merge_gap": merge_gap,
            "raw_ids": baseline_stats["num_tracklets"],
            "repaired_ids": stats["num_tracklets"],
            "id_reduction": baseline_stats["num_tracklets"] - stats["num_tracklets"],
            "raw_gaps": baseline_stats["total_internal_gaps"],
            "repaired_gaps": stats["total_internal_gaps"],
            "gap_reduction": baseline_stats["total_internal_gaps"] - stats["total_internal_gaps"],
            "merges": len(full["merge_map"]),
            "appearance_rejected_candidates": diagnostics["appearance_rejected_candidates"],
            "ambiguous_candidates_skipped": diagnostics["ambiguous_candidates_skipped"],
            "safety_violations": diagnostics["safety_violations"],
            "mean_accepted_similarity": sum(similarities) / len(similarities) if similarities else None,
            "min_accepted_similarity": min(similarities) if similarities else None,
            "borderline_merges": diagnostics["borderline_merges"],
        }
        row["safe_score"] = _safe_score(row)
        rows.append(row)

    rows.sort(key=lambda row: (-row["safe_score"], row["appearance_threshold"],
                               row["appearance_weight"], row["ambiguity_margin"],
                               row["max_global_merge_gap"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(rows, output_dir / "sweep_results.csv")
    _save_markdown(rows, output_dir / "sweep_results.md")
    (output_dir / "best_safe_config.json").write_text(
        json.dumps(rows[0], indent=2) + "\n", encoding="utf-8"
    )
    return rows


def _safe_score(row: dict) -> float:
    """Reward continuity while penalizing uncertain or unsafe associations."""
    moderate_id_reduction = min(row["id_reduction"], max(row["raw_ids"] * 0.25, 1.0))
    minimum = row["min_accepted_similarity"]
    low_similarity_penalty = 0.0 if minimum is None else (1.0 - minimum) * row["merges"]
    return float(
        row["gap_reduction"] + 0.5 * moderate_id_reduction
        - 100.0 * row["safety_violations"]
        - row["borderline_merges"] - low_similarity_penalty
        - 0.1 * row["ambiguous_candidates_skipped"]
    )


def _save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_markdown(rows: list[dict], path: Path) -> None:
    columns = list(rows[0])
    lines = ["# Appearance-global sensitivity sweep", "",
             "Proxy-level tracklet continuity results; no identity ground truth is used.", "",
             f"| {' | '.join(columns)} |", f"| {' | '.join('---' for _ in columns)} |"]
    for row in rows:
        values = ["" if row[column] is None else str(row[column]) for column in columns]
        lines.append(f"| {' | '.join(values)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run appearance-global threshold sensitivity sweep.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-pattern", default="{frame_id:06d}.jpg")
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--max-center-distance", type=float, default=80.0)
    parser.add_argument("--max-size-ratio", type=float, default=1.5)
    parser.add_argument("--max-speed", type=float, default=80.0)
    parser.add_argument("--short-threshold", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sweep(**vars(args))


if __name__ == "__main__":
    main()
