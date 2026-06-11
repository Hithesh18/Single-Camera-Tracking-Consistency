"""Tests for the four-way tracklet repair ablation."""

from __future__ import annotations

import json

import pandas as pd

from tracklet_repair.src.evaluation.run_ablation import (
    VARIANT_NAMES,
    build_ablation_variants,
    run_ablation,
)
from tracklet_repair.src.utils.io import TRACK_COLUMNS


def _baseline_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, 10, 0.0, 0.0, 10.0, 10.0, 0.9, 0),
            (2, 10, 2.0, 0.0, 10.0, 10.0, 0.9, 0),
            (5, 11, 6.0, 0.0, 10.0, 10.0, 0.9, 0),
            (6, 11, 8.0, 0.0, 10.0, 10.0, 0.9, 0),
        ],
        columns=TRACK_COLUMNS,
    )


def _write_sample_json(path) -> None:
    data = {
        "1": [{"object sc id": 10, "bbox_visible": [0, 0, 10, 10]}],
        "2": [{"object sc id": 10, "bbox_visible": [2, 0, 12, 10]}],
        "5": [{"object sc id": 11, "bbox_visible": [6, 0, 16, 10]}],
        "6": [{"object sc id": 11, "bbox_visible": [8, 0, 18, 10]}],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_builds_all_variants_without_mutating_baseline() -> None:
    baseline = _baseline_tracks()
    original = baseline.copy(deep=True)

    variants = build_ablation_variants(
        baseline,
        max_gap=2,
        max_merge_gap=2,
        max_center_distance=20.0,
        max_size_ratio=1.5,
    )

    assert tuple(variants) == VARIANT_NAMES
    pd.testing.assert_frame_equal(baseline, original)


def test_full_repair_merges_before_interpolation() -> None:
    variants = build_ablation_variants(
        _baseline_tracks(),
        max_gap=2,
        max_merge_gap=2,
        max_center_distance=20.0,
        max_size_ratio=1.5,
    )

    interpolation_only = variants["interpolation_only"]
    merge_only = variants["merge_only"]
    full_repair = variants["full_repair"]

    assert len(interpolation_only["tracks"]) == 4
    assert interpolation_only["interpolated_detections"] == 0
    assert merge_only["merge_map"] == {11: 10}
    assert len(merge_only["tracks"]) == 4

    assert full_repair["merge_map"] == {11: 10}
    assert full_repair["interpolated_detections"] == 2
    assert full_repair["tracks"]["frame_id"].tolist() == [1, 2, 3, 4, 5, 6]
    assert set(full_repair["tracks"]["track_id"]) == {10}


def test_runner_writes_tracks_and_tracklet_level_summaries(tmp_path) -> None:
    input_json = tmp_path / "tracks.json"
    output_dir = tmp_path / "ablation"
    _write_sample_json(input_json)

    report = run_ablation(
        input_json=input_json,
        output_dir=output_dir,
        max_gap=2,
        max_merge_gap=2,
        max_center_distance=20.0,
        max_size_ratio=1.5,
        short_threshold=3,
    )

    for variant_name in VARIANT_NAMES:
        assert (output_dir / f"{variant_name}_tracks.txt").exists()
    assert (output_dir / "ablation.json").exists()
    assert (output_dir / "ablation.md").exists()

    assert tuple(report["variants"]) == VARIANT_NAMES
    assert report["configuration"]["full_repair_order"] == [
        "conservative_merge",
        "interpolate_gaps",
    ]

    json_text = (output_dir / "ablation.json").read_text(encoding="utf-8").lower()
    markdown_text = (output_dir / "ablation.md").read_text(encoding="utf-8").lower()
    for unsupported_metric in ("idf1", "hota", "mota", "identity switch"):
        assert unsupported_metric not in json_text
        assert unsupported_metric not in markdown_text
