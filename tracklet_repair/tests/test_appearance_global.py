"""Tests for crop features and appearance-supported global association."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from tracklet_repair.src.features.appearance import (
    cosine_similarity,
    extract_crop,
    histogram_feature,
)
from tracklet_repair.src.evaluation.run_appearance_sweep import run_sweep
from tracklet_repair.src.postprocess.merge import merge_tracklets
from tracklet_repair.src.utils.io import TRACK_COLUMNS


def _tracks(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def _row(frame, track, x=2.0, class_id=0):
    return (frame, track, x, 2.0, 8.0, 8.0, 0.9, class_id)


def _save_frame(directory, frame_id, color):
    image = np.full((16, 16, 3), color, dtype=np.uint8)
    Image.fromarray(image).save(directory / f"{frame_id:06d}.jpg")


def _run(tracks, frames_dir, **options):
    parameters = {
        "merge_mode": "appearance_global", "frames_dir": frames_dir,
        "max_global_merge_gap": 3, "max_center_distance": 20.0,
        "max_size_ratio": 1.5, "max_speed": 20.0,
        "appearance_threshold": 0.7, "ambiguity_margin": 0.05,
        "return_diagnostics": True,
    }
    parameters.update(options)
    return merge_tracklets(tracks, **parameters)


def test_crop_extraction_clips_bbox_and_rejects_empty_crop() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    crop = extract_crop(frame, (-3, -2, 8, 7))

    assert crop is not None
    assert crop.shape == (5, 5, 3)
    assert extract_crop(frame, (20, 20, 3, 3)) is None


def test_histogram_similarity_distinguishes_color_crops() -> None:
    red = np.full((10, 10, 3), (240, 10, 10), dtype=np.uint8)
    similar_red = np.full((10, 10, 3), (245, 12, 12), dtype=np.uint8)
    blue = np.full((10, 10, 3), (10, 10, 240), dtype=np.uint8)

    red_feature = histogram_feature(red, "combined")
    similar_feature = histogram_feature(similar_red, "combined")
    blue_feature = histogram_feature(blue, "combined")

    assert cosine_similarity(red_feature, similar_feature) > 0.95
    assert cosine_similarity(red_feature, blue_feature) < 0.7


def test_appearance_global_accepts_similar_and_rejects_dissimilar(tmp_path) -> None:
    for frame_id in (1, 2, 4, 5):
        _save_frame(tmp_path, frame_id, (220, 20, 20))
    tracks = _tracks(_row(1, 10), _row(2, 10), _row(4, 11), _row(5, 11))

    merged, merge_map, diagnostics = _run(tracks, tmp_path)
    assert merge_map == {11: 10}
    assert set(merged["track_id"]) == {10}
    assert diagnostics["safety_violations"] == 0

    _save_frame(tmp_path, 4, (20, 20, 220))
    _save_frame(tmp_path, 5, (20, 20, 220))
    rejected, rejected_map, diagnostics = _run(tracks, tmp_path)
    assert rejected_map == {}
    assert set(rejected["track_id"]) == {10, 11}
    assert diagnostics["appearance_rejected_candidates"] == 1


def test_appearance_global_keeps_hard_constraints(tmp_path) -> None:
    for frame_id in (1, 2, 3, 4):
        _save_frame(tmp_path, frame_id, (200, 30, 30))
    overlap = _tracks(_row(1, 10), _row(3, 10), _row(3, 11), _row(4, 11))
    mismatch = _tracks(
        _row(1, 10, class_id=0), _row(2, 10, class_id=0),
        _row(3, 11, class_id=1), _row(4, 11, class_id=1),
    )

    assert _run(overlap, tmp_path)[1] == {}
    assert _run(mismatch, tmp_path)[1] == {}


def test_appearance_global_abstains_on_ambiguity(tmp_path) -> None:
    for frame_id in (1, 2, 4, 5):
        _save_frame(tmp_path, frame_id, (200, 30, 30))
    tracks = _tracks(
        _row(1, 10), _row(2, 10),
        _row(4, 11, 4.0), _row(5, 11, 5.0),
        _row(4, 12, 4.1), _row(5, 12, 5.1),
    )

    merged, merge_map, diagnostics = _run(tracks, tmp_path, ambiguity_margin=0.2)
    assert merge_map == {}
    assert set(merged["track_id"]) == {10, 11, 12}
    assert diagnostics["ambiguous_candidates_skipped"] >= 2


def test_global_assignment_prevents_duplicate_target_and_cycles(tmp_path) -> None:
    for frame_id in (1, 2, 4, 5, 7, 8):
        _save_frame(tmp_path, frame_id, (210, 20, 20))
    tracks = _tracks(
        _row(1, 10, 0.0), _row(2, 10, 1.0),
        _row(1, 20, 15.0), _row(2, 20, 14.0),
        _row(4, 30, 3.0), _row(5, 30, 4.0),
        _row(7, 40, 6.0), _row(8, 40, 7.0),
    )

    merged, merge_map, diagnostics = _run(
        tracks, tmp_path, ambiguity_margin=0.0, appearance_threshold=0.6
    )
    assert len([target for target in merge_map if target == 30]) <= 1
    assert not merged.duplicated(subset=["frame_id", "track_id"]).any()
    assert diagnostics["safety_violations"] == 0


def test_missing_appearance_rejects_without_explicit_fallback(tmp_path) -> None:
    tracks = _tracks(_row(1, 10), _row(2, 10), _row(4, 11), _row(5, 11))
    _, merge_map, diagnostics = _run(tracks, tmp_path)
    assert merge_map == {}
    assert diagnostics["appearance_missing_candidates"] == 1


def test_sweep_writes_safety_summaries(tmp_path) -> None:
    input_json = tmp_path / "tracks.json"
    frames_dir = tmp_path / "frames"
    output_dir = tmp_path / "sweep"
    frames_dir.mkdir()
    input_json.write_text(
        '{"1":[{"object sc id":10,"bbox_visible":[2,2,10,10]}],'
        '"2":[{"object sc id":10,"bbox_visible":[3,2,11,10]}],'
        '"4":[{"object sc id":11,"bbox_visible":[5,2,13,10]}],'
        '"5":[{"object sc id":11,"bbox_visible":[6,2,14,10]}]}',
        encoding="utf-8",
    )
    for frame_id in (1, 2, 4, 5):
        _save_frame(frames_dir, frame_id, (220, 20, 20))

    rows = run_sweep(input_json, frames_dir, output_dir)

    assert len(rows) == 81
    assert (output_dir / "sweep_results.csv").exists()
    assert (output_dir / "sweep_results.md").exists()
    assert (output_dir / "best_safe_config.json").exists()
    assert all(row["safety_violations"] == 0 for row in rows)
