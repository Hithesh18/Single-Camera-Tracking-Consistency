"""Regression tests for conservative tracklet merging."""

import pandas as pd

from tracklet_repair.src.postprocess.merge import (
    conservative_merge_tracklets,
    merge_tracklets,
    motion_aware_merge_tracklets,
)
from tracklet_repair.src.utils.io import TRACK_COLUMNS


def _tracks(*rows: tuple) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def _row(
    frame_id: int,
    track_id: int,
    x: float,
    *,
    y: float = 0.0,
    width: float = 10.0,
    height: float = 10.0,
    class_id: int = 0,
) -> tuple:
    return (frame_id, track_id, x, y, width, height, 0.9, class_id)


def test_merges_temporally_and_spatially_compatible_tracklets() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(2, 10, 2.0),
        _row(4, 11, 4.0),
        _row(5, 11, 6.0),
    )

    merged, merge_map = conservative_merge_tracklets(
        tracks,
        max_merge_gap=1,
        max_center_distance=10.0,
    )

    assert merge_map == {11: 10}
    assert merged["track_id"].tolist() == [10, 10, 10, 10]


def test_rejects_tracklets_that_overlap_in_time() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(3, 10, 2.0),
        _row(3, 11, 3.0),
        _row(4, 11, 4.0),
    )

    merged, merge_map = conservative_merge_tracklets(tracks)

    assert merge_map == {}
    assert set(merged["track_id"]) == {10, 11}


def test_rejects_class_mismatch() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0, class_id=0),
        _row(2, 10, 1.0, class_id=0),
        _row(3, 11, 2.0, class_id=1),
        _row(4, 11, 3.0, class_id=1),
    )

    merged, merge_map = conservative_merge_tracklets(
        tracks,
        max_center_distance=10.0,
        require_same_class=True,
    )

    assert merge_map == {}
    assert set(merged["track_id"]) == {10, 11}


def test_rejects_candidate_above_spatial_threshold() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(2, 10, 1.0),
        _row(3, 11, 100.0),
        _row(4, 11, 101.0),
    )

    merged, merge_map = conservative_merge_tracklets(
        tracks,
        max_center_distance=20.0,
    )

    assert merge_map == {}
    assert set(merged["track_id"]) == {10, 11}


def test_rejects_candidate_above_temporal_gap_threshold() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(2, 10, 1.0),
        _row(6, 11, 2.0),
        _row(7, 11, 3.0),
    )

    merged, merge_map = conservative_merge_tracklets(
        tracks,
        max_merge_gap=2,
        max_center_distance=10.0,
    )

    assert merge_map == {}
    assert set(merged["track_id"]) == {10, 11}


def test_rejects_candidate_above_size_ratio_threshold() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0, width=10.0, height=10.0),
        _row(2, 10, 1.0, width=10.0, height=10.0),
        _row(3, 11, 2.0, width=20.0, height=20.0),
        _row(4, 11, 3.0, width=20.0, height=20.0),
    )

    merged, merge_map = conservative_merge_tracklets(
        tracks,
        max_center_distance=10.0,
        max_size_ratio=1.5,
    )

    assert merge_map == {}
    assert set(merged["track_id"]) == {10, 11}


def test_accepts_values_exactly_on_merge_thresholds() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(2, 10, 0.0),
        _row(5, 11, 77.5, y=-2.5, width=15.0, height=15.0),
        _row(6, 11, 78.5, y=-2.5, width=15.0, height=15.0),
    )

    merged, merge_map = conservative_merge_tracklets(
        tracks,
        max_merge_gap=2,
        max_center_distance=80.0,
        max_size_ratio=1.5,
    )

    assert merge_map == {11: 10}
    assert set(merged["track_id"]) == {10}


def test_preserves_stable_tracks_and_sorts_output_deterministically() -> None:
    tracks = _tracks(
        _row(5, 11, 6.0),
        _row(2, 99, 500.0),
        _row(2, 10, 2.0),
        _row(4, 11, 4.0),
        _row(1, 99, 499.0),
        _row(1, 10, 0.0),
    )

    first, first_map = conservative_merge_tracklets(
        tracks,
        max_merge_gap=1,
        max_center_distance=10.0,
    )
    second, second_map = conservative_merge_tracklets(
        tracks,
        max_merge_gap=1,
        max_center_distance=10.0,
    )

    assert first_map == second_map == {11: 10}
    assert list(zip(first["frame_id"], first["track_id"])) == [
        (1, 10),
        (1, 99),
        (2, 10),
        (2, 99),
        (4, 10),
        (5, 10),
    ]
    assert len(first.loc[first["track_id"] == 99]) == 2
    pd.testing.assert_frame_equal(first, second)


def test_chained_merge_does_not_create_overlapping_final_tracklet() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(2, 10, 1.0),
        _row(4, 11, 2.0),
        _row(5, 11, 3.0),
        _row(5, 12, 3.5),
        _row(6, 12, 4.5),
    )

    merged, merge_map = conservative_merge_tracklets(
        tracks,
        max_merge_gap=2,
        max_center_distance=10.0,
    )

    assert merge_map == {11: 10}
    assert set(merged["track_id"]) == {10, 12}
    assert not merged.duplicated(subset=["frame_id", "track_id"]).any()


def test_conservative_merge_mode_matches_existing_function() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(2, 10, 2.0),
        _row(4, 11, 4.0),
        _row(5, 11, 6.0),
    )

    expected_tracks, expected_map = conservative_merge_tracklets(
        tracks,
        max_merge_gap=1,
        max_center_distance=10.0,
    )
    actual_tracks, actual_map = merge_tracklets(
        tracks,
        merge_mode="conservative",
        max_merge_gap=1,
        max_center_distance=10.0,
    )

    assert actual_map == expected_map
    pd.testing.assert_frame_equal(actual_tracks, expected_tracks)


def test_motion_aware_rejects_overlapping_frames() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(3, 10, 2.0),
        _row(3, 11, 3.0),
        _row(4, 11, 4.0),
    )

    merged, merge_map = motion_aware_merge_tracklets(tracks)

    assert merge_map == {}
    assert set(merged["track_id"]) == {10, 11}


def test_motion_aware_rejects_class_mismatch() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0, class_id=0),
        _row(2, 10, 2.0, class_id=0),
        _row(4, 11, 6.0, class_id=1),
        _row(5, 11, 8.0, class_id=1),
    )

    merged, merge_map = motion_aware_merge_tracklets(tracks)

    assert merge_map == {}
    assert set(merged["track_id"]) == {10, 11}


def test_motion_aware_skips_ambiguous_candidates() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(2, 10, 2.0),
        _row(4, 11, 6.0),
        _row(5, 11, 8.0),
        _row(4, 12, 6.2),
        _row(5, 12, 8.2),
    )

    _, conservative_map = conservative_merge_tracklets(
        tracks,
        max_merge_gap=2,
        max_center_distance=10.0,
    )
    merged, merge_map = motion_aware_merge_tracklets(
        tracks,
        max_merge_gap=2,
        max_center_distance=10.0,
        ambiguity_margin=0.05,
        max_speed=20.0,
    )

    assert len(conservative_map) == 1
    assert merge_map == {}
    assert set(merged["track_id"]) == {10, 11, 12}


def test_motion_aware_uses_constant_velocity_prediction() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(2, 10, 10.0),
        _row(4, 11, 30.0),
        _row(5, 11, 40.0),
    )

    _, conservative_map = conservative_merge_tracklets(
        tracks,
        max_merge_gap=2,
        max_center_distance=5.0,
    )
    merged, merge_map = motion_aware_merge_tracklets(
        tracks,
        max_merge_gap=2,
        max_center_distance=5.0,
        max_speed=20.0,
    )

    assert conservative_map == {}
    assert merge_map == {11: 10}
    assert set(merged["track_id"]) == {10}


def test_motion_aware_rejects_impossible_jump() -> None:
    tracks = _tracks(
        _row(1, 10, 0.0),
        _row(2, 10, 1.0),
        _row(4, 11, 200.0),
        _row(5, 11, 201.0),
    )

    merged, merge_map = motion_aware_merge_tracklets(
        tracks,
        max_merge_gap=2,
        max_center_distance=500.0,
        max_speed=20.0,
    )

    assert merge_map == {}
    assert set(merged["track_id"]) == {10, 11}
