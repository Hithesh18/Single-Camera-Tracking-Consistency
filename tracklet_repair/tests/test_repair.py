"""Regression tests for short-gap interpolation."""

import pandas as pd

from tracklet_repair.src.postprocess.repair import interpolate_track_gaps
from tracklet_repair.src.utils.io import TRACK_COLUMNS


def _tracks(*rows: tuple) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def test_interpolates_short_gap_linearly_and_preserves_originals() -> None:
    tracks = _tracks(
        (1, 5, 0.0, 10.0, 20.0, 30.0, 0.6, 2),
        (4, 5, 30.0, 40.0, 26.0, 36.0, 1.0, 2),
    )
    original = tracks.copy(deep=True)

    repaired = interpolate_track_gaps(tracks, max_gap=2)

    pd.testing.assert_frame_equal(tracks, original)
    assert repaired["frame_id"].tolist() == [1, 2, 3, 4]
    assert repaired["is_interpolated"].tolist() == [False, True, True, False]

    frame_2 = repaired.loc[repaired["frame_id"] == 2].iloc[0]
    frame_3 = repaired.loc[repaired["frame_id"] == 3].iloc[0]
    assert frame_2[["x", "y", "width", "height"]].tolist() == [
        10.0,
        20.0,
        22.0,
        32.0,
    ]
    assert frame_3[["x", "y", "width", "height"]].tolist() == [
        20.0,
        30.0,
        24.0,
        34.0,
    ]
    assert frame_2["score"] == 0.8
    assert frame_3["score"] == 0.8
    assert frame_2["track_id"] == 5
    assert frame_2["class_id"] == 2


def test_does_not_interpolate_gap_above_maximum() -> None:
    tracks = _tracks(
        (1, 3, 0.0, 0.0, 10.0, 10.0, 0.9, 0),
        (5, 3, 4.0, 0.0, 10.0, 10.0, 0.9, 0),
    )

    repaired = interpolate_track_gaps(tracks, max_gap=2)

    assert repaired["frame_id"].tolist() == [1, 5]
    assert repaired["is_interpolated"].tolist() == [False, False]


def test_interpolation_output_order_is_deterministic() -> None:
    tracks = _tracks(
        (3, 2, 3.0, 0.0, 10.0, 10.0, 0.9, 0),
        (2, 1, 2.0, 0.0, 10.0, 10.0, 0.9, 0),
        (1, 2, 1.0, 0.0, 10.0, 10.0, 0.9, 0),
    )

    first = interpolate_track_gaps(tracks, max_gap=1)
    second = interpolate_track_gaps(tracks, max_gap=1)

    assert list(zip(first["frame_id"], first["track_id"])) == [
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 2),
    ]
    pd.testing.assert_frame_equal(first, second)
