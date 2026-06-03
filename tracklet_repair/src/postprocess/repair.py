"""Repair small internal gaps in tracklets."""

from __future__ import annotations

import pandas as pd

from tracklet_repair.src.utils.io import TRACK_COLUMNS, validate_tracking_dataframe


BOX_COLUMNS = ["x", "y", "width", "height"]


def interpolate_track_gaps(df: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Fill short missing-frame gaps with linear box interpolation."""
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative.")

    repaired = df.copy()
    validate_tracking_dataframe(repaired)
    repaired["is_interpolated"] = False

    new_rows = []
    for _, track in repaired.groupby("track_id", sort=False):
        track = track.sort_values("frame_id").reset_index(drop=True)

        for row_index in range(len(track) - 1):
            previous = track.iloc[row_index]
            next_row = track.iloc[row_index + 1]
            missing_count = int(next_row["frame_id"] - previous["frame_id"] - 1)

            if missing_count <= 0 or missing_count > max_gap:
                continue

            for step in range(1, missing_count + 1):
                ratio = step / (missing_count + 1)
                interpolated = previous[TRACK_COLUMNS].copy()
                interpolated["frame_id"] = int(previous["frame_id"] + step)

                for column in BOX_COLUMNS:
                    interpolated[column] = (
                        previous[column] + ratio * (next_row[column] - previous[column])
                    )

                interpolated["score"] = (previous["score"] + next_row["score"]) / 2
                interpolated["track_id"] = int(previous["track_id"])
                interpolated["class_id"] = int(previous["class_id"])
                interpolated["is_interpolated"] = True
                new_rows.append(interpolated.to_dict())

    if new_rows:
        repaired = pd.concat([repaired, pd.DataFrame(new_rows)], ignore_index=True)

    return repaired.sort_values(["frame_id", "track_id"]).reset_index(drop=True)


def repair_short_gaps(tracks: pd.DataFrame, max_gap_frames: int) -> pd.DataFrame:
    """Compatibility wrapper for the old scaffold name."""
    return interpolate_track_gaps(tracks, max_gap=max_gap_frames)
