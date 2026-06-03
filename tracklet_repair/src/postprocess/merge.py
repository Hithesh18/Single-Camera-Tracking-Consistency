"""Conservative merging for fragmented tracklets."""

from __future__ import annotations

from math import sqrt

import pandas as pd

from tracklet_repair.src.utils.io import TRACK_COLUMNS, validate_tracking_dataframe


def conservative_merge_tracklets(
    df: pd.DataFrame,
    max_merge_gap: int = 10,
    max_center_distance: float = 80.0,
    max_size_ratio: float = 1.5,
    require_same_class: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Merge short tracklet fragments when temporal and box checks agree."""
    if max_merge_gap < 0:
        raise ValueError("max_merge_gap must be non-negative.")
    if max_center_distance < 0:
        raise ValueError("max_center_distance must be non-negative.")
    if max_size_ratio < 1:
        raise ValueError("max_size_ratio must be at least 1.")

    merged = df.copy()
    validate_tracking_dataframe(merged)

    summaries = _tracklet_summaries(merged)
    merge_map: dict[int, int] = {}

    for target in sorted(summaries, key=lambda item: (item["start_frame"], item["track_id"])):
        target_id = int(target["track_id"])
        if target_id in merge_map:
            continue

        candidates = []
        for source in summaries:
            source_id = int(source["track_id"])
            if source_id == target_id:
                continue
            candidate = _build_candidate(
                source,
                target,
                max_merge_gap=max_merge_gap,
                max_center_distance=max_center_distance,
                max_size_ratio=max_size_ratio,
                require_same_class=require_same_class,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            continue

        candidates.sort(key=lambda item: (item["center_distance"], item["temporal_gap"]))
        best = candidates[0]
        merge_map[target_id] = _resolve_track_id(best["source_id"], merge_map)

    if merge_map:
        merged["track_id"] = merged["track_id"].map(lambda track_id: merge_map.get(track_id, track_id))

    merged = merged[TRACK_COLUMNS].sort_values(["frame_id", "track_id"]).reset_index(drop=True)
    return merged, merge_map


def _tracklet_summaries(df: pd.DataFrame) -> list[dict]:
    """Collect the first and last detection for each tracklet."""
    summaries = []
    for track_id, track in df.groupby("track_id", sort=False):
        track = track.sort_values("frame_id")
        first = track.iloc[0]
        last = track.iloc[-1]
        summaries.append(
            {
                "track_id": int(track_id),
                "start_frame": int(first["frame_id"]),
                "end_frame": int(last["frame_id"]),
                "first": first,
                "last": last,
                "class_id": int(first["class_id"]) if "class_id" in track.columns else None,
            }
        )
    return summaries


def _build_candidate(
    source: dict,
    target: dict,
    max_merge_gap: int,
    max_center_distance: float,
    max_size_ratio: float,
    require_same_class: bool,
) -> dict | None:
    """Return a candidate merge if all conservative checks pass."""
    temporal_gap = int(target["start_frame"] - source["end_frame"] - 1)
    if temporal_gap < 0 or temporal_gap > max_merge_gap:
        return None

    if require_same_class and source["class_id"] != target["class_id"]:
        return None

    distance = _center_distance(source["last"], target["first"])
    if distance > max_center_distance:
        return None

    if _ratio(source["last"]["width"], target["first"]["width"]) > max_size_ratio:
        return None
    if _ratio(source["last"]["height"], target["first"]["height"]) > max_size_ratio:
        return None

    return {
        "source_id": int(source["track_id"]),
        "target_id": int(target["track_id"]),
        "temporal_gap": temporal_gap,
        "center_distance": distance,
    }


def _center_distance(row_a: pd.Series, row_b: pd.Series) -> float:
    """Compute Euclidean distance between bounding-box centers."""
    center_a_x = row_a["x"] + row_a["width"] / 2
    center_a_y = row_a["y"] + row_a["height"] / 2
    center_b_x = row_b["x"] + row_b["width"] / 2
    center_b_y = row_b["y"] + row_b["height"] / 2
    return float(sqrt((center_a_x - center_b_x) ** 2 + (center_a_y - center_b_y) ** 2))


def _ratio(value_a: float, value_b: float) -> float:
    """Return the larger ratio between two positive values."""
    if value_a <= 0 or value_b <= 0:
        return float("inf")
    return float(max(value_a / value_b, value_b / value_a))


def _resolve_track_id(track_id: int, merge_map: dict[int, int]) -> int:
    """Resolve chained merges to the final kept track ID."""
    while track_id in merge_map:
        track_id = merge_map[track_id]
    return int(track_id)


def conservative_merge(tracks: pd.DataFrame, min_similarity_score: float) -> pd.DataFrame:
    """Compatibility wrapper for the old scaffold name."""
    _ = min_similarity_score
    merged, _ = conservative_merge_tracklets(tracks)
    return merged
