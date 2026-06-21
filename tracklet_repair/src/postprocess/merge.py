"""Conservative merging for fragmented tracklets."""

from __future__ import annotations

from math import sqrt

import pandas as pd

from tracklet_repair.src.utils.io import TRACK_COLUMNS, validate_tracking_dataframe


MERGE_MODES = ("conservative", "motion_aware", "appearance_global")


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
    components = {int(summary["track_id"]): summary for summary in summaries}
    merge_map: dict[int, int] = {}

    for target in sorted(summaries, key=lambda item: (item["start_frame"], item["track_id"])):
        target_id = int(target["track_id"])
        if target_id not in components:
            continue

        current_target = components[target_id]
        candidates = []
        for source_id, source in components.items():
            if source_id == target_id:
                continue
            candidate = _build_candidate(
                source,
                current_target,
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
        source_id = int(best["source_id"])
        merge_map[target_id] = source_id
        components[source_id] = _merge_summaries(
            components[source_id],
            current_target,
        )
        del components[target_id]

    if merge_map:
        merged["track_id"] = merged["track_id"].map(lambda track_id: merge_map.get(track_id, track_id))

    merged = merged[TRACK_COLUMNS].sort_values(["frame_id", "track_id"]).reset_index(drop=True)
    return merged, merge_map


def merge_tracklets(
    df: pd.DataFrame,
    merge_mode: str = "conservative",
    max_merge_gap: int = 10,
    max_center_distance: float = 80.0,
    max_size_ratio: float = 1.5,
    require_same_class: bool = True,
    velocity_window: int = 3,
    ambiguity_margin: float = 0.10,
    max_speed: float = 80.0,
    frames_dir=None,
    frame_pattern: str = "{frame_id:06d}.jpg",
    appearance_window: int = 3,
    appearance_threshold: float = 0.65,
    appearance_backend: str = "combined",
    allow_geometry_fallback: bool = False,
    max_global_merge_gap: int | None = None,
    appearance_weight: float = 0.40,
    motion_weight: float = 0.25,
    geometry_weight: float = 0.15,
    temporal_weight: float = 0.10,
    size_weight: float = 0.10,
    return_diagnostics: bool = False,
) -> tuple:
    """Run the selected tracklet merge strategy."""
    if merge_mode == "conservative":
        result = conservative_merge_tracklets(
            df,
            max_merge_gap=max_merge_gap,
            max_center_distance=max_center_distance,
            max_size_ratio=max_size_ratio,
            require_same_class=require_same_class,
        )
        return (*result, {}) if return_diagnostics else result
    if merge_mode == "motion_aware":
        result = motion_aware_merge_tracklets(
            df,
            max_merge_gap=max_merge_gap,
            max_center_distance=max_center_distance,
            max_size_ratio=max_size_ratio,
            require_same_class=require_same_class,
            velocity_window=velocity_window,
            ambiguity_margin=ambiguity_margin,
            max_speed=max_speed,
        )
        return (*result, {}) if return_diagnostics else result
    if merge_mode == "appearance_global":
        from tracklet_repair.src.postprocess.appearance_global import (
            appearance_global_merge_tracklets,
        )

        merged, merge_map, diagnostics = appearance_global_merge_tracklets(
            df,
            frames_dir=frames_dir,
            frame_pattern=frame_pattern,
            appearance_window=appearance_window,
            appearance_threshold=appearance_threshold,
            appearance_backend=appearance_backend,
            allow_geometry_fallback=allow_geometry_fallback,
            max_global_merge_gap=(
                max_merge_gap if max_global_merge_gap is None else max_global_merge_gap
            ),
            max_center_distance=max_center_distance,
            max_size_ratio=max_size_ratio,
            require_same_class=require_same_class,
            velocity_window=velocity_window,
            max_speed=max_speed,
            ambiguity_margin=ambiguity_margin,
            appearance_weight=appearance_weight,
            motion_weight=motion_weight,
            geometry_weight=geometry_weight,
            temporal_weight=temporal_weight,
            size_weight=size_weight,
        )
        if return_diagnostics:
            return merged, merge_map, diagnostics
        return merged, merge_map
    raise ValueError(f"Unknown merge mode: {merge_mode}. Expected one of {MERGE_MODES}.")


def motion_aware_merge_tracklets(
    df: pd.DataFrame,
    max_merge_gap: int = 10,
    max_center_distance: float = 80.0,
    max_size_ratio: float = 1.5,
    require_same_class: bool = True,
    velocity_window: int = 3,
    ambiguity_margin: float = 0.10,
    max_speed: float = 80.0,
) -> tuple[pd.DataFrame, dict]:
    """Merge fragments using motion prediction and ambiguity rejection."""
    _validate_motion_parameters(
        max_merge_gap,
        max_center_distance,
        max_size_ratio,
        velocity_window,
        ambiguity_margin,
        max_speed,
    )

    merged = df.copy()
    validate_tracking_dataframe(merged)
    components = {
        int(summary["track_id"]): summary
        for summary in _motion_tracklet_summaries(merged)
    }
    merge_map: dict[int, int] = {}

    while True:
        proposals = _motion_proposals(
            components,
            max_merge_gap=max_merge_gap,
            max_center_distance=max_center_distance,
            max_size_ratio=max_size_ratio,
            require_same_class=require_same_class,
            velocity_window=velocity_window,
            ambiguity_margin=ambiguity_margin,
            max_speed=max_speed,
        )
        if not proposals:
            break

        used_components: set[int] = set()
        accepted = 0
        for proposal in sorted(
            proposals,
            key=lambda item: (item["score"], item["source_id"], item["target_id"]),
        ):
            source_id = int(proposal["source_id"])
            target_id = int(proposal["target_id"])
            if source_id not in components or target_id not in components:
                continue
            if source_id in used_components or target_id in used_components:
                continue

            merge_map[target_id] = source_id
            components[source_id] = _merge_motion_summaries(
                components[source_id],
                components[target_id],
            )
            del components[target_id]
            used_components.update((source_id, target_id))
            accepted += 1

        if accepted == 0:
            break

    if merge_map:
        merged["track_id"] = merged["track_id"].map(
            lambda track_id: _resolve_motion_track_id(int(track_id), merge_map)
        )

    merged = merged[TRACK_COLUMNS].sort_values(["frame_id", "track_id"]).reset_index(drop=True)
    if merged.duplicated(subset=["frame_id", "track_id"]).any():
        raise RuntimeError("Motion-aware merging created overlapping final tracklets.")
    return merged, merge_map


def _validate_motion_parameters(
    max_merge_gap: int,
    max_center_distance: float,
    max_size_ratio: float,
    velocity_window: int,
    ambiguity_margin: float,
    max_speed: float,
) -> None:
    if max_merge_gap < 0:
        raise ValueError("max_merge_gap must be non-negative.")
    if max_center_distance < 0:
        raise ValueError("max_center_distance must be non-negative.")
    if max_size_ratio < 1:
        raise ValueError("max_size_ratio must be at least 1.")
    if velocity_window < 2:
        raise ValueError("velocity_window must be at least 2.")
    if ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be non-negative.")
    if max_speed <= 0:
        raise ValueError("max_speed must be positive.")


def _motion_tracklet_summaries(df: pd.DataFrame) -> list[dict]:
    summaries = []
    for track_id, track in df.groupby("track_id", sort=False):
        track = track.sort_values("frame_id").reset_index(drop=True)
        first = track.iloc[0]
        last = track.iloc[-1]
        centers = [
            (
                int(row["frame_id"]),
                float(row["x"] + row["width"] / 2),
                float(row["y"] + row["height"] / 2),
            )
            for _, row in track.iterrows()
        ]
        summaries.append(
            {
                "track_id": int(track_id),
                "start_frame": int(first["frame_id"]),
                "end_frame": int(last["frame_id"]),
                "first": first,
                "last": last,
                "class_id": int(first["class_id"]),
                "frame_ids": set(track["frame_id"].astype(int)),
                "centers": centers,
                "length": int(len(track)),
                "mean_score": float(track["score"].mean()),
            }
        )
    return summaries


def _motion_proposals(
    components: dict[int, dict],
    max_merge_gap: int,
    max_center_distance: float,
    max_size_ratio: float,
    require_same_class: bool,
    velocity_window: int,
    ambiguity_margin: float,
    max_speed: float,
) -> list[dict]:
    proposals = []
    for source_id, source in sorted(
        components.items(),
        key=lambda item: (item[1]["end_frame"], item[0]),
    ):
        candidates = []
        for target_id, target in components.items():
            if source_id == target_id:
                continue
            candidate = _build_motion_candidate(
                source,
                target,
                max_merge_gap=max_merge_gap,
                max_center_distance=max_center_distance,
                max_size_ratio=max_size_ratio,
                require_same_class=require_same_class,
                velocity_window=velocity_window,
                max_speed=max_speed,
            )
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(
            key=lambda item: (item["score"], item["target_start"], item["target_id"])
        )
        if not candidates:
            continue
        if (
            len(candidates) > 1
            and candidates[1]["score"] - candidates[0]["score"] < ambiguity_margin
        ):
            continue
        proposals.append(candidates[0])
    return proposals


def _build_motion_candidate(
    source: dict,
    target: dict,
    max_merge_gap: int,
    max_center_distance: float,
    max_size_ratio: float,
    require_same_class: bool,
    velocity_window: int,
    max_speed: float,
) -> dict | None:
    if source["frame_ids"] & target["frame_ids"]:
        return None

    temporal_gap = int(target["start_frame"] - source["end_frame"] - 1)
    if temporal_gap < 0 or temporal_gap > max_merge_gap:
        return None
    if require_same_class and source["class_id"] != target["class_id"]:
        return None

    width_ratio = _ratio(source["last"]["width"], target["first"]["width"])
    height_ratio = _ratio(source["last"]["height"], target["first"]["height"])
    size_ratio = max(width_ratio, height_ratio)
    if size_ratio > max_size_ratio:
        return None

    source_center = source["centers"][-1][1:]
    target_center = target["centers"][0][1:]
    displacement = _subtract(target_center, source_center)
    frame_delta = int(target["start_frame"] - source["end_frame"])
    raw_distance = _vector_length(displacement)
    speed = raw_distance / frame_delta
    if speed > max_speed:
        return None

    source_velocity, source_has_velocity = _estimate_velocity(
        source["centers"], velocity_window, from_start=False
    )
    target_velocity, target_has_velocity = _estimate_velocity(
        target["centers"], velocity_window, from_start=True
    )
    if source_has_velocity:
        predicted_center = (
            source_center[0] + source_velocity[0] * frame_delta,
            source_center[1] + source_velocity[1] * frame_delta,
        )
    else:
        predicted_center = source_center
    predicted_distance = _vector_length(_subtract(target_center, predicted_center))
    if predicted_distance > max_center_distance:
        return None

    prediction_cost = _normalized(predicted_distance, max_center_distance)
    raw_distance_cost = _normalized(
        raw_distance,
        max(max_center_distance * frame_delta, 1.0),
    )
    speed_cost = _normalized(speed, max_speed)
    direction_cost = _direction_cost(source_velocity, displacement)
    velocity_cost = _direction_cost(source_velocity, target_velocity)
    if not source_has_velocity:
        direction_cost = 0.5
    if not source_has_velocity or not target_has_velocity:
        velocity_cost = 0.5

    size_cost = _size_cost(size_ratio, max_size_ratio)
    temporal_cost = _normalized(temporal_gap, max(max_merge_gap, 1))
    confidence = max(
        0.0,
        min(1.0, (float(source["last"]["score"]) + float(target["first"]["score"])) / 2),
    )
    confidence_cost = 1.0 - confidence
    reliability_cost = 1.0 / min(source["length"], target["length"], 10)

    score = (
        0.40 * prediction_cost
        + 0.10 * raw_distance_cost
        + 0.10 * speed_cost
        + 0.125 * direction_cost
        + 0.075 * velocity_cost
        + 0.10 * size_cost
        + 0.05 * temporal_cost
        + 0.025 * confidence_cost
        + 0.025 * reliability_cost
    )
    return {
        "source_id": int(source["track_id"]),
        "target_id": int(target["track_id"]),
        "target_start": int(target["start_frame"]),
        "score": float(score),
    }


def _merge_motion_summaries(source: dict, target: dict) -> dict:
    total_length = source["length"] + target["length"]
    mean_score = (
        source["mean_score"] * source["length"]
        + target["mean_score"] * target["length"]
    ) / total_length
    return {
        "track_id": int(source["track_id"]),
        "start_frame": int(source["start_frame"]),
        "end_frame": int(target["end_frame"]),
        "first": source["first"],
        "last": target["last"],
        "class_id": source["class_id"],
        "frame_ids": source["frame_ids"] | target["frame_ids"],
        "centers": sorted(source["centers"] + target["centers"]),
        "length": total_length,
        "mean_score": float(mean_score),
    }


def _estimate_velocity(
    centers: list[tuple[int, float, float]],
    window: int,
    from_start: bool,
) -> tuple[tuple[float, float], bool]:
    selected = centers[:window] if from_start else centers[-window:]
    if len(selected) < 2:
        return (0.0, 0.0), False
    first = selected[0]
    last = selected[-1]
    frame_delta = last[0] - first[0]
    if frame_delta <= 0:
        return (0.0, 0.0), False
    return (
        (last[1] - first[1]) / frame_delta,
        (last[2] - first[2]) / frame_delta,
    ), True


def _subtract(vector_a: tuple[float, float], vector_b: tuple[float, float]) -> tuple[float, float]:
    return vector_a[0] - vector_b[0], vector_a[1] - vector_b[1]


def _vector_length(vector: tuple[float, float]) -> float:
    return float(sqrt(vector[0] ** 2 + vector[1] ** 2))


def _direction_cost(vector_a: tuple[float, float], vector_b: tuple[float, float]) -> float:
    length_a = _vector_length(vector_a)
    length_b = _vector_length(vector_b)
    if length_a == 0 and length_b == 0:
        return 0.0
    if length_a == 0 or length_b == 0:
        return 0.5
    cosine = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (
        length_a * length_b
    )
    cosine = max(-1.0, min(1.0, cosine))
    return float((1.0 - cosine) / 2.0)


def _normalized(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0 if value == 0 else 1.0
    return float(min(value / maximum, 1.0))


def _size_cost(size_ratio: float, max_size_ratio: float) -> float:
    if max_size_ratio == 1:
        return 0.0
    return float((size_ratio - 1.0) / (max_size_ratio - 1.0))


def _resolve_motion_track_id(track_id: int, merge_map: dict[int, int]) -> int:
    visited = set()
    while track_id in merge_map:
        if track_id in visited:
            raise RuntimeError("Cycle detected in motion-aware merge map.")
        visited.add(track_id)
        track_id = int(merge_map[track_id])
    return track_id


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
                "frame_ids": set(track["frame_id"].astype(int)),
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
    if source["frame_ids"] & target["frame_ids"]:
        return None

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


def _merge_summaries(source: dict, target: dict) -> dict:
    """Update a source summary after accepting a later target component."""
    return {
        "track_id": int(source["track_id"]),
        "start_frame": int(source["start_frame"]),
        "end_frame": int(target["end_frame"]),
        "first": source["first"],
        "last": target["last"],
        "class_id": source["class_id"],
        "frame_ids": source["frame_ids"] | target["frame_ids"],
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


def conservative_merge(tracks: pd.DataFrame, min_similarity_score: float) -> pd.DataFrame:
    """Compatibility wrapper for the old scaffold name."""
    _ = min_similarity_score
    merged, _ = conservative_merge_tracklets(tracks)
    return merged
