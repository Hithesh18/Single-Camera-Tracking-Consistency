"""Appearance-supported global association for fragmented tracklets."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from tracklet_repair.src.features.appearance import AppearanceFeatureCache, cosine_similarity
from tracklet_repair.src.utils.io import TRACK_COLUMNS, validate_tracking_dataframe


def appearance_global_merge_tracklets(
    df: pd.DataFrame,
    *,
    frames_dir: Path | None,
    frame_pattern: str = "{frame_id:06d}.jpg",
    appearance_window: int = 3,
    appearance_threshold: float = 0.65,
    appearance_backend: str = "combined",
    allow_geometry_fallback: bool = False,
    max_global_merge_gap: int = 8,
    max_center_distance: float = 80.0,
    max_size_ratio: float = 1.5,
    require_same_class: bool = True,
    velocity_window: int = 3,
    max_speed: float = 80.0,
    ambiguity_margin: float = 0.10,
    appearance_weight: float = 0.40,
    motion_weight: float = 0.25,
    geometry_weight: float = 0.15,
    temporal_weight: float = 0.10,
    size_weight: float = 0.10,
) -> tuple[pd.DataFrame, dict[int, int], dict]:
    """Globally select non-conflicting, appearance-supported merge edges."""
    _validate_parameters(
        appearance_window, appearance_threshold, max_global_merge_gap,
        max_center_distance, max_size_ratio, velocity_window, max_speed,
        ambiguity_margin, appearance_weight, motion_weight, geometry_weight,
        temporal_weight, size_weight,
    )
    tracks = df.copy(deep=True)
    validate_tracking_dataframe(tracks)
    summaries = _summaries(tracks)
    cache = None
    if frames_dir is not None:
        cache = AppearanceFeatureCache(Path(frames_dir), frame_pattern, appearance_backend)

    diagnostics = {
        "candidate_edges": 0,
        "appearance_rejected_candidates": 0,
        "appearance_missing_candidates": 0,
        "ambiguous_candidates_skipped": 0,
        "safety_violations": 0,
        "accepted_appearance_similarities": [],
        "borderline_merges": 0,
        "selection_method": "hungarian",
    }
    candidates = []
    for source in summaries:
        for target in summaries:
            if source["track_id"] == target["track_id"]:
                continue
            candidate = _candidate(
                source, target, cache=cache, appearance_window=appearance_window,
                appearance_threshold=appearance_threshold,
                allow_geometry_fallback=allow_geometry_fallback,
                max_global_merge_gap=max_global_merge_gap,
                max_center_distance=max_center_distance,
                max_size_ratio=max_size_ratio,
                require_same_class=require_same_class,
                velocity_window=velocity_window, max_speed=max_speed,
                appearance_weight=appearance_weight, motion_weight=motion_weight,
                geometry_weight=geometry_weight, temporal_weight=temporal_weight,
                size_weight=size_weight, diagnostics=diagnostics,
            )
            if candidate is not None:
                candidates.append(candidate)

    candidates = _reject_ambiguous(candidates, ambiguity_margin, diagnostics)
    selected = _global_assignment(candidates, summaries)
    merge_map = {int(item["target_id"]): int(item["source_id"]) for item in selected}
    resolved_map = {target: _resolve(source, merge_map) for target, source in merge_map.items()}

    if resolved_map:
        tracks["track_id"] = tracks["track_id"].map(
            lambda track_id: _resolve(int(track_id), resolved_map)
        )
    tracks = tracks[TRACK_COLUMNS].sort_values(["frame_id", "track_id"]).reset_index(drop=True)
    duplicates = int(tracks.duplicated(subset=["frame_id", "track_id"]).sum())
    diagnostics["safety_violations"] = duplicates
    if duplicates:
        raise RuntimeError("Appearance-global merging created overlapping final tracklets.")

    similarities = [
        float(item["appearance_similarity"])
        for item in selected if item["appearance_similarity"] is not None
    ]
    diagnostics["accepted_appearance_similarities"] = similarities
    diagnostics["borderline_merges"] = sum(
        similarity < appearance_threshold + 0.05 for similarity in similarities
    )
    diagnostics["accepted_candidates"] = len(selected)
    return tracks, merge_map, diagnostics


def _summaries(df: pd.DataFrame) -> list[dict]:
    summaries = []
    for track_id, rows in df.groupby("track_id", sort=True):
        rows = rows.sort_values("frame_id").reset_index(drop=True)
        first, last = rows.iloc[0], rows.iloc[-1]
        summaries.append({
            "track_id": int(track_id), "rows": rows,
            "start_frame": int(first["frame_id"]), "end_frame": int(last["frame_id"]),
            "first": first, "last": last, "class_id": int(first["class_id"]),
            "frame_ids": set(rows["frame_id"].astype(int)), "length": len(rows),
            "mean_score": float(rows["score"].mean()),
        })
    return summaries


def _candidate(source: dict, target: dict, **options) -> dict | None:
    diagnostics = options["diagnostics"]
    if source["frame_ids"] & target["frame_ids"]:
        return None
    gap = target["start_frame"] - source["end_frame"] - 1
    if gap < 0 or gap > options["max_global_merge_gap"]:
        return None
    if options["require_same_class"] and source["class_id"] != target["class_id"]:
        return None

    width_ratio = _ratio(source["last"]["width"], target["first"]["width"])
    height_ratio = _ratio(source["last"]["height"], target["first"]["height"])
    size_ratio = max(width_ratio, height_ratio)
    if size_ratio > options["max_size_ratio"]:
        return None

    source_center = _center(source["last"])
    target_center = _center(target["first"])
    frame_delta = target["start_frame"] - source["end_frame"]
    displacement = _distance(source_center, target_center)
    speed = displacement / frame_delta
    if speed > options["max_speed"]:
        return None
    velocity = _velocity(source["rows"], options["velocity_window"], from_start=False)
    target_velocity = _velocity(target["rows"], options["velocity_window"], from_start=True)
    predicted = (source_center[0] + velocity[0] * frame_delta,
                 source_center[1] + velocity[1] * frame_delta)
    predicted_distance = _distance(predicted, target_center)
    if predicted_distance > options["max_center_distance"]:
        return None

    diagnostics["candidate_edges"] += 1
    similarity = None
    cache = options["cache"]
    if cache is not None:
        source_feature = cache.endpoint_feature(source["rows"], False, options["appearance_window"])
        target_feature = cache.endpoint_feature(target["rows"], True, options["appearance_window"])
        similarity = cosine_similarity(source_feature, target_feature)
    if similarity is None:
        diagnostics["appearance_missing_candidates"] += 1
        if not options["allow_geometry_fallback"]:
            return None
        appearance_cost = 0.5
    elif similarity < options["appearance_threshold"]:
        diagnostics["appearance_rejected_candidates"] += 1
        return None
    else:
        appearance_cost = 1.0 - similarity

    displacement_vector = (
        target_center[0] - source_center[0], target_center[1] - source_center[1]
    )
    motion_cost = (
        0.50 * min(predicted_distance / max(options["max_center_distance"], 1.0), 1.0)
        + 0.25 * min(speed / options["max_speed"], 1.0)
        + 0.125 * _direction_cost(velocity, displacement_vector)
        + 0.125 * _direction_cost(velocity, target_velocity)
    )
    geometry_cost = min(displacement / max(options["max_center_distance"] * frame_delta, 1.0), 1.0)
    temporal_cost = gap / max(options["max_global_merge_gap"], 1)
    size_cost = (size_ratio - 1.0) / max(options["max_size_ratio"] - 1.0, 1e-9)
    reliability_cost = 0.5 * (1.0 / min(source["length"], target["length"], 10))
    reliability_cost += 0.5 * (1.0 - min(source["mean_score"], target["mean_score"], 1.0))
    weighted = (
        options["appearance_weight"] * appearance_cost
        + options["motion_weight"] * motion_cost
        + options["geometry_weight"] * geometry_cost
        + options["temporal_weight"] * temporal_cost
        + options["size_weight"] * size_cost
    ) / _weight_total(options)
    score = 0.95 * weighted + 0.05 * reliability_cost
    return {
        "source_id": source["track_id"], "target_id": target["track_id"],
        "score": float(score), "appearance_similarity": similarity,
        "temporal_gap": int(gap), "center_distance": float(displacement),
        "predicted_distance": float(predicted_distance), "size_ratio": float(size_ratio),
    }


def _reject_ambiguous(candidates: list[dict], margin: float, diagnostics: dict) -> list[dict]:
    rejected: set[tuple[int, int]] = set()
    for key_name in ("source_id", "target_id"):
        grouped: dict[int, list[dict]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate[key_name], []).append(candidate)
        for group in grouped.values():
            group.sort(key=lambda item: (item["score"], item["source_id"], item["target_id"]))
            if len(group) > 1 and group[1]["score"] - group[0]["score"] < margin:
                rejected.update((item["source_id"], item["target_id"]) for item in group)
    diagnostics["ambiguous_candidates_skipped"] = len(rejected)
    return [item for item in candidates if (item["source_id"], item["target_id"]) not in rejected]


def _global_assignment(candidates: list[dict], summaries: list[dict]) -> list[dict]:
    if not candidates:
        return []
    ids = sorted(summary["track_id"] for summary in summaries)
    index = {track_id: position for position, track_id in enumerate(ids)}
    count = len(ids)
    costs = np.full((count, count * 2), 1e6, dtype=float)
    costs[:, count:] = 1.0
    lookup = {}
    for item in candidates:
        row, column = index[item["source_id"]], index[item["target_id"]]
        cost = item["score"] + row * 1e-10 + column * 1e-12
        costs[row, column] = min(costs[row, column], cost)
        lookup[(row, column)] = item
    rows, columns = linear_sum_assignment(costs)
    selected = [lookup[(row, column)] for row, column in zip(rows, columns)
                if column < count and (row, column) in lookup and costs[row, column] < 1.0]
    return sorted(selected, key=lambda item: (item["score"], item["source_id"], item["target_id"]))


def _validate_parameters(window, threshold, gap, distance, ratio, velocity_window,
                         speed, margin, *weights) -> None:
    if window <= 0 or velocity_window < 2:
        raise ValueError("Appearance window must be positive and velocity window at least 2.")
    if not 0 <= threshold <= 1 or gap < 0 or distance < 0 or ratio < 1 or speed <= 0 or margin < 0:
        raise ValueError("Invalid appearance-global threshold or constraint.")
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Appearance-global weights must be non-negative with a positive sum.")


def _weight_total(options: dict) -> float:
    return sum(options[name] for name in ("appearance_weight", "motion_weight", "geometry_weight", "temporal_weight", "size_weight"))


def _center(row) -> tuple[float, float]:
    return float(row["x"] + row["width"] / 2), float(row["y"] + row["height"] / 2)


def _distance(a, b) -> float:
    return float(sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def _direction_cost(a, b) -> float:
    length_a, length_b = _distance(a, (0.0, 0.0)), _distance(b, (0.0, 0.0))
    if length_a == 0 or length_b == 0:
        return 0.5
    cosine = (a[0] * b[0] + a[1] * b[1]) / (length_a * length_b)
    return float((1.0 - max(-1.0, min(1.0, cosine))) / 2.0)


def _velocity(rows, window: int, from_start: bool) -> tuple[float, float]:
    selected = rows.iloc[:window] if from_start else rows.iloc[-window:]
    if len(selected) < 2:
        return 0.0, 0.0
    first, last = selected.iloc[0], selected.iloc[-1]
    delta = int(last["frame_id"] - first["frame_id"])
    if delta <= 0:
        return 0.0, 0.0
    first_center, last_center = _center(first), _center(last)
    return (last_center[0] - first_center[0]) / delta, (last_center[1] - first_center[1]) / delta


def _ratio(a, b) -> float:
    a, b = float(a), float(b)
    if a <= 0 or b <= 0:
        return float("inf")
    return max(a / b, b / a)


def _resolve(track_id: int, merge_map: dict[int, int]) -> int:
    visited = set()
    while track_id in merge_map:
        if track_id in visited:
            raise RuntimeError("Cycle detected in appearance-global merge map.")
        visited.add(track_id)
        track_id = int(merge_map[track_id])
    return track_id
