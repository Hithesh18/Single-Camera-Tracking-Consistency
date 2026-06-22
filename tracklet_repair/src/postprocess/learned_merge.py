"""Learned tracklet merging — uses the trained matcher instead of if/else rules.

For every ordered candidate pair (source ends before target starts) the trained
LearnedMatcher returns P(same identity) from deep ReID appearance + motion.
Pairs are accepted greedily by descending probability, building identity chains
(A->B->C) while keeping each tracklet to at most one predecessor and one
successor. No temporal cap is required, so the model can re-identify an object
that left the view and returned.
"""

from __future__ import annotations

import pandas as pd

from tracklet_repair.src.matcher.features import (
    build_embedding_index,
    build_tracklets,
    ordered_candidate_pairs,
    pair_features,
)
from tracklet_repair.src.matcher.model import LearnedMatcher
from tracklet_repair.src.utils.io import TRACK_COLUMNS, validate_tracking_dataframe


def learned_merge_tracklets(
    df: pd.DataFrame,
    matcher_path: str,
    embed_dir: str,
    prob_threshold: float = 0.5,
    max_gap: int | None = None,
    require_same_class: bool = True,
    match_iou: float = 0.7,
) -> tuple[pd.DataFrame, dict, dict]:
    """Merge fragments whose learned same-identity probability clears a threshold."""
    merged = df.copy()
    validate_tracking_dataframe(merged)

    matcher = LearnedMatcher.load(matcher_path)
    embed_index = build_embedding_index(embed_dir)
    tracklets = build_tracklets(merged, embed_index, match_iou=match_iou)

    proposals = []
    for source_id, target_id in ordered_candidate_pairs(tracklets, max_gap=max_gap):
        source, target = tracklets[source_id], tracklets[target_id]
        if require_same_class and source.class_id != target.class_id:
            continue
        prob = matcher.probability(pair_features(source, target))
        if prob >= prob_threshold:
            proposals.append((prob, source_id, target_id))

    proposals.sort(key=lambda item: item[0], reverse=True)

    merge_map: dict[int, int] = {}
    has_successor: set[int] = set()
    has_predecessor: set[int] = set()
    accepted = []
    for prob, source_id, target_id in proposals:
        if source_id in has_successor or target_id in has_predecessor:
            continue
        merge_map[target_id] = source_id
        has_successor.add(source_id)
        has_predecessor.add(target_id)
        accepted.append({"source": source_id, "target": target_id, "probability": round(prob, 4)})

    if merge_map:
        merged["track_id"] = merged["track_id"].map(lambda tid: _resolve(int(tid), merge_map))

    merged = merged[TRACK_COLUMNS].sort_values(["frame_id", "track_id"]).reset_index(drop=True)
    diagnostics = {
        "candidate_proposals": len(proposals),
        "accepted_merges": len(accepted),
        "prob_threshold": prob_threshold,
        "merges": accepted,
    }
    return merged, merge_map, diagnostics


def _resolve(track_id: int, merge_map: dict[int, int]) -> int:
    visited = set()
    while track_id in merge_map:
        if track_id in visited:
            raise RuntimeError("Cycle detected in learned merge map.")
        visited.add(track_id)
        track_id = int(merge_map[track_id])
    return track_id
