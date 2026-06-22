"""Join deep ReID embeddings to tracklets and build learned-matcher features.

This is the "visual appearance" backbone the supervisor asked for: instead of
geometric if/else rules, every candidate tracklet merge is described by a vector
that is dominated by deep ReID appearance similarity (OSNet, 512-d), with motion
and temporal cues alongside. A learned model (see model.py) maps this vector to
P(same identity).

Embeddings are matched to tracked detections exactly on (frame, x1, y1, x2, y2):
the BoT-SORT output box equals the detection box, and the embedding filename
encodes that box, so no fuzzy matching is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BOX_COLS = ["x", "y", "width", "height"]

# Order of the pair-feature vector. Kept as a module constant so training and
# inference always agree, and so the report can name each learned input.
PAIR_FEATURE_NAMES = [
    "mean_appearance_cosine",
    "endpoint_appearance_cosine",
    "max_appearance_cosine",
    "temporal_gap_norm",
    "predicted_center_distance_norm",
    "raw_center_distance_norm",
    "size_ratio",
    "same_class",
    "velocity_direction_consistency",
]
PAIR_FEATURE_DIM = len(PAIR_FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Embedding store
# --------------------------------------------------------------------------- #

def build_embedding_index(embed_dir: str) -> dict[int, list[tuple[np.ndarray, str]]]:
    """Index embedding .npy files per frame as (box_xywh, path).

    Filename layout: feature_<frame>_<idx>_<x1>_<x2>_<y1>_<y2>_<conf>_cls<c>.npy

    Boxes are matched to tracked detections by IoU (not exact equality) because
    BoT-SORT's Kalman filter shifts the output box a pixel or two away from the
    raw detection box, so an exact (frame, box) key misses almost everything.
    """
    index: dict[int, list[tuple[np.ndarray, str]]] = {}
    for path in Path(embed_dir).glob("feature_*_cls*.npy"):
        parts = path.stem.split("_")
        if len(parts) < 8:
            continue
        try:
            frame = int(parts[1])
            x1, x2, y1, y2 = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
        except ValueError:
            continue
        box = np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)
        index.setdefault(frame, []).append((box, str(path)))
    return index


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _load_normalized(path: str) -> np.ndarray:
    vec = np.load(path).astype(np.float32).ravel()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# --------------------------------------------------------------------------- #
# Tracklet summaries with appearance
# --------------------------------------------------------------------------- #

@dataclass
class Tracklet:
    track_id: int
    class_id: int
    frames: list[int]
    boxes: list[np.ndarray]          # each [x, y, w, h]
    embeddings: list[np.ndarray]     # L2-normalized, may be shorter than frames
    mean_embedding: np.ndarray = field(default=None)

    @property
    def start_frame(self) -> int:
        return self.frames[0]

    @property
    def end_frame(self) -> int:
        return self.frames[-1]

    def head_embedding(self, k: int = 3) -> np.ndarray | None:
        return _mean_unit(self.embeddings[:k]) if self.embeddings else None

    def tail_embedding(self, k: int = 3) -> np.ndarray | None:
        return _mean_unit(self.embeddings[-k:]) if self.embeddings else None

    def center(self, at_start: bool) -> np.ndarray:
        box = self.boxes[0] if at_start else self.boxes[-1]
        return np.array([box[0] + box[2] / 2, box[1] + box[3] / 2], dtype=float)

    def velocity(self, window: int, at_start: bool) -> np.ndarray | None:
        idx = range(min(window, len(self.frames)))
        sel = list(idx) if at_start else [len(self.frames) - 1 - i for i in idx][::-1]
        if len(sel) < 2:
            return None
        f0, f1 = self.frames[sel[0]], self.frames[sel[-1]]
        if f1 == f0:
            return None
        c0 = self.boxes[sel[0]]; c1 = self.boxes[sel[-1]]
        p0 = np.array([c0[0] + c0[2] / 2, c0[1] + c0[3] / 2])
        p1 = np.array([c1[0] + c1[2] / 2, c1[1] + c1[3] / 2])
        return (p1 - p0) / (f1 - f0)


def _mean_unit(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    mean = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def build_tracklets(tracks: pd.DataFrame, embed_index: dict, match_iou: float = 0.7) -> dict[int, Tracklet]:
    """Group a track DataFrame into Tracklet objects with attached embeddings.

    Each tracked box is joined to the best-IoU embedding box in the same frame
    (>= match_iou), tolerating the small Kalman-filter offset.
    """
    tracklets: dict[int, Tracklet] = {}
    for track_id, group in tracks.groupby("track_id", sort=False):
        group = group.sort_values("frame_id")
        frames, boxes, embeddings = [], [], []
        for _, row in group.iterrows():
            frame = int(row["frame_id"])
            box = np.array([row["x"], row["y"], row["width"], row["height"]], dtype=float)
            frames.append(frame)
            boxes.append(box)
            best_path, best_iou = None, match_iou
            for cand_box, cand_path in embed_index.get(frame, []):
                iou = _iou_xywh(box, cand_box)
                if iou >= best_iou:
                    best_path, best_iou = cand_path, iou
            if best_path is not None:
                embeddings.append(_load_normalized(best_path))
        class_id = int(group.iloc[0]["class_id"]) if "class_id" in group.columns else 0
        tracklet = Tracklet(int(track_id), class_id, frames, boxes, embeddings)
        tracklet.mean_embedding = _mean_unit(embeddings)
        tracklets[int(track_id)] = tracklet
    return tracklets


# --------------------------------------------------------------------------- #
# Pair features (the learned-matcher input)
# --------------------------------------------------------------------------- #

def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def pair_features(
    source: Tracklet,
    target: Tracklet,
    center_norm: float = 200.0,
    gap_norm: float = 30.0,
    endpoint_k: int = 3,
    velocity_window: int = 3,
) -> np.ndarray:
    """Build the feature vector for an ordered candidate merge (source -> target).

    `source` ends before `target` begins. Appearance similarities use the deep
    ReID embeddings; motion uses source velocity extrapolated to the target.
    """
    src_tail = source.tail_embedding(endpoint_k)
    tgt_head = target.head_embedding(endpoint_k)

    mean_cos = _cos(source.mean_embedding, target.mean_embedding)
    endpoint_cos = _cos(src_tail, tgt_head)
    max_cos = 0.0
    if source.embeddings and target.embeddings:
        tail = np.stack(source.embeddings[-endpoint_k:])
        head = np.stack(target.embeddings[:endpoint_k])
        max_cos = float(np.clip((tail @ head.T).max(), -1.0, 1.0))

    temporal_gap = target.start_frame - source.end_frame - 1
    gap_norm_val = min(max(temporal_gap, 0) / gap_norm, 2.0)

    src_center = source.center(at_start=False)
    tgt_center = target.center(at_start=True)
    raw_distance = float(np.linalg.norm(tgt_center - src_center))
    raw_norm = min(raw_distance / center_norm, 2.0)

    velocity = source.velocity(velocity_window, at_start=False)
    frame_delta = max(target.start_frame - source.end_frame, 1)
    if velocity is not None:
        predicted = src_center + velocity * frame_delta
        predicted_distance = float(np.linalg.norm(tgt_center - predicted))
    else:
        predicted_distance = raw_distance
    predicted_norm = min(predicted_distance / center_norm, 2.0)

    src_box = source.boxes[-1]; tgt_box = target.boxes[0]
    size_ratio = max(
        _ratio(src_box[2], tgt_box[2]),
        _ratio(src_box[3], tgt_box[3]),
    )
    size_ratio = min(size_ratio, 5.0)

    same_class = 1.0 if source.class_id == target.class_id else 0.0

    tgt_velocity = target.velocity(velocity_window, at_start=True)
    direction_consistency = 0.0
    if velocity is not None and tgt_velocity is not None:
        nv, nt = np.linalg.norm(velocity), np.linalg.norm(tgt_velocity)
        if nv > 0 and nt > 0:
            direction_consistency = float(np.clip(np.dot(velocity, tgt_velocity) / (nv * nt), -1.0, 1.0))

    return np.array([
        mean_cos, endpoint_cos, max_cos,
        gap_norm_val, predicted_norm, raw_norm,
        size_ratio, same_class, direction_consistency,
    ], dtype=np.float32)


def _ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 5.0
    return float(max(a / b, b / a))


def ordered_candidate_pairs(
    tracklets: dict[int, Tracklet],
    max_gap: int | None = None,
) -> list[tuple[int, int]]:
    """All (source_id, target_id) where source ends strictly before target starts.

    With `max_gap=None` no temporal limit is applied — this is what lets the
    learned matcher consider long-gap re-identification (object leaves view /
    goes behind an obstacle and returns), which the geometric merge cannot.
    """
    items = list(tracklets.values())
    pairs = []
    for source in items:
        for target in items:
            if source.track_id == target.track_id:
                continue
            gap = target.start_frame - source.end_frame - 1
            if gap < 0:
                continue
            if max_gap is not None and gap > max_gap:
                continue
            pairs.append((source.track_id, target.track_id))
    return pairs
