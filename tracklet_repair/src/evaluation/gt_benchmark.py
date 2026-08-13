"""Single-camera evaluation against ground truth + gap root-cause analysis.

Computes MOT metrics (IDF1, MOTA, ID switches, fragmentations) for baseline vs
repaired tracker output against the AIC25 ground truth, and classifies why each
internal gap happened (occlusion/exit vs missed detection vs low confidence vs
association failure). CPU-only, runs on existing single-camera JSONs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import motmetrics as mm
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "gt_benchmark requires motmetrics. Install with: pip install motmetrics"
    ) from error

from tracklet_repair.src.utils.json_adapter import load_single_camera_json_as_dataframe


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

BOX_COLS = ["x", "y", "width", "height"]


def load_tracks(path: str) -> pd.DataFrame:
    """Load a tracker output as a frame/track/xywh DataFrame.

    Accepts either a BoT-SORT single-camera JSON file or an MOT-style .txt file
    (frame_id, track_id, x, y, width, height, score, class_id).
    """
    path = str(path)
    if path.endswith(".json"):
        return load_single_camera_json_as_dataframe(path)
    df = pd.read_csv(
        path,
        header=None,
        names=["frame_id", "track_id", "x", "y", "width", "height", "score", "class_id"],
    )
    return df


_GT_CACHE: dict[str, dict] = {}


def _load_gt_json(gt_json_path: str) -> dict:
    """Parse the (large) ground-truth JSON once and cache it across cameras."""
    if gt_json_path not in _GT_CACHE:
        with open(gt_json_path, "r", encoding="utf-8") as file:
            _GT_CACHE[gt_json_path] = json.load(file)
    return _GT_CACHE[gt_json_path]


def load_gt_camera(gt_json_path: str, camera: str) -> pd.DataFrame:
    """Load ground truth for one camera as frame/track/xywh.

    The AIC25 ground truth lists each object's 2D box only for the cameras where
    it is visible, so objects absent from `camera` at a frame are simply not
    returned (which is exactly the occlusion/out-of-view signal we want).
    The parsed JSON is cached so a multi-camera sweep parses the 48 MB file once.
    """
    data = _load_gt_json(gt_json_path)

    rows = []
    for frame_key, objects in data.items():
        frame_id = int(frame_key)
        for obj in objects:
            box = obj.get("2d bounding box visible", {}).get(camera)
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            rows.append(
                {
                    "frame_id": frame_id,
                    "track_id": int(obj["object id"]),
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                }
            )
    if not rows:
        raise ValueError(f"No ground-truth boxes found for camera '{camera}'.")
    return pd.DataFrame(rows).sort_values(["frame_id", "track_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Metrics vs ground truth
# --------------------------------------------------------------------------- #

METRIC_KEYS = [
    "idf1", "idp", "idr", "recall", "precision",
    "num_unique_objects", "mostly_tracked", "mostly_lost",
    "num_fragmentations", "num_switches",
    "num_false_positives", "num_misses", "mota", "motp",
]


def _iou_distance_matrix(gt_boxes: np.ndarray, hyp_boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Return an (n_gt, n_hyp) distance matrix (1 - IoU), NaN where IoU < threshold.

    Boxes are in [x, y, width, height]. NaN marks pairs too far apart to match,
    which is the convention motmetrics expects.
    """
    n, m = len(gt_boxes), len(hyp_boxes)
    dist = np.full((n, m), np.nan, dtype=float)
    for i in range(n):
        for j in range(m):
            iou = _iou(gt_boxes[i], hyp_boxes[j])
            if iou >= iou_threshold:
                dist[i, j] = 1.0 - iou
    return dist


def evaluate_against_gt(
    gt: pd.DataFrame,
    hyp: pd.DataFrame,
    iou_threshold: float = 0.5,
    frame_window: tuple[int, int] | None = None,
) -> dict:
    """Compute MOT metrics for one hypothesis DataFrame against ground truth.

    If `frame_window` is given, only GT/hyp frames inside [lo, hi] are scored.
    This keeps absolute metrics meaningful when the tracker covers only part of
    the sequence (e.g. a 1000-frame cap against 9000-frame ground truth).
    """
    if frame_window is not None:
        lo, hi = frame_window
        gt = gt[(gt["frame_id"] >= lo) & (gt["frame_id"] <= hi)]
        hyp = hyp[(hyp["frame_id"] >= lo) & (hyp["frame_id"] <= hi)]

    acc = mm.MOTAccumulator(auto_id=False)
    frames = sorted(set(gt["frame_id"]).union(set(hyp["frame_id"])))

    for frame in frames:
        g = gt[gt["frame_id"] == frame]
        h = hyp[hyp["frame_id"] == frame]
        gt_ids = g["track_id"].tolist()
        hyp_ids = h["track_id"].tolist()
        gt_boxes = g[BOX_COLS].to_numpy(dtype=float).reshape(-1, 4)
        hyp_boxes = h[BOX_COLS].to_numpy(dtype=float).reshape(-1, 4)
        if not gt_ids and not hyp_ids:
            continue
        dist = _iou_distance_matrix(gt_boxes, hyp_boxes, iou_threshold)
        acc.update(gt_ids, hyp_ids, dist, frameid=frame)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=METRIC_KEYS, name="acc")
    return {key: summary.loc["acc", key] for key in METRIC_KEYS}


def compare_against_gt(
    gt: pd.DataFrame,
    baseline: pd.DataFrame,
    repaired: pd.DataFrame,
    iou_threshold: float = 0.5,
    clip_to_hyp_frames: bool = True,
) -> dict:
    """Evaluate baseline and repaired tracks against GT and report deltas.

    By default the evaluation is clipped to the frame window the tracker actually
    covered (union of baseline + repaired), so a partial run is not unfairly
    penalised for frames it never processed.
    """
    window = None
    if clip_to_hyp_frames:
        frames = pd.concat([baseline["frame_id"], repaired["frame_id"]])
        window = (int(frames.min()), int(frames.max()))

    base_metrics = evaluate_against_gt(gt, baseline, iou_threshold, window)
    rep_metrics = evaluate_against_gt(gt, repaired, iou_threshold, window)
    deltas = {key: rep_metrics[key] - base_metrics[key] for key in METRIC_KEYS}
    return {
        "baseline": base_metrics,
        "repaired": rep_metrics,
        "delta": deltas,
        "frame_window": window,
    }


# --------------------------------------------------------------------------- #
# Gap root-cause analysis
# --------------------------------------------------------------------------- #

def load_detections_txt(path: str) -> dict[int, list[tuple[np.ndarray, float]]]:
    """Load raw detections (Camera,frame,cls,x1,y1,x2,y2,score) per frame."""
    detections: dict[int, list[tuple[np.ndarray, float]]] = {}
    if not Path(path).exists():
        return detections
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame = int(parts[1])
            x1, y1, x2, y2 = (float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6]))
            score = float(parts[7])
            box = np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)
            detections.setdefault(frame, []).append((box, score))
    return detections


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match_track_to_gt(track: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float) -> int | None:
    """Vote for the GT object id that best overlaps this tracklet over time."""
    votes: dict[int, int] = {}
    for _, row in track.iterrows():
        g = gt[gt["frame_id"] == row["frame_id"]]
        if g.empty:
            continue
        tbox = row[BOX_COLS].to_numpy(dtype=float)
        best_id, best_iou = None, iou_threshold
        for _, grow in g.iterrows():
            iou = _iou(tbox, grow[BOX_COLS].to_numpy(dtype=float))
            if iou >= best_iou:
                best_id, best_iou = int(grow["track_id"]), iou
        if best_id is not None:
            votes[best_id] = votes.get(best_id, 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)


GAP_CATEGORIES = (
    "occlusion_or_exit",        # GT object not visible in this camera during the gap
    "missed_detection",         # GT visible, but the detector produced no box
    "low_confidence_detection", # detector saw it but below the tracking threshold
    "association_failure",      # a confident detection existed; tracker failed to link it
    "unmatched_tracklet",       # tracklet could not be matched to any GT id
)


def classify_gaps(
    hyp: pd.DataFrame,
    gt: pd.DataFrame,
    detections: dict[int, list[tuple[np.ndarray, float]]],
    iou_threshold: float = 0.5,
    det_iou_threshold: float = 0.3,
    confidence_threshold: float = 0.5,
) -> dict:
    """Classify every internal gap in the hypothesis tracklets by root cause."""
    counts = {category: 0 for category in GAP_CATEGORIES}
    per_gap = []

    for track_id, track in hyp.groupby("track_id", sort=False):
        track = track.sort_values("frame_id").reset_index(drop=True)
        gt_id = _match_track_to_gt(track, gt, iou_threshold)

        for i in range(len(track) - 1):
            prev, nxt = track.iloc[i], track.iloc[i + 1]
            missing = int(nxt["frame_id"] - prev["frame_id"] - 1)
            if missing <= 0:
                continue

            for step in range(1, missing + 1):
                frame = int(prev["frame_id"] + step)
                ratio = step / (missing + 1)
                expected = np.array(
                    [prev[c] + ratio * (nxt[c] - prev[c]) for c in BOX_COLS], dtype=float
                )

                if gt_id is None:
                    category = "unmatched_tracklet"
                else:
                    visible = not gt[
                        (gt["frame_id"] == frame) & (gt["track_id"] == gt_id)
                    ].empty
                    if not visible:
                        category = "occlusion_or_exit"
                    else:
                        best_score = 0.0
                        for box, score in detections.get(frame, []):
                            if _iou(expected, box) >= det_iou_threshold:
                                best_score = max(best_score, score)
                        if best_score == 0.0:
                            category = "missed_detection"
                        elif best_score < confidence_threshold:
                            category = "low_confidence_detection"
                        else:
                            category = "association_failure"

                counts[category] += 1
                per_gap.append(
                    {"track_id": int(track_id), "frame": frame, "gt_id": gt_id, "cause": category}
                )

    total = sum(counts.values())
    return {
        "total_gap_frames": total,
        "counts": counts,
        "percent": {k: (100.0 * v / total if total else 0.0) for k, v in counts.items()},
        "detail": per_gap,
    }


# --------------------------------------------------------------------------- #
# Orchestration + reporting
# --------------------------------------------------------------------------- #

def run(
    gt_json: str,
    camera: str,
    baseline: str,
    repaired: str,
    detections_txt: str | None,
    output_dir: str,
    iou_threshold: float = 0.5,
) -> dict:
    """Run the full GT benchmark + gap analysis and write a report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gt_df = load_gt_camera(gt_json, camera)
    base_df = load_tracks(baseline)
    rep_df = load_tracks(repaired)

    metrics = compare_against_gt(gt_df, base_df, rep_df, iou_threshold)

    detections = load_detections_txt(detections_txt) if detections_txt else {}
    gaps = classify_gaps(base_df, gt_df, detections, iou_threshold)

    report = {
        "camera": camera,
        "iou_threshold": iou_threshold,
        "inputs": {"baseline": baseline, "repaired": repaired, "detections": detections_txt},
        "metrics": metrics,
        "gap_root_cause": {k: v for k, v in gaps.items() if k != "detail"},
    }
    (out / "gt_benchmark.json").write_text(
        json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, out / "gt_benchmark.md")
    _print_summary(report)
    return report


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def _write_markdown(report: dict, path: Path) -> None:
    m = report["metrics"]
    lines = [
        f"# Single-Camera GT Benchmark — {report['camera']}",
        "",
        f"IoU threshold: {report['iou_threshold']}",
        "",
        "## Metrics vs ground truth (baseline vs repaired)",
        "",
        "| metric | baseline | repaired | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in METRIC_KEYS:
        lines.append(
            f"| {key} | {_fmt(m['baseline'][key])} | {_fmt(m['repaired'][key])} | {_fmt(m['delta'][key])} |"
        )
    g = report["gap_root_cause"]
    lines += [
        "",
        "## Internal-gap root cause (baseline tracklets)",
        "",
        f"Total gap frames: {g['total_gap_frames']}",
        "",
        "| cause | count | percent |",
        "| --- | ---: | ---: |",
    ]
    for cause in GAP_CATEGORIES:
        lines.append(f"| {cause} | {g['counts'][cause]} | {g['percent'][cause]:.1f}% |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(report: dict) -> None:
    m = report["metrics"]
    print(f"\n=== GT benchmark: {report['camera']} ===")
    for key in ["idf1", "mota", "num_switches", "num_fragmentations"]:
        print(f"  {key:20s} baseline={_fmt(m['baseline'][key])}  repaired={_fmt(m['repaired'][key])}  Δ={_fmt(m['delta'][key])}")
    g = report["gap_root_cause"]
    print(f"  gap root cause ({g['total_gap_frames']} gap frames):")
    for cause in GAP_CATEGORIES:
        if g["counts"][cause]:
            print(f"    {cause:26s} {g['counts'][cause]:5d}  ({g['percent'][cause]:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-camera GT benchmark + gap root-cause.")
    parser.add_argument("--gt-json", required=True, help="Path to ground_truth.json")
    parser.add_argument("--camera", required=True, help="Camera name, e.g. Camera_02")
    parser.add_argument("--baseline", required=True, help="Baseline tracker JSON or MOT .txt")
    parser.add_argument("--repaired", required=True, help="Repaired tracker JSON or MOT .txt")
    parser.add_argument("--detections-txt", default=None, help="Raw detection .txt (for gap cause)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        gt_json=args.gt_json,
        camera=args.camera,
        baseline=args.baseline,
        repaired=args.repaired,
        detections_txt=args.detections_txt,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()
