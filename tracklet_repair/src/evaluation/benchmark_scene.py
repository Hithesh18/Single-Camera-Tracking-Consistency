"""Scene-level GT benchmark: raw vs tracklet_repair, aggregated over cameras.

For every camera in a scene this:
  1. loads the raw single-camera tracker JSON,
  2. produces a repaired version with tracklet_repair (merge -> interpolate),
  3. evaluates raw vs repaired against the ground truth (IDF1, MOTA, ID switches,
     fragmentations, ...),
  4. classifies the root cause of every internal gap in the raw output,
and finally writes one aggregate report across all cameras.

This is the headline experiment for Subproject 1: it shows, with ground truth,
whether the repair step makes single-camera tracks more consistent, and explains
why tracks break in the first place. CPU-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tracklet_repair.src.evaluation.gt_benchmark import (
    GAP_CATEGORIES,
    METRIC_KEYS,
    classify_gaps,
    compare_against_gt,
    load_detections_txt,
    load_gt_camera,
    load_tracks,
)
from tracklet_repair.src.postprocess.merge import merge_tracklets
from tracklet_repair.src.postprocess.repair import interpolate_track_gaps


def repair_tracks(
    raw: pd.DataFrame,
    max_gap: int,
    max_merge_gap: int,
    max_center_distance: float,
    max_size_ratio: float,
    merge_mode: str,
    matcher_path: str | None = None,
    embed_dir: str | None = None,
    prob_threshold: float = 0.5,
) -> pd.DataFrame:
    """Apply the tracklet_repair pipeline (merge then interpolate).

    With merge_mode='learned', the trained matcher decides merges using deep
    ReID appearance; matcher_path and embed_dir must be supplied.
    """
    merged, _ = merge_tracklets(
        raw,
        merge_mode=merge_mode,
        max_merge_gap=max_merge_gap,
        max_center_distance=max_center_distance,
        max_size_ratio=max_size_ratio,
        matcher_path=matcher_path,
        embed_dir=embed_dir,
        prob_threshold=prob_threshold,
    )
    return interpolate_track_gaps(merged, max_gap=max_gap)


def discover_cameras(tracking_dir: Path, scene: str) -> list[str]:
    """Find all cameras with a single-camera tracking JSON for this scene."""
    scene_dir = tracking_dir / "Singlecamera" / scene
    if not scene_dir.is_dir():
        return []
    cameras = []
    for child in sorted(scene_dir.iterdir()):
        if child.is_dir() and (child / f"{child.name}.json").exists():
            cameras.append(child.name)
        elif child.suffix == ".json":
            cameras.append(child.stem)
    return cameras


def _find_raw_json(tracking_dir: Path, scene: str, camera: str) -> Path | None:
    candidates = [
        tracking_dir / "Singlecamera" / scene / camera / f"{camera}.json",
        tracking_dir / "Singlecamera" / scene / f"{camera}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_scene(
    gt_json: str,
    scene: str,
    cameras: list[str],
    tracking_dir: str,
    detection_dir: str | None,
    output_dir: str,
    max_gap: int = 5,
    max_merge_gap: int = 10,
    max_center_distance: float = 80.0,
    max_size_ratio: float = 1.5,
    merge_mode: str = "conservative",
    iou_threshold: float = 0.5,
    matcher_path: str | None = None,
    embed_root: str | None = None,
    prob_threshold: float = 0.5,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracking_path = Path(tracking_dir)

    if not cameras:
        cameras = discover_cameras(tracking_path, scene)
        print(f"Auto-discovered {len(cameras)} cameras: {cameras}")

    per_camera = {}
    for camera in cameras:
        raw_path = _find_raw_json(tracking_path, scene, camera)
        if raw_path is None:
            print(f"[skip] {camera}: no raw single-camera JSON found")
            continue

        raw = load_tracks(str(raw_path))
        embed_dir = str(Path(embed_root) / scene / camera) if embed_root else None
        repaired = repair_tracks(
            raw, max_gap, max_merge_gap, max_center_distance, max_size_ratio, merge_mode,
            matcher_path=matcher_path, embed_dir=embed_dir, prob_threshold=prob_threshold,
        )

        gt = load_gt_camera(gt_json, camera)
        metrics = compare_against_gt(gt, raw, repaired, iou_threshold)

        detections = {}
        if detection_dir:
            det_path = Path(detection_dir) / scene / f"{camera}.txt"
            detections = load_detections_txt(str(det_path))
        gaps = classify_gaps(raw, gt, detections, iou_threshold)

        per_camera[camera] = {
            "metrics": metrics,
            "gap_root_cause": {k: v for k, v in gaps.items() if k != "detail"},
        }
        print(
            f"[ok] {camera}: IDF1 {metrics['baseline']['idf1']:.3f}->{metrics['repaired']['idf1']:.3f}"
            f" | IDSW {metrics['baseline']['num_switches']}->{metrics['repaired']['num_switches']}"
            f" | Frag {metrics['baseline']['num_fragmentations']}->{metrics['repaired']['num_fragmentations']}"
        )

    report = {"scene": scene, "iou_threshold": iou_threshold, "per_camera": per_camera}
    report["aggregate"] = _aggregate(per_camera)
    (out / "scene_benchmark.json").write_text(
        json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, out / "scene_benchmark.md")
    print(f"\nSaved scene report to {out}")
    return report


def _aggregate(per_camera: dict) -> dict:
    """Sum counting metrics and average rate metrics across cameras."""
    rate_keys = ["idf1", "idp", "idr", "recall", "precision", "mota", "motp"]
    sum_keys = [k for k in METRIC_KEYS if k not in rate_keys]
    agg = {"baseline": {}, "repaired": {}}
    cams = list(per_camera.values())
    if not cams:
        return agg
    for variant in ("baseline", "repaired"):
        for key in sum_keys:
            agg[variant][key] = sum(c["metrics"][variant][key] for c in cams)
        for key in rate_keys:
            agg[variant][key] = sum(c["metrics"][variant][key] for c in cams) / len(cams)
    gap_counts = {cat: sum(c["gap_root_cause"]["counts"][cat] for c in cams) for cat in GAP_CATEGORIES}
    agg["gap_counts"] = gap_counts
    return agg


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        f"# Scene GT Benchmark — {report['scene']}",
        "",
        f"IoU threshold: {report['iou_threshold']}",
        "",
        "## Per-camera: raw -> repaired (key metrics)",
        "",
        "| camera | IDF1 | MOTA | ID switches | Fragmentations |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cam, data in report["per_camera"].items():
        m = data["metrics"]
        lines.append(
            f"| {cam} "
            f"| {m['baseline']['idf1']:.3f} -> {m['repaired']['idf1']:.3f} "
            f"| {m['baseline']['mota']:.3f} -> {m['repaired']['mota']:.3f} "
            f"| {m['baseline']['num_switches']} -> {m['repaired']['num_switches']} "
            f"| {m['baseline']['num_fragmentations']} -> {m['repaired']['num_fragmentations']} |"
        )

    agg = report.get("aggregate", {})
    if agg.get("baseline"):
        lines += [
            "",
            "## Aggregate (sums for counts, mean for rates)",
            "",
            "| metric | baseline | repaired |",
            "| --- | ---: | ---: |",
        ]
        for key in METRIC_KEYS:
            lines.append(f"| {key} | {_fmt(agg['baseline'][key])} | {_fmt(agg['repaired'][key])} |")
        lines += [
            "",
            "## Internal-gap root cause (all cameras, raw output)",
            "",
            "| cause | count |",
            "| --- | ---: |",
        ]
        total = sum(agg["gap_counts"].values())
        for cause in GAP_CATEGORIES:
            count = agg["gap_counts"][cause]
            pct = f" ({100.0 * count / total:.1f}%)" if total else ""
            lines.append(f"| {cause} | {count}{pct} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scene-level GT benchmark: raw vs tracklet_repair.")
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--scene", required=True, help="e.g. Warehouse_016")
    parser.add_argument("--cameras", nargs="*", default=None, help="e.g. Camera Camera_01 Camera_02; omit to auto-discover all")
    parser.add_argument("--tracking-dir", default="Tracking", help="dir containing Singlecamera/<scene>/...")
    parser.add_argument("--detection-dir", default="Detection", help="dir with <scene>/<camera>.txt (gap cause)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--max-merge-gap", type=int, default=10)
    parser.add_argument("--max-center-distance", type=float, default=80.0)
    parser.add_argument("--max-size-ratio", type=float, default=1.5)
    parser.add_argument("--merge-mode", default="conservative", help="conservative|motion_aware|appearance_global|learned")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--matcher-path", default=None, help="trained matcher prefix (learned mode)")
    parser.add_argument("--embed-root", default="EmbedFeature", help="EmbedFeature root (learned mode)")
    parser.add_argument("--prob-threshold", type=float, default=0.5, help="learned-merge probability threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_scene(
        gt_json=args.gt_json,
        scene=args.scene,
        cameras=args.cameras,
        tracking_dir=args.tracking_dir,
        detection_dir=args.detection_dir,
        output_dir=args.output_dir,
        max_gap=args.max_gap,
        max_merge_gap=args.max_merge_gap,
        max_center_distance=args.max_center_distance,
        max_size_ratio=args.max_size_ratio,
        merge_mode=args.merge_mode,
        iou_threshold=args.iou_threshold,
        matcher_path=args.matcher_path,
        embed_root=args.embed_root,
        prob_threshold=args.prob_threshold,
    )


if __name__ == "__main__":
    main()
