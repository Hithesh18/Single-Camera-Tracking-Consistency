"""Tune BoT-SORT tracker parameters against ground truth (Direction 1).

For each parameter combination this re-runs single-camera tracking, evaluates the
raw tracker output against the ground truth (IDF1, MOTA, ID switches,
fragmentations), and ranks the combinations. This is the "tune BoT-SORT
parameters (matching thresholds, motion models)" direction: the tracker's own
matching thresholds are searched, not just the post-processing.

Tracking re-runs need the detections + embeddings to already exist (produced by
the detection/embedding steps), so run this after those steps — locally if the
tracker runs on CPU, otherwise on Colab.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

from tracklet_repair.src.evaluation.gt_benchmark import (
    evaluate_against_gt,
    load_gt_camera,
    load_tracks,
)

# Parameters that BoTSORT actually consumes (see BoT-SORT/tracker/bot_sort.py).
TUNABLE = ["track_high_thresh", "track_low_thresh", "new_track_thresh",
           "track_buffer", "match_thresh", "proximity_thresh", "appearance_thresh"]

DEFAULT_GRID = {
    "match_thresh": [0.7, 0.8, 0.9],
    "track_buffer": [30, 60, 90],
    "track_high_thresh": [0.4, 0.5, 0.6],
}


def _track_one(python: str, scene: str, dataset: str, camera: str,
               params: dict, limit_frames: int | None) -> bool:
    cmd = [python, "BoT-SORT/single_camera_tracking.py", "-s", scene, "-c", camera, "--dataset", dataset]
    if limit_frames:
        cmd += ["--limit_frames", str(limit_frames)]
    for key, value in params.items():
        cmd += [f"--{key}", str(value)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print("\n".join(result.stdout.strip().splitlines()[-15:]))
    return result.returncode == 0


def _score_combo(gt_json: str, scene: str, cameras: list[str], tracking_dir: str,
                 iou_threshold: float) -> dict:
    """Aggregate raw-output metrics across cameras for the current tracking run."""
    keys = ["idf1", "mota", "num_switches", "num_fragmentations"]
    totals = {k: 0.0 for k in keys}
    n = 0
    for camera in cameras:
        raw_path = Path(tracking_dir) / "Singlecamera" / scene / camera / f"{camera}.json"
        if not raw_path.exists():
            continue
        raw = load_tracks(str(raw_path))
        gt = load_gt_camera(gt_json, camera)
        window = (int(raw["frame_id"].min()), int(raw["frame_id"].max()))
        m = evaluate_against_gt(gt, raw, iou_threshold, window)
        for k in keys:
            totals[k] += m[k]
        n += 1
    if n == 0:
        return {k: float("nan") for k in keys}
    totals["idf1"] /= n
    totals["mota"] /= n
    return totals


def run_sweep(python: str, gt_json: str, scene: str, dataset: str, cameras: list[str],
              grid: dict, tracking_dir: str, output_dir: str,
              limit_frames: int | None, iou_threshold: float) -> list[dict]:
    keys = list(grid.keys())
    combos = [dict(zip(keys, values)) for values in itertools.product(*grid.values())]
    print(f"Sweeping {len(combos)} combinations over {keys} on cameras {cameras}")

    results = []
    for i, params in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] params={params}")
        ok = all(_track_one(python, scene, dataset, cam, params, limit_frames) for cam in cameras)
        if not ok:
            print("  tracking failed, skipping")
            continue
        metrics = _score_combo(gt_json, scene, cameras, tracking_dir, iou_threshold)
        print(f"  IDF1={metrics['idf1']:.3f} MOTA={metrics['mota']:.3f} "
              f"IDSW={metrics['num_switches']:.0f} Frag={metrics['num_fragmentations']:.0f}")
        results.append({"params": params, "metrics": metrics})

    results.sort(key=lambda r: (-r["metrics"]["idf1"], r["metrics"]["num_switches"]))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "botsort_sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_markdown(results, keys, out / "botsort_sweep.md")
    if results:
        print(f"\nBest: {results[0]['params']}  IDF1={results[0]['metrics']['idf1']:.3f}")
    print(f"Saved sweep to {out}")
    return results


def _write_markdown(results: list[dict], keys: list[str], path: Path) -> None:
    lines = ["# BoT-SORT Parameter Sweep (ranked by IDF1)", "",
             "| " + " | ".join(keys) + " | IDF1 | MOTA | ID switches | Fragmentations |",
             "| " + " | ".join(["---"] * (len(keys) + 4)) + " |"]
    for r in results:
        row = [str(r["params"][k]) for k in keys]
        m = r["metrics"]
        row += [f"{m['idf1']:.3f}", f"{m['mota']:.3f}", f"{m['num_switches']:.0f}", f"{m['num_fragmentations']:.0f}"]
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune BoT-SORT params against ground truth.")
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset", default="Val")
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--grid", default=None, help="JSON file mapping param -> list of values")
    parser.add_argument("--tracking-dir", default="Tracking")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    grid = json.loads(Path(args.grid).read_text()) if args.grid else DEFAULT_GRID
    unknown = set(grid) - set(TUNABLE)
    if unknown:
        raise ValueError(f"Unknown tunable params: {unknown}. Allowed: {TUNABLE}")

    run_sweep(args.python, args.gt_json, args.scene, args.dataset, args.cameras, grid,
              args.tracking_dir, args.output_dir, args.limit_frames, args.iou_threshold)


if __name__ == "__main__":
    main()
