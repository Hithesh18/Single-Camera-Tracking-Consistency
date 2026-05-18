"""
Convert ground_truth.json and tracker output into the format TrackEval expects.

Usage:
    python prepare_eval_data.py --scene Warehouse_016 --exp exp1 --dataset Val
"""
import os
import json
import argparse
import numpy as np


def convert_gt(gt_json_path, out_path):
    """Convert ground_truth.json to TrackEval raw format.

    Output line format (space-separated, 11 fields):
        scene_id  cam_id  global_id  frame_id  x  y  z  l  w  h  conf
    Fields 0,1,10 are ignored by main.py — we fill them with 0.
    """
    with open(gt_json_path) as f:
        gt = json.load(f)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []
    for frame_str, objects in gt.items():
        frame_id = int(frame_str) + 1  # GT is 0-indexed, TrackEval expects 1-indexed
        for obj in objects:
            gid = obj['object id']
            x, y, z = obj['3d location']
            l, w, h = obj['3d bounding box scale']
            lines.append(f"0 0 {gid} {frame_id} {x:.6f} {y:.6f} {z:.6f} {l:.6f} {w:.6f} {h:.6f} 1.0")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"GT written: {out_path}  ({len(lines)} detections, {len(gt)} frames)")


def convert_tracker(tracker_json_path, out_path):
    """Convert fixed_whole_tracking_results.json (multi-camera) to TrackEval raw format.

    Output line format (space-separated, 11 fields):
        scene_id  cam_id  global_id  frame_id  x  y  z  l  w  h  conf
    """
    with open(tracker_json_path) as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []

    for cam_id, frames in data.items():
        for frame_str, objects in frames.items():
            frame_id = int(frame_str)
            if not isinstance(objects, list):
                objects = [objects]
            for obj in objects:
                gid = obj.get('GlobalOfflineID', obj.get('object mc id', obj.get('object sc id', -1)))
                loc = obj.get('3d location', [0, 0, 0])
                scale = obj.get('3d bounding box scale', [1, 1, 1])
                x, y, z = loc[0], loc[1], loc[2]
                l, w, h = scale[0], scale[1], scale[2]
                lines.append(f"0 0 {gid} {frame_id} {x:.6f} {y:.6f} {z:.6f} {l:.6f} {w:.6f} {h:.6f} 0.9")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Tracker written: {out_path}  ({len(lines)} detections)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene', type=str, required=True, help='Scene ID e.g. Warehouse_016')
    parser.add_argument('--exp', type=str, default='exp1', help='Experiment name used as tracker/seq label')
    parser.add_argument('--dataset', type=str, default='Val')
    parser.add_argument('--base_dir', type=str, default='..')
    args = parser.parse_args()

    base = args.base_dir
    scene = args.scene
    exp = args.exp

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt_json = os.path.join(base, 'AIC25_Track1', args.dataset, scene, 'ground_truth.json')
    gt_out  = os.path.join('aicity_25_data', scene, 'gt.txt')
    if os.path.exists(gt_json):
        convert_gt(gt_json, gt_out)
    else:
        print(f"[WARN] GT not found: {gt_json}")

    # ── Tracker output ────────────────────────────────────────────────────────
    tracker_json = os.path.join(base, 'Tracking', 'Multicamera', scene, exp,
                                'fixed_whole_tracking_results.json')
    tracker_out  = os.path.join('aicity_25_data', scene, f'{exp}.txt')
    if os.path.exists(tracker_json):
        convert_tracker(tracker_json, tracker_out)
    else:
        print(f"[WARN] Tracker output not found: {tracker_json}")
        print("       Run multi_camera_fix.py first to generate it.")

    print("\nDone. Now run:  python main.py --scene", scene, "--exp", exp)


if __name__ == '__main__':
    main()
