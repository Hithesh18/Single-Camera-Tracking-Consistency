import os
import cv2
from multiprocessing import Pool
import argparse
import os.path as osp
import shutil

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def make_parser():
    parser = argparse.ArgumentParser("extract frame")
    parser.add_argument("root_path", type=str, default=None)
    parser.add_argument("-s", "--scene", type=str, default=None)
    return parser

args = make_parser().parse_args()
data_root = args.root_path
scene = args.scene

IMAGE_FORMAT = ".jpg"
LOG_EVERY = 300   # print a progress line every N frames when tqdm is unavailable


def video2image(parameter_set):
    worker_id, total_workers, scenario, camera, camera_dir = parameter_set
    imgs_dir = f"{camera_dir}/Frame"

    if os.path.exists(imgs_dir):
        print(f"[SKIP {worker_id+1}/{total_workers}] {camera} — Frame/ already exists", flush=True)
        return

    os.makedirs(imgs_dir)

    video_file = None
    for f in os.listdir(camera_dir):
        if f.endswith(".mp4"):
            video_file = os.path.join(camera_dir, f)
            break

    if video_file is None:
        print(f"[ERROR] No .mp4 found in {camera_dir}", flush=True)
        return

    cap = cv2.VideoCapture(video_file)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    desc = f"[{worker_id+1}/{total_workers}] {camera}"

    print(f"[START] {desc}  ({total} frames)", flush=True)

    current = 1
    ret, frame = cap.read()

    if HAS_TQDM:
        pbar = _tqdm(total=total or None, desc=desc, unit="fr",
                     dynamic_ncols=True, leave=True, mininterval=2.0)

    while ret:
        cv2.imwrite(f"{imgs_dir}/{str(current).zfill(6)}{IMAGE_FORMAT}", frame)
        ret, frame = cap.read()
        if HAS_TQDM:
            pbar.update(1)
        elif current % LOG_EVERY == 0:
            pct = f"{100*current//total}%" if total else "?"
            print(f"  {desc}  {current}/{total}  ({pct})", flush=True)
        current += 1

    cap.release()

    if HAS_TQDM:
        pbar.close()

    print(f"[DONE ] {desc}  — {current-1} frames saved", flush=True)


def main():
    scenario_dir = osp.join(data_root, scene + "/videos")
    if not osp.exists(scenario_dir):
        print(f"[ERROR] Directory not found: {scenario_dir}")
        return

    # Move any loose .mp4 files into their own sub-folder
    for fname in sorted(os.listdir(scenario_dir)):
        if not fname.endswith(".mp4"):
            continue
        video_name = osp.splitext(fname)[0]
        target_dir  = osp.join(scenario_dir, video_name)
        target_path = osp.join(target_dir, fname)
        if not osp.exists(target_dir):
            os.makedirs(target_dir)
        shutil.move(osp.join(scenario_dir, fname), target_path)
        print(f"Moved {fname} → {target_path}", flush=True)

    cameras = sorted(
        c for c in os.listdir(scenario_dir)
        if "map" not in c and osp.isdir(osp.join(scenario_dir, c))
    )
    total_workers = len(cameras)
    print(f"\n[INFO] Extracting frames from {total_workers} cameras in scene '{scene}' …\n",
          flush=True)

    parameter_sets = [
        [idx, total_workers, scene, cam, osp.join(scenario_dir, cam)]
        for idx, cam in enumerate(cameras)
    ]

    pool = Pool(processes=total_workers)
    pool.map(video2image, parameter_sets)
    pool.close()
    pool.join()

    print(f"\n[INFO] All {total_workers} cameras complete.", flush=True)


if __name__ == "__main__":
    main()
