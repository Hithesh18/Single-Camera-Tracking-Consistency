import os
import argparse
from trackeval import metrics
from trackeval.eval import Evaluator
from trackeval.datasets.aicity_3d import AICity3D


def convert_to_trackeval_format(input_path, output_path, is_gt=True):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) != 11:
                continue
            _, _, global_id, frame_id, x, y, z, w, l, h, _ = parts
            confidence = 1.0 if is_gt else 0.9
            line_out = (f"{int(frame_id)},{int(global_id)},"
                        f"{float(x):.6f},{float(y):.6f},{float(z):.6f},"
                        f"{float(l):.6f},{float(w):.6f},{float(h):.6f},"
                        f"{confidence:.6f},-1,-1,-1\n")
            outfile.write(line_out)
            count += 1

    print(f"Converted {input_path} -> {output_path}  ({count} lines)")


def run_hota(scene, exp):
    dataset = AICity3D(config={
        'GT_FOLDER':       f'data/{scene}/gt',
        'TRACKERS_FOLDER': f'data/{scene}/tracker',
        'SEQ_TO_EVAL':     [exp],
        'TRACKERS_TO_EVAL':[exp],
        'GT_LOC_FORMAT':   '{gt_folder}/gt.txt',
    })

    evaluator = Evaluator({
        'PRINT_CONFIG': False,
        'DISPLAY_LESS_PROGRESS': True,
    })

    evaluator.evaluate([dataset], [metrics.HOTA3D()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene', type=str, default='Warehouse_016')
    parser.add_argument('--exp',         type=str, default='exp1',
                        help='Experiment name — must match prepare_eval_data.py --exp')
    args = parser.parse_args()

    raw_gt      = f'aicity_25_data/{args.scene}/gt.txt'
    raw_tracker = f'aicity_25_data/{args.scene}/{args.exp}.txt'
    out_gt      = f'data/{args.scene}/gt/gt.txt'
    out_tracker = f'data/{args.scene}/tracker/{args.exp}/{args.exp}.txt'

    if not os.path.exists(raw_gt):
        print(f"[ERROR] GT file missing: {raw_gt}")
        print("        Run:  python prepare_eval_data.py -s", args.scene)
        return
    if not os.path.exists(raw_tracker):
        print(f"[ERROR] Tracker file missing: {raw_tracker}")
        print("        Run:  python prepare_eval_data.py -s", args.scene, "--exp", args.exp)
        return

    convert_to_trackeval_format(raw_gt, out_gt, is_gt=True)
    convert_to_trackeval_format(raw_tracker, out_tracker, is_gt=False)
    run_hota(args.scene, args.exp)


if __name__ == '__main__':
    main()
