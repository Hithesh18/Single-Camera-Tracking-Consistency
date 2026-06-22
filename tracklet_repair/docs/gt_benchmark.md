# Ground-Truth Benchmark + Gap Root-Cause (Deliverable 1)

Addresses supervisor feedback: *"test against and compare with ground truth"* and
*"internal gaps — how is it coming?"*, and the goal's *"more stable tracklets,
fewer ID switches"*. CPU-only; runs on existing single-camera outputs.

## Modules

- `tracklet_repair/src/evaluation/gt_benchmark.py` — one camera: evaluate a
  baseline and a repaired track file against ground truth, and classify gaps.
- `tracklet_repair/src/evaluation/benchmark_scene.py` — whole scene: apply
  tracklet_repair to each camera's raw output, benchmark raw vs repaired vs GT,
  aggregate, and report gap root causes.

## What it measures (vs ground truth)

Uses `motmetrics`. Key metrics: **IDF1**, **MOTA/MOTP**, **num_switches (IDSW)**,
**num_fragmentations (Frag)**, MT/ML, FP/FN, precision/recall. IDSW is the direct
"fewer ID switches" number; Frag is the direct "tracks break less" number.

The GT (`ground_truth.json`) and the tracker JSON share a schema: frame-indexed,
each object has `2d bounding box visible[<camera>]` ([x1,y1,x2,y2]) and an id
(`object id` in GT, `object sc id` in tracker output). Matching is IoU >= 0.5.

Evaluation is clipped to the frame window the tracker actually covered
(`clip_to_hyp_frames`), so a capped run (e.g. 1000 frames) is not penalised for
the other 8000 GT frames.

## Gap root cause — why tracks break

For each internal gap in a tracklet, the missing frames are classified using the
GT's per-camera visibility (it lists a box only for cameras that can see the
object) and the raw detection file:

| cause | meaning |
| --- | --- |
| `occlusion_or_exit` | GT object not visible in this camera during the gap → genuine break |
| `missed_detection` | GT visible, but the detector produced no box |
| `low_confidence_detection` | detector saw it, but below the tracking threshold (0.5) |
| `association_failure` | a confident detection existed; the tracker failed to link it |
| `unmatched_tracklet` | tracklet could not be matched to any GT id |

This turns "20 tracklets have gaps" into an actionable breakdown of *why*.

## Run

```bash
# One camera
python -m tracklet_repair.src.evaluation.gt_benchmark \
  --gt-json AIC25_Track1/Val/Warehouse_016/ground_truth.json --camera Camera \
  --baseline Tracking/Singlecamera/Warehouse_016/Camera/Camera.json \
  --repaired Tracking/Singlecamera/Warehouse_016/Camera/fixed_Camera.json \
  --detections-txt Detection/Warehouse_016/Camera.txt \
  --output-dir tracklet_repair/results/gt_benchmark/Camera

# Whole scene (applies tracklet_repair to each raw output)
python -m tracklet_repair.src.evaluation.benchmark_scene \
  --gt-json AIC25_Track1/Val/Warehouse_016/ground_truth.json \
  --scene Warehouse_016 --cameras Camera Camera_01 Camera_02 \
  --output-dir tracklet_repair/results/scene_benchmark/Warehouse_016
```

## Local validation (camera `Camera`, 100-frame sample)

Smoke-tested end to end. Repair reduced fragmentations 2 -> 1; IDSW 0 -> 0;
IDF1 ~0.632. Gap root cause: association_failure 77%, missed_detection 13%,
low_confidence 10%, occlusion 0% — i.e. on this sample most breaks are the
tracker dropping a confident detection, not true occlusion. (Sample is tiny;
the real numbers come from full-length Colab runs.)

## Note on dependency

`motmetrics 1.4.0` calls `np.asfarray`, removed in NumPy 2.0, so we build the
IoU distance matrix ourselves and use motmetrics only for the accumulator/metrics.
