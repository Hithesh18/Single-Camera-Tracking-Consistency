# Experiment Log

## Project Scaffold

The repository structure was created on branch `hakan/project-setup`.

Main folders:

- `src`
- `configs`
- `scripts`
- `docs`
- `results`
- `paper`
- `examples`

This step only prepared the project layout and placeholder code.

## Tracklet Statistics Analysis

Implemented tracking file loading, validation, and
`compute_tracklet_statistics`.

Test input:

```bash
examples/sample_tracks.txt
```

Main output before repair:

- `total_detections`: 11
- `num_tracklets`: 3
- `num_tracklets_with_gaps`: 1
- `total_internal_gaps`: 1
- `mean_tracklet_length`: 3.67

The sample contains one track with an internal frame gap.

## Short-Gap Interpolation

Implemented interpolation for small missing frame gaps inside the same
`track_id`.

Test command:

```bash
python -m src.postprocess.run_postprocess --input examples/sample_tracks.txt --output results/postprocess/sample_repaired.txt --max-gap 5
```

Result:

- original detections: 11
- repaired detections: 13
- interpolated detections: 2

Analysis after repair:

- `total_detections`: 13
- `num_tracklets`: 3
- `num_tracklets_with_gaps`: 0
- `total_internal_gaps`: 0
- `mean_tracklet_length`: 4.33

The synthetic sample gap was removed without changing the number of tracklets.

## Conservative Tracklet Merging

Implemented conservative merging for fragmented tracklets. This is verified on
a synthetic example; real-data evaluation is still required.

Test input:

```bash
examples/sample_fragmented_tracks.txt
```

Test command:

```bash
python -m src.postprocess.run_postprocess --input examples/sample_fragmented_tracks.txt --output results/postprocess/fragmented_repaired.txt --max-gap 5 --enable-merge --max-merge-gap 5 --max-center-distance 80 --max-size-ratio 1.5
```

Result:

- original detections: 12
- repaired detections: 14
- interpolated detections: 2
- merged tracklets: 1
- merge map: `{11: 10}`

Analysis before repair:

- `total_detections`: 12
- `num_tracklets`: 4
- `mean_tracklet_length`: 3.0
- `num_tracklets_with_gaps`: 0

Analysis after repair:

- `total_detections`: 14
- `num_tracklets`: 3
- `mean_tracklet_length`: 4.67
- `num_tracklets_with_gaps`: 0

The synthetic fragmented identity was merged into one longer tracklet, while
the number of detections increased because the remaining short gap was
interpolated.

## Baseline vs Repaired Comparison

Implemented a comparison script for baseline and repaired tracking outputs.

Test baseline:

```bash
examples/sample_fragmented_tracks.txt
```

Test repaired output:

```bash
results/postprocess/fragmented_repaired.txt
```

Test command:

```bash
python -m src.evaluation.evaluate_tracking --baseline examples/sample_fragmented_tracks.txt --repaired results/postprocess/fragmented_repaired.txt --output-json results/evaluation/fragmented_comparison.json --output-md results/evaluation/fragmented_comparison.md --short-threshold 10
```

Main comparison result:

- `total_detections`: 12 -> 14, diff +2
- `num_tracklets`: 4 -> 3, diff -1
- `mean_tracklet_length`: 3.00 -> 4.67, diff +1.67
- `median_tracklet_length`: 3.00 -> 3.00, diff 0.00
- `num_short_tracklets`: 4 -> 3, diff -1
- `percent_short_tracklets`: 100.00 -> 100.00, diff 0.00
- `num_tracklets_with_gaps`: 0 -> 0, diff 0
- `total_internal_gaps`: 0 -> 0, diff 0

The synthetic fragmented sample shows that the repair pipeline can reduce the
number of tracklets and increase mean tracklet length. The added detections come
from interpolation. This is still a synthetic sanity check, not a real-data
result.

## Tracklet Repair Ablation

Added a JSON ablation runner to compare the existing repair components on the
same input. This supports the final report by separating the effects of
interpolation only, conservative merging only, and the full repair pipeline.

Files added:

- `tracklet_repair/src/evaluation/run_ablation.py`
- `tracklet_repair/tests/test_ablation.py`

Test command:

```bash
python -m pytest tracklet_repair/tests -v
```

Test result: 14 tests passed.

Example command:

```bash
python -m tracklet_repair.src.evaluation.run_ablation --input-json tracklet_repair/examples/sample_single_camera.json --output-dir tracklet_repair/results/ablation/sample_single_camera --max-gap 5 --max-merge-gap 5 --max-center-distance 80 --max-size-ratio 1.5 --short-threshold 10
```

Example result:

- baseline: 8 detections, 4 tracklets, 1 internal gap
- interpolation only: 10 detections, 4 tracklets, 0 internal gaps
- merge only: 8 detections, 3 tracklets, 1 internal gap
- full repair: 10 detections, 3 tracklets, 0 internal gaps

Generated outputs:

- `baseline_tracks.txt`
- `interpolation_only_tracks.txt`
- `merge_only_tracks.txt`
- `full_repair_tracks.txt`
- `ablation.json`
- `ablation.md`

The comparison uses tracklet-level statistics only. It does not add ReID
matching or claim IDF1, HOTA, or MOTA improvements. Merge quality still depends
on the existing geometric thresholds. The consolidated table can later be used
as an ablation table in the final report.

## Real-Data Tracklet Repair Ablation

The four-way ablation was run at commit `6988fd5` on three raw single-camera
outputs and their corresponding fixed-output controls. Every run used the same
parameters: `max_gap=5`, `max_merge_gap=5`, `max_center_distance=80`,
`max_size_ratio=1.5`, and `short_threshold=10`.

Commands:

```bash
python -m tracklet_repair.src.evaluation.run_ablation --input-json tracklet_repair/local_inputs/Camera.json --output-dir tracklet_repair/results/ablation/Camera_raw --max-gap 5 --max-merge-gap 5 --max-center-distance 80 --max-size-ratio 1.5 --short-threshold 10

python -m tracklet_repair.src.evaluation.run_ablation --input-json tracklet_repair/local_inputs/fixed_Camera.json --output-dir tracklet_repair/results/ablation/Camera_fixed --max-gap 5 --max-merge-gap 5 --max-center-distance 80 --max-size-ratio 1.5 --short-threshold 10

python -m tracklet_repair.src.evaluation.run_ablation --input-json tracklet_repair/local_inputs/Camera_02_1000/Camera_02.json --output-dir tracklet_repair/results/ablation/Camera_02_raw_1000 --max-gap 5 --max-merge-gap 5 --max-center-distance 80 --max-size-ratio 1.5 --short-threshold 10

python -m tracklet_repair.src.evaluation.run_ablation --input-json tracklet_repair/local_inputs/Camera_02_1000/fixed_Camera_02.json --output-dir tracklet_repair/results/ablation/Camera_02_fixed_1000 --max-gap 5 --max-merge-gap 5 --max-center-distance 80 --max-size-ratio 1.5 --short-threshold 10

python -m tracklet_repair.src.evaluation.run_ablation --input-json tracklet_repair/local_inputs/Camera_03_1000/Camera_03.json --output-dir tracklet_repair/results/ablation/Camera_03_raw_1000 --max-gap 5 --max-merge-gap 5 --max-center-distance 80 --max-size-ratio 1.5 --short-threshold 10

python -m tracklet_repair.src.evaluation.run_ablation --input-json tracklet_repair/local_inputs/Camera_03_1000/fixed_Camera_03.json --output-dir tracklet_repair/results/ablation/Camera_03_fixed_1000 --max-gap 5 --max-merge-gap 5 --max-center-distance 80 --max-size-ratio 1.5 --short-threshold 10
```

Results:

| Input | Variant | Detections | Tracklets | Mean length | Median length | Short | Gap tracks | Internal gaps | Interpolated | Merged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Camera raw | baseline | 467 | 13 | 35.92 | 8.00 | 7 | 3 | 6 | 0 | 0 |
| Camera raw | interpolation only | 477 | 13 | 36.69 | 8.00 | 7 | 1 | 2 | 10 | 0 |
| Camera raw | merge only | 467 | 12 | 38.92 | 17.50 | 6 | 4 | 7 | 0 | 1 |
| Camera raw | full repair | 480 | 12 | 40.00 | 17.50 | 6 | 1 | 2 | 13 | 1 |
| Camera fixed | baseline | 486 | 6 | 81.00 | 92.50 | 0 | 0 | 0 | 0 | 0 |
| Camera fixed | interpolation only | 486 | 6 | 81.00 | 92.50 | 0 | 0 | 0 | 0 | 0 |
| Camera fixed | merge only | 486 | 6 | 81.00 | 92.50 | 0 | 0 | 0 | 0 | 0 |
| Camera fixed | full repair | 486 | 6 | 81.00 | 92.50 | 0 | 0 | 0 | 0 | 0 |
| Camera 02 raw, 1000 frames | baseline | 5225 | 44 | 118.75 | 10.50 | 22 | 15 | 53 | 0 | 0 |
| Camera 02 raw, 1000 frames | interpolation only | 5296 | 44 | 120.36 | 11.00 | 21 | 11 | 14 | 71 | 0 |
| Camera 02 raw, 1000 frames | merge only | 5225 | 40 | 130.62 | 32.00 | 18 | 17 | 57 | 0 | 4 |
| Camera 02 raw, 1000 frames | full repair | 5300 | 40 | 132.50 | 33.50 | 17 | 11 | 14 | 75 | 4 |
| Camera 02 fixed, 1000 frames | baseline | 4952 | 17 | 291.29 | 311.00 | 0 | 0 | 0 | 0 | 0 |
| Camera 02 fixed, 1000 frames | interpolation only | 4952 | 17 | 291.29 | 311.00 | 0 | 0 | 0 | 0 | 0 |
| Camera 02 fixed, 1000 frames | merge only | 4952 | 17 | 291.29 | 311.00 | 0 | 0 | 0 | 0 | 0 |
| Camera 02 fixed, 1000 frames | full repair | 4952 | 17 | 291.29 | 311.00 | 0 | 0 | 0 | 0 | 0 |
| Camera 03 raw, 1000-frame range | baseline | 3844 | 40 | 96.10 | 19.50 | 15 | 21 | 46 | 0 | 0 |
| Camera 03 raw, 1000-frame range | interpolation only | 3913 | 40 | 97.83 | 21.50 | 15 | 12 | 17 | 69 | 0 |
| Camera 03 raw, 1000-frame range | merge only | 3844 | 36 | 106.78 | 24.00 | 11 | 22 | 50 | 0 | 4 |
| Camera 03 raw, 1000-frame range | full repair | 3920 | 36 | 108.89 | 27.50 | 11 | 12 | 17 | 76 | 4 |
| Camera 03 fixed, 1000-frame range | baseline | 3961 | 20 | 198.05 | 83.00 | 0 | 0 | 0 | 0 | 0 |
| Camera 03 fixed, 1000-frame range | interpolation only | 3961 | 20 | 198.05 | 83.00 | 0 | 0 | 0 | 0 | 0 |
| Camera 03 fixed, 1000-frame range | merge only | 3961 | 20 | 198.05 | 83.00 | 0 | 0 | 0 | 0 | 0 |
| Camera 03 fixed, 1000-frame range | full repair | 3961 | 20 | 198.05 | 83.00 | 0 | 0 | 0 | 0 | 0 |

Each output directory contains:

- `baseline_tracks.txt`
- `interpolation_only_tracks.txt`
- `merge_only_tracks.txt`
- `full_repair_tracks.txt`
- `ablation.json`
- `ablation.md`

On all three raw outputs, interpolation reduced internal gaps while preserving
the number of tracklets. Conservative merging reduced the tracklet count, but
the newly joined trajectories contained gaps until interpolation was applied.
The full repair variant combined both effects and was less fragmented than the
baseline: it reduced tracklet count, short tracklets, gap-track counts, and
internal gaps. All fixed-output controls remained unchanged across all four
variants, which is consistent with conservative behavior on already stable
tracks.

These are tracklet-level statistics only. There is no ground-truth IDF1, HOTA,
MOTA, or identity-switch claim, and the ablation does not use ReID or global
matching. Merge quality still depends on geometric thresholds, and the same
thresholds were used for every sequence. These results can support the
real-data ablation table in the final report.

## Camera_04 Colab Ablation

The four-way ablation was also run on the first 1000 video frames from
`Warehouse_016/Camera_04`. The input was a BoT-SORT/AIC-style single-camera
JSON output. The run used the same parameters as the earlier real-data
experiments: `max_gap=5`, `max_merge_gap=5`, `max_center_distance=80`,
`max_size_ratio=1.5`, and `short_threshold=10`.

Raw output results:

| Variant | Detections | Tracklets | Mean length | Median length | Short | Short (%) | Gap tracks | Internal gaps | Interpolated | Merged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1154 | 21 | 54.95 | 11 | 10 | 47.62 | 6 | 28 | 0 | 0 |
| interpolation only | 1188 | 21 | 56.57 | 11 | 10 | 47.62 | 5 | 11 | 34 | 0 |
| merge only | 1154 | 17 | 67.88 | 12 | 6 | 35.29 | 8 | 32 | 0 | 4 |
| full repair | 1194 | 17 | 70.24 | 12 | 6 | 35.29 | 5 | 11 | 40 | 4 |

For the fixed-output control, all four variants were unchanged:

- total detections: 1231
- tracklets: 7
- mean tracklet length: 175.86
- median tracklet length: 166
- short tracklets: 0
- internal gaps: 0
- interpolated detections: 0
- merged tracklets: 0

On the raw output, full repair reduced the number of tracklets from 21 to 17
and internal gaps from 28 to 11. It added 40 interpolated detections and
accepted four conservative merges. Merge-only increased internal gaps from 28
to 32 because joining separate fragments made the missing frames between them
part of one longer component. Full repair applies merging first and then fills
eligible short gaps with interpolation. The unchanged fixed control is
consistent with conservative behavior on an already stable output.

This is a tracklet-level continuity evaluation and does not prove identity
correctness. No IDF1, HOTA, MOTA, ReID, or global matching improvement is
claimed.
