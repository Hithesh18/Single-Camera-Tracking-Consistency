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
