# Reproducibility

The `tracklet_repair` helper analyzes single-camera tracklets, repairs short
gaps, applies conservative merging, and compares baseline and repaired outputs.
Run all commands from the repository root.

## Synthetic MOT-style test

```bash
python -m tracklet_repair.src.postprocess.run_postprocess \
  --input tracklet_repair/examples/sample_fragmented_tracks.txt \
  --output tracklet_repair/results/postprocess/fragmented_repaired.txt \
  --max-gap 5 \
  --enable-merge \
  --max-merge-gap 5 \
  --max-center-distance 80 \
  --max-size-ratio 1.5

python -m tracklet_repair.src.evaluation.evaluate_tracking \
  --baseline tracklet_repair/examples/sample_fragmented_tracks.txt \
  --repaired tracklet_repair/results/postprocess/fragmented_repaired.txt \
  --output-json tracklet_repair/results/evaluation/fragmented_comparison.json \
  --output-md tracklet_repair/results/evaluation/fragmented_comparison.md \
  --short-threshold 10
```

## Synthetic JSON pipeline test

```bash
python -m tracklet_repair.src.pipeline.run_json_tracklet_pipeline \
  --input-json tracklet_repair/examples/sample_single_camera.json \
  --output-dir tracklet_repair/results/json_pipeline/sample_single_camera \
  --max-gap 5 \
  --enable-merge \
  --max-merge-gap 5 \
  --max-center-distance 80 \
  --max-size-ratio 1.5 \
  --short-threshold 10
```

## Local real-data runs

Raw single-camera output:

```bash
python -m tracklet_repair.src.pipeline.run_json_tracklet_pipeline \
  --input-json tracklet_repair/local_inputs/Camera.json \
  --output-dir tracklet_repair/results/json_pipeline/Camera_raw \
  --max-gap 5 \
  --enable-merge \
  --max-merge-gap 5 \
  --max-center-distance 80 \
  --max-size-ratio 1.5 \
  --short-threshold 10
```

Output from the existing single-camera fix step:

```bash
python -m tracklet_repair.src.pipeline.run_json_tracklet_pipeline \
  --input-json tracklet_repair/local_inputs/fixed_Camera.json \
  --output-dir tracklet_repair/results/json_pipeline/Camera_fixed \
  --max-gap 5 \
  --enable-merge \
  --max-merge-gap 5 \
  --max-center-distance 80 \
  --max-size-ratio 1.5 \
  --short-threshold 10
```

Real input JSON files are local only and ignored by Git. Generated outputs are
written under `tracklet_repair/results/` and are also ignored.

The current comparison is tracklet-level analysis, not full MOT evaluation.
IDF1, HOTA, MOTA, and identity-switch metrics require ground truth and TrackEval
integration.
