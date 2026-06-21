# Tracklet Repair and Conservative Merging for Single-Camera Tracking Consistency

This module is part of the Deep Learning Lab project on multi-camera tracking.
It contributes to Subproject 1 by analyzing and improving the stability of
single-camera tracklets before later pipeline stages.

The module is a lightweight post-processing and evaluation helper. It reads
existing tracker outputs, measures tracklet continuity, repairs short gaps, and
conservatively joins compatible fragments. It does not train a ReID model and
does not implement global multi-camera matching.

## Implemented Functionality

- Tracklet statistics for project-style JSON and MOT-style text inputs.
- Linear interpolation of short internal frame gaps.
- Conservative merging based on temporal gap, bounding-box distance, size
  ratio, and object class when available.
- An end-to-end JSON repair pipeline.
- Four-way ablation comparison:
  - `baseline`
  - `interpolation_only`
  - `merge_only`
  - `full_repair`
- Regression tests for interpolation, merging, ordering, and overlap safety.
- Experiment logging for synthetic and real tracker outputs.

The full repair order is conservative merging followed by short-gap
interpolation.

## Structure

```text
tracklet_repair/
|-- configs/          Configuration examples
|-- docs/             Method, experiment, and reproducibility notes
|-- examples/         Small committed synthetic inputs
|-- scripts/          Shell commands for common workflows
|-- src/
|   |-- analysis/     Tracklet statistics
|   |-- evaluation/   Before/after comparison and ablation
|   |-- pipeline/     End-to-end JSON helper
|   |-- postprocess/  Interpolation and conservative merging
|   `-- utils/        JSON adapters and tracking file I/O
`-- tests/            Regression tests
```

## Input Format

The primary project input is a BoT-SORT/AIC-style single-camera JSON file.
Frames are stored as JSON keys, and each frame maps to a list of tracked
objects. Each usable object must provide:

- a single-camera track ID, such as `object sc id` or `object_sc_id`;
- a visible two-dimensional bounding box; and
- optionally a score and object class.

The JSON adapter handles the project-specific key variants and converts
bounding boxes to the internal columns:

```text
frame_id, track_id, x, y, width, height, score, class_id
```

Missing scores and classes use neutral defaults. The module also accepts
comma-separated tracking text with these eight columns and no header.

## Setup

Run commands from the repository root:

```bash
pip install -r tracklet_repair/requirements.txt
```

The regression suite uses `pytest`, which must also be available in the active
environment.

## Run Tests

```bash
python -m pytest tracklet_repair/tests -v
```

## Run the JSON Pipeline

The pipeline converts one JSON output, computes baseline statistics, applies
repair, and writes a before/after comparison:

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

## Run the Ablation

The ablation evaluates the same input with no repair, interpolation only,
merging only, and the full repair order:

```bash
python -m tracklet_repair.src.evaluation.run_ablation \
  --input-json tracklet_repair/examples/sample_single_camera.json \
  --output-dir tracklet_repair/results/ablation_sample \
  --max-gap 5 \
  --max-merge-gap 5 \
  --max-center-distance 80 \
  --max-size-ratio 1.5 \
  --short-threshold 10
```

The output directory contains one tracking text file per variant together with
`ablation.json` and `ablation.md`.

## Real-Data Evaluation

Real single-camera tracker outputs are kept locally under
`tracklet_repair/local_inputs/`. Generated tables and tracking files are
written under `tracklet_repair/results/`. Both locations are ignored by Git.

Commands, parameters, and committed experiment summaries are recorded in
[`docs/experiment_log.md`](docs/experiment_log.md). The documented real-data
ablations use the same thresholds for every sequence:

- maximum interpolation gap: 5 frames;
- maximum merge gap: 5 frames;
- maximum center distance: 80 pixels;
- maximum width or height ratio: 1.5; and
- short-tracklet threshold: 10 detections.

See [`docs/reproducibility.md`](docs/reproducibility.md) for repeatable sample
and local-data commands.

## Limitations

- The reported measurements are tracklet-level continuity statistics.
- Fewer internal gaps, fewer tracklets, or longer tracklets do not prove that
  every identity association is correct.
- No IDF1, HOTA, MOTA, ReID, or global matching improvement is claimed without
  suitable ground truth and full tracking evaluation.
- Conservative geometric rules may miss valid merges when motion or
  bounding-box changes are large.
