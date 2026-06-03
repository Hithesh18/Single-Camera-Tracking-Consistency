# Tracklet Repair and Conservative Merging for More Stable Single-Camera Tracking

This folder is a helper module for tracklet analysis, repair, and comparison
inside the Single-Camera Tracking Consistency project. It is kept isolated from
the main BoT-SORT and TrackEval pipeline while we test the post-processing
steps.

This repository contains a starter project for a Deep Learning Lab topic on
single-camera tracking consistency inside a multi-camera multi-target tracking
pipeline.

The goal is to improve the stability of single-camera tracking outputs by
analyzing fragmented tracklets, repairing short gaps, and conservatively merging
tracklets that likely belong to the same target. This scaffold does not yet
implement the full algorithms. It provides a clean structure, documented
placeholder functions, and scripts for future experiments.

## Project Idea

Modern multi-camera tracking systems often depend on strong single-camera
tracking results. When a tracker frequently switches IDs, loses objects for a
few frames, or creates fragmented tracklets, the downstream multi-camera
association step becomes harder.

This project focuses on a post-processing stage that runs after a baseline
single-camera tracker:

1. Analyze tracklets and detect fragmentation patterns.
2. Repair short temporal gaps inside plausible trajectories.
3. Conservatively merge tracklets only when evidence is strong.
4. Evaluate whether the repaired output improves tracking consistency.

## Repository Structure

```text
.
├── configs/              # YAML configuration files
├── docs/                 # Planning notes and experiment logs
├── paper/                # LaTeX paper skeleton
├── results/              # Output directory for generated results
├── scripts/              # Shell scripts for common commands
├── src/                  # Python source code
│   ├── analysis/         # Tracklet statistics and diagnostics
│   ├── evaluation/       # Metric computation placeholders
│   ├── postprocess/      # Repair and merge placeholders
│   └── utils/            # Shared helpers
├── README.md
└── requirements.txt
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Expected Data Format

The starter code assumes tracking files are stored as CSV files with one
detection per row:

```text
frame,track_id,x,y,w,h,score,camera_id
1,12,450.0,210.0,80.0,160.0,0.92,c001
2,12,454.0,211.5,80.5,160.0,0.91,c001
```

Column meanings:

- `frame`: Integer frame index.
- `track_id`: Tracker-assigned identity inside one camera.
- `x`, `y`, `w`, `h`: Bounding box top-left coordinate, width, and height.
- `score`: Detector or tracker confidence.
- `camera_id`: Camera name or numeric camera identifier.

Future experiments may adapt this to MOTChallenge text files or project-specific
multi-camera data.

## How to Run

Run the analysis placeholder:

```bash
bash scripts/run_analysis.sh
```

Run the post-processing placeholder:

```bash
bash scripts/run_postprocess.sh
```

Run short-gap interpolation on the sample tracks:

```bash
python -m src.postprocess.run_postprocess --input examples/sample_tracks.txt --output results/postprocess/sample_repaired.txt --max-gap 5
```

Run the evaluation placeholder:

```bash
bash scripts/run_evaluation.sh
```

Compare baseline and repaired sample outputs:

```bash
python -m src.evaluation.evaluate_tracking --baseline examples/sample_fragmented_tracks.txt --repaired results/postprocess/fragmented_repaired.txt --output-json results/evaluation/fragmented_comparison.json --output-md results/evaluation/fragmented_comparison.md --short-threshold 10
```

The scripts currently use the example paths defined in the config files. Update
the paths in `configs/baseline.yaml` and `configs/postprocess.yaml` once real
tracking data is available.

## Current Status

This is a clean scaffold only. The next implementation steps are:

- Add robust loading for real tracker output files.
- Implement tracklet statistics and fragmentation diagnostics.
- Add simple gap repair based on temporal and spatial constraints.
- Add conservative merge rules.
- Compare baseline and repaired results with standard tracking metrics.
