# Project Plan

## Title

Tracklet Fragmentation Analysis and Repair for Single-Camera Tracking

## Motivation

Single-camera tracking quality strongly affects the later multi-camera
association stage. Fragmented tracklets and avoidable ID switches can create
noisy identity candidates across cameras.

## Current Contribution

This project currently provides a lightweight post-processing pipeline for
single-camera tracking outputs, starting with tracklet analysis, short-gap
interpolation, conservative merging, and a baseline vs repaired comparison.

The pipeline has been checked on synthetic samples and two local real
single-camera outputs. Full ground-truth MOT evaluation has not been completed.

## Main Questions

- Which tracking failures appear most often in the baseline output?
- Can short gaps be repaired without introducing many false links?
- Can fragmented tracklets be merged conservatively enough to improve stability?
- Which metrics best capture the improvement for this project?

## Completed Work

- Project scaffold.
- Tracklet statistics analysis.
- Short-gap interpolation.
- Conservative tracklet merging.
- Baseline vs repaired comparison script.
- Real raw `Camera.json` tested.
- Real `fixed_Camera.json` tested.
- Generated real-data results kept local and ignored.

## Planned Next Steps

1. Add visual failure-case examples.
2. Optionally add TrackEval/MOT metrics if ground truth is available.
3. Prepare final result tables for paper and presentation.

## Risks

- Over-aggressive merging may create identity errors.
- Missing appearance embeddings may limit reliable merge decisions.
- Improvements in simple statistics may not always improve standard MOT metrics.

## TODO

- Decide final dataset and file format.
- Add dataset-specific notes.
- Define the exact evaluation protocol with the lab supervisor.
