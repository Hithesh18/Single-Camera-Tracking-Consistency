#!/usr/bin/env bash
set -euo pipefail

python -m tracklet_repair.src.evaluation.evaluate_tracking \
  --baseline tracklet_repair/examples/sample_fragmented_tracks.txt \
  --repaired tracklet_repair/results/postprocess/fragmented_repaired.txt \
  --output-json tracklet_repair/results/evaluation/fragmented_comparison.json \
  --output-md tracklet_repair/results/evaluation/fragmented_comparison.md \
  --short-threshold 10
