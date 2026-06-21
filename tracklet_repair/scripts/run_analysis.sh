#!/usr/bin/env bash
set -euo pipefail

python -m tracklet_repair.src.analysis.analyze_tracklets \
  --input tracklet_repair/examples/sample_tracks.txt \
  --output tracklet_repair/results/analysis/sample_stats.json
