#!/usr/bin/env bash
set -euo pipefail

python -m tracklet_repair.src.postprocess.run_postprocess \
  --input tracklet_repair/examples/sample_fragmented_tracks.txt \
  --output tracklet_repair/results/postprocess/fragmented_repaired.txt \
  --max-gap 5 \
  --enable-merge \
  --max-merge-gap 5 \
  --max-center-distance 80 \
  --max-size-ratio 1.5
