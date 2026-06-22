# Learned Tracklet Matcher (Deliverable 2)

Replaces the geometric if/else merge cost with a **learned** P(same identity),
dominated by **deep ReID visual appearance**. Directly addresses supervisor
feedback: "it should include ML / learning is very important", "instead of
if/else implement color/dynamic visual appearance", "if object tilted bbox
changes" (appearance survives tilt), "object goes behind obstacle and comes
back" (long-gap re-id via appearance, no temporal cap).

## Modules

- `tracklet_repair/src/matcher/features.py` — joins OSNet 512-d embeddings to
  tracked detections (IoU match, tolerant of BoT-SORT's Kalman offset) and builds
  the pair-feature vector.
- `tracklet_repair/src/matcher/model.py` — small MLP (9 -> 32 -> 16 -> 1) +
  `LearnedMatcher` inference wrapper (standardises features, returns P(same)).
- `tracklet_repair/src/matcher/train_matcher.py` — builds GT-labelled pairs and
  trains the model. CPU-only.

## Pair-feature vector (the learned input)

`mean_appearance_cosine`, `endpoint_appearance_cosine`, `max_appearance_cosine`,
`temporal_gap_norm`, `predicted_center_distance_norm`, `raw_center_distance_norm`,
`size_ratio`, `same_class`, `velocity_direction_consistency`.

Three of nine inputs are deep-appearance cosines; geometry/motion are present but
the model *learns* their weights instead of hard thresholds.

## Labels

A tracklet is matched to a GT `object id` by IoU vote. An ordered pair
(source ends before target starts) is positive if both map to the same GT id.
`max_gap=None` forms long-gap pairs so the model can learn re-identification.

## Run (local, CPU)

```bash
python -m tracklet_repair.src.matcher.train_matcher \
  --gt-json AIC25_Track1/Val/Warehouse_016/ground_truth.json \
  --scene Warehouse_016 \
  --out tracklet_repair/models/matcher
```

## Status

Pipeline validated locally end to end on camera `Camera`: 466/467 embeddings
matched, model trains and saves. BUT the local sample (100 frames, 13 tracklets)
yields only ~9 labelled pairs — enough to prove the code, far too few to train a
real model. **Next: generate full-length multi-camera outputs on Colab to get
thousands of pairs, retrain, then wire the matcher into a `learned` merge mode
and re-benchmark with gt_benchmark.** The integration step (learned merge mode)
is not built yet.
