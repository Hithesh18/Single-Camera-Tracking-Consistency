# Method Notes

## Baseline

The baseline is assumed to be a single-camera tracker that outputs one row per
object detection and frame.

The current input format is a comma-separated MOT-style file:

```text
frame_id, track_id, x, y, width, height, score, class_id
```

## Current Pipeline

1. Load MOT-style tracking output.
2. Compute tracklet-level statistics.
3. Optionally merge conservative tracklet candidates.
4. Detect small internal gaps inside each `track_id`.
5. Fill short gaps with linear interpolation of `x`, `y`, `width`, and `height`.
6. Save repaired tracking output in the same standard format.

## Tracklet Analysis

The analysis currently reports simple counts and gap statistics:

- Number of detections and tracklets.
- Tracklet length summary.
- Number of tracklets with internal frame gaps.
- Total internal gap count.
- Per-class tracklet counts.

The current result is verified on a small synthetic sample, not on real tracker
output yet.

## Short-Gap Interpolation

For each `track_id`, the repair step checks consecutive detections. If the
number of missing frames is at most `max_gap`, it creates interpolated detections
between the two boxes.

Interpolated fields:

- `x`, `y`, `width`, `height`: linear interpolation.
- `score`: average of the previous and next scores.
- `track_id`, `class_id`: copied from the same track.

This step helps with missed detections or very short occlusions where the same
`track_id` continues later.

It does not solve fragmented identities where the object receives a new
`track_id` after an occlusion.

## Conservative Merging

The current merging step tries to connect fragmented tracklets only when the
match is simple and low risk. Candidate tracklets must satisfy these rules:

- They must not overlap in time.
- The temporal gap must be at most `max_merge_gap`.
- `class_id` must match when `require_same_class` is enabled.
- The center of the first tracklet's last box and the second tracklet's first
  box must be close.
- Width and height ratios must stay within `max_size_ratio`.

The method prefers safe merges over aggressive merges. It does not yet use
appearance embeddings or ReID features, so it may miss difficult merges when
objects move quickly or detection boxes shift strongly.

## Evaluation Comparison

The evaluation script compares baseline and repaired tracking files using the
same tracklet statistics. It reports absolute differences as repaired minus
baseline and computes relative percentage changes where meaningful.

This is a tracklet-level comparison. It does not replace full MOT metrics such
as IDF1, HOTA, or MOTA when ground truth is available.

## TODO

- Record examples of successful and failed repairs.
- Compare baseline and repaired outputs on real tracking data.
