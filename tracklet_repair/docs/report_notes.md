# Report and presentation notes

## Project context

This work belongs to the single-camera tracking consistency subproject. The
goal is to analyze and improve broken or fragmented tracklets before
multi-camera global matching.

The `tracklet_repair` helper is not a replacement for the existing BoT-SORT
pipeline. It is an isolated analysis, post-processing, and evaluation layer for
single-camera outputs.

## Problem statement

Raw single-camera tracking can contain short frame gaps, fragmented identities,
and short unstable tracklets. These local tracking errors can create unreliable
identity candidates and propagate into later multi-camera matching.

## Method

The helper pipeline:

1. Loads project-style single-camera JSON or MOT-style track text.
2. Computes tracklet-level statistics.
3. Interpolates short internal gaps.
4. Conservatively merges likely tracklet fragments.
5. Compares the baseline and repaired outputs.

## Conservative design

It is safer to miss a possible merge than to combine different identities. A
merge candidate must be temporally close, spatially close, and similar in
bounding-box size. The object class must also match when class information is
available.

## Current real-data results

| Input | detections | tracklets | mean length | gap tracklets | internal gaps | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| raw Camera.json baseline | 467 | 13 | 35.92 | 3 | 6 | before helper |
| raw Camera.json repaired | 480 | 12 | 40.00 | 1 | 2 | after helper |
| fixed_Camera.json baseline | 486 | 6 | 81.00 | 0 | 0 | before helper |
| fixed_Camera.json repaired | 486 | 6 | 81.00 | 0 | 0 | after helper |
| raw Camera_02.json baseline | 5225 | 44 | 118.75 | 15 | 53 | before helper |
| raw Camera_02.json repaired | 5300 | 40 | 132.50 | 11 | 14 | after helper |
| fixed_Camera_02.json baseline | 4952 | 17 | 291.29 | 0 | 0 | before helper |
| fixed_Camera_02.json repaired | 4952 | 17 | 291.29 | 0 | 0 | after helper |
| raw Camera_03.json baseline | 3844 | 40 | 96.10 | 21 | 46 | before helper |
| raw Camera_03.json repaired | 3920 | 36 | 108.89 | 12 | 17 | after helper |
| fixed_Camera_03.json baseline | 3961 | 20 | 198.05 | 0 | 0 | before helper |
| fixed_Camera_03.json repaired | 3961 | 20 | 198.05 | 0 | 0 | after helper |

For the raw output, the helper added 13 interpolated detections and performed
one conservative merge with merge map `{7: 6}`. It made no additional changes
to the fixed output, suggesting that the existing `single_camera_fix.py` result
is already stable on this sample under the current tracklet-level metrics.

The Camera_02 Colab run used the first 1000 frames and the ByteTrack/MOT17
fallback checkpoint with a one-class YOLOX experiment because the trained AIC25
six-class detector was unavailable. OSNet x1_0 extraction produced features
for 7400 detections. BoT-SORT produced 5225 tracked objects across 44 track IDs,
while `single_camera_fix.py` produced 4952 objects across 17 track IDs.

On the raw output, the helper added 75 interpolated detections and performed
four conservative merges with merge map
`{20: 19, 30: 29, 34: 33, 42: 41}`. Internal gaps decreased from 53 to 14,
median tracklet length increased from 10.50 to 33.50, and short tracklets
decreased from 22 to 17. The fixed output was unchanged. This run supports
reproducibility and robustness, but it is not the final AIC25 detector result.

The Camera_03 Colab run also used the first 1000 frames and the same fallback
detector with OSNet x1_0 ReID extraction. Detection and ReID each produced 4556
outputs. BoT-SORT produced 3844 tracked objects across 40 track IDs and 976
frames, while `single_camera_fix.py` produced 3961 objects across 20 track IDs
and 999 frames.

On the raw Camera_03 output, the helper added 76 interpolated detections and
performed four conservative merges with merge map
`{9: 7, 10: 7, 25: 23, 32: 31}`. Internal gaps decreased from 46 to 17,
tracklets with gaps decreased from 21 to 12, median tracklet length increased
from 19.50 to 27.50, and short tracklets decreased from 15 to 11. The fixed
output was unchanged. This is another robustness result with the fallback
detector, not the final AIC25 detector result.

## Presentation points

- We target local gaps and identity fragmentation before global multi-camera
  association.
- The helper supports project JSON and MOT-style text, then reports consistent
  tracklet statistics.
- Repair is conservative because incorrect identity merges are more harmful
  than missed merges.
- Across three real raw/fixed pairs, the helper reduced raw fragmentation while
  leaving already stable fixed outputs unchanged.
- Future work is broader real-data testing, visual examples, and ground-truth
  MOT evaluation.

## Limitations

- The current result is tracklet-level analysis, not full MOT evaluation.
- Only three real raw/fixed output pairs have been tested so far. Camera_02 and
  Camera_03 used a fallback detector on the first 1000 frames.
- IDF1, HOTA, MOTA, and identity-switch metrics require ground truth and
  TrackEval integration.
- Results from more cameras and scenes would make the experiment stronger.

## Next steps

1. Run the same pipeline on additional real single-camera outputs, preferably
   with the trained AIC25 detector.
2. Compare with TrackEval metrics if suitable ground truth is available.
3. Add before-and-after visual examples if the corresponding frames are
   available.
