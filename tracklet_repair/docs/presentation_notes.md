# Tracklet repair presentation notes

## One-sentence contribution

This work adds an isolated helper for tracklet-level analysis, conservative
repair, and before-and-after comparison of single-camera tracking outputs.

## Problem

Raw single-camera tracking can contain fragmented identities, short unstable
tracklets, and missing detections inside otherwise continuous trajectories.
These local errors can produce unreliable identity candidates and propagate
into later multi-camera global matching.

## Relation to the existing pipeline

The existing BoT-SORT pipeline produces raw `Camera.json` files, and
`single_camera_fix.py` produces `fixed_Camera.json` files. The helper does not
replace either component. It reads their outputs to measure tracklet
continuity, apply conservative repairs, and compare the result with the
original output.

## Method

1. Load a project-style single-camera JSON file or MOT-style track text.
2. Compute tracklet-level statistics.
3. Interpolate short internal frame gaps.
4. Conservatively merge safe tracklet fragments.
5. Compare baseline and repaired outputs.

## Conservative design

It is safer to miss a possible merge than to merge different identities. A
merge is accepted only when the fragments are close in time and space, have
similar bounding-box sizes, and have the same class when class information is
available.

## Main results

| Run | Output | Internal gaps | Tracklets | Median length | Short tracklets |
| --- | --- | ---: | ---: | ---: | ---: |
| First real pair | Raw | 6 -> 2 | 13 -> 12 | 8.00 -> 17.50 | 7 -> 6 |
| First real pair | Fixed | 0 -> 0 | 6 -> 6 | stable | 0 -> 0 |
| Camera_02, first 1000 frames | Raw | 53 -> 14 | 44 -> 40 | 10.50 -> 33.50 | 22 -> 17 |
| Camera_02, first 1000 frames | Fixed | 0 -> 0 | 17 -> 17 | 311.00 -> 311.00 | 0 -> 0 |

For the Camera_02 run, 9000 frames were extracted from the video, but only the
first 1000 were processed. Detection used the ByteTrack/MOT17 fallback
checkpoint with the one-class YOLOX experiment because the AIC25 six-class
checkpoint was unavailable.

On the raw Camera_02 output, the helper interpolated 75 detections and merged
four conservative tracklet pairs. The fixed output remained unchanged because
it already had no internal gaps or short tracklets.

## One-minute speaking script

Our contribution focuses on tracklet consistency inside one camera, before
multi-camera identity matching. Raw tracker outputs can contain short gaps,
fragmented identities, and unstable short tracklets. We added an isolated
helper that reads the existing BoT-SORT JSON output, measures tracklet
statistics, interpolates small internal gaps, and conservatively merges only
fragments that are close in time and position with similar box sizes.

On the first real raw output, internal gaps decreased from 6 to 2 and median
tracklet length increased from 8 to 17.5. On the Camera_02 1000-frame run,
internal gaps decreased from 53 to 14, median length increased from 10.5 to
33.5, and short tracklets decreased from 22 to 17. The helper made no changes
to either fixed output because those tracklets were already stable. This
suggests that the method repairs visible raw fragmentation without
over-modifying stable results. These are tracklet-level results; full MOT
metrics remain future work.

## Limitations

- The current evaluation is tracklet-level, not a full MOT evaluation.
- IDF1, HOTA, and MOTA have not yet been measured.
- Ground truth and TrackEval integration are required for full MOT metrics.
- The Camera_02 run used a fallback detector, not the final AIC25 detector.

## Future work

- Run the helper on more cameras and scenes.
- Repeat the experiment with the AIC25 detector checkpoint if available.
- Add TrackEval metrics when suitable ground truth is available.
- Integrate the helper after `single_camera_fix.py` if the team chooses to use
  it in the main pipeline.
