# Real-data tracklet repair run

## Inputs

- `Camera.json`: raw single-camera output.
- `fixed_Camera.json`: output after the existing `single_camera_fix.py`.

The input files and generated results are local only and are not committed.

## Raw Camera.json

| Metric | Before | After |
| --- | ---: | ---: |
| Total detections | 467 | 480 |
| Tracklets | 13 | 12 |
| Mean tracklet length | 35.92 | 40.00 |
| Median tracklet length | 8.00 | 17.50 |
| Short tracklets | 7 | 6 |
| Percent short tracklets | 53.85 | 50.00 |
| Tracklets with gaps | 3 | 1 |
| Total internal gaps | 6 | 2 |

- Interpolated detections: 13
- Merged tracklets: 1
- Merge map: `{7: 6}`

## Fixed fixed_Camera.json

| Metric | Before | After |
| --- | ---: | ---: |
| Total detections | 486 | 486 |
| Tracklets | 6 | 6 |
| Mean tracklet length | 81.00 | 81.00 |
| Short tracklets | 0 | 0 |
| Tracklets with gaps | 0 | 0 |
| Total internal gaps | 0 | 0 |

- Interpolated detections: 0
- Merged tracklets: 0

## Interpretation

The helper improves the raw single-camera output by interpolating short gaps
and conservatively merging one fragmented tracklet. The fixed output already
has stable tracklets under the current statistics, so the helper makes no
additional changes. This is expected and indicates conservative behavior.

## Camera_02 Colab one-camera run, first 1000 frames

This run used `Warehouse_016`, `Camera_02`, and the first 1000 frames.
Detection used the ByteTrack/MOT17 fallback checkpoint with the one-class
YOLOX experiment because the trained AIC25 six-class detector checkpoint was
not available. ReID features were extracted with OSNet x1_0.

- Detector output: 7400 detections
- Raw BoT-SORT output: 5225 tracked objects across 44 track IDs
- Existing `single_camera_fix.py` output: 4952 objects across 17 track IDs

This is an additional reproducibility and robustness run with the available
fallback detector, not the final AIC25 detector result.

### Raw Camera_02.json

| Metric | Before | After |
| --- | ---: | ---: |
| Total detections | 5225 | 5300 |
| Tracklets | 44 | 40 |
| Mean tracklet length | 118.75 | 132.50 |
| Median tracklet length | 10.50 | 33.50 |
| Short tracklets | 22 | 17 |
| Percent short tracklets | 50.00 | 42.50 |
| Tracklets with gaps | 15 | 11 |
| Total internal gaps | 53 | 14 |

- Interpolated detections: 75
- Merged tracklets: 4
- Merge map: `{20: 19, 30: 29, 34: 33, 42: 41}`

### Fixed fixed_Camera_02.json

| Metric | Before | After |
| --- | ---: | ---: |
| Total detections | 4952 | 4952 |
| Tracklets | 17 | 17 |
| Mean tracklet length | 291.29 | 291.29 |
| Median tracklet length | 311.00 | 311.00 |
| Short tracklets | 0 | 0 |
| Percent short tracklets | 0.00 | 0.00 |
| Tracklets with gaps | 0 | 0 |
| Total internal gaps | 0 | 0 |

- Interpolated detections: 0
- Merged tracklets: 0

### Camera_02 interpretation

On the raw output, internal gaps decreased from 53 to 14, median tracklet
length increased from 10.50 to 33.50, and short tracklets decreased from 22 to
17. Four conservative tracklet pairs were merged. The fixed output already had
no internal gaps or short tracklets, so the helper made no changes. This
follows the pattern seen in the first real-data pair: the helper repairs raw
fragmentation without over-modifying an already stable fixed output.

This is a tracklet-level comparison, not a full MOT evaluation. Ground-truth
metrics such as IDF1, HOTA, MOTA, and ID switches remain future work if ground
truth is available.
