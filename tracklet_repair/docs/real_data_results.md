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

This is a tracklet-level comparison, not a full MOT evaluation. Ground-truth
metrics such as IDF1, HOTA, MOTA, and ID switches remain future work if ground
truth is available.
