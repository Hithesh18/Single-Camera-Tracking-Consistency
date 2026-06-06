# Colab one-camera run

## 1. Goal

This guide runs one validation scene and one camera instead of downloading or
processing the full dataset.

- Dataset split: `Val`
- Scene: `Warehouse_016`
- Camera: `Camera_01`

The tracking targets are:

```text
Tracking/Singlecamera/Warehouse_016/Camera_01/Camera_01.json
Tracking/Singlecamera/Warehouse_016/Camera_01/fixed_Camera_01.json
```

## 2. Colab setup

1. Open a new Google Colab notebook.
2. Select **Runtime > Change runtime type > GPU**.
3. Clone the repository and select the project branch:

```bash
!git clone https://github.com/Hithesh18/Single-Camera-Tracking-Consistency.git /content/repo
%cd /content/repo
!git checkout hakan/tracklet-repair-evaluation
```

4. Install the project dependencies and model requirements:

```bash
!bash colab_setup.sh
```

Confirm that Colab sees a GPU:

```bash
!nvidia-smi
```

## 3. Detector checkpoint

If the trained AIC25 detector checkpoint is available, place it at:

```text
/content/repo/BoT-SORT/ai_city_ckpt.pth.tar
```

For example, after mounting Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!cp /content/drive/MyDrive/AIC25/models/ai_city_ckpt.pth.tar \
    /content/repo/BoT-SORT/ai_city_ckpt.pth.tar
!ls -lh /content/repo/BoT-SORT/ai_city_ckpt.pth.tar
```

Using this checkpoint avoids detector training. If it is unavailable, the
pretrained ByteTrack detector downloaded by `colab_setup.sh` may be used as a
fallback, but its AIC25 results may be weaker.

## 4. Download only Camera_01

Install the Hugging Face client and download only the selected camera and scene
metadata. Do not request `depth_maps/**` or the complete dataset.

```python
!pip install -q huggingface_hub

from huggingface_hub import login, snapshot_download

# Use a Hugging Face token if the dataset requires authentication.
# login(token="YOUR_HF_TOKEN")

snapshot_download(
    repo_id="nvidia/PhysicalAI-SmartSpaces",
    repo_type="dataset",
    local_dir="/content/hf_data",
    allow_patterns=[
        "MTMC_Tracking_2025/val/Warehouse_016/videos/Camera_01/**",
        "MTMC_Tracking_2025/val/Warehouse_016/videos/Camera_01.mp4",
        "MTMC_Tracking_2025/val/Warehouse_016/calibration.json",
        "MTMC_Tracking_2025/val/Warehouse_016/ground_truth.json",
    ],
)
```

Map the downloaded lowercase `val` path to the project’s expected `Val` path:

```python
from pathlib import Path
import shutil

source = Path(
    "/content/hf_data/MTMC_Tracking_2025/val/Warehouse_016"
)
target = Path(
    "/content/repo/AIC25_Track1/Val/Warehouse_016"
)
target.mkdir(parents=True, exist_ok=True)

for name in ("calibration.json", "ground_truth.json"):
    src = source / name
    if src.exists():
        shutil.copy2(src, target / name)

target_videos = target / "videos"
target_videos.mkdir(parents=True, exist_ok=True)

camera_dir = source / "videos" / "Camera_01"
camera_mp4 = source / "videos" / "Camera_01.mp4"

if camera_dir.exists():
    shutil.copytree(
        camera_dir,
        target_videos / "Camera_01",
        dirs_exist_ok=True,
    )
elif camera_mp4.exists():
    shutil.copy2(camera_mp4, target_videos / "Camera_01.mp4")
else:
    raise FileNotFoundError("Camera_01 video was not downloaded")

print("Prepared:", target)
```

## 5. Expected local structure

Before frame extraction, one of these video layouts should exist:

```text
AIC25_Track1/Val/Warehouse_016/
├── calibration.json
├── ground_truth.json
└── videos/
    └── Camera_01/
        └── Camera_01.mp4
```

or:

```text
AIC25_Track1/Val/Warehouse_016/videos/Camera_01.mp4
```

The extraction script moves a loose `Camera_01.mp4` into its camera directory.
After extraction, the important path is:

```text
AIC25_Track1/Val/Warehouse_016/videos/Camera_01/Frame/
├── 000001.jpg
├── 000002.jpg
└── ...
```

Depth maps are not required for this one-camera test. The tracker uses
calibration-based ground coordinates when depth use is disabled.

## 6. Run the pipeline in order

Run these commands from `/content/repo`.

### A. Extract frames

Only `Camera_01` should be present under `videos/`, so the scene-level extractor
processes one camera:

```bash
%cd /content/repo
!python tools/extract_frames_25.py ./AIC25_Track1/Val -s Warehouse_016
```

### B. Run detection

With the trained AIC25 checkpoint:

```bash
%cd /content/repo
!python BoT-SORT/tools/aic25_get_detection.py \
    --scene Warehouse_016 \
    --dataset Val \
    --camera Camera_01 \
    -f BoT-SORT/yolox/exps/example/mot/yolox_x_AI_City_25.py \
    -c BoT-SORT/ai_city_ckpt.pth.tar \
    ./
```

Expected files:

```text
Detection/Warehouse_016/Camera_01.txt
Detection/Warehouse_016/Camera_01.json
```

If the AIC25 checkpoint is unavailable, the documented fallback is:

```bash
!python BoT-SORT/tools/aic25_get_detection.py \
    --scene Warehouse_016 \
    --dataset Val \
    --camera Camera_01 \
    -f BoT-SORT/yolox/exps/example/mot/yolox_x_mix_det.py \
    -c BoT-SORT/pretrained/bytetrack_x_mot17.pth.tar \
    ./
```

### C. Extract ReID embeddings

The script processes detection text files in the selected scene. Keep only the
`Camera_01` detection files there for this run.

```bash
%cd /content/repo/deep-person-reid
!python torchreid/aic25_extract.py \
    -s Warehouse_016 \
    --dataset Val \
    ../
```

Expected output:

```text
EmbedFeature/Warehouse_016/Camera_01/*.npy
```

The command also fills the `NpyPath` fields in
`Detection/Warehouse_016/Camera_01.json`.

### D. Run single-camera tracking

```bash
%cd /content/repo
!python BoT-SORT/single_camera_tracking.py \
    -s Warehouse_016 \
    -c Camera_01 \
    --dataset Val
```

Expected output:

```text
Tracking/Singlecamera/Warehouse_016/Camera_01/Camera_01.json
```

### E. Run the existing single-camera fix

```bash
%cd /content/repo
!python BoT-SORT/single_camera_fix.py \
    -s Warehouse_016 \
    -c Camera_01 \
    --dataset Val \
    --nms
```

Expected output:

```text
Tracking/Singlecamera/Warehouse_016/Camera_01/fixed_Camera_01.json
```

## 7. Run tracklet_repair

Analyze and repair the raw tracker output:

```bash
%cd /content/repo
!python -m tracklet_repair.src.pipeline.run_json_tracklet_pipeline \
    --input-json Tracking/Singlecamera/Warehouse_016/Camera_01/Camera_01.json \
    --output-dir tracklet_repair/results/json_pipeline/Camera_01_raw \
    --max-gap 5 \
    --enable-merge \
    --max-merge-gap 5 \
    --max-center-distance 80 \
    --max-size-ratio 1.5 \
    --short-threshold 10
```

Run the same comparison on the existing fixed output:

```bash
!python -m tracklet_repair.src.pipeline.run_json_tracklet_pipeline \
    --input-json Tracking/Singlecamera/Warehouse_016/Camera_01/fixed_Camera_01.json \
    --output-dir tracklet_repair/results/json_pipeline/Camera_01_fixed \
    --max-gap 5 \
    --enable-merge \
    --max-merge-gap 5 \
    --max-center-distance 80 \
    --max-size-ratio 1.5 \
    --short-threshold 10
```

## 8. Outputs to save

For both the raw and fixed runs, keep these files:

```text
comparison.md
comparison.json
baseline_stats.json
repaired_stats.json
```

The result directories are:

```text
tracklet_repair/results/json_pipeline/Camera_01_raw/
tracklet_repair/results/json_pipeline/Camera_01_fixed/
```

Copy these small result files to Google Drive before the Colab runtime ends if
they need to persist.

## 9. Troubleshooting

### CUDA is not available

Confirm that the runtime type is set to GPU and run `!nvidia-smi`. Reconnect the
runtime after changing hardware.

### Detector checkpoint is missing

Check:

```bash
!ls -lh /content/repo/BoT-SORT/ai_city_ckpt.pth.tar
```

Copy the trained checkpoint from Drive or use the weaker ByteTrack fallback.
Detector training is not part of this one-camera run.

### `Val` and `val` path mismatch

Hugging Face stores the split under lowercase `val`. The project scripts expect
`AIC25_Track1/Val/...`. Run the path-mapping cell before frame extraction.

### Camera folder name mismatch

The required name is exactly `Camera_01`. Check:

```bash
!find AIC25_Track1/Val/Warehouse_016/videos -maxdepth 2 -type d
```

### Detection output is missing

Check that frames exist and that the detector command uses
`--camera Camera_01`:

```bash
!ls AIC25_Track1/Val/Warehouse_016/videos/Camera_01/Frame | head
!ls -lh Detection/Warehouse_016/
```

### EmbedFeature output is missing

Check that both detection files exist and that the OSNet checkpoint was
installed:

```bash
!ls -lh Detection/Warehouse_016/Camera_01.*
!ls -lh deep-person-reid/checkpoints/osnet_ms_m_c.pth.tar
!find EmbedFeature/Warehouse_016/Camera_01 -type f | head
```

### Colab storage is nearly full

Download only `Camera_01`. Avoid depth maps and other cameras. Frames consume
more space than the original video, so copy final results to Drive and remove
temporary frames only after tracking and visualization are finished.

### Detection files contain stale appended data

The detection script opens its output files in append mode. Before rerunning
detection, remove only the previous `Camera_01` detection outputs:

```bash
!rm -f Detection/Warehouse_016/Camera_01.txt
!rm -f Detection/Warehouse_016/Camera_01.json
```

Do this only when intentionally restarting the one-camera detection step.

## 10. Safety

- Do not commit downloaded videos, extracted frames, checkpoints, embeddings,
  tracking outputs, or local result files.
- Do not commit generated files under `tracklet_repair/results/`.
- Keep large data in the Colab runtime or Google Drive.
- Review `git status` before every commit.
