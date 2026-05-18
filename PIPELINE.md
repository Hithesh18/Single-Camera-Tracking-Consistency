# Running the Pipeline (Google Colab)

---

## One-time setup (first Colab session)

### Step 1 — Clone repo & install dependencies
```bash
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /content/repo
%cd /content/repo
!bash colab_setup.sh
```

### Step 2 — Download dataset from HuggingFace (skip depth maps to save space)

> ⚠️ Full dataset is 3.31 TB. Download only what you need per session.
> Colab free tier has ~100 GB disk. Depth maps alone are 30-80 GB per warehouse — skip them for now.

```python
from huggingface_hub import snapshot_download
import os, shutil

# Download Warehouse_016 Val — videos + calibration + ground_truth only (NO depth maps)
snapshot_download(
    repo_id="nvidia/PhysicalAI-SmartSpaces",
    repo_type="dataset",
    local_dir="/content/hf_data",
    allow_patterns=[
        "MTMC_Tracking_2025/val/Warehouse_016/videos/**",
        "MTMC_Tracking_2025/val/Warehouse_016/calibration.json",
        "MTMC_Tracking_2025/val/Warehouse_016/ground_truth.json",
    ],
    # ignore_patterns=["*depth_maps*"]  # already excluded above
)
```

### Step 3 — Remap HuggingFace paths to what the code expects

> HuggingFace uses `val/` (lowercase) and `depth_maps/` (plural).
> The code expects `Val/` (capital) and `depth_map/` (singular).

```python
import os, shutil

HF   = "/content/hf_data/MTMC_Tracking_2025"
DEST = "/content/repo/AIC25_Track1"

# Val — Warehouse_016
src = f"{HF}/val/Warehouse_016"
dst = f"{DEST}/Val/Warehouse_016"
os.makedirs(dst, exist_ok=True)

# Copy calibration + ground_truth
for f in ["calibration.json", "ground_truth.json"]:
    if os.path.exists(f"{src}/{f}"):
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")

# Link videos (symlink to avoid copying GBs)
if os.path.exists(f"{src}/videos"):
    os.symlink(f"{src}/videos", f"{dst}/videos")

# depth_maps → depth_map (rename, only if downloaded)
if os.path.exists(f"{src}/depth_maps"):
    os.symlink(f"{src}/depth_maps", f"{dst}/depth_map")

print("Paths mapped.")
```

### Step 4 — Copy trained detector checkpoint (if already trained)
```bash
!cp /content/drive/MyDrive/AIC25/models/ai_city_ckpt.pth.tar /content/repo/BoT-SORT/ai_city_ckpt.pth.tar
```
> If you don't have `ai_city_ckpt.pth.tar` yet, run Training step [A] below first.

---

## Each session (after reconnecting to Colab)

Re-run Steps 1–4 above — all `/content/` data resets on disconnect.
Save outputs to Google Drive to persist them:
```bash
!mkdir -p /content/drive/MyDrive/AIC25/outputs
!ln -sfn /content/drive/MyDrive/AIC25/outputs /content/repo/Detection
!ln -sfn /content/drive/MyDrive/AIC25/outputs /content/repo/EmbedFeature
!ln -sfn /content/drive/MyDrive/AIC25/outputs /content/repo/Tracking
```

---

## Pipeline Steps

### [A] Train the detector (only once, ~6-10h on T4)
> Skip if you already have `BoT-SORT/ai_city_ckpt.pth.tar`

```bash
%cd /content/repo

# Extract training frames first
!python tools/extract_frames_25.py ./AIC25_Track1/Train -s Warehouse_016

# Generate COCO annotations
!python tools/convert_to_coco_25.py --BASE_DIR ./AIC25_Track1 -s Train

# Save training outputs to Drive so they survive disconnects
!mkdir -p /content/drive/MyDrive/AIC25/YOLOX_outputs
!ln -sfn /content/drive/MyDrive/AIC25/YOLOX_outputs /content/repo/YOLOX_outputs

# Train
!python BoT-SORT/yolox/train.py \
    -f BoT-SORT/yolox/exps/example/mot/yolox_x_AI_City_25.py \
    -d 1 -b 4 --fp16 \
    -c BoT-SORT/pretrained/bytetrack_x_mot17.pth.tar

# Copy best checkpoint after training finishes
!cp YOLOX_outputs/yolox_x_AI_City_25/best_ckpt.pth.tar BoT-SORT/ai_city_ckpt.pth.tar
!cp YOLOX_outputs/yolox_x_AI_City_25/best_ckpt.pth.tar /content/drive/MyDrive/AIC25/models/ai_city_ckpt.pth.tar
```

---

### [B] Extract frames
```bash
%cd /content/repo
!python tools/extract_frames_25.py ./AIC25_Track1/Val -s Warehouse_016
```

### [C] Run detection
```bash
%cd /content/repo/BoT-SORT
!python tools/aic25_get_detection.py -s Warehouse_016 ../
```

### [D] Extract Re-ID embeddings
```bash
%cd /content/repo/deep-person-reid
!python torchreid/aic25_extract.py -s Warehouse_016 ../
```

### [E] Single-camera tracking
```bash
%cd /content/repo
for cam in Camera Camera_01 Camera_02 Camera_03 Camera_04 Camera_05 Camera_06 Camera_07 Camera_08 Camera_09 Camera_10 Camera_11; do
    python BoT-SORT/single_camera_tracking.py -s Warehouse_016 -c $cam
done
```

### [F] Fix single-camera results
```bash
!python BoT-SORT/single_camera_fix.py
```

### [G] Multi-camera tracking
```bash
!python BoT-SORT/multi_camera_revised.py
!python BoT-SORT/multi_camera_fix.py
```

### [H] Evaluate
```bash
!python TrackEval/main.py
```

---

## Files checklist

| File | Location in repo | Source |
|------|-----------------|--------|
| Video `.mp4` files | `AIC25_Track1/Val/Warehouse_016/videos/Camera*/` | HuggingFace (Step 2) |
| `calibration.json` | `AIC25_Track1/Val/Warehouse_016/` | HuggingFace (Step 2) |
| `ground_truth.json` | `AIC25_Track1/Val/Warehouse_016/` | HuggingFace (Step 2) |
| `depth_map/*.h5` | `AIC25_Track1/Val/Warehouse_016/depth_map/` | HuggingFace — optional (2-7 GB each) |
| `bytetrack_x_mot17.pth.tar` | `BoT-SORT/pretrained/` | Auto-downloaded by `colab_setup.sh` |
| `osnet_ms_m_c.pth.tar` | `deep-person-reid/checkpoints/` | Auto-downloaded by `colab_setup.sh` |
| `ai_city_ckpt.pth.tar` | `BoT-SORT/` | Generated by training step [A] |
