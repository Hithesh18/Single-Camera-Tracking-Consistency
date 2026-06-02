#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# colab_setup.sh
# Run this once at the start of each Colab session (after cloning the repo).
# Usage:  bash colab_setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "[1/6] Repo root: $REPO_ROOT"

# ── 1. Install BoT-SORT + YOLOX dependencies ─────────────────────────────────
echo "[2/6] Installing BoT-SORT dependencies..."
cd "$REPO_ROOT/BoT-SORT"
python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q -r requirements.txt cython_bbox pycocotools
python -m pip install -q faiss-gpu || python -m pip install -q faiss-cpu || true
python setup.py develop --quiet

# ── 2. Install torchreid (deep-person-reid) ───────────────────────────────────
echo "[3/6] Installing torchreid..."
cd "$REPO_ROOT/deep-person-reid"
python -m pip install -q -r requirements.txt
python setup.py develop --quiet

# ── 3. Install tracking requirements ─────────────────────────────────────────
echo "[4/6] Installing tracking requirements..."
cd "$REPO_ROOT"
python -m pip install -q -r tracking/requirements.txt
python - <<'PY'
from loguru import logger
from thop import profile
from yolox.exp import get_exp
from tracker.bot_sort import BoTSORT
print("Dependency imports OK")
PY

# ── 4. Download OSNet Re-ID model ─────────────────────────────────────────────
# osnet_ms_m_c.pth.tar — OSNet trained on multiple datasets (torchreid model zoo)
echo "[5/6] Downloading OSNet Re-ID model..."
OSNET_PATH="$REPO_ROOT/deep-person-reid/checkpoints/osnet_ms_m_c.pth.tar"
if [ ! -f "$OSNET_PATH" ]; then
    python -m pip install -q gdown
    # Google Drive file ID for osnet_ms_m_c.pth.tar from KaiyangZhou/deep-person-reid
    gdown --id 1IosIFlLiulGIjwW3H8uMCC3YvMyr9gZ2 -O "$OSNET_PATH"
    echo "  OSNet saved to: $OSNET_PATH"
else
    echo "  OSNet already exists, skipping."
fi

# ── 5. Download ByteTrack base model ─────────────────────────────────────────
# bytetrack_x_mot17.pth.tar — YOLOX_x pretrained on MOT17 (ByteTrack model zoo)
echo "[6/6] Downloading ByteTrack base model..."
BYTETRACK_PATH="$REPO_ROOT/BoT-SORT/pretrained/bytetrack_x_mot17.pth.tar"
if [ ! -f "$BYTETRACK_PATH" ]; then
    # Google Drive file ID from ByteTrack official release
    gdown --id 1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5 -O "$BYTETRACK_PATH"
    echo "  ByteTrack model saved to: $BYTETRACK_PATH"
else
    echo "  ByteTrack model already exists, skipping."
fi

echo ""
echo "Setup complete. Models downloaded and packages installed."
echo ""
echo "Next steps:"
echo "  1. Mount Google Drive and link your AIC25 dataset:"
echo "     ln -s /content/drive/MyDrive/AIC25/AIC25_Track1  \$REPO_ROOT/AIC25_Track1_data"
echo "  2. Check if ai_city_ckpt.pth.tar exists in BoT-SORT/:"
echo "     ls \$REPO_ROOT/BoT-SORT/ai_city_ckpt.pth.tar"
echo "     If missing, run training first: bash scripts/train_detector.sh"
echo "  3. Then run the pipeline: see PIPELINE.md"
