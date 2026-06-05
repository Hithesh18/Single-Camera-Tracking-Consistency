#!/bin/bash
set -e

SCENE="Warehouse_017"
DATASET="Test"
CAMERAS=("Camera" "Camera_01" "Camera_02" "Camera_03" "Camera_04" "Camera_05" "Camera_06" "Camera_07")

cd /home/seco/deepLearning/Single-Camera-Tracking-Consistency

echo "=== Step 1: Extract frames ==="
conda run -n botsort_env python3 tools/extract_frames_25.py ./AIC25_Track1/$DATASET -s $SCENE

echo "=== Step 2: Detection ==="
cp ./detector/aic25_get_detection.py ./BoT-SORT/tools/
conda run -n botsort_env python3 BoT-SORT/tools/aic25_get_detection.py -s $SCENE ./

echo "=== Step 3: ReID embeddings ==="
cp ./embedder/aic25_extract.py ./deep-person-reid/torchreid/
conda run -n torchreid python3 deep-person-reid/torchreid/aic25_extract.py -s $SCENE ./

echo "=== Step 4: Single-camera tracking ==="
for CAM in "${CAMERAS[@]}"; do
    echo "  Tracking $CAM..."
    conda run -n botsort_env python3 BoT-SORT/single_camera_tracking.py -s $SCENE -c $CAM
done

echo "=== Step 5: Single-camera fix ==="
for CAM in "${CAMERAS[@]}"; do
    echo "  Fixing $CAM..."
    conda run -n botsort_env python3 BoT-SORT/single_camera_fix.py -s $SCENE -c $CAM --dataset $DATASET --nms
done

echo ""
echo "Done! Output files are in:"
echo "  Tracking/Singlecamera/$SCENE/<Camera>/Camera.json"
echo "  Tracking/Singlecamera/$SCENE/<Camera>/fixed_Camera.json"
