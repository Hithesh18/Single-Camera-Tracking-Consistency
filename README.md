# Single-Camera Tracking Consistency

> **Multi-Camera Tracking Improvement — Single-Camera Tracking Consistency**

## Authors

- Hithesh Chettenahalli Honnegowda
- Hakan Berke Şiranur

---

> Based on [**BoT-SORT: Robust Associations Multi-Pedestrian Tracking**](https://arxiv.org/abs/2206.14651)
> by Hamidreza-Hashempoor
>
> Original repository: [https://github.com/Hamidreza-Hashempoor/Glance-MCMT](https://github.com/Hamidreza-Hashempoor/Glance-MCMT)

---

## Run it (Google Colab)

There are three notebooks, each self-contained (clone + install + download + run — just open and run top to bottom):

| Notebook | What it does | Where |
|---|---|---|
| **[`AIC25_Pipeline.ipynb`](AIC25_Pipeline.ipynb)** | Main pipeline: detection → ReID → single-camera tracking → tracklet-repair ablation (Path A, 2 cameras by default, no ground truth), plus an optional heavier Path B for multi-camera + official 3D-HOTA. Start here to confirm the pipeline runs. | Colab, T4 GPU |
| **[`Tier2_HOTA.ipynb`](Tier2_HOTA.ipynb)** | Cross-scene generalization: trains the learned ReID tracklet matcher on one warehouse, tests it on a different unseen one (raw → conservative → learned, against ground truth). | Colab, T4 GPU |
| **[`Local_CrossScene.ipynb`](Local_CrossScene.ipynb)** | Same cross-scene experiment as above, for a local machine with an NVIDIA GPU instead of Colab. | Local, NVIDIA GPU |

To run any of them on Colab:

1. Open [Google Colab](https://colab.research.google.com/), then **File → Upload notebook** and pick the file (or open it straight from GitHub via **File → Open notebook → GitHub**).
2. **Runtime → Change runtime type → T4 GPU.**
3. Add a HuggingFace token as a Colab secret (key icon on the left panel): name `HF_TOKEN`, value from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Accept the dataset terms once at [huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces).
4. Run all cells top to bottom.

**`AIC25_Pipeline.ipynb` in more detail:** Path A defaults to 2 cameras (`Camera`, `Camera_01`) on `Warehouse_016`, so a full run fits in one Colab session — edit the `CAMERAS` and `SCENE` variables in its config cell to cover more cameras or another scene. Path B (depth maps, all cameras, official 3D-HOTA) is a separate, clearly marked section further down — optional, and only worth running once Path A works.

All three notebooks cache outputs and downloaded models to Google Drive, so re-running after a disconnect only repeats the parts that changed.

---

## Code Flow / Architecture

Each stage below is a separate script, called from `AIC25_Pipeline.ipynb` in this order. Inputs/outputs are real file paths, not simplified.

```text
HuggingFace dataset (nvidia/PhysicalAI-SmartSpaces)
        │  videos/, calibration.json, ground_truth.json
        ▼
[1] Detection            BoT-SORT/tools/aic25_get_detection.py
        │  → Detection/<Scene>/<Camera>.json   (per-frame bounding boxes)
        ▼
[2] ReID embeddings      deep-person-reid/torchreid/aic25_extract.py   (OSNet backbone)
        │  → EmbedFeature/<Scene>/...          (per-detection appearance vectors)
        ▼
[3] Single-camera track  BoT-SORT/single_camera_tracking.py            (BoT-SORT tracker)
        │  → Tracking/Singlecamera/<Scene>/<Camera>/<Camera>.json
        ▼
[4] Single-camera fix    BoT-SORT/single_camera_fix.py                 (NMS-based tracklet merge)
        │  → Tracking/Singlecamera/<Scene>/<Camera>/fixed_<Camera>.json
        ▼
[5] Tracklet repair      tracklet_repair/src/evaluation/run_ablation.py
        │  (baseline vs interpolation-only vs merge-only vs full repair)
        │  → tracklet_repair/results/ablation_<Scene>_<Camera>/ablation.{json,md}
        ▼
   Path A stops here — before/after comparison table.

── Path B (optional, heavier) ──────────────────────────────────────────────
[6] Depth maps           streamed per camera inside the notebook (Step B1)
        ▼
[3'/4'] re-run [3] and [4] with --use_depth True → 3D world coordinates
        ▼
[7] Multi-camera track   BoT-SORT/multi_camera_revised.py + multi_camera_fix.py
        │  → Tracking/Multicamera/<Scene>/<exp>/output_result/
        ▼
[8] Evaluation           TrackEval/prepare_eval_data.py → TrackEval/main.py
        │  → HOTA / DetA / AssA / LocA scores
```

**Subproject 1's ground-truth result** (the one that matters for the report) lives in `Tier2_HOTA.ipynb` / `Local_CrossScene.ipynb`, not in `AIC25_Pipeline.ipynb` — it trains on one scene and evaluates on a held-out one:

```text
Single-camera JSONs (train scene)
        ▼
tracklet_repair/src/matcher/train_matcher.py     trains a small MLP on GT-labelled tracklet pairs
        │  → tracklet_repair/models/<TrainScene>_matcher
        ▼
Single-camera JSONs (unseen test scene)
        ▼
tracklet_repair/src/evaluation/benchmark_scene.py   scores raw vs conservative (heuristic) vs learned
        │  against ground_truth.json — IDF1, MOTA, ID switches, fragmentations
        ▼
   comparison table (raw / conservative / learned, per camera + aggregate)
```

`tracklet_repair/src/evaluation/tune_botsort.py` (BoT-SORT matching-threshold sweep against ground truth) is an optional add-on, also runnable from either notebook.

See [`tracklet_repair/README.md`](tracklet_repair/README.md) for the module's own structure (`analysis/`, `evaluation/`, `postprocess/`, `matcher/`, `utils/`) and [`tracklet_repair/docs/`](tracklet_repair/docs/) for method notes and experiment logs.

---

## Project Overview

**Primary Goal:** Build and improve a system to track multiple objects across multiple cameras while keeping their identities consistent.

**Example Scenario:** If a person or robot appears in one camera and later in another, the system should assign the same ID.

### The Tracking Pipeline

1. Object detection in each camera.
2. Formation of short trajectories (tracklets) over time.
3. Global matching of tracklets across different cameras.

---

## Subproject 1: Single-Camera Tracking Consistency

**Goal:** Improve the quality and stability of tracking within each individual camera.

**Task:** Analyze why tracklets break and design methods to make them consistent.

### Directions

- Tune BoT-SORT parameters (matching thresholds, motion models).
- Improve tracklet merging (Sequential NMS) — see [`tracklet_repair/`](tracklet_repair/README.md).
- Handle occlusions and missed detections.

**Expected Outcome:** More stable tracklets with fewer ID switches.
