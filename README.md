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

**[`Tier2_HOTA.ipynb`](Tier2_HOTA.ipynb) is the main notebook** — it produces the actual result for this project: a learned ReID tracklet matcher trained on one warehouse and tested on a different, unseen one (raw → conservative (heuristic) → learned, scored against ground truth: IDF1, MOTA, ID switches, fragmentations).

| Notebook | What it does | Where |
|---|---|---|
| **[`Tier2_HOTA.ipynb`](Tier2_HOTA.ipynb)** ⭐ | **Main result.** Cross-scene generalization: trains the matcher on `TRAIN_SCENE`, tests it on unseen `TEST_SCENE`. | Colab, T4 GPU |
| [`Local_CrossScene.ipynb`](Local_CrossScene.ipynb) | Same cross-scene experiment, for a local machine with an NVIDIA GPU instead of Colab. | Local, NVIDIA GPU |
| [`AIC25_Pipeline.ipynb`](AIC25_Pipeline.ipynb) | Secondary/optional: single-scene pipeline sanity check (no train/test split, no ground truth) + official 3D-HOTA score for a paper comparison. Not needed for the main result. | Colab, T4 GPU |

To run `Tier2_HOTA.ipynb` on Colab:

1. Open [Google Colab](https://colab.research.google.com/), then **File → Upload notebook** and pick `Tier2_HOTA.ipynb` (or open it straight from GitHub via **File → Open notebook → GitHub**).
2. **Runtime → Change runtime type → T4 GPU.**
3. Add a HuggingFace token as a Colab secret (key icon on the left panel): name `HF_TOKEN`, value from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Accept the dataset terms once at [huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces).
4. In the **Configuration** cell, set `TRAIN_SCENE` / `TEST_SCENE` (must differ) and `CAMERAS` — defaults to 2 cameras (`Camera`, `Camera_01`) so a run fits in one session; the cell has `# <-- change this` comments marking what to edit. Widen `CAMERAS` (or set it to `None` for all 12) for fuller coverage.
5. Run all cells top to bottom. Results (`comparison.md`, raw vs conservative vs learned per camera + aggregate) are cached to Google Drive.

All three notebooks are self-contained (clone + install + download + run) and cache outputs/models to Google Drive, so re-running after a disconnect only repeats the parts that changed.

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
