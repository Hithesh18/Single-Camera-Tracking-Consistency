# Repairing Identity Fragmentation in Single-Camera Multi-Object Tracking

> A post-hoc repair stage that merges fragmented tracklets and interpolates short gaps in BoT-SORT output, without
> touching the tracker. The merge decision is a learned nine-feature classifier (dominated by deep ReID appearance
> similarity) instead of a hand-tuned geometric cascade. Evaluated cross-scene — trained on one warehouse, tested on
> a different, unseen one — on the MTMC_Tracking_2025 benchmark.

## Authors

- Hakan Berke Şiranur — Institute of Artificial Intelligence, University of Stuttgart
- Hithesh Chettenahalli Honnegowda — Institute of Artificial Intelligence, University of Stuttgart

---

> Extends [**BoT-SORT: Robust Associations Multi-Pedestrian Tracking**](https://arxiv.org/abs/2206.14651) (Aharon et al.)
> as used by the single-camera stage of [**Glance-MCMT**](https://github.com/Hamidreza-Hashempoor/Glance-MCMT) (Hashempoor).

---

## Headline result

The learned matcher beats the rule-based (geometric) merge on both unseen test scenes:

| Metric | Warehouse_013 (train on 012) — Raw → Learned | Warehouse_015 (train on 016) — Raw → Learned |
|---|---|---|
| IDF1 ↑ | 0.571 → **0.638** (+0.067) | 0.606 → **0.628** (+0.022) |
| Identity switches ↓ | 64 → **26** (−59.4%) | 1528 → **1291** (−15.5%) |

The rule-based variant only recovers +0.005 and +0.015 IDF1 on the same scenes — most of the gain comes from learning the merge decision rather than hand-setting its thresholds. Full results, per-camera breakdown, and the gap-cause attribution are in the project report (*Repairing Identity Fragmentation in Single-Camera Multi-Object Tracking*, Şiranur & Chettenahalli Honnegowda, 2026).

---

## Run it (Google Colab)

**[`Tier2_HOTA.ipynb`](Tier2_HOTA.ipynb) is the main notebook** — it reproduces the headline result above: trains the learned ReID tracklet matcher on one warehouse and tests it on a different, unseen one (raw → conservative (heuristic) → learned, scored against ground truth: IDF1, MOTA, ID switches, fragmentations).

| Notebook | What it does | Where |
|---|---|---|
| **[`Tier2_HOTA.ipynb`](Tier2_HOTA.ipynb)** ⭐ | **Main result.** Cross-scene generalization: trains the matcher on `TRAIN_SCENE`, tests it on unseen `TEST_SCENE`. | Colab, T4 GPU |
| [`Local_CrossScene.ipynb`](Local_CrossScene.ipynb) | Same cross-scene experiment, for a local machine with an NVIDIA GPU instead of Colab. | Local, NVIDIA GPU |

To run `Tier2_HOTA.ipynb` on Colab:

1. Open [Google Colab](https://colab.research.google.com/), then **File → Upload notebook** and pick `Tier2_HOTA.ipynb` (or open it straight from GitHub via **File → Open notebook → GitHub**).
2. **Runtime → Change runtime type → T4 GPU.**
3. Add a HuggingFace token as a Colab secret (key icon on the left panel): name `HF_TOKEN`, value from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Accept the dataset terms once at [huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces).
4. In the **Configuration** cell, set `TRAIN_SCENE` / `TEST_SCENE` (must differ) and `CAMERAS` — defaults to 2 cameras (`Camera`, `Camera_01`) so a run fits in one session; the cell has `# <-- change this` comments marking what to edit. Widen `CAMERAS` (or set it to `None` for all 12, matching the paper's eleven-camera setup) for fuller coverage.
5. Run all cells top to bottom. Results (`comparison.md`, raw vs conservative vs learned per camera + aggregate) are cached to Google Drive.

Both notebooks are self-contained (clone + install + download + run) and cache outputs/models to Google Drive, so re-running after a disconnect only repeats the parts that changed.

---

## Code Flow / Architecture

Each stage below is a separate script, called from `Tier2_HOTA.ipynb` / `Local_CrossScene.ipynb` (via `tracklet_repair/src/pipeline/cross_scene_runner.py`) in this order. Inputs/outputs are real file paths, not simplified.

```text
HuggingFace dataset (nvidia/PhysicalAI-SmartSpaces)
        │  videos/, calibration.json, ground_truth.json     — for TRAIN_SCENE, then TEST_SCENE
        ▼
[1] Detection            BoT-SORT/tools/aic25_get_detection.py
        │  → Detection/<Scene>/<Camera>.json   (per-frame bounding boxes)
        ▼
[2] ReID embeddings      deep-person-reid/torchreid/aic25_extract.py   (OSNet backbone)
        │  → EmbedFeature/<Scene>/...          (per-detection appearance vectors)
        ▼
[3] Single-camera track  BoT-SORT/single_camera_tracking.py            (BoT-SORT tracker, stock settings)
        │  → Tracking/Singlecamera/<Scene>/<Camera>/<Camera>.json
        ▼
[4] Single-camera fix    BoT-SORT/single_camera_fix.py                 (NMS-based tracklet merge)
        │  → Tracking/Singlecamera/<Scene>/<Camera>/fixed_<Camera>.json
        ▼
[5] Train matcher (TRAIN_SCENE only)   tracklet_repair/src/matcher/train_matcher.py
        │  9-feature pair descriptor (3 OSNet cosines + motion + temporal + shape/class)
        │  → 9 → 32 → 16 → 1 MLP, <900 params, CPU, full-batch Adam, 300 iters
        │  → tracklet_repair/models/<TrainScene>_matcher
        ▼
[6] Benchmark (TEST_SCENE, unseen)     tracklet_repair/src/evaluation/benchmark_scene.py
        │  raw vs conservative (geometric merge, Section 4.3) vs learned (Section 4.4)
        │  scored against ground_truth.json — IDF1, MOTA, ID switches, fragmentations
        ▼
   comparison.md — raw / conservative / learned, per camera + aggregate
```

`tracklet_repair/src/evaluation/tune_botsort.py` (BoT-SORT matching-threshold sweep against ground truth) is an optional add-on, also runnable from either notebook.

See [`tracklet_repair/README.md`](tracklet_repair/README.md) for the module's own structure (`analysis/`, `evaluation/`, `postprocess/`, `matcher/`, `utils/`) and [`tracklet_repair/docs/`](tracklet_repair/docs/) for method notes and experiment logs. The full method — candidate generation, the rule-based criterion, the pair descriptor, training, gap interpolation ordering, and the gap-cause attribution cascade — is written up in the project report.

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

---

## Optional: `AIC25_Pipeline.ipynb`

Not required for the headline result above — kept as a secondary notebook for two things the main experiment doesn't cover:

- A **single-scene** sanity check (no train/test split, no ground truth): detection → ReID → tracking → the four-way `tracklet_repair` ablation (baseline / interpolation-only / merge-only / full-repair), useful as a quick "does the pipeline still run" check.
- The **official 3D-HOTA score** (its optional Path B: depth maps + all cameras + `TrackEval`), for comparing against the Glance-MCMT paper's reported 43–51 HOTA. This report does not compute HOTA (see Section 5.6 of the paper) — Path B exists only if a paper-comparable number is wanted later.

Same Colab steps as above (upload, T4 GPU, `HF_TOKEN` secret, run top to bottom); its config cell defaults to `Warehouse_016` with 2 cameras.
