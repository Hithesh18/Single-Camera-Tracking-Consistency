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
- Improve tracklet merging (Sequential NMS).
- Handle occlusions and missed detections.

**Expected Outcome:** More stable tracklets with fewer ID switches.
