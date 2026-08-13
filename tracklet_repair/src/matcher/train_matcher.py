"""Train the learned tracklet matcher on ground-truth-labelled pairs.

For each camera: load raw tracks with OSNet embeddings, match tracklets to GT
object ids by IoU vote, form ordered candidate pairs (source ends before
target starts) labelled 1 if they share a GT id, and train a small MLP on
appearance+motion features to predict P(same identity). No temporal cap on
pairs, so it can learn long-gap re-identification. CPU-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tracklet_repair.src.evaluation.gt_benchmark import (
    _match_track_to_gt,
    load_gt_camera,
    load_tracks,
)
from tracklet_repair.src.evaluation.benchmark_scene import discover_cameras
from tracklet_repair.src.matcher.features import (
    BOX_COLS,
    build_embedding_index,
    build_tracklets,
    ordered_candidate_pairs,
    pair_features,
)
from tracklet_repair.src.matcher.model import LearnedMatcher, MatcherMLP


def build_dataset(
    gt_json: str,
    scene: str,
    cameras: list[str],
    tracking_dir: str,
    embed_root: str,
    iou_threshold: float = 0.5,
    max_gap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return feature matrix X, labels y, and the camera each pair came from."""
    features, labels, sources = [], [], []
    tracking_path = Path(tracking_dir)

    for camera in cameras:
        raw_path = tracking_path / "Singlecamera" / scene / camera / f"{camera}.json"
        if not raw_path.exists():
            print(f"[skip] {camera}: no tracking JSON")
            continue
        tracks = load_tracks(str(raw_path))
        gt = load_gt_camera(gt_json, camera)
        embed_index = build_embedding_index(str(Path(embed_root) / scene / camera))
        tracklets = build_tracklets(tracks, embed_index)

        # map each tracklet -> GT id
        gt_id_of = {}
        for tid, tl in tracklets.items():
            track_df = tracks[tracks["track_id"] == tid]
            gt_id_of[tid] = _match_track_to_gt(track_df, gt, iou_threshold)

        n_pairs = 0
        for src_id, tgt_id in ordered_candidate_pairs(tracklets, max_gap=max_gap):
            gi, gj = gt_id_of[src_id], gt_id_of[tgt_id]
            if gi is None or gj is None:
                continue
            vec = pair_features(tracklets[src_id], tracklets[tgt_id])
            features.append(vec)
            labels.append(1.0 if gi == gj else 0.0)
            sources.append(camera)
            n_pairs += 1
        n_emb = sum(len(t.embeddings) for t in tracklets.values())
        print(f"[ok] {camera}: {len(tracklets)} tracklets, {n_emb} embeddings matched, {n_pairs} labelled pairs")

    if not features:
        raise RuntimeError("No labelled pairs produced — check tracking/embedding paths.")
    return np.stack(features), np.array(labels, dtype=np.float32), sources


def train(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 300,
    lr: float = 1e-2,
    val_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[LearnedMatcher, dict]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    mean, std = X.mean(axis=0), X.std(axis=0)
    std_safe = np.where(std < 1e-6, 1.0, std)
    Xz = (X - mean) / std_safe

    idx = rng.permutation(len(Xz))
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    Xtr = torch.tensor(Xz[train_idx], dtype=torch.float32)
    ytr = torch.tensor(y[train_idx], dtype=torch.float32)
    Xva = torch.tensor(Xz[val_idx], dtype=torch.float32)
    yva = y[val_idx]

    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0])

    model = MatcherMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        optimizer.step()

    matcher = LearnedMatcher(model, mean, std_safe)
    metrics = _evaluate(matcher, Xva.numpy() * std_safe + mean, yva, X[train_idx], y[train_idx], mean, std_safe)
    metrics["train_pairs"] = int(len(train_idx))
    metrics["val_pairs"] = int(len(val_idx))
    metrics["positives"] = int(y.sum())
    metrics["total_pairs"] = int(len(y))
    return matcher, metrics


def _evaluate(matcher, Xva_raw, yva, Xtr_raw, ytr, mean, std) -> dict:
    def auc(probs, labels):
        pos = probs[labels == 1]
        neg = probs[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return float(wins / (len(pos) * len(neg)))

    va_probs = matcher.probabilities(Xva_raw)
    tr_probs = matcher.probabilities(Xtr_raw)
    return {
        "val_auc": auc(va_probs, yva),
        "train_auc": auc(tr_probs, ytr),
        "val_accuracy@0.5": float(((va_probs > 0.5) == (yva > 0.5)).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the learned tracklet matcher.")
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--cameras", nargs="*", default=None, help="omit to auto-discover")
    parser.add_argument("--tracking-dir", default="Tracking")
    parser.add_argument("--embed-root", default="EmbedFeature")
    parser.add_argument("--out", default="tracklet_repair/models/matcher", help="model path prefix")
    parser.add_argument("--max-gap", type=int, default=None, help="None = allow long-gap re-id pairs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    cameras = args.cameras or discover_cameras(Path(args.tracking_dir), args.scene)
    print(f"Cameras: {cameras}")
    X, y, _ = build_dataset(
        args.gt_json, args.scene, cameras, args.tracking_dir, args.embed_root,
        iou_threshold=args.iou_threshold, max_gap=args.max_gap,
    )
    print(f"\nDataset: {len(y)} pairs, {int(y.sum())} positive ({100*y.mean():.1f}%)")

    matcher, metrics = train(X, y, epochs=args.epochs)
    print("\n=== training result ===")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    matcher.save(args.out)
    print(f"\nSaved model to {args.out}.pt / {args.out}.json")


if __name__ == "__main__":
    main()
