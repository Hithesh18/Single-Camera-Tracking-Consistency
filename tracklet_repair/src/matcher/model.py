"""Small learned tracklet-affinity model (replaces the geometric if/else cost).

A compact MLP maps a pair-feature vector (appearance + motion + temporal) to
P(same identity). It is deliberately small so it trains on CPU and does not
overfit the modest number of ground-truth tracklet pairs available per scene.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tracklet_repair.src.matcher.features import PAIR_FEATURE_DIM, PAIR_FEATURE_NAMES


class MatcherMLP(nn.Module):
    def __init__(self, in_dim: int = PAIR_FEATURE_DIM, hidden: tuple[int, int] = (32, 16)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits


class LearnedMatcher:
    """Inference wrapper: standardizes features, returns P(same identity)."""

    def __init__(self, model: MatcherMLP, mean: np.ndarray, std: np.ndarray):
        self.model = model.eval()
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def probability(self, features: np.ndarray) -> float:
        x = (features - self.mean) / self.std
        with torch.no_grad():
            logit = self.model(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
            return float(torch.sigmoid(logit).item())

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        x = (features - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x.astype(np.float32)))
            return torch.sigmoid(logits).numpy()

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path.with_suffix(".pt"))
        meta = {
            "feature_names": PAIR_FEATURE_NAMES,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "LearnedMatcher":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        model = MatcherMLP()
        model.load_state_dict(torch.load(path.with_suffix(".pt"), map_location="cpu"))
        return cls(model, np.array(meta["mean"]), np.array(meta["std"]))
