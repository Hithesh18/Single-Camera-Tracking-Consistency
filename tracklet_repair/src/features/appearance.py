"""Lightweight appearance features extracted from tracked object crops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


APPEARANCE_BACKENDS = ("hsv", "rgb", "combined")


def frame_path(frames_dir: Path, frame_pattern: str, frame_id: int) -> Path:
    """Resolve one frame path from a format pattern containing ``frame_id``."""
    try:
        name = frame_pattern.format(frame_id=int(frame_id))
    except (KeyError, ValueError) as error:
        raise ValueError("frame_pattern must contain a valid {frame_id} field.") from error
    return frames_dir / name


def load_frame(path: Path) -> np.ndarray | None:
    """Load an RGB frame, returning None for missing or unreadable images."""
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    except (OSError, ValueError):
        return None


def clip_bbox(
    bbox: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[int, int, int, int] | None:
    """Clip an x/y/width/height box to an image boundary."""
    x, y, width, height = (float(value) for value in bbox)
    x1 = max(0, min(image_width, int(np.floor(x))))
    y1 = max(0, min(image_height, int(np.floor(y))))
    x2 = max(0, min(image_width, int(np.ceil(x + width))))
    y2 = max(0, min(image_height, int(np.ceil(y + height))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def extract_crop(
    frame: np.ndarray | None, bbox: tuple[float, float, float, float]
) -> np.ndarray | None:
    """Extract a clipped RGB crop, returning None for invalid input."""
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        return None
    clipped = clip_bbox(bbox, frame.shape[1], frame.shape[0])
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    crop = frame[y1:y2, x1:x2, :3]
    return crop.copy() if crop.size else None


def histogram_feature(
    crop: np.ndarray | None, backend: str = "combined", bins: int = 16
) -> np.ndarray | None:
    """Build a normalized RGB, HSV, or combined color histogram feature."""
    if backend not in APPEARANCE_BACKENDS:
        raise ValueError(f"Unknown appearance backend: {backend}.")
    if crop is None or crop.size == 0 or bins <= 0:
        return None

    image = Image.fromarray(crop.astype(np.uint8), mode="RGB")
    parts = []
    if backend in ("rgb", "combined"):
        parts.append(_channel_histograms(np.asarray(image), bins, 256))
    if backend in ("hsv", "combined"):
        parts.append(_channel_histograms(np.asarray(image.convert("HSV")), bins, 256))
    feature = np.concatenate(parts).astype(np.float64)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else None


def cosine_similarity(feature_a: np.ndarray | None, feature_b: np.ndarray | None) -> float | None:
    """Return cosine similarity for two compatible features."""
    if feature_a is None or feature_b is None or feature_a.shape != feature_b.shape:
        return None
    denominator = np.linalg.norm(feature_a) * np.linalg.norm(feature_b)
    if denominator == 0:
        return None
    return float(np.clip(np.dot(feature_a, feature_b) / denominator, -1.0, 1.0))


class AppearanceFeatureCache:
    """Cache frame and crop features used by repeated candidate scoring."""

    def __init__(self, frames_dir: Path, frame_pattern: str, backend: str) -> None:
        if backend not in APPEARANCE_BACKENDS:
            raise ValueError(f"Unknown appearance backend: {backend}.")
        self.frames_dir = Path(frames_dir)
        self.frame_pattern = frame_pattern
        self.backend = backend
        self._frames: dict[int, np.ndarray | None] = {}
        self._features: dict[tuple, np.ndarray | None] = {}

    def feature(self, frame_id: int, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
        key = (int(frame_id), *(round(float(value), 3) for value in bbox), self.backend)
        if key not in self._features:
            if frame_id not in self._frames:
                self._frames[frame_id] = load_frame(
                    frame_path(self.frames_dir, self.frame_pattern, frame_id)
                )
            crop = extract_crop(self._frames[frame_id], bbox)
            self._features[key] = histogram_feature(crop, self.backend)
        return self._features[key]

    def endpoint_feature(self, rows, from_start: bool, window: int) -> np.ndarray | None:
        """Average the first or last K valid crop features from track rows."""
        if window <= 0:
            raise ValueError("appearance_window must be positive.")
        ordered = rows.sort_values("frame_id", ascending=from_start)
        features = []
        for _, row in ordered.iterrows():
            bbox = (row["x"], row["y"], row["width"], row["height"])
            feature = self.feature(int(row["frame_id"]), bbox)
            if feature is not None:
                features.append(feature)
            if len(features) == window:
                break
        if not features:
            return None
        mean = np.mean(features, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else None


def _channel_histograms(image: np.ndarray, bins: int, maximum: int) -> np.ndarray:
    histograms = [
        np.histogram(image[..., channel], bins=bins, range=(0, maximum))[0]
        for channel in range(3)
    ]
    feature = np.concatenate(histograms).astype(np.float64)
    total = feature.sum()
    return feature / total if total > 0 else feature
