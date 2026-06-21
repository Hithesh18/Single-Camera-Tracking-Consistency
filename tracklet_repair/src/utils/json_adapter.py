"""Adapters for BoT-SORT single-camera JSON tracking outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tracklet_repair.src.utils.io import TRACK_COLUMNS, validate_tracking_dataframe


CLASS_NAME_TO_ID = {
    "Person": 0,
    "Forklift": 1,
    "NovaCarter": 2,
    "Transporter": 3,
    "FourierGR1T2": 4,
    "AgilityDigit": 5,
}


def load_single_camera_json(path: str) -> dict:
    """Load a BoT-SORT single-camera JSON file."""
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Single-camera JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Single-camera JSON must be a frame-indexed object.")
    return data


def single_camera_json_to_dataframe(data: dict) -> pd.DataFrame:
    """Convert frame-indexed single-camera JSON to the standard track DataFrame."""
    rows = []
    saw_object = False
    saw_bbox = False

    for frame_key, frame_objects in data.items():
        frame_id = _parse_frame_id(frame_key)
        objects = _as_object_list(frame_objects)

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            saw_object = True

            track_id = _extract_track_id(obj)
            bbox_info = _extract_bbox(obj)
            class_id = _extract_class_id(obj)
            score = _extract_score(obj)

            if track_id is None:
                continue
            if bbox_info is None:
                continue
            saw_bbox = True

            bbox, bbox_format = bbox_info
            x, y, width, height = _bbox_to_xywh(bbox, bbox_format)
            rows.append(
                {
                    "frame_id": frame_id,
                    "track_id": track_id,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "score": score,
                    "class_id": class_id,
                }
            )

    if not saw_object:
        raise ValueError("No track objects found in single-camera JSON.")
    if not saw_bbox:
        raise ValueError("No usable 2D bounding boxes found in single-camera JSON.")
    if not rows:
        raise ValueError("No usable tracks found with both track IDs and bboxes.")

    df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
    validate_tracking_dataframe(df)
    return df.sort_values(["frame_id", "track_id"]).reset_index(drop=True)


def load_single_camera_json_as_dataframe(path: str) -> pd.DataFrame:
    """Load and convert a single-camera JSON file in one step."""
    return single_camera_json_to_dataframe(load_single_camera_json(path))


def _parse_frame_id(frame_key: Any) -> int:
    try:
        return int(frame_key)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid frame id in JSON key: {frame_key}") from error


def _as_object_list(frame_objects: Any) -> list:
    if frame_objects is None:
        return []
    if isinstance(frame_objects, list):
        return frame_objects
    if isinstance(frame_objects, dict):
        return [frame_objects]
    return []


def _extract_track_id(obj: dict) -> int | None:
    for key in (
        "object sc id",
        "object_sc_id",
        "object_scid",
        "track_id",
        "id",
        "OfflineID",
    ):
        value = obj.get(key)
        if value is not None:
            return int(value)
    return None


def _extract_bbox(obj: dict) -> tuple[list[float], str] | None:
    direct_keys = {
        "bbox": "auto",
        "bbox_2d": "auto",
        "2d_bbox": "auto",
        "tlwh": "xywh",
        "xywh": "xywh",
        "2d bounding box": "auto",
    }
    for key, bbox_format in direct_keys.items():
        bbox = obj.get(key)
        if _is_bbox(bbox):
            return [float(value) for value in bbox], bbox_format

    visible = obj.get("2d bounding box visible") or obj.get("bbox_visible")
    if isinstance(visible, dict):
        for bbox in visible.values():
            if _is_bbox(bbox):
                return [float(value) for value in bbox], "xyxy"
    elif _is_bbox(visible):
        return [float(value) for value in visible], "xyxy"

    return None


def _extract_class_id(obj: dict) -> int:
    for key in ("class_id", "ClassID", "Class", "object class id"):
        value = obj.get(key)
        if value is not None:
            return int(value)

    class_name = obj.get("object type") or obj.get("class_name") or obj.get("class")
    return CLASS_NAME_TO_ID.get(str(class_name), 0)


def _extract_score(obj: dict) -> float:
    for key in ("score", "confidence", "Confidence", "track_score"):
        value = obj.get(key)
        if value is not None:
            return float(value)
    return 1.0


def _is_bbox(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 4


def _bbox_to_xywh(bbox: list[float], bbox_format: str) -> tuple[float, float, float, float]:
    x1, y1, third, fourth = bbox
    if bbox_format == "xyxy" or (
        bbox_format == "auto" and third > x1 and fourth > y1
    ):
        return x1, y1, third - x1, fourth - y1
    return x1, y1, third, fourth
