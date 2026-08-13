"""Input and output helpers for tracking files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TRACK_COLUMNS = [
    "frame_id",
    "track_id",
    "x",
    "y",
    "width",
    "height",
    "score",
    "class_id",
]


def load_tracking_file(path: str) -> pd.DataFrame:
    """Load a MOT-style tracking text file (no header row) into a sorted DataFrame."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Tracking file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, header=None, sep=",", skipinitialspace=True)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Tracking file is empty: {file_path}") from error

    if df.shape[1] != len(TRACK_COLUMNS):
        raise ValueError(
            f"Expected {len(TRACK_COLUMNS)} columns in {file_path}, "
            f"but found {df.shape[1]}."
        )

    df.columns = TRACK_COLUMNS
    validate_tracking_dataframe(df)

    df = df.sort_values(["frame_id", "track_id"]).reset_index(drop=True)
    return df


def validate_tracking_dataframe(df: pd.DataFrame) -> None:
    """Validate and convert tracking columns in place."""
    missing_columns = [column for column in TRACK_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required tracking columns: {missing_columns}")

    if df[TRACK_COLUMNS].isna().any().any():
        raise ValueError("Tracking data contains missing values.")

    for column in TRACK_COLUMNS:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = converted.isna()
        if invalid_mask.any():
            bad_value = df.loc[invalid_mask, column].iloc[0]
            raise ValueError(f"Invalid numeric value in column '{column}': {bad_value}")
        df[column] = converted

    integer_columns = ["frame_id", "track_id", "class_id"]
    for column in integer_columns:
        if not (df[column] % 1 == 0).all():
            raise ValueError(f"Column '{column}' must contain integer values.")
        df[column] = df[column].astype(int)


def save_tracking_file(df: pd.DataFrame, path: str) -> None:
    """Save tracking data as a comma-separated text file without a header."""
    output_path = Path(path)
    tracks = df.copy()
    validate_tracking_dataframe(tracks)

    sorted_df = tracks.sort_values(["frame_id", "track_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_df[TRACK_COLUMNS].to_csv(output_path, index=False, header=False)


def load_tracks(path: Path) -> pd.DataFrame:
    """Compatibility wrapper for older scaffold modules."""
    return load_tracking_file(str(path))


def save_tracks(tracks: pd.DataFrame, path: Path) -> None:
    """Compatibility wrapper for older scaffold modules."""
    save_tracking_file(tracks, str(path))
