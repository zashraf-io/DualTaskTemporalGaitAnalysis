"""
Module 1 — input_loader.py
Responsibility: Load raw coordinate data from a Sports2D .trc file and return a
standardised trajectory DataFrame.

Coordinate system (output):
    x  — anterior-posterior axis (direction of travel; positive = forward)
    y  — vertical axis (positive = upward)
    All units in metres when Sports2D was run with --to_meters true.

TRC column layout (from Sports2D process.py:trc_data_from_XYZtime):
    time | kpt1_X  kpt1_Y  kpt1_Z | kpt2_X  kpt2_Y  kpt2_Z | …
    Z is always 0 for 2D video; we discard it.

Keypoint names used (Sports2D Body_with_feet / HALPE_26 model,
Config_demo.toml custom skeleton block):
    LHeel, RHeel, LBigToe, RBigToe
    (LAnkle / RAnkle loaded optionally for downstream use)

Output schema:
    DataFrame with columns:
        frame          int   — 0-based frame index
        time_s         float — frame / fps
        left_heel_x    float
        left_heel_y    float
        right_heel_x   float
        right_heel_y   float
        left_toe_x     float
        left_toe_y     float
        right_toe_x    float
        right_toe_y    float
    Optional extra columns (present if keypoints found in file):
        left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public constants — keypoint names as written by Sports2D (HALPE_26 model)
# ---------------------------------------------------------------------------
_KPT_LEFT_HEEL  = "LHeel"
_KPT_RIGHT_HEEL = "RHeel"
_KPT_LEFT_TOE   = "LBigToe"
_KPT_RIGHT_TOE  = "RBigToe"
_KPT_LEFT_ANK   = "LAnkle"
_KPT_RIGHT_ANK  = "RAnkle"

_REQUIRED_KEYPOINTS = [_KPT_LEFT_HEEL, _KPT_RIGHT_HEEL,
                       _KPT_LEFT_TOE,  _KPT_RIGHT_TOE]
_OPTIONAL_KEYPOINTS = [_KPT_LEFT_ANK, _KPT_RIGHT_ANK]


def load_trc(trc_path: str | Path, fps: float = 120.0) -> tuple[pd.DataFrame, float]:
    """
    Parse a Sports2D .trc file and return the standardised trajectory DataFrame
    along with the detected frame rate.

    Parameters
    ----------
    trc_path : str or Path
        Path to the .trc file produced by Sports2D with --to_meters true.
        The file may be named like  *_m_person00.trc  (metres variant).
    fps : float
        Recording frame rate used to compute time_s = frame / fps.
        Sports2D embeds the frame rate in the TRC header; this value is used
        as a fallback if the header cannot be parsed.

    Returns
    -------
    tuple[pd.DataFrame, float]
        A tuple of (trajectory_df, detected_fps).
        trajectory_df columns: frame, time_s,
                 left_heel_x, left_heel_y,
                 right_heel_x, right_heel_y,
                 left_toe_x, left_toe_y,
                 right_toe_x, right_toe_y
        (plus ankle columns if present in the file)
        detected_fps: the frame rate read from the TRC header (or the
                 fallback value if the header could not be parsed).

    Raises
    ------
    FileNotFoundError
        If trc_path does not exist.
    ValueError
        If required keypoints (LHeel, RHeel, LBigToe, RBigToe) are missing.
    """
    trc_path = Path(trc_path)
    if not trc_path.exists():
        raise FileNotFoundError(f"TRC file not found: {trc_path}")

    # ------------------------------------------------------------------
    # 1. Read raw TRC text and locate key header lines
    # ------------------------------------------------------------------
    with open(trc_path, "r") as fh:
        lines = fh.readlines()

    # TRC files vary in header layout. We auto-detect by finding:
    #   - The numeric header line containing fps (first line with a float)
    #   - The marker-name line (contains "Frame#" or marker names like LHeel)
    #   - The coordinate label line (contains X1 Y1 Z1 ...)
    # Typical layout from Sports2D v0.8.29:
    #   Line 0: PathFileType ...
    #   Line 1: DataRate  CameraRate  NumFrames  NumMarkers  Units ...  (labels)
    #   Line 2: 30.0      30.0        178        22          m     ...  (values)
    #   Line 3: Frame#  Time  Hip  RHip  RKnee  ...                    (markers)
    #   Line 4:         X1 Y1 Z1 X2 Y2 Z2 ...                         (coords)
    #   Line 5+: data rows

    fps_header_line = None   # the line with numeric fps value
    marker_line_idx = None   # the line with marker names
    data_start_line = None   # first data row

    for i, line in enumerate(lines[:10]):
        stripped = line.strip()
        fields = stripped.split("\t")

        # Find marker-name line: contains "Frame#" at start
        if fields[0].strip() == "Frame#":
            marker_line_idx = i
            continue

        # Find coordinate label line: contains X1, Y1, Z1 pattern
        if any(f.strip() in ("X1", "Y1", "Z1") for f in fields):
            data_start_line = i + 1
            continue

        # Find fps line: first field is a parseable float and not a header keyword
        if fps_header_line is None and i > 0:
            try:
                val = float(fields[0])
                fps_header_line = i
            except (ValueError, IndexError):
                pass

    # Fallback to legacy layout if auto-detect didn't find markers
    if marker_line_idx is None:
        marker_line_idx = 2
    if data_start_line is None:
        data_start_line = marker_line_idx + 2

    fps_from_header = _parse_fps_from_header(
        lines, fps_line=fps_header_line, fallback=fps
    )

    # ------------------------------------------------------------------
    # 2. Extract marker names from the marker-name line
    # ------------------------------------------------------------------
    marker_line = lines[marker_line_idx].rstrip("\n").split("\t")
    # First tokens are Frame#, Time; every 3rd token after that is a name
    # Sports2D writes: Frame#\tTime\tKpt1\t\t\tKpt2\t\t\t …
    marker_names: list[str] = []
    for tok in marker_line[2:]:
        tok = tok.strip()
        if tok:
            marker_names.append(tok)

    # ------------------------------------------------------------------
    # 3. Read numeric data (skip header rows)
    # ------------------------------------------------------------------
    data = pd.read_csv(
        trc_path,
        sep="\t",
        skiprows=data_start_line,
        header=None,
        engine="python",
    )
    # Drop any all-NaN trailing columns that some editors leave
    data.dropna(axis=1, how="all", inplace=True)
    # Coerce all columns to numeric (handles stray strings/empty fields)
    data = data.apply(pd.to_numeric, errors="coerce")

    # Column layout: Frame  Time  X1 Y1 Z1  X2 Y2 Z2 …
    n_markers = len(marker_names)
    expected_cols = 2 + 3 * n_markers
    if data.shape[1] < expected_cols:
        raise ValueError(
            f"TRC data has {data.shape[1]} columns but expected "
            f"{expected_cols} (2 + 3×{n_markers} markers)."
        )

    frame_col = data.iloc[:, 0].astype(int)
    time_col  = data.iloc[:, 1]

    # Build a dict: marker_name → (x_series, y_series)
    kpt_data: dict[str, tuple[pd.Series, pd.Series]] = {}
    for idx, name in enumerate(marker_names):
        col_x = 2 + 3 * idx
        col_y = 3 + 3 * idx
        kpt_data[name] = (data.iloc[:, col_x].reset_index(drop=True),
                          data.iloc[:, col_y].reset_index(drop=True))

    # ------------------------------------------------------------------
    # 4. Validate required keypoints
    # ------------------------------------------------------------------
    missing = [k for k in _REQUIRED_KEYPOINTS if k not in kpt_data]
    if missing:
        raise ValueError(
            f"Required keypoints {missing} not found in TRC file.\n"
            f"Available keypoints: {list(kpt_data.keys())}"
        )

    # ------------------------------------------------------------------
    # 5. Assemble output DataFrame
    # ------------------------------------------------------------------
    n = len(frame_col)
    out = pd.DataFrame({
        "frame":        frame_col.values,
        "time_s":       time_col.values,
        "left_heel_x":  kpt_data[_KPT_LEFT_HEEL][0].values,
        "left_heel_y":  kpt_data[_KPT_LEFT_HEEL][1].values,
        "right_heel_x": kpt_data[_KPT_RIGHT_HEEL][0].values,
        "right_heel_y": kpt_data[_KPT_RIGHT_HEEL][1].values,
        "left_toe_x":   kpt_data[_KPT_LEFT_TOE][0].values,
        "left_toe_y":   kpt_data[_KPT_LEFT_TOE][1].values,
        "right_toe_x":  kpt_data[_KPT_RIGHT_TOE][0].values,
        "right_toe_y":  kpt_data[_KPT_RIGHT_TOE][1].values,
    })

    # The TRC 'Time' column (read into out["time_s"]) already contains the 
    # correct timestamps (including any segment offsets from stitching), 
    # so we DO NOT recompute it from frame/fps here.

    # Optionally include ankles
    for side, kpt_name in [("left",  _KPT_LEFT_ANK),
                            ("right", _KPT_RIGHT_ANK)]:
        if kpt_name in kpt_data:
            out[f"{side}_ankle_x"] = kpt_data[kpt_name][0].values
            out[f"{side}_ankle_y"] = kpt_data[kpt_name][1].values

    out.reset_index(drop=True, inplace=True)
    return out, fps_from_header


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_fps_from_header(
    lines: list[str], fps_line: int | None, fallback: float
) -> float:
    """
    Extract DataRate (fps) from the auto-detected TRC header line.

    Parameters
    ----------
    lines : list[str]
        All lines of the TRC file.
    fps_line : int or None
        Index of the line containing numeric fps value. None to use fallback.
    fallback : float
        Value to return if parsing fails.
    """
    if fps_line is None:
        return fallback
    try:
        fields = lines[fps_line].strip().split("\t")
        return float(fields[0])
    except Exception:
        return fallback


def load_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alternative entry-point: accept a pre-built DataFrame that already
    conforms to the output schema and return it unchanged (identity loader).

    Useful for testing or when the caller has already constructed the
    trajectory table from a non-TRC source (e.g., CSV from another tool).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at minimum: frame, time_s, left_heel_x, left_heel_y,
        right_heel_x, right_heel_y, left_toe_x, left_toe_y,
        right_toe_x, right_toe_y.

    Returns
    -------
    pd.DataFrame
        The same DataFrame, validated.
    """
    required = [
        "frame", "time_s",
        "left_heel_x", "left_heel_y",
        "right_heel_x", "right_heel_y",
        "left_toe_x", "left_toe_y",
        "right_toe_x", "right_toe_y",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")
    return df.copy()
