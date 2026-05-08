"""
Module 2 — preprocessor.py
Responsibility: Apply any remaining smoothing to the trajectory data.

⚠ NO-OP STATUS (Sports2D default workflow):
    Sports2D applies a Butterworth low-pass filter (4th order, 6 Hz) to all
    keypoint coordinates BEFORE writing the .trc file.  This is confirmed at
    Sports2D/process.py lines 2101–2132 (pixel coordinates are filtered, then
    converted to metres).  The _m_personXX.trc file therefore contains
    already-filtered coordinates.

    When the input .trc came from Sports2D with:
        filter = true
        filter_type = 'butterworth'
        cut_off_frequency = 6 Hz
        order = 4
    … this module is a NO-OP and simply passes the data through.

    If the caller sets apply_filter=True (e.g., raw CSV from another tool
    that has NOT been pre-filtered), a zero-phase 4th-order Butterworth LP
    filter at 6 Hz is applied — matching Stenum et al. and the DUO-GAIT
    preprocessing approach.

Coordinate system (input and output):
    x — anterior-posterior (positive = forward)
    y — vertical (positive = upward)  ← confirmed Y-up from Sports2D
                                         convert_px_to_meters (process.py:1377)

Input schema (from Module 1):
    frame, time_s, left_heel_x, left_heel_y, right_heel_x, right_heel_y,
    left_toe_x, left_toe_y, right_toe_x, right_toe_y
    (optional: left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y)

Output schema: identical to input schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# Columns that carry coordinate data (subject to filtering)
_COORD_COLS = [
    "left_heel_x",  "left_heel_y",
    "right_heel_x", "right_heel_y",
    "left_toe_x",   "left_toe_y",
    "right_toe_x",  "right_toe_y",
    # ankle columns are included if present
    "left_ankle_x",  "left_ankle_y",
    "right_ankle_x", "right_ankle_y",
]

# Default Butterworth parameters — match Sports2D Config_demo.toml defaults
# and Stenum et al. / DUO-GAIT preprocessing approach
_DEFAULT_CUTOFF_HZ = 6.0
_DEFAULT_ORDER     = 4


def preprocess(
    traj_df: pd.DataFrame,
    fps: float = 120.0,
    apply_filter: bool = False,
    cutoff_hz: float = _DEFAULT_CUTOFF_HZ,
    order: int = _DEFAULT_ORDER,
    ensure_y_up: bool = True,
    force_invert_y: bool = False,
) -> pd.DataFrame:
    """
    Optionally filter trajectory data and ensure the coordinate convention.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Output of input_loader.load_trc() — standardised trajectory DataFrame.
    fps : float
        Sampling rate in Hz (needed only when apply_filter=True).
    apply_filter : bool
        If False (default), this function is a no-op — appropriate when the
        .trc was produced by Sports2D which pre-filters internally.
        If True, apply a zero-phase Butterworth LP filter at cutoff_hz Hz.
    cutoff_hz : float
        Low-pass cut-off frequency (Hz). Default 6 Hz matches Sports2D and
        DUO-GAIT / Stenum et al.
    order : int
        Filter order. Default 4 (zero-phase, so effective order is 8).
    ensure_y_up : bool
        If True (default), verify that the median heel-Y value is positive
        (indicating Y-up convention from Sports2D metres output).  If the
        median is negative, negate all Y columns so that ground is near 0 and
        "up" is positive.  This guards against accidentally loading a
        pixel-coordinate TRC where Y increases downward.
    force_invert_y : bool
        If True, unconditionally negate all Y columns regardless of the
        ensure_y_up heuristic.  Use this for subjects whose Sports2D output
        has an inverted Y-axis that the automatic heuristic does not catch.

    Returns
    -------
    pd.DataFrame
        Same schema as traj_df, with coordinates optionally filtered and
        Y-axis orientation corrected.
    """
    df = traj_df.copy()

    # ------------------------------------------------------------------
    # 1. Y-axis orientation check
    # ------------------------------------------------------------------
    if force_invert_y:
        # Manual override: unconditionally negate all Y columns
        y_cols = [c for c in df.columns if c.endswith("_y")]
        df[y_cols] = -df[y_cols]
    elif ensure_y_up:
        median_heel_y = np.nanmedian(df["left_heel_y"].values)
        if median_heel_y < 0:
            # Y is likely downward — negate all Y columns
            y_cols = [c for c in df.columns if c.endswith("_y")]
            df[y_cols] = -df[y_cols]

    # ------------------------------------------------------------------
    # 2. Filtering (no-op by default)
    # ------------------------------------------------------------------
    if not apply_filter:
        # Sports2D pre-filters before writing the TRC file.
        # Applying a second filter would distort the data.
        return df

    # Apply zero-phase (forward-backward) Butterworth LP filter
    nyquist = fps / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"cutoff_hz ({cutoff_hz}) must be less than Nyquist ({nyquist}). "
            f"Reduce cutoff_hz or increase fps."
        )
    b, a = butter(order, cutoff_hz / nyquist, btype="low", analog=False)

    coord_cols_present = [c for c in _COORD_COLS if c in df.columns]
    for col in coord_cols_present:
        signal = df[col].values.astype(float)
        # Replace NaN with interpolated values for filtering, then restore NaN
        nan_mask = np.isnan(signal)
        if nan_mask.all():
            continue
        if nan_mask.any():
            idx = np.arange(len(signal))
            signal = np.interp(idx, idx[~nan_mask], signal[~nan_mask])
        filtered = filtfilt(b, a, signal)
        filtered[nan_mask] = np.nan
        df[col] = filtered

    return df
