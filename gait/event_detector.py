"""
Module 3 — event_detector.py
Responsibility: Detect heel-strike (HS = IC) and toe-off (TO = FO) gait events
from filtered keypoint trajectories, replicating the logical structure of the
DUO-GAIT / Tunca event detector as closely as the different signal domain allows.

=== DUO-GAIT approach (IMU domain) ===
Source: LFRF_parameters/event_detection/imu_event_detection.py
  1. Stance phases identified as periods where gyroscope magnitude < threshold
     (gyro_threshold_stance, line 6-51).
  2. Step boundaries = transitions from swing to stance (step_begins, line 74).
  3. Within each step, a tilt signal is computed from the highest-variance gyro
     axis integrated over time (lines 88-94).
  4. IC (Initial Contact / heel-strike) and FO (Foot Off / toe-off) are found
     as peaks in ±tilt_diff within search regions bounded by tilt peaks
     (lines 99-214).
  - prominence thresholds: fo_prom_threshold=1.5, ic_prom_threshold=0.1
  - search region prominence: fo/ic_search_threshold=0.7

=== Video-domain equivalent (this module) ===
Signal analogies:
  IMU gyro magnitude ↔ foot speed derived from heel/toe displacement
  Stance (gyro < threshold) ↔ heel-Y near ground (below adaptive threshold)
  IC (heel-strike) ↔ local MINIMUM in heel-Y trajectory
      (heel is lowest when it contacts the ground; Y-up convention)
  FO (toe-off) ↔ local MINIMUM in toe-Y trajectory just before toe lifts
      (toe is at its lowest ground contact point before swing onset)

Peak detection uses scipy.signal.find_peaks with prominence thresholds tuned
to produce results equivalent to the Tunca algorithm at 120 fps.

Output schema:
    pd.DataFrame with columns:
        foot        str   — 'left' or 'right'
        event_type  str   — 'HS' (heel-strike / IC) or 'TO' (toe-off / FO)
        frame       int   — frame index in the trajectory DataFrame
        time_s      float — timestamp in seconds

This module is replaceable: any event detector returning this schema can be
dropped in without changing downstream modules.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Peak-detection parameters (tuned for 120 fps; scale with fps if needed)
# ---------------------------------------------------------------------------

# Minimum vertical prominence (metres) for a heel-Y local minimum to count as IC.
# Equivalent to requiring a clear foot-lift above ground contact level.
# NOT FOUND IN DUO-GAIT SOURCE (domain-translated) — ASSUMPTION: 0.005 m
_IC_PROMINENCE_M = 0.005

# Minimum vertical prominence (metres) for a toe-Y local minimum to count as FO.
# Toe-Y has more noise than heel-Y, so a higher threshold is needed.
# NOT FOUND IN DUO-GAIT SOURCE (domain-translated) — ASSUMPTION: 0.008 m
_FO_PROMINENCE_M = 0.008

# Minimum number of frames between successive IC events of the same foot.
# At 120 fps, a stride shorter than 0.5 s (60 frames) is physiologically
# implausible; guards against double-detections.
# NOT FOUND IN DUO-GAIT SOURCE — ASSUMPTION based on ~0.5 s minimum stride
_MIN_IC_DISTANCE_S = 0.70  # seconds (real-time minimum between heel strikes)

# Minimum seconds between successive FO events (real-time)
_MIN_FO_DISTANCE_S = 0.70  # seconds


def detect_events(
    traj_df: pd.DataFrame,
    fps: float = 120.0,
    speed_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Detect heel-strike (HS) and toe-off (TO) gait events from trajectory data.

    Parameters
    ----------
    traj_df : pd.DataFrame
        Output of preprocessor.preprocess() — standardised, filtered trajectory.
        Must contain: frame, time_s, left_heel_y, right_heel_y,
                      left_toe_y, right_toe_y.
    fps : float
        Sampling rate (Hz) of the video (playback fps, NOT recording fps).
    speed_factor : float
        Slow-motion correction factor.  If the video was recorded at 240 fps
        and plays at 30 fps, speed_factor = 8.0.  A real-time 1-second stride
        appears as 8 seconds (240 frames) in the video.  The minimum-distance
        parameters are scaled by this factor so that events are not
        over-detected on slow-motion footage.

    Returns
    -------
    pd.DataFrame
        Columns: foot (str), event_type (str), frame (int), time_s (float).
        Sorted by time_s.
    """
    events: list[dict] = []

    for side in ("left", "right"):
        heel_y = traj_df[f"{side}_heel_y"].values.astype(float)
        toe_y  = traj_df[f"{side}_toe_y"].values.astype(float)
        frames = traj_df["frame"].values
        times  = traj_df["time_s"].values

        # Interpolate NaNs before peak detection (same spirit as DUO-GAIT
        # which operates on continuously sampled IMU data without NaN)
        heel_y = _interpolate_nans(heel_y)
        toe_y  = _interpolate_nans(toe_y)

        # Scale min-distance parameter to account for playback fps AND
        # slow-motion factor.  A real-time minimum of _MIN_IC_DISTANCE_S
        # translates to (min_s * speed_factor * fps) frames in the video.
        ic_dist = max(1, int(_MIN_IC_DISTANCE_S * speed_factor * fps))
        fo_dist = max(1, int(_MIN_FO_DISTANCE_S * speed_factor * fps))

        # ------------------------------------------------------------------
        # IC (heel-strike): local MINIMUM in heel_y
        # find_peaks works on maxima; invert signal to find minima
        # ------------------------------------------------------------------
        ic_idx, _ = find_peaks(
            -heel_y,
            prominence=_IC_PROMINENCE_M,
            distance=ic_dist,
        )

        # Filter IC events to remove anomalous peaks (e.g. at interval boundaries)
        # by requiring them to be within a small margin of the median peak value.
        if len(ic_idx) > 0:
            median_ic_y = np.median(heel_y[ic_idx])
            ic_idx = [idx for idx in ic_idx if abs(heel_y[idx] - median_ic_y) <= 0.05]

        for idx in ic_idx:
            events.append({
                "foot":       side,
                "event_type": "HS",
                "frame":      int(frames[idx]),
                "time_s":     float(times[idx]),
            })

        # ------------------------------------------------------------------
        # FO (toe-off): local MINIMUM in toe_y
        # ------------------------------------------------------------------
        fo_idx, _ = find_peaks(
            -toe_y,
            prominence=_FO_PROMINENCE_M,
            distance=fo_dist,
        )

        # Filter FO events to remove anomalous peaks
        if len(fo_idx) > 0:
            median_fo_y = np.median(toe_y[fo_idx])
            fo_idx = [idx for idx in fo_idx if abs(toe_y[idx] - median_fo_y) <= 0.05]

        for idx in fo_idx:
            events.append({
                "foot":       side,
                "event_type": "TO",
                "frame":      int(frames[idx]),
                "time_s":     float(times[idx]),
            })

    events_df = pd.DataFrame(events, columns=["foot", "event_type", "frame", "time_s"])
    events_df.sort_values("time_s", inplace=True, ignore_index=True)
    return events_df


def events_to_gait_event_dict(events_df: pd.DataFrame) -> dict:
    """
    Convert the flat events DataFrame into the nested dict structure used
    internally by GaitParameters (mirrors DUO-GAIT's event_detector output).

    Structure (matches DUO-GAIT event_detector.py:47-51):
        {
            "stance_begin": "IC",
            "stance_end":   "FO",
            "left":  {"samples": {"IC": np.array, "FO": np.array},
                      "times":   {"IC": list,     "FO": list}},
            "right": { … }
        }

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of detect_events().

    Returns
    -------
    dict
        Nested gait event dictionary compatible with parameter_calculator.py.
    """
    result: dict = {
        "stance_begin": "IC",
        "stance_end":   "FO",
    }

    for side in ("left", "right"):
        side_df = events_df[events_df["foot"] == side].copy()
        ic_df   = side_df[side_df["event_type"] == "HS"].sort_values("time_s")
        fo_df   = side_df[side_df["event_type"] == "TO"].sort_values("time_s")

        result[side] = {
            "samples": {
                "IC": ic_df["frame"].to_numpy(dtype=int),
                "FO": fo_df["frame"].to_numpy(dtype=int),
            },
            "times": {
                "IC": ic_df["time_s"].tolist(),
                "FO": fo_df["time_s"].tolist(),
            },
        }

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate_nans(signal: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN values in a 1-D array."""
    if not np.any(np.isnan(signal)):
        return signal
    idx = np.arange(len(signal))
    nan_mask = np.isnan(signal)
    if nan_mask.all():
        return np.zeros_like(signal)
    return np.interp(idx, idx[~nan_mask], signal[~nan_mask])
