"""
Module 4 — parameter_calculator.py
Responsibility: Compute stride-by-stride spatio-temporal gait parameters using
EXACTLY the same formulas and definitions as the DUO-GAIT pipeline.

All formulas are cited to the DUO-GAIT source file and line number.

=== DUO-GAIT formula sources ===
File: src/LFRF_parameters/pipeline/gait_parameters.py

    stride_time  = IC[i+1] - IC[i]          (lines 254-256)
    stance_time  = FO[i]   - IC[i]          (lines 397-398)
    swing_time   = IC[i+1] - FO[i]          (lines 386-388)
    stride_length = ‖pos_xy[IC_{i+1}] − pos_xy[IC_i]‖  (lines 225-247)
    stance_ratio = stance_time / stride_time  (line 94, summary_raw)
    cadence      = 120 / stride_time          (aggregate_gait_parameters.py:39)
                   (steps/min; 120 because a stride = 2 steps)
    speed        = stride_length / stride_time  (aggregate_gait_parameters.py:40)

Additional parameters NOT in DUO-GAIT (flagged):
    step_time        = time from HS of one foot to HS of contralateral foot
                       # NOT FOUND IN DUO-GAIT SOURCE — computed for completeness
    double_support_time = time from HS of one foot to TO of contralateral foot
                       # NOT FOUND IN DUO-GAIT SOURCE — computed for completeness

Outlier detection (gait_parameters.py:127-201):
    angle_change > 0.2  → is_outlier + turning_step
    stride_length < 0.2 → is_outlier
    stride_time   > 2.0 → is_outlier
    stance_ratio  < 0.5 → is_outlier
    |z-score| > 3 on:   stride_lengths, stride_times, swing_times,
                         stance_times, stance_ratios
    (angle_change not applicable in video domain; turning detection uses
     anterior-posterior heel velocity reversal — see outlier_remover.py)

Input:
    gait_events : dict  — output of event_detector.events_to_gait_event_dict()
    traj_df     : pd.DataFrame — filtered trajectory (output of preprocessor)

Output schema (one row per stride):
    pd.DataFrame with columns:
        foot            str   — 'left' or 'right'
        stride_index    int   — sequential index within this foot's strides
        timestamps      float — time of stride start (IC[i]), seconds
        stride_lengths  float — metres
        stride_times    float — seconds
        swing_times     float — seconds
        stance_times    float — seconds
        stance_ratios   float — dimensionless
        fo_times        float — seconds (FO event time for this stride)
        ic_times        float — seconds (IC[i+1] time)
        fo_samples      int   — frame index of FO
        ic_samples      int   — frame index of IC[i+1]
        step_time       float — seconds  [NOT IN DUO-GAIT]
        double_support_time float — seconds  [NOT IN DUO-GAIT]
        is_outlier      bool
        turning_step    bool
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_parameters(
    gait_events: dict,
    traj_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute stride-by-stride gait parameters for both feet.

    Parameters
    ----------
    gait_events : dict
        Nested dict from event_detector.events_to_gait_event_dict().
        Keys: 'stance_begin', 'stance_end', 'left', 'right'.
        Each foot sub-dict has 'samples' and 'times' with 'IC' and 'FO' arrays.
    traj_df : pd.DataFrame
        Filtered trajectory DataFrame (from preprocessor.preprocess()).
        Must have: frame, time_s, left_heel_x, left_heel_y,
                   right_heel_x, right_heel_y.

    Returns
    -------
    pd.DataFrame
        One row per stride for each foot, combined and sorted by timestamps.
    """
    stance_begin = gait_events["stance_begin"]  # "IC"
    stance_end   = gait_events["stance_end"]    # "FO"

    # Build a frame-indexed lookup for heel positions (for stride_length)
    frame_arr = traj_df["frame"].values

    all_strides: list[pd.DataFrame] = []

    for side in ("left", "right"):
        ic_times   = np.array(gait_events[side]["times"][stance_begin])   # IC timestamps
        fo_times   = np.array(gait_events[side]["times"][stance_end])     # FO timestamps
        ic_samples = np.array(gait_events[side]["samples"][stance_begin]) # IC frame indices
        fo_samples = np.array(gait_events[side]["samples"][stance_end])   # FO frame indices

        # ------------------------------------------------------------------
        # Align IC and FO arrays so that every stride has exactly one FO
        # between its start IC and end IC.
        # DUO-GAIT adjust_data (gait_parameters.py:22-64) enforces:
        #   - first event must be IC (stance_begin)
        #   - last  event must be IC (stance_begin)
        #   - len(IC) == len(FO) + 1
        # ------------------------------------------------------------------
        ic_times, fo_times, ic_samples, fo_samples = _align_events(
            ic_times, fo_times, ic_samples, fo_samples
        )

        if len(ic_times) < 2 or len(fo_times) < 1:
            continue  # not enough events to form even one stride

        n_strides = len(ic_times) - 1  # DUO-GAIT: strides = IC pairs

        # ------------------------------------------------------------------
        # Core temporal parameters — DUO-GAIT gait_parameters.py
        # ------------------------------------------------------------------
        # stride_time = IC[i+1] - IC[i]   (line 254-256)
        stride_times = ic_times[1:] - ic_times[:-1]

        # stance_time = FO[i] - IC[i]     (line 397-398)
        stance_times = fo_times - ic_times[:n_strides]

        # swing_time = IC[i+1] - FO[i]   (line 386-388)
        swing_times = ic_times[1:] - fo_times

        # stance_ratio = stance_time / stride_time  (line 94)
        stance_ratios = stance_times / stride_times

        # ------------------------------------------------------------------
        # Stride length — DUO-GAIT gait_parameters.py line 225-247
        # Uses x-y position of heel (proxy for foot position) at IC events.
        # In the IMU domain, foot trajectories are estimated via integration;
        # in the video domain we read heel position directly from traj_df.
        # ------------------------------------------------------------------
        stride_lengths = _compute_stride_lengths(
            side, ic_samples, traj_df
        )

        # ------------------------------------------------------------------
        # Additional parameters NOT in DUO-GAIT
        # # NOT FOUND IN DUO-GAIT SOURCE — ASSUMPTION: standard definition
        # step_time = time from IC of this foot to IC of contralateral foot
        # double_support_time = time from IC of this foot to TO of contra foot
        # These require cross-foot alignment; computed post-hoc in the
        # combined DataFrame.
        # ------------------------------------------------------------------
        step_times          = np.full(n_strides, np.nan)
        double_support_times = np.full(n_strides, np.nan)

        # ------------------------------------------------------------------
        # Assemble stride-level DataFrame — mirrors DUO-GAIT summary_raw
        # (gait_parameters.py:66-108)
        # ------------------------------------------------------------------
        df = pd.DataFrame({
            "foot":          side,
            "stride_index":  np.arange(n_strides),
            "timestamps":    ic_times[:n_strides],          # IC[i]  (line 86)
            "stride_lengths": stride_lengths,
            "stride_times":  stride_times,
            "swing_times":   swing_times,
            "stance_times":  stance_times,
            "stance_ratios": stance_ratios,
            "fo_times":      fo_times,                      # line 95
            "ic_times":      ic_times[1:],                  # IC[i+1] line 96
            "fo_samples":    fo_samples,                    # line 97
            "ic_samples":    ic_samples[1:],                # line 98
            "step_time":     step_times,
            "double_support_time": double_support_times,
        })

        df = _init_outlier_columns(df)
        all_strides.append(df)

    if not all_strides:
        return pd.DataFrame()

    combined = pd.concat(all_strides, ignore_index=True)

    # ------------------------------------------------------------------
    # Cross-foot step time and double-support time
    # NOT FOUND IN DUO-GAIT SOURCE — ASSUMPTION: standard gait definitions
    # ------------------------------------------------------------------
    combined = _fill_cross_foot_params(combined, gait_events)

    combined.sort_values("timestamps", inplace=True, ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Outlier columns — initialisation
# ---------------------------------------------------------------------------

def _init_outlier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_outlier and turning_step columns initialised to False."""
    df = df.copy()
    df["is_outlier"]   = False
    df["turning_step"] = False
    return df


# ---------------------------------------------------------------------------
# Outlier flagging — threshold-based detection
# File: LFRF_parameters/pipeline/gait_parameters.py lines 127-201
# ---------------------------------------------------------------------------

def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark strides with non-physiological parameter values as outliers.

    Two sequential passes mirror the DUO-GAIT approach
    (gait_parameters.py lines 127-201):

    Pass 1 — Threshold-based detection
        Hard physiological limits; strides outside these bounds are almost
        certainly artifacts from phantom tracking or corrupted pose estimation.
        Applied across both feet combined.

    Pass 2 — Z-score detection (|z| > 3) per foot
        Applied ONLY to strides that survived Pass 1 so that extreme
        threshold-violating values do not skew the z-score statistics.
        Replicates DUO-GAIT gait_parameters.py lines 127-201.
        Minimum of 4 valid strides per foot required; fewer and the
        z-score is undefined/unreliable, so Pass 2 is skipped for that foot.

    This function should be called AFTER speed factor correction so that
    thresholds are evaluated against real-time values.
    """
    if df.empty:
        return df

    df = df.copy()

    # ------------------------------------------------------------------
    # Pass 1: Threshold-based outlier detection
    # Values outside these ranges are non-physiological for human gait.
    # ------------------------------------------------------------------
    checks = {
        "stride_times":   (0.4, 3.0),   # Normal: 0.8–1.4 s
        "stride_lengths": (0.2, 3.0),   # Normal: 1.0–1.8 m
        "stance_ratios":  (0.1, 0.95),  # Relaxed from DUO-GAIT's (0.3, 0.9)
                                         # — video-domain FO detection yields
                                         #   systematically lower stance ratios
        "swing_times":    (0.1, 2.0),   # Normal: 0.3–0.5 s
        "stance_times":   (0.2, 2.5),   # Normal: 0.5–0.9 s
    }
    for col, (lo, hi) in checks.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            df.loc[mask, "is_outlier"] = True

    # ------------------------------------------------------------------
    # Pass 2: Z-score detection — DUO-GAIT gait_parameters.py lines 127-201
    # Applied per foot on strides that passed the threshold filter.
    # z-score statistics are computed from valid (non-threshold-outlier)
    # strides of each foot to avoid contamination by extreme values.
    # ------------------------------------------------------------------
    _Z_SCORE_THRESHOLD = 3.0
    _MIN_STRIDES_FOR_ZSCORE = 4

    z_cols = [
        "stride_lengths", "stride_times",
        "swing_times", "stance_times", "stance_ratios",
    ]

    for side in ("left", "right"):
        side_mask  = df["foot"] == side
        valid_mask = side_mask & (df["is_outlier"] == False)

        for col in z_cols:
            if col not in df.columns:
                continue

            valid_vals = df.loc[valid_mask, col].dropna()
            if len(valid_vals) < _MIN_STRIDES_FOR_ZSCORE:
                continue  # not enough data for reliable z-score

            mean = valid_vals.mean()
            std  = valid_vals.std(ddof=1)
            if std == 0.0:
                continue  # constant values — z-score undefined

            # Compute z-scores for all strides of this foot (not just valid
            # ones) so that already-threshold-flagged strides are not
            # doubly-flagged but outlier detection is complete.
            col_vals = df.loc[side_mask, col]
            z_scores = (col_vals - mean).abs() / std
            df.loc[side_mask & (z_scores > _Z_SCORE_THRESHOLD), "is_outlier"] = True

    return df


# ---------------------------------------------------------------------------
# Stride length — DUO-GAIT gait_parameters.py lines 225-247
# ---------------------------------------------------------------------------

def _compute_stride_lengths(
    side: str,
    ic_samples: np.ndarray,
    traj_df: pd.DataFrame,
) -> np.ndarray:
    """
    Compute stride length as the anterior-posterior (x-axis) displacement of
    the heel between successive IC events.

    The coordinate system produced by Sports2D has:
        x — anterior-posterior (direction of walking, positive = forward)
        y — vertical (positive = upward)

    DUO-GAIT uses the norm of the foot trajectory in the floor plane
    (gait_parameters.py line 247: np.linalg.norm(step[0:2])).  In the IMU
    domain those two axes are both horizontal (AP + ML).  In the sagittal
    video domain the two available axes are AP (x) and vertical (y).
    Including y inflates stride length by the heel's vertical oscillation
    during swing phase, which is not part of the walking distance.  Only
    the AP displacement abs(dx) is used here.
    """
    frame_to_row = {f: i for i, f in enumerate(traj_df["frame"].values)}
    x_col = f"{side}_heel_x"

    lengths: list[float] = []
    n_strides = len(ic_samples) - 1
    for i in range(n_strides):
        s_frame = ic_samples[i]
        e_frame = ic_samples[i + 1]
        try:
            s_row = frame_to_row[s_frame]
            e_row = frame_to_row[e_frame]
        except KeyError:
            lengths.append(np.nan)
            continue

        dx = traj_df[x_col].iat[e_row] - traj_df[x_col].iat[s_row]
        lengths.append(abs(float(dx)))

    return np.array(lengths)


# ---------------------------------------------------------------------------
# Event alignment — mirrors DUO-GAIT adjust_data (gait_parameters.py:22-64)
# ---------------------------------------------------------------------------

def _align_events(
    ic_times:   np.ndarray,
    fo_times:   np.ndarray,
    ic_samples: np.ndarray,
    fo_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure the recording begins with IC and ends with IC, and that each
    stride has exactly one FO between its bounding ICs.
    Replicates gait_parameters.py adjust_data() (lines 22-64) in the video
    domain where events are frame indices rather than IMU sample numbers.

    FO selection: within each (IC[i], IC[i+1]) interval, the **LAST** FO is
    chosen.  In normal gait the sequence is IC → stance → FO → swing → IC,
    so the FO naturally occurs late in the stride cycle.  Picking the first
    FO would select spurious toe-Y minima right after heel-strike.
    """
    if len(ic_times) == 0 or len(fo_times) == 0:
        return ic_times, fo_times, ic_samples, fo_samples

    # Drop leading FO events that precede the first IC (line 33-43)
    while len(fo_times) > 0 and fo_times[0] <= ic_times[0]:
        fo_times   = fo_times[1:]
        fo_samples = fo_samples[1:]

    # Drop trailing FO that follows the last IC (line 45-56)
    while len(fo_times) > 0 and fo_times[-1] >= ic_times[-1]:
        fo_times   = fo_times[:-1]
        fo_samples = fo_samples[:-1]

    if len(fo_times) == 0:
        return ic_times[:1], fo_times, ic_samples[:1], fo_samples

    # ------------------------------------------------------------------
    # Select exactly one FO per IC interval.
    # For each consecutive IC pair (IC[i], IC[i+1]), keep only the LAST
    # FO that falls within that interval.  This is physiologically
    # correct: FO (toe-off) naturally occurs late in the stance phase,
    # just before swing begins.
    # ------------------------------------------------------------------
    kept_ic_times:   list[float] = []
    kept_ic_samples: list[int]   = []
    kept_fo_times:   list[float] = []
    kept_fo_samples: list[int]   = []

    for i in range(len(ic_times) - 1):
        t_start = ic_times[i]
        t_end   = ic_times[i + 1]

        # Collect all FOs within (t_start, t_end)
        fos_in_interval = [
            (fo_times[j], fo_samples[j])
            for j in range(len(fo_times))
            if t_start < fo_times[j] < t_end
        ]

        if fos_in_interval:
            # Pick the LAST FO in the interval (closest to next IC = end of stance)
            best_fo_t, best_fo_s = fos_in_interval[-1]
            kept_ic_times.append(t_start)
            kept_ic_samples.append(ic_samples[i])
            kept_fo_times.append(best_fo_t)
            kept_fo_samples.append(best_fo_s)
        # else: skip this IC pair (no valid FO in the interval)

    # Add the final IC that closes the last stride
    if kept_fo_times:
        last_fo_t = kept_fo_times[-1]
        for i in range(len(ic_times)):
            if ic_times[i] > last_fo_t:
                kept_ic_times.append(ic_times[i])
                kept_ic_samples.append(ic_samples[i])
                break

    ic_times   = np.array(kept_ic_times)
    fo_times   = np.array(kept_fo_times)
    ic_samples = np.array(kept_ic_samples, dtype=int)
    fo_samples = np.array(kept_fo_samples, dtype=int)

    # Final sanity check: len(IC) must be len(FO) + 1
    n_fo = len(fo_times)
    if len(ic_times) > n_fo + 1:
        ic_times   = ic_times[:n_fo + 1]
        ic_samples = ic_samples[:n_fo + 1]

    return ic_times, fo_times, ic_samples, fo_samples


# ---------------------------------------------------------------------------
# Cross-foot parameters — NOT IN DUO-GAIT SOURCE
# ---------------------------------------------------------------------------

def _fill_cross_foot_params(
    combined: pd.DataFrame,
    gait_events: dict,
) -> pd.DataFrame:
    """
    Compute step_time and double_support_time for each stride by finding the
    nearest contralateral IC / FO event.

    # NOT FOUND IN DUO-GAIT SOURCE — ASSUMPTION: standard gait definitions:
    #   step_time        = time from IC(this) to IC(contra)
    #   double_support_time = time from IC(this) to FO(contra)
    """
    stance_begin = gait_events["stance_begin"]
    stance_end   = gait_events["stance_end"]

    for side in ("left", "right"):
        contra = "right" if side == "left" else "left"
        contra_ic = np.array(gait_events[contra]["times"][stance_begin])
        contra_fo = np.array(gait_events[contra]["times"][stance_end])

        mask = combined["foot"] == side
        for idx in combined[mask].index:
            t_ic = combined.at[idx, "timestamps"]

            # step_time: contra IC immediately after this IC
            future_contra_ics = contra_ic[contra_ic > t_ic]
            if len(future_contra_ics) > 0:
                combined.at[idx, "step_time"] = future_contra_ics[0] - t_ic

            # double_support_time: contra FO immediately after this IC
            future_contra_fos = contra_fo[contra_fo > t_ic]
            if len(future_contra_fos) > 0:
                combined.at[idx, "double_support_time"] = future_contra_fos[0] - t_ic

    return combined
