"""
batch_runner.py
Batch processing module for running the gait analysis pipeline across an
entire dataset directory.

Input directory structure:
    maindir/
      01/
        single.mp4, dual.mp4, single.csv, dual.csv, master.csv
      02/
        ...

master.csv (per-participant, one row):
    height,fps,speed_factor
    1.72,30,8.0

Output directory structure:
    out/
      sub_01/   (regular per-participant output)
      sub_02/
      ...
      master/
        master.csv   (combined results + Average row)
        dtc_bar_chart.png
        st_vs_dt_comparison.png
        dtc_heatmap.png
        parameter_boxplots.png
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

try:
    from PySide6.QtCore import QThread, Signal, QMetaObject, Qt, Q_ARG
except ImportError:
    from PyQt6.QtCore import QThread, pyqtSignal as Signal

# Pipeline modules
from gait import input_loader
from gait import preprocessor
from gait import event_detector
from gait import parameter_calculator
from gait import outlier_remover
from gait import aggregator
from gait import dtc_calculator

logger = logging.getLogger(__name__)

# Required files per participant folder
_REQUIRED_FILES = ["single.mp4", "dual.mp4", "single.csv", "dual.csv", "master.csv"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParticipantConfig:
    """Parsed configuration for one participant from their folder."""
    participant_id: str
    folder: Path
    st_video: Path
    dt_video: Path
    st_boundaries_csv: Path
    dt_boundaries_csv: Path
    height_m: float = 1.70
    fps: float = 30.0
    speed_factor: float = 1.0
    invert_y: bool = False


class ErrorAction(Enum):
    SKIP = "skip"
    RETRY = "retry"
    CANCEL = "cancel"


# ---------------------------------------------------------------------------
# Directory discovery & validation
# ---------------------------------------------------------------------------

def discover_participants(input_dir: Path) -> list[Path]:
    """
    Scan input_dir for numbered subdirectories, sorted naturally.

    Returns list of directory Paths (e.g., [input_dir/01, input_dir/02, ...]).
    """
    folders = sorted(
        [d for d in input_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if not folders:
        raise FileNotFoundError(
            f"No subdirectories found in {input_dir}. "
            f"Expected numbered folders like 01/, 02/, etc."
        )
    return folders


def validate_participant_folder(folder: Path) -> list[str]:
    """
    Check that a participant folder contains all required files.

    Returns list of missing file names (empty if all present).
    """
    missing = []
    for fname in _REQUIRED_FILES:
        if not (folder / fname).exists():
            missing.append(fname)
    return missing


def parse_master_csv(csv_path: Path) -> dict:
    """
    Parse the per-participant master.csv (one row: height, fps, speed_factor).

    Returns dict with keys: height, fps, speed_factor.

    Height unit handling
    --------------------
    The ``height`` field must be in **metres** (e.g. 1.72) because it is
    passed directly to Sports2D's ``--first_person_height`` argument, which
    expects metres.  As a safety net, any value greater than 3.0 is treated
    as centimetres and silently converted (a human taller than 3 m is
    physiologically impossible, so the value must be in cm).  A warning is
    logged so the data-entry error is visible in the run log.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"master.csv is empty: {csv_path}")

    row = df.iloc[0]
    height = float(row.get("height", 1.70))

    # Guard against height entered in centimetres instead of metres.
    # 3.0 m is physiologically impossible; anything above it must be cm.
    if height > 3.0:
        logger.warning(
            f"{csv_path}: height value {height} exceeds 3.0 — "
            f"treating as centimetres and converting to {height / 100:.3f} m. "
            f"Update the master.csv to use metres to silence this warning."
        )
        height /= 100.0

    # invert_y: accept 1/0, true/false, yes/no (case-insensitive)
    invert_raw = str(row.get("invert_y", "false")).strip().lower()
    invert_y = invert_raw in ("1", "true", "yes")

    return {
        "height": height,
        "fps": float(row.get("fps", 30.0)),
        "speed_factor": float(row.get("speed_factor", 1.0)),
        "invert_y": invert_y,
    }


def build_participant_config(folder: Path) -> ParticipantConfig:
    """Build a ParticipantConfig from a validated folder."""
    meta = parse_master_csv(folder / "master.csv")
    return ParticipantConfig(
        participant_id=f"sub_{folder.name}",
        folder=folder,
        st_video=folder / "single.mp4",
        dt_video=folder / "dual.mp4",
        st_boundaries_csv=folder / "single.csv",
        dt_boundaries_csv=folder / "dual.csv",
        height_m=meta["height"],
        fps=meta["fps"],
        speed_factor=meta["speed_factor"],
        invert_y=meta["invert_y"],
    )


# ---------------------------------------------------------------------------
# Synchronous single-participant pipeline
# ---------------------------------------------------------------------------

def run_single_participant_sync(
    config: ParticipantConfig,
    output_dir: Path,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Run the full pipeline for one participant synchronously.

    Uses PipelineRunner's stage logic but runs in the calling thread.
    Returns dict of all stage results.
    """
    from runners.pipeline_runner import PipelineRunner

    runner = PipelineRunner(
        participant_id=config.participant_id,
        st_input=str(config.st_video),
        dt_input=str(config.dt_video),
        st_is_video=True,
        dt_is_video=True,
        height_m=config.height_m,
        fps=config.fps,
        output_dir=str(output_dir),
        st_boundaries_csv=str(config.st_boundaries_csv),
        dt_boundaries_csv=str(config.dt_boundaries_csv),
        speed_factor=config.speed_factor,
        invert_y=config.invert_y,
    )

    # Run synchronously (call run() directly instead of start())
    runner.run()

    return runner._results


# ---------------------------------------------------------------------------
# Master output generation
# ---------------------------------------------------------------------------

def generate_master_output(
    all_results: list[dict],
    participant_configs: list[ParticipantConfig],
    master_dir: Path,
) -> pd.DataFrame:
    """
    Collect all per-participant results into a master CSV with an Average row,
    and generate summary graphs.

    The Average row's DTC is recomputed from averaged ST/DT values,
    NOT from averaging per-participant DTCs.

    Returns the master DataFrame.
    """
    master_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    st_agg_list = []
    dt_agg_list = []

    for i, results in enumerate(all_results):
        cfg = participant_configs[i]
        pid = cfg.participant_id

        st_agg = results.get("aggregate_st")
        dt_agg = results.get("aggregate_dt")
        dtc_payload = results.get("dtc", {})
        dtc_df = dtc_payload.get("dtc") if isinstance(dtc_payload, dict) else None

        if st_agg is None or dt_agg is None or dtc_df is None:
            logger.warning(f"Skipping {pid} in master output — missing aggregated data")
            continue

        st_agg_list.append(st_agg)
        dt_agg_list.append(dt_agg)

        # Build one row per participant
        row = {"sub": pid, "height_m": cfg.height_m, "fps": cfg.fps,
               "speed_factor": cfg.speed_factor}

        # ST columns (suffix _st)
        for col in st_agg.columns:
            if col in ("sub", "condition"):
                continue
            row[f"{col}_st"] = float(st_agg[col].iloc[0])

        # DT columns (suffix _dt)
        for col in dt_agg.columns:
            if col in ("sub", "condition"):
                continue
            row[f"{col}_dt"] = float(dt_agg[col].iloc[0])

        # DTC columns (keep _DTC suffix as-is)
        for col in dtc_df.columns:
            if col in ("sub", "condition"):
                continue
            row[col] = float(dtc_df[col].iloc[0])

        rows.append(row)

    if not rows:
        logger.error("No participant data to generate master output.")
        return pd.DataFrame()

    master_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Average row: mean of ST/DT columns, DTC recomputed from averages
    # ------------------------------------------------------------------
    avg_row = {"sub": "AVERAGE", "height_m": np.nan, "fps": np.nan,
               "speed_factor": np.nan}

    # Average all _st and _dt columns
    st_cols = [c for c in master_df.columns if c.endswith("_st")]
    dt_cols = [c for c in master_df.columns if c.endswith("_dt")]
    dtc_cols = [c for c in master_df.columns if c.endswith("_DTC")]

    for col in st_cols:
        avg_row[col] = master_df[col].mean(skipna=True)

    for col in dt_cols:
        avg_row[col] = master_df[col].mean(skipna=True)

    # Recompute DTC from averaged ST/DT values (NOT from averaging DTCs)
    # DTC(%) = (X_ST - X_DT) / X_ST * 100
    for dtc_col in dtc_cols:
        # Map DTC column back to ST/DT column names
        # e.g., "stride_lengths_avg_DTC" -> ST: "stride_lengths_avg_st", DT: "stride_lengths_avg_dt"
        base_name = dtc_col.replace("_DTC", "")
        st_col_name = f"{base_name}_st"
        dt_col_name = f"{base_name}_dt"

        if st_col_name in avg_row and dt_col_name in avg_row:
            val_st = avg_row[st_col_name]
            val_dt = avg_row[dt_col_name]
            if pd.notna(val_st) and val_st != 0:
                avg_row[dtc_col] = (val_st - val_dt) / val_st * 100.0
            else:
                avg_row[dtc_col] = np.nan
        else:
            avg_row[dtc_col] = np.nan

    master_df = pd.concat([master_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save master CSV
    master_csv_path = master_dir / "master.csv"
    master_df.to_csv(master_csv_path, index=False)
    logger.info(f"Master CSV saved to {master_csv_path}")

    # Generate graphs
    try:
        _generate_graphs(master_df, master_dir)
    except Exception as e:
        logger.error(f"Graph generation failed: {e}")

    return master_df


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def _generate_graphs(master_df: pd.DataFrame, master_dir: Path):
    """Generate all summary graphs and save as PNG in master_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Separate participant rows from average row
    participants = master_df[master_df["sub"] != "AVERAGE"].copy()
    avg_row = master_df[master_df["sub"] == "AVERAGE"].iloc[0] if "AVERAGE" in master_df["sub"].values else None

    if participants.empty:
        return

    dtc_cols = [c for c in participants.columns if c.endswith("_DTC")]
    # Focus on _avg_ DTC columns for main charts
    avg_dtc_cols = [c for c in dtc_cols if "_avg_DTC" in c and "_left" not in c and "_right" not in c]

    # Color palette
    C_BG = "#1e1e1e"
    C_SURFACE = "#2a2a2a"
    C_TEXT = "#e0e0e0"
    C_MUTED = "#888888"
    C_POSITIVE = "#e05c5c"
    C_NEGATIVE = "#5cb85c"
    C_ACCENT = "#4a90d9"
    C_ORANGE = "#f0a030"

    # ── 1. DTC Bar Chart ──────────────────────────────────────────────
    if avg_dtc_cols and avg_row is not None:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor(C_SURFACE)

        labels = [c.replace("_avg_DTC", "").replace("_", " ").title() for c in avg_dtc_cols]
        values = [float(avg_row[c]) if pd.notna(avg_row[c]) else 0 for c in avg_dtc_cols]
        colors = [C_POSITIVE if v > 0 else C_NEGATIVE for v in values]

        bars = ax.bar(range(len(labels)), values, color=colors, width=0.6, edgecolor="#444")
        ax.axhline(0, color=C_MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, color=C_TEXT)
        ax.set_ylabel("DTC (%)", color=C_TEXT, fontsize=11)
        ax.set_title("Group Mean Dual-Task Cost (%)", color=C_TEXT, fontsize=13, fontweight="bold")
        ax.tick_params(colors=C_MUTED)
        for spine in ax.spines.values():
            spine.set_color("#444")

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                    color=C_TEXT, fontsize=8)

        fig.tight_layout()
        fig.savefig(master_dir / "dtc_bar_chart.png", dpi=150, facecolor=C_BG)
        plt.close(fig)
        logger.info("Saved dtc_bar_chart.png")

    # ── 2. ST vs DT Comparison ────────────────────────────────────────
    st_avg_cols = [c for c in participants.columns
                   if c.endswith("_avg_st") and "_left" not in c and "_right" not in c]
    if st_avg_cols and avg_row is not None:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor(C_SURFACE)

        labels = [c.replace("_avg_st", "").replace("_", " ").title() for c in st_avg_cols]
        st_vals = [float(avg_row[c]) if pd.notna(avg_row[c]) else 0 for c in st_avg_cols]
        dt_vals = [float(avg_row[c.replace("_st", "_dt")]) if c.replace("_st", "_dt") in avg_row.index and pd.notna(avg_row[c.replace("_st", "_dt")]) else 0 for c in st_avg_cols]

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, st_vals, w, label="Single-Task", color=C_ACCENT, edgecolor="#444")
        ax.bar(x + w/2, dt_vals, w, label="Dual-Task", color=C_ORANGE, edgecolor="#444")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, color=C_TEXT)
        ax.set_ylabel("Value", color=C_TEXT, fontsize=11)
        ax.set_title("ST vs DT — Group Mean Parameters", color=C_TEXT, fontsize=13, fontweight="bold")
        ax.legend(facecolor=C_SURFACE, labelcolor=C_TEXT)
        ax.tick_params(colors=C_MUTED)
        for spine in ax.spines.values():
            spine.set_color("#444")

        fig.tight_layout()
        fig.savefig(master_dir / "st_vs_dt_comparison.png", dpi=150, facecolor=C_BG)
        plt.close(fig)
        logger.info("Saved st_vs_dt_comparison.png")

    # ── 3. DTC Heatmap ────────────────────────────────────────────────
    if avg_dtc_cols and len(participants) > 1:
        fig, ax = plt.subplots(figsize=(max(10, len(avg_dtc_cols) * 0.8), max(4, len(participants) * 0.5)))
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor(C_SURFACE)

        heatmap_data = participants[avg_dtc_cols].values.astype(float)
        param_labels = [c.replace("_avg_DTC", "").replace("_", " ").title() for c in avg_dtc_cols]
        sub_labels = participants["sub"].tolist()

        # Replace NaN with 0 for display
        heatmap_data = np.nan_to_num(heatmap_data, nan=0.0)

        vmax = max(abs(heatmap_data.min()), abs(heatmap_data.max()), 1)
        im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(param_labels)))
        ax.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=9, color=C_TEXT)
        ax.set_yticks(range(len(sub_labels)))
        ax.set_yticklabels(sub_labels, fontsize=9, color=C_TEXT)
        ax.set_title("Dual-Task Cost Heatmap (%) — Per Participant × Parameter",
                      color=C_TEXT, fontsize=12, fontweight="bold")

        # Annotate cells
        for i in range(len(sub_labels)):
            for j in range(len(param_labels)):
                val = heatmap_data[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        color="white" if abs(val) > vmax * 0.5 else C_TEXT, fontsize=7)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(colors=C_MUTED)
        cbar.set_label("DTC (%)", color=C_TEXT)

        fig.tight_layout()
        fig.savefig(master_dir / "dtc_heatmap.png", dpi=150, facecolor=C_BG)
        plt.close(fig)
        logger.info("Saved dtc_heatmap.png")

    # ── 4. Box Plots — ST vs DT distribution ─────────────────────────
    if st_avg_cols and len(participants) > 1:
        n_params = len(st_avg_cols)
        fig, axes = plt.subplots(1, n_params, figsize=(max(12, n_params * 2.5), 5))
        fig.patch.set_facecolor(C_BG)
        if n_params == 1:
            axes = [axes]

        for i, st_col in enumerate(st_avg_cols):
            ax = axes[i]
            ax.set_facecolor(C_SURFACE)
            dt_col = st_col.replace("_st", "_dt")

            st_data = participants[st_col].dropna().values
            dt_data = participants[dt_col].dropna().values if dt_col in participants.columns else np.array([])

            bp = ax.boxplot(
                [st_data, dt_data], labels=["ST", "DT"],
                patch_artist=True, widths=0.5,
                medianprops=dict(color=C_TEXT, linewidth=2),
                whiskerprops=dict(color=C_MUTED),
                capprops=dict(color=C_MUTED),
                flierprops=dict(markerfacecolor=C_MUTED, markersize=4),
            )
            bp["boxes"][0].set_facecolor(C_ACCENT)
            bp["boxes"][0].set_alpha(0.7)
            if len(bp["boxes"]) > 1:
                bp["boxes"][1].set_facecolor(C_ORANGE)
                bp["boxes"][1].set_alpha(0.7)

            param_name = st_col.replace("_avg_st", "").replace("_", " ").title()
            ax.set_title(param_name, color=C_TEXT, fontsize=9)
            ax.tick_params(colors=C_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#444")

        fig.suptitle("Parameter Distributions — ST vs DT", color=C_TEXT,
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(master_dir / "parameter_boxplots.png", dpi=150, facecolor=C_BG)
        plt.close(fig)
        logger.info("Saved parameter_boxplots.png")


# ---------------------------------------------------------------------------
# CLI batch entry point
# ---------------------------------------------------------------------------

def run_batch_cli(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Run the full batch pipeline from the command line.

    Discovers participants, validates folders, runs pipeline for each,
    and generates master output. Skips participants with missing files
    (logs a warning).

    Returns the master DataFrame.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    folders = discover_participants(input_dir)
    logger.info(f"Found {len(folders)} participant folders in {input_dir}")

    # Use a single list of (config, results) tuples so that a participant
    # failure never misaligns configs with results.  Previously the code
    # maintained two separate lists and patched configs[i] = None on failure,
    # but all_results was not padded with a placeholder — causing zip() to
    # pair surviving configs with the *wrong* results when any failure occurred
    # before a subsequent success, and silently dropping the last participant.
    completed: list[tuple[ParticipantConfig, dict]] = []

    for folder in folders:
        missing = validate_participant_folder(folder)
        if missing:
            logger.warning(
                f"Skipping {folder.name}: missing files: {', '.join(missing)}"
            )
            continue

        try:
            cfg = build_participant_config(folder)
        except Exception as e:
            logger.error(f"Error parsing config for {folder.name}: {e}")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing {cfg.participant_id} ({len(completed)+1} of {len(folders)})")
        print(f"{'='*60}")

        try:
            results = run_single_participant_sync(cfg, output_dir)
            completed.append((cfg, results))
        except Exception as e:
            logger.error(f"ERROR processing {cfg.participant_id}: {e}")
            import traceback
            traceback.print_exc()
            # Nothing is appended — the pair is simply skipped.  Both lists
            # stay perfectly aligned because they are the same list.

    if not completed:
        logger.error("All participants failed. No master output generated.")
        return pd.DataFrame()

    valid_configs = [c for c, r in completed]
    valid_results = [r for c, r in completed]

    master_df = generate_master_output(valid_results, valid_configs,
                                        master_dir=output_dir / "master")

    print(f"\n{'='*60}")
    print(f"  Batch complete: {len(valid_results)}/{len(folders)} participants processed")
    print(f"  Master output: {output_dir / 'master'}")
    print(f"{'='*60}")

    return master_df


# ---------------------------------------------------------------------------
# BatchPipelineRunner QThread (for UI)
# ---------------------------------------------------------------------------

class BatchPipelineRunner(QThread):
    """
    QThread-based batch runner for the UI.

    Signals
    -------
    batch_progress(participant_id, current_index, total, stage_name, status, pct)
        Emitted during pipeline execution per participant.
    participant_complete(participant_id, index, total)
        Emitted when a participant finishes successfully.
    file_error(participant_id, folder_path, missing_files)
        Emitted when required files are missing — UI should show
        Skip/Retry/Cancel dialog and call set_error_response().
    batch_finished(master_df_dict)
        Emitted on successful batch completion.
    batch_error(message)
        Emitted on unrecoverable failure.
    sports2d_progress(cond, pct, fps, eta)
        Forwarded from inner PipelineRunner.
    """

    batch_progress       = Signal(str, int, int, str, str, int)
    participant_complete  = Signal(str, int, int)
    file_error           = Signal(str, str, str)   # pid, folder, missing (comma-separated)
    batch_finished       = Signal(object)           # master DataFrame
    batch_error          = Signal(str)
    sports2d_progress    = Signal(str, int, float, str)

    def __init__(self, input_dir: str, output_dir: str, save_video: bool = False, segment_mode: bool = False, parent=None):
        super().__init__(parent)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.save_video = save_video
        self.segment_mode = segment_mode
        self._cancel_requested = False

        # Error response mechanism (thread-safe)
        self._error_response: Optional[ErrorAction] = None
        self._error_event = threading.Event()
        self._retry_folder: Optional[str] = None

    def request_cancel(self):
        """Request cancellation of the batch (checked between participants)."""
        self._cancel_requested = True

    def set_error_response(self, action: ErrorAction, retry_folder: str = ""):
        """Called from the UI thread after showing Skip/Retry/Cancel dialog."""
        self._error_response = action
        self._retry_folder = retry_folder if retry_folder else None
        self._error_event.set()

    def run(self):
        try:
            folders = discover_participants(self.input_dir)
        except FileNotFoundError as e:
            self.batch_error.emit(str(e))
            return

        total = len(folders)
        configs: list[ParticipantConfig] = []
        all_results: list[dict] = []

        # Process each folder
        idx = 0
        folder_list = list(folders)

        while idx < len(folder_list):
            if self._cancel_requested:
                self.batch_error.emit("Batch cancelled by user.")
                return

            folder = folder_list[idx]
            pid = f"sub_{folder.name}"

            # Validate
            missing = validate_participant_folder(folder)
            if missing:
                # Signal error to UI and wait for response
                self.file_error.emit(pid, str(folder), ", ".join(missing))
                self._error_event.wait()
                self._error_event.clear()

                action = self._error_response
                if action == ErrorAction.CANCEL:
                    self.batch_error.emit("Batch cancelled by user.")
                    return
                elif action == ErrorAction.SKIP:
                    logger.warning(f"Skipping {pid}: missing {', '.join(missing)}")
                    idx += 1
                    continue
                elif action == ErrorAction.RETRY:
                    if self._retry_folder:
                        folder_list[idx] = Path(self._retry_folder)
                    continue  # retry same index

            # Build config
            try:
                cfg = build_participant_config(folder_list[idx])
            except Exception as e:
                logger.error(f"Config parse error for {pid}: {e}")
                idx += 1
                continue

            # Run pipeline
            self.batch_progress.emit(pid, idx + 1, total, "Starting", "running", 0)

            try:
                from runners.pipeline_runner import PipelineRunner

                runner = PipelineRunner(
                    participant_id=cfg.participant_id,
                    st_input=str(cfg.st_video),
                    dt_input=str(cfg.dt_video),
                    st_is_video=True,
                    dt_is_video=True,
                    height_m=cfg.height_m,
                    fps=cfg.fps,
                    output_dir=str(self.output_dir),
                    st_boundaries_csv=str(cfg.st_boundaries_csv),
                    dt_boundaries_csv=str(cfg.dt_boundaries_csv),
                    speed_factor=cfg.speed_factor,
                    save_video=self.save_video,
                    segment_mode=self.segment_mode,
                    invert_y=cfg.invert_y,
                )

                # Forward progress signals
                runner.progress.connect(
                    lambda sname, status, pct, _pid=pid, _i=idx, _t=total:
                        self.batch_progress.emit(_pid, _i + 1, _t, sname, status, pct)
                )
                runner.sports2d_progress.connect(self.sports2d_progress.emit)

                # Run synchronously in this thread
                runner.run()

                configs.append(cfg)
                all_results.append(runner._results)

                self.participant_complete.emit(pid, idx + 1, total)

            except Exception as e:
                logger.error(f"Pipeline failed for {pid}: {e}")
                import traceback
                traceback.print_exc()

            idx += 1

        # Generate master output
        if all_results:
            master_df = generate_master_output(
                all_results, configs,
                master_dir=self.output_dir / "master",
            )
            self.batch_finished.emit(master_df)
        else:
            self.batch_error.emit("No participants completed successfully.")
