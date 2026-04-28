"""
ui_main.py — PyQt6 / PySide6 dark-themed gait analysis desktop UI.

Layout:
    Left panel (280 px fixed) — participant input + controls
    Right area — tabbed results (Annotated Video, ST Params, DT Params,
                 Dual-Task Cost, Raw Data Table)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from PySide6.QtCore    import Qt, QThread, Signal, Slot, QTimer
    from PySide6.QtGui     import QColor, QFont, QPalette, QAction
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QLineEdit, QPushButton, QFileDialog, QProgressBar,
        QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
        QSplitter, QScrollArea, QComboBox, QDoubleSpinBox, QSpinBox,
        QGroupBox, QFrame, QRadioButton, QButtonGroup, QSizePolicy,
        QMessageBox, QCheckBox, QTextEdit, QAbstractItemView,
    )
    BACKEND = "PySide6"
except ImportError:
    from PyQt6.QtCore    import Qt, QThread, pyqtSignal as Signal, pyqtSlot as Slot, QTimer
    from PyQt6.QtGui     import QColor, QFont, QPalette, QAction
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QLineEdit, QPushButton, QFileDialog, QProgressBar,
        QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
        QSplitter, QScrollArea, QComboBox, QDoubleSpinBox, QSpinBox,
        QGroupBox, QFrame, QRadioButton, QButtonGroup, QSizePolicy,
        QMessageBox, QCheckBox, QTextEdit, QAbstractItemView,
    )
    BACKEND = "PyQt6"

# Qt Multimedia for video playback
try:
    if BACKEND == "PySide6":
        from PySide6.QtMultimediaWidgets import QVideoWidget
        from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PySide6.QtCore import QUrl
    else:
        from PyQt6.QtMultimediaWidgets import QVideoWidget
        from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PyQt6.QtCore import QUrl
    HAS_MULTIMEDIA = True
except ImportError:
    HAS_MULTIMEDIA = False

# Attempt pyqtgraph; fall back to matplotlib if unavailable
try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from runners.pipeline_runner import PipelineRunner, StageStatus, STATUS_ICON


# ---------------------------------------------------------------------------
# Color palette (Addendum 1 design specs)
# ---------------------------------------------------------------------------
C_BG        = "#1e1e1e"
C_SURFACE   = "#2a2a2a"
C_ACCENT    = "#4a90d9"
C_TEXT      = "#e0e0e0"
C_MUTED     = "#888888"
C_POSITIVE  = "#e05c5c"   # DTC positive (worsened) — red
C_NEGATIVE  = "#5cb85c"   # DTC negative (improved) — green
C_OUTLIER   = "#5c2a2a"   # Muted red for outlier rows in table
C_LEFT_FOOT = "#4a90d9"   # Blue
C_RIGHT_FOOT= "#f0a030"   # Orange


def _apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(C_BG))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(C_TEXT))
    pal.setColor(QPalette.ColorRole.Base,            QColor(C_SURFACE))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor("#252525"))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(C_BG))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(C_TEXT))
    pal.setColor(QPalette.ColorRole.Text,            QColor(C_TEXT))
    pal.setColor(QPalette.ColorRole.Button,          QColor(C_SURFACE))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(C_TEXT))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(C_ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(C_TEXT))
    pal.setColor(QPalette.ColorRole.Link,            QColor(C_ACCENT))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QGroupBox {{
            border: 1px solid #444; border-radius: 4px;
            margin-top: 1em; color: {C_MUTED}; font-size: 11px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; }}
        QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
            background: #333; border: 1px solid #555;
            border-radius: 3px; color: {C_TEXT}; padding: 2px 4px;
        }}
        QPushButton {{
            background: {C_SURFACE}; border: 1px solid #555;
            border-radius: 4px; color: {C_TEXT}; padding: 5px 12px;
        }}
        QPushButton:hover {{ background: #3a3a3a; border-color: {C_ACCENT}; }}
        QPushButton#run_btn {{
            background: {C_ACCENT}; border: none; font-weight: bold;
            color: white; padding: 8px 16px; border-radius: 5px;
        }}
        QPushButton#run_btn:hover {{ background: #5aa0e9; }}
        QPushButton#run_btn:disabled {{ background: #444; color: {C_MUTED}; }}
        QTabWidget::pane {{ border: 1px solid #444; }}
        QTabBar::tab {{
            background: {C_SURFACE}; color: {C_MUTED};
            padding: 6px 14px; border-radius: 3px 3px 0 0;
        }}
        QTabBar::tab:selected {{ background: #363636; color: {C_TEXT}; }}
        QTableWidget {{ gridline-color: #444; }}
        QHeaderView::section {{
            background: #333; color: {C_MUTED};
            border: 1px solid #444; padding: 4px;
        }}
        QProgressBar {{
            border: 1px solid #555; border-radius: 3px;
            background: #333; text-align: center; color: {C_TEXT};
        }}
        QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 3px; }}
        QScrollBar:vertical {{ background: {C_SURFACE}; width: 10px; }}
        QScrollBar::handle:vertical {{ background: #555; border-radius: 5px; }}
        QRadioButton {{ color: {C_TEXT}; }}
        QCheckBox {{ color: {C_TEXT}; }}
        QLabel {{ color: {C_TEXT}; }}
    """)


# ---------------------------------------------------------------------------
# Left panel: Input & Controls
# ---------------------------------------------------------------------------

class InputPanel(QWidget):
    """Left panel for participant input, file browsing, and pipeline control."""

    run_requested = Signal(dict)   # emits config dict when user clicks Run
    batch_run_requested = Signal(dict)  # emits config for batch mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(290)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────
        hdr = QLabel("Gait Analysis")
        hdr.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {C_ACCENT};")
        root.addWidget(hdr)

        # ── Mode selector ─────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single Participant", "Batch Processing"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        root.addLayout(mode_row)

        # ── Participant ID ────────────────────────────────────────────
        grp_id = QGroupBox("Participant")
        lay_id = QVBoxLayout(grp_id)
        self.pid_edit = QLineEdit("sub_01")
        lay_id.addWidget(self.pid_edit)
        root.addWidget(grp_id)

        # ── ST input ──────────────────────────────────────────────────
        self.st_group = self._build_file_input_group(
            "Single-Task (ST)", "st"
        )
        root.addWidget(self.st_group)

        # ── DT input ──────────────────────────────────────────────────
        self.dt_group = self._build_file_input_group(
            "Dual-Task (DT)", "dt"
        )
        root.addWidget(self.dt_group)

        # ── Numeric params ────────────────────────────────────────────
        grp_num = QGroupBox("Recording Parameters")
        lay_num = QVBoxLayout(grp_num)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Height (m):"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 2.5)
        self.height_spin.setValue(1.70)
        self.height_spin.setSingleStep(0.01)
        row1.addWidget(self.height_spin)
        lay_num.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 500)
        self.fps_spin.setValue(120)
        row2.addWidget(self.fps_spin)
        lay_num.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Speed Factor:"))
        self.speed_factor_spin = QDoubleSpinBox()
        self.speed_factor_spin.setRange(0.1, 100.0)
        self.speed_factor_spin.setValue(1.0)
        self.speed_factor_spin.setSingleStep(0.1)
        self.speed_factor_spin.setToolTip(
            "Playback speed correction. Set to 8.0 if the video is "
            "8\u00d7 slower than real-time (e.g. 240fps recorded, 30fps playback)."
        )
        row3.addWidget(self.speed_factor_spin)
        lay_num.addLayout(row3)
        root.addWidget(grp_num)

        # ── Boundary timestamps (optional) ────────────────────────────
        grp_bnd = QGroupBox("Boundary Timestamps (optional)")
        lay_bnd = QVBoxLayout(grp_bnd)

        # ST boundaries
        lay_bnd.addWidget(QLabel("ST enter/exit CSV:"))
        row_st_bnd = QHBoxLayout()
        self.st_boundaries_edit = QLineEdit()
        self.st_boundaries_edit.setPlaceholderText("Optional — enter/exit timestamps")
        row_st_bnd.addWidget(self.st_boundaries_edit)
        btn_st_bnd = QPushButton("…")
        btn_st_bnd.setFixedWidth(30)
        btn_st_bnd.clicked.connect(lambda: self._browse_csv(self.st_boundaries_edit))
        row_st_bnd.addWidget(btn_st_bnd)
        lay_bnd.addLayout(row_st_bnd)

        # DT boundaries
        lay_bnd.addWidget(QLabel("DT enter/exit CSV:"))
        row_dt_bnd = QHBoxLayout()
        self.dt_boundaries_edit = QLineEdit()
        self.dt_boundaries_edit.setPlaceholderText("Optional — enter/exit timestamps")
        row_dt_bnd.addWidget(self.dt_boundaries_edit)
        btn_dt_bnd = QPushButton("…")
        btn_dt_bnd.setFixedWidth(30)
        btn_dt_bnd.clicked.connect(lambda: self._browse_csv(self.dt_boundaries_edit))
        row_dt_bnd.addWidget(btn_dt_bnd)
        lay_bnd.addLayout(row_dt_bnd)

        root.addWidget(grp_bnd)

        # Store single-mode widgets for show/hide
        self._single_widgets = [grp_id, self.st_group, self.dt_group, grp_num, grp_bnd]

        # ── Batch-mode widgets ────────────────────────────────────────
        self._batch_group = QGroupBox("Batch Processing")
        batch_lay = QVBoxLayout(self._batch_group)
        batch_lay.addWidget(QLabel("Dataset Directory:"))
        row_ds = QHBoxLayout()
        self.dataset_dir_edit = QLineEdit()
        self.dataset_dir_edit.setPlaceholderText("Path to dataset root folder…")
        row_ds.addWidget(self.dataset_dir_edit)
        btn_ds = QPushButton("…")
        btn_ds.setFixedWidth(30)
        btn_ds.clicked.connect(self._browse_dataset_dir)
        row_ds.addWidget(btn_ds)
        batch_lay.addLayout(row_ds)
        self._batch_group.setVisible(False)
        root.addWidget(self._batch_group)

        # ── Run button + progress ─────────────────────────────────────
        self.run_btn = QPushButton("▶  Run Analysis")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self._on_run)
        root.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        # Stage checklist
        self.stage_log = QTextEdit()
        self.stage_log.setReadOnly(True)
        self.stage_log.setMaximumHeight(230)
        self.stage_log.setStyleSheet(
            f"background:{C_SURFACE}; color:{C_MUTED}; font-family: monospace; font-size: 11px;"
        )
        root.addWidget(self.stage_log)

        # ── Processing Options (output dir + toggles) ─────────────────
        grp_out = QGroupBox("Processing Options")
        lay_out = QVBoxLayout(grp_out)
        row_out = QHBoxLayout()
        self.out_edit = QLineEdit(str(Path(".").resolve() / "out"))
        row_out.addWidget(self.out_edit)
        btn_out = QPushButton("…")
        btn_out.setFixedWidth(30)
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(btn_out)
        lay_out.addLayout(row_out)

        # Checkboxes in a horizontal row to save vertical space
        cb_row = QHBoxLayout()
        self.save_video_cb = QCheckBox("Save Video")
        self.save_video_cb.setChecked(False)
        self.save_video_cb.setToolTip(
            "Save Sports2D annotated video output (slower processing)."
        )
        cb_row.addWidget(self.save_video_cb)

        self.segment_mode_cb = QCheckBox("Segment Mode")
        self.segment_mode_cb.setChecked(False)
        self.segment_mode_cb.setToolTip(
            "Split video into valid segments using boundary timestamps\n"
            "before processing. Eliminates phantom tracking in\n"
            "out-of-frame gaps. Requires a boundaries CSV."
        )
        cb_row.addWidget(self.segment_mode_cb)
        lay_out.addLayout(cb_row)

        root.addWidget(grp_out)

        root.addStretch()

    def _build_file_input_group(self, label: str, cond: str) -> QGroupBox:
        grp = QGroupBox(label)
        lay = QVBoxLayout(grp)

        # Radio buttons: TRC vs Video (Addendum 2)
        btn_grp = QButtonGroup(grp)
        rdo_trc = QRadioButton(".trc file")
        rdo_vid = QRadioButton("Video file")
        rdo_trc.setChecked(True)
        btn_grp.addButton(rdo_trc, 0)
        btn_grp.addButton(rdo_vid, 1)
        row_rdo = QHBoxLayout()
        row_rdo.addWidget(rdo_trc)
        row_rdo.addWidget(rdo_vid)
        lay.addLayout(row_rdo)

        path_edit = QLineEdit()
        path_edit.setPlaceholderText("Path to file…")
        browse_btn = QPushButton("Browse")

        def browse():
            if btn_grp.checkedId() == 0:  # TRC
                filt = "TRC files (*.trc *.csv);;All files (*)"
            else:                          # Video
                filt = "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
            path, _ = QFileDialog.getOpenFileName(grp, f"Select {label} file", "", filt)
            if path:
                path_edit.setText(path)

        browse_btn.clicked.connect(browse)
        row_path = QHBoxLayout()
        row_path.addWidget(path_edit)
        row_path.addWidget(browse_btn)
        lay.addLayout(row_path)

        # Store references
        setattr(self, f"{cond}_rdo_grp",  btn_grp)
        setattr(self, f"{cond}_path_edit", path_edit)
        setattr(self, f"{cond}_rdo_trc",  rdo_trc)
        setattr(self, f"{cond}_rdo_vid",  rdo_vid)
        return grp

    def _browse_csv(self, target_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Boundary CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            target_edit.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.out_edit.setText(path)

    def _on_run(self):
        if self.mode_combo.currentIndex() == 1:
            self._on_batch_run()
            return
        config = {
            "participant_id": self.pid_edit.text().strip() or "sub_01",
            "st_input":       getattr(self, "st_path_edit").text().strip(),
            "dt_input":       getattr(self, "dt_path_edit").text().strip(),
            "st_is_video":    getattr(self, "st_rdo_grp").checkedId() == 1,
            "dt_is_video":    getattr(self, "dt_rdo_grp").checkedId() == 1,
            "height_m":       self.height_spin.value(),
            "fps":            self.fps_spin.value(),
            "output_dir":     self.out_edit.text().strip(),
            "st_boundaries_csv": self.st_boundaries_edit.text().strip(),
            "dt_boundaries_csv": self.dt_boundaries_edit.text().strip(),
            "speed_factor":   self.speed_factor_spin.value(),
            "save_video":     self.save_video_cb.isChecked(),
            "segment_mode":   self.segment_mode_cb.isChecked(),
        }
        if not config["st_input"] or not config["dt_input"]:
            QMessageBox.warning(self, "Missing Input", "Please provide ST and DT file paths.")
            return
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.stage_log.clear()
        self.run_requested.emit(config)

    def update_stage(self, stage_name: str, status: str, pct: int):
        icon = STATUS_ICON.get(StageStatus(status), "?")
        self.stage_log.append(f"{icon} {stage_name}")
        self.progress_bar.setValue(pct)

    def on_finished(self):
        self.run_btn.setEnabled(True)
        self.progress_bar.setValue(100)

    def on_error(self, msg: str):
        self.run_btn.setEnabled(True)
        QMessageBox.critical(self, "Pipeline Error", msg[:800])

    # ------------------------------------------------------------------
    # Batch mode helpers
    # ------------------------------------------------------------------

    def _on_mode_changed(self, index: int):
        is_batch = index == 1
        for w in self._single_widgets:
            w.setVisible(not is_batch)
        self._batch_group.setVisible(is_batch)

    def _browse_dataset_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if path:
            self.dataset_dir_edit.setText(path)

    def _on_batch_run(self):
        dataset_dir = self.dataset_dir_edit.text().strip()
        output_dir = self.out_edit.text().strip()
        if not dataset_dir:
            QMessageBox.warning(self, "Missing Input", "Please select a dataset directory.")
            return
        config = {
            "dataset_dir": dataset_dir, 
            "output_dir": output_dir,
            "save_video": self.save_video_cb.isChecked(),
            "segment_mode": self.segment_mode_cb.isChecked(),
        }
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.stage_log.clear()
        self.batch_run_requested.emit(config)

    def update_batch_stage(self, pid: str, idx: int, total: int,
                           stage: str, status: str, pct: int):
        overall = int(((idx - 1) / total) * 100 + pct / total)
        self.progress_bar.setValue(overall)
        self.stage_log.append(f"[{idx}/{total}] {pid} — {stage} ({status})")


# ---------------------------------------------------------------------------
# Results Tabs
# ---------------------------------------------------------------------------

class ParameterTab(QWidget):
    """Tab showing time-series stride-by-stride parameters with outliers shown."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._scroll.setWidget(self._container)
        self._layout.addWidget(self._scroll)

    def load_data(self, strides_df: pd.DataFrame):
        """Render time-series charts for each parameter, one graph per param."""
        # Clear previous content
        while self._container_layout.count():
            child = self._container_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if strides_df is None or strides_df.empty:
            self._container_layout.addWidget(QLabel("No data available."))
            return

        params = [
            "stride_times", "stance_times", "swing_times",
            "stride_lengths", "stance_ratios", "step_time",
            "double_support_time",
        ]

        for param in params:
            if param not in strides_df.columns:
                continue
            chart = self._make_chart(strides_df, param)
            if chart:
                self._container_layout.addWidget(chart)

        self._container_layout.addStretch()

    def _make_chart(self, df: pd.DataFrame, param: str) -> Optional[QWidget]:
        if HAS_PYQTGRAPH:
            return self._pg_chart(df, param)
        return self._mpl_chart(df, param)

    def _pg_chart(self, df: pd.DataFrame, param: str) -> QWidget:
        pw = pg.PlotWidget(title=param.replace("_", " ").title())
        pw.setBackground(C_SURFACE)
        pw.setFixedHeight(160)
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setLabel("left",   param.replace("_", " "), color=C_MUTED)
        pw.setLabel("bottom", "Time (s)",               color=C_MUTED)

        colors = {"left": C_LEFT_FOOT, "right": C_RIGHT_FOOT}

        for side, col_hex in colors.items():
            sub = df[df["foot"] == side].copy()
            if sub.empty:
                continue
            valid   = sub[sub["is_outlier"] == False]
            invalid = sub[sub["is_outlier"] == True]

            col_rgb = QColor(col_hex)
            pen = pg.mkPen(color=col_rgb, width=2)
            pw.plot(valid["timestamps"].values, valid[param].values,
                    pen=pen, name=side)
            # Greyed-out outlier points
            grey = pg.ScatterPlotItem(
                x=invalid["timestamps"].values,
                y=invalid[param].values,
                brush=pg.mkBrush(QColor("#555555")),
                pen=pg.mkPen(None), size=6
            )
            pw.addItem(grey)

        pw.addLegend()
        return pw

    def _mpl_chart(self, df: pd.DataFrame, param: str) -> QWidget:
        fig, ax = plt.subplots(figsize=(7, 2.2))
        fig.patch.set_facecolor(C_SURFACE)
        ax.set_facecolor(C_BG)
        ax.tick_params(colors=C_MUTED)
        ax.spines[:].set_color(C_MUTED)
        ax.set_title(param.replace("_", " ").title(), color=C_TEXT, fontsize=10)
        ax.set_xlabel("Time (s)", color=C_MUTED, fontsize=9)
        ax.set_ylabel(param.replace("_", " "), color=C_MUTED, fontsize=9)

        colors = {"left": C_LEFT_FOOT, "right": C_RIGHT_FOOT}
        for side, col in colors.items():
            sub   = df[df["foot"] == side]
            valid = sub[sub["is_outlier"] == False]
            inv   = sub[sub["is_outlier"] == True]
            ax.plot(valid["timestamps"], valid[param], color=col,
                    linewidth=1.5, label=side)
            ax.scatter(inv["timestamps"], inv[param],
                       color="#555555", s=20, zorder=5)

        ax.legend(facecolor=C_SURFACE, labelcolor=C_TEXT, fontsize=8)
        fig.tight_layout(pad=0.5)
        canvas = FigureCanvas(fig)
        canvas.setFixedHeight(200)
        plt.close(fig)
        return canvas


class DTCTab(QWidget):
    """Tab displaying Dual-Task Cost bar chart and summary table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self._chart_holder = QWidget()
        self._chart_holder.setFixedHeight(320)
        lay.addWidget(self._chart_holder)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self._table)

    def load_data(self, dtc_df: pd.DataFrame, dtc_summary: pd.DataFrame):
        if dtc_df is None or dtc_df.empty:
            return
        self._render_chart(dtc_summary)
        self._render_table(dtc_summary)

    def _render_chart(self, summary: pd.DataFrame):
        layout = self._chart_holder.layout()
        if layout:
            while layout.count():
                c = layout.takeAt(0)
                if c.widget():
                    c.widget().deleteLater()
        else:
            layout = QVBoxLayout(self._chart_holder)

        params  = summary["parameter"].tolist()
        values  = summary["DTC_pct"].tolist()
        colors  = [C_POSITIVE if v > 0 else (C_NEGATIVE if v < 0 else "#888888")
                   for v in values]

        if HAS_PYQTGRAPH:
            pw = pg.PlotWidget(title="Dual-Task Cost (%)")
            pw.setBackground(C_SURFACE)
            pw.showGrid(y=True, alpha=0.3)
            pw.setLabel("left", "DTC (%)", color=C_MUTED)
            x = list(range(len(params)))
            bars = pg.BarGraphItem(x=x, height=values,
                                   width=0.6, brushes=colors)
            pw.addItem(bars)
            pw.addLine(y=0, pen=pg.mkPen(C_MUTED, width=1, style=Qt.PenStyle.DashLine))
            ticks = [(i, p.replace("_", "\n")) for i, p in enumerate(params)]
            pw.getAxis("bottom").setTicks([ticks])
            layout.addWidget(pw)
        else:
            fig, ax = plt.subplots(figsize=(9, 3))
            fig.patch.set_facecolor(C_SURFACE)
            ax.set_facecolor(C_BG)
            ax.tick_params(colors=C_MUTED, labelsize=7)
            ax.spines[:].set_color(C_MUTED)
            ax.bar(params, values, color=colors)
            ax.axhline(0, color=C_MUTED, linewidth=1, linestyle="--")
            ax.set_title("Dual-Task Cost (%)", color=C_TEXT)
            ax.set_ylabel("DTC (%)", color=C_MUTED)
            ax.set_xticklabels(params, rotation=30, ha="right")
            fig.tight_layout()
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            plt.close(fig)

    def _render_table(self, summary: pd.DataFrame):
        self._table.clear()
        if summary.empty:
            return
        self._table.setRowCount(len(summary))
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Parameter", "DTC (%)", "Direction"])
        for row, r in summary.iterrows():
            self._table.setItem(row, 0, QTableWidgetItem(str(r["parameter"])))
            val_item = QTableWidgetItem(f"{r['DTC_pct']:.3f}")
            val_item.setForeground(
                QColor(C_POSITIVE) if r["DTC_pct"] > 0 else
                QColor(C_NEGATIVE) if r["DTC_pct"] < 0 else
                QColor(C_MUTED)
            )
            self._table.setItem(row, 1, val_item)
            self._table.setItem(row, 2, QTableWidgetItem(str(r["direction"])))
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


class RawDataTab(QWidget):
    """Tab showing full stride-by-stride data with outlier highlighting, sort, and filter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # Filter bar
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Type to filter rows by any value…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        bar.addWidget(self.filter_edit)
        lay.addLayout(bar)

        self._table = QTableWidget()
        self._table.setSortingEnabled(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        lay.addWidget(self._table)

        self._full_df: Optional[pd.DataFrame] = None

    def load_data(self, strides_st: pd.DataFrame, strides_dt: pd.DataFrame):
        dfs = []
        for df, cond in [(strides_st, "st"), (strides_dt, "dt")]:
            if df is not None and not df.empty:
                d = df.copy()
                d.insert(0, "condition", cond)
                dfs.append(d)
        if not dfs:
            return
        self._full_df = pd.concat(dfs, ignore_index=True)
        self._render(self._full_df)

    def _render(self, df: pd.DataFrame):
        self._table.setRowCount(0)
        if df.empty:
            return
        cols = list(df.columns)
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(df))

        outlier_col = cols.index("is_outlier") if "is_outlier" in cols else -1

        for row_i, (_, row) in enumerate(df.iterrows()):
            is_out = bool(row.get("is_outlier", False))
            for col_i, col in enumerate(cols):
                val = row[col]
                item = QTableWidgetItem(
                    f"{val:.4f}" if isinstance(val, float) else str(val)
                )
                if is_out:
                    item.setBackground(QColor(C_OUTLIER))
                self._table.setItem(row_i, col_i, item)

    def _apply_filter(self, text: str):
        if self._full_df is None:
            return
        if not text:
            self._render(self._full_df)
            return
        mask = self._full_df.apply(
            lambda col: col.astype(str).str.contains(text, case=False, na=False)
        ).any(axis=1)
        self._render(self._full_df[mask].reset_index(drop=True))


class VideoTab(QWidget):
    """Tab for playing back the Sports2D annotated video."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_paths: dict[str, str] = {}  # {"st": path, "dt": path}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # Top bar: condition selector + controls
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Condition:"))
        self._cond_combo = QComboBox()
        self._cond_combo.addItems(["Single-Task (ST)", "Dual-Task (DT)"])
        self._cond_combo.currentIndexChanged.connect(self._on_condition_changed)
        self._cond_combo.setFixedWidth(180)
        top_bar.addWidget(self._cond_combo)
        top_bar.addStretch()

        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedWidth(80)
        self._play_btn.clicked.connect(self._toggle_play)
        top_bar.addWidget(self._play_btn)
        lay.addLayout(top_bar)

        # Video area
        if HAS_MULTIMEDIA:
            self._video_widget = QVideoWidget()
            self._video_widget.setStyleSheet(f"background: {C_BG};")
            self._player = QMediaPlayer()
            self._audio = QAudioOutput()
            self._player.setAudioOutput(self._audio)
            self._player.setVideoOutput(self._video_widget)
            self._player.playbackStateChanged.connect(self._on_state_changed)
            lay.addWidget(self._video_widget, stretch=1)

            # Seek slider
            try:
                if BACKEND == "PySide6":
                    from PySide6.QtWidgets import QSlider
                else:
                    from PyQt6.QtWidgets import QSlider
            except ImportError:
                QSlider = None

            if QSlider:
                seek_bar = QHBoxLayout()
                self._slider = QSlider(Qt.Orientation.Horizontal)
                self._slider.setRange(0, 0)
                self._slider.sliderMoved.connect(self._seek)
                self._player.positionChanged.connect(self._update_slider_pos)
                self._player.durationChanged.connect(
                    lambda d: self._slider.setRange(0, d)
                )
                self._time_label = QLabel("0:00 / 0:00")
                self._time_label.setStyleSheet(f"color: {C_MUTED}; font-family: monospace;")
                self._time_label.setFixedWidth(120)
                seek_bar.addWidget(self._slider)
                seek_bar.addWidget(self._time_label)
                lay.addLayout(seek_bar)
        else:
            self._player = None
            self._placeholder = QLabel(
                "Video playback requires QtMultimedia.\n\n"
                "Install with: pip install PySide6-Addons\n\n"
                "The video file path will be shown below when available."
            )
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setStyleSheet(f"color: {C_MUTED}; font-size: 13px;")
            lay.addWidget(self._placeholder, stretch=1)

        self._path_label = QLabel("")
        self._path_label.setStyleSheet(f"color: {C_MUTED}; font-size: 10px;")
        self._path_label.setWordWrap(True)
        lay.addWidget(self._path_label)

        # Sports2D progress bar — shown only while pipeline is running
        self._s2d_label = QLabel("")
        self._s2d_label.setStyleSheet(f"color: {C_ACCENT}; font-size: 11px;")
        self._s2d_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._s2d_label)

        self._s2d_bar = QProgressBar()
        self._s2d_bar.setRange(0, 100)
        self._s2d_bar.setValue(0)
        self._s2d_bar.setTextVisible(False)
        self._s2d_bar.setFixedHeight(8)
        self._s2d_bar.setStyleSheet(
            f"QProgressBar {{ background: {C_SURFACE}; border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 4px; }}"
        )
        lay.addWidget(self._s2d_bar)

        # Hidden initially
        self._s2d_bar.setVisible(False)
        self._s2d_label.setVisible(False)

    def load_videos(self, st_path: str = "", dt_path: str = ""):
        """Set the annotated video paths and load the ST video."""
        self._video_paths = {"st": st_path or "", "dt": dt_path or ""}
        self._load_video_for_condition(0)  # load ST by default

    def _on_condition_changed(self, index: int):
        self._load_video_for_condition(index)

    def _load_video_for_condition(self, index: int):
        cond = "st" if index == 0 else "dt"
        path = self._video_paths.get(cond, "")

        if not path or not Path(path).exists():
            self._path_label.setText(
                f"No annotated video available for {cond.upper()}.\n"
                "Run the pipeline with video input to generate one."
            )
            if self._player:
                self._player.stop()
                self._player.setSource(QUrl())
            return

        self._path_label.setText(f"Video: {path}")
        if self._player:
            self._player.setSource(QUrl.fromLocalFile(path))
            self._player.play()

    def _toggle_play(self):
        if not self._player:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("Pause")
            self.hide_s2d_progress()   # hide progress bar while video plays
        else:
            self._play_btn.setText("Play")

    def _seek(self, position: int):
        if self._player:
            self._player.setPosition(position)

    def _update_slider_pos(self, position: int):
        if hasattr(self, "_slider"):
            self._slider.blockSignals(True)
            self._slider.setValue(position)
            self._slider.blockSignals(False)
            # Update time label
            dur = self._player.duration() if self._player else 0
            self._time_label.setText(
                f"{self._fmt_time(position)} / {self._fmt_time(dur)}"
            )

    @staticmethod
    def _fmt_time(ms: int) -> str:
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"

    # ------------------------------------------------------------------
    # Sports2D progress bar
    # ------------------------------------------------------------------

    def show_s2d_progress(self, cond: str, pct: int, fps: float, eta: str):
        """Update the Sports2D progress bar with live tqdm data."""
        self._s2d_bar.setVisible(True)
        self._s2d_label.setVisible(True)
        self._s2d_bar.setValue(pct)
        label = f"Processing {cond.upper()} — {pct}%"
        if fps > 0:
            label += f"  |  {fps:.1f} it/s"
        if eta:
            label += f"  |  ETA {eta}"
        self._s2d_label.setText(label)

    def hide_s2d_progress(self):
        """Hide the Sports2D progress bar (finished or video playing)."""
        self._s2d_bar.setVisible(False)
        self._s2d_label.setVisible(False)


class DiagnosticsTab(QWidget):
    """Tab showing event detection diagnostics: counts, trajectory plots
    with HS/TO markers, and enter/exit boundary lines."""

    # Available trajectory views — (display_name, y_col, event_type, marker_col_suffix)
    _VIEWS = [
        ("Heel Y  (HS events)", "heel_y", "HS", "heel_y"),
        ("Toe Y   (TO events)", "toe_y",  "TO", "toe_y"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: dict = {}           # keyed by cond: {traj, events, raw, clean, boundaries}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        # Top bar: condition + marker selectors
        top = QHBoxLayout()
        top.addWidget(QLabel("Condition:"))
        self._cond_combo = QComboBox()
        self._cond_combo.addItems(["Single-Task (ST)", "Dual-Task (DT)"])
        self._cond_combo.currentIndexChanged.connect(self._refresh)
        self._cond_combo.setFixedWidth(180)
        top.addWidget(self._cond_combo)

        top.addSpacing(20)
        top.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        for name, *_ in self._VIEWS:
            self._view_combo.addItem(name)
        self._view_combo.currentIndexChanged.connect(self._refresh)
        self._view_combo.setFixedWidth(200)
        top.addWidget(self._view_combo)
        top.addStretch()
        lay.addLayout(top)

        # Summary cards row
        self._cards_widget = QWidget()
        self._cards_layout = QHBoxLayout(self._cards_widget)
        self._cards_layout.setContentsMargins(0, 4, 0, 4)

        card_style = (
            f"background: {C_SURFACE}; border-radius: 8px; padding: 10px;"
            f"font-family: 'Segoe UI', sans-serif;"
        )
        self._event_card = QLabel("No data")
        self._event_card.setStyleSheet(card_style)
        self._event_card.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._event_card.setMinimumWidth(240)
        self._event_card.setWordWrap(True)

        self._stride_card = QLabel("No data")
        self._stride_card.setStyleSheet(card_style)
        self._stride_card.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._stride_card.setMinimumWidth(320)
        self._stride_card.setWordWrap(True)

        self._cards_layout.addWidget(self._event_card)
        self._cards_layout.addWidget(self._stride_card)
        self._cards_layout.addStretch()
        lay.addWidget(self._cards_widget)

        # Plot area
        self._plot_holder = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_holder)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._plot_holder, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_data(
        self,
        traj_st: "pd.DataFrame | None" = None,
        traj_dt: "pd.DataFrame | None" = None,
        events_st: "pd.DataFrame | None" = None,
        events_dt: "pd.DataFrame | None" = None,
        raw_strides_st: "pd.DataFrame | None" = None,
        raw_strides_dt: "pd.DataFrame | None" = None,
        clean_strides_st: "pd.DataFrame | None" = None,
        clean_strides_dt: "pd.DataFrame | None" = None,
        boundaries_csv_st: str = "",
        boundaries_csv_dt: str = "",
        speed_factor: float = 1.0,
    ):
        """Load all diagnostic data. Call after pipeline finishes."""
        self._data = {}
        for cond, traj, evts, raw, clean, bcsv in [
            ("st", traj_st, events_st, raw_strides_st, clean_strides_st, boundaries_csv_st),
            ("dt", traj_dt, events_dt, raw_strides_dt, clean_strides_dt, boundaries_csv_dt),
        ]:
            if traj is not None and not traj.empty:
                self._data[cond] = {
                    "traj": traj,
                    "events": evts,
                    "raw": raw,
                    "clean": clean,
                    "boundaries_csv": bcsv,
                }
        self._refresh()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _current_cond(self) -> str:
        return "st" if self._cond_combo.currentIndex() == 0 else "dt"

    def _current_view(self) -> tuple:
        idx = self._view_combo.currentIndex()
        return self._VIEWS[idx]

    def _refresh(self):
        cond = self._current_cond()
        data = self._data.get(cond)

        if not data:
            self._event_card.setText("No data loaded for this condition.")
            self._stride_card.setText("")
            self._clear_plot()
            return

        self._update_cards(data, cond)
        self._update_plot(data, cond)

    # ------------------------------------------------------------------
    # Summary cards
    # ------------------------------------------------------------------

    def _update_cards(self, data: dict, cond: str):
        evts = data.get("events")
        raw = data.get("raw")
        clean = data.get("clean")

        # Event counts card
        lines = [f"<b style='color:{C_ACCENT};'>Detected Events ({cond.upper()})</b><br>"]
        if evts is not None and not evts.empty:
            for foot, col in [("Left", C_LEFT_FOOT), ("Right", C_RIGHT_FOOT)]:
                side = foot.lower()
                hs = len(evts[(evts["foot"] == side) & (evts["event_type"] == "HS")])
                to = len(evts[(evts["foot"] == side) & (evts["event_type"] == "TO")])
                lines.append(
                    f"<span style='color:{col};'>{foot}</span>: "
                    f"<b>{hs}</b> HS &nbsp;&nbsp; <b>{to}</b> TO<br>"
                )
        else:
            lines.append("No events detected.<br>")
        self._event_card.setText("".join(lines))

        # Stride counts card
        lines = [f"<b style='color:{C_ACCENT};'>Stride Counts ({cond.upper()})</b><br>"]
        if clean is not None and not clean.empty:
            total = len(clean)
            outlier = int(clean["is_outlier"].sum()) if "is_outlier" in clean.columns else 0
            valid = total - outlier
            valid_l = len(clean[(clean["is_outlier"] != True) & (clean["foot"] == "left")])
            valid_r = len(clean[(clean["is_outlier"] != True) & (clean["foot"] == "right")])

            warn_l = " <span style='color:#e05c5c;'>(!)</span>" if valid_l < 8 else ""
            warn_r = " <span style='color:#e05c5c;'>(!)</span>" if valid_r < 8 else ""

            lines.append(f"Total strides: <b>{total}</b><br>")
            lines.append(f"Outliers removed: <b>{outlier}</b><br>")
            lines.append(
                f"Valid: <b>{valid}</b> &nbsp; "
                f"(<span style='color:{C_LEFT_FOOT};'>L:{valid_l}</span>{warn_l} / "
                f"<span style='color:{C_RIGHT_FOOT};'>R:{valid_r}</span>{warn_r})<br>"
            )

            # Removal reasons breakdown
            if "removal_reason" in clean.columns:
                reasons = clean[clean["is_outlier"] == True]["removal_reason"].value_counts()
                if len(reasons) > 0:
                    reason_str = ", ".join(f"{r}: {c}" for r, c in reasons.items())
                    lines.append(f"<span style='color:{C_MUTED};'>Reasons: {reason_str}</span><br>")

            # Stride time stats
            valid_df = clean[clean["is_outlier"] != True]
            if "stride_times" in valid_df.columns and len(valid_df) > 0:
                st_vals = valid_df["stride_times"]
                lines.append(
                    f"<br>Stride time: <b>{st_vals.mean():.3f}s</b> "
                    f"(range {st_vals.min():.3f} - {st_vals.max():.3f}s)"
                )
        elif raw is not None and not raw.empty:
            lines.append(f"Raw strides: <b>{len(raw)}</b> (no cleaning done)<br>")
        else:
            lines.append("No strides computed.<br>")

        self._stride_card.setText("".join(lines))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _clear_plot(self):
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

    def _update_plot(self, data: dict, cond: str):
        self._clear_plot()
        _, y_suffix, event_type, _ = self._current_view()

        traj = data["traj"]
        evts = data.get("events")
        boundaries = self._parse_boundary_csv(data.get("boundaries_csv", ""))

        if HAS_PYQTGRAPH:
            self._pg_plot(traj, evts, boundaries, y_suffix, event_type, cond)
        else:
            self._mpl_plot(traj, evts, boundaries, y_suffix, event_type, cond)

    def _parse_boundary_csv(self, csv_path: str) -> list:
        """Parse boundary CSV into list of {'time_s': float, 'event': str}."""
        if not csv_path:
            return []
        from pathlib import Path
        p = Path(csv_path)
        if not p.exists() or p.stat().st_size == 0:
            return []
        try:
            df = pd.read_csv(p)
        except Exception:
            return []
        if "time_s" not in df.columns or "event" not in df.columns:
            return []

        events = []
        for _, row in df.iterrows():
            t_str = str(row["time_s"]).strip()
            try:
                if ":" in t_str:
                    parts = t_str.split(":")
                    t = 0.0
                    for part in parts:
                        t = t * 60 + float(part)
                else:
                    t = float(t_str)
            except ValueError:
                continue
            ev = str(row["event"]).strip().lower()
            if ev in ("enter", "exit"):
                events.append({"time_s": t, "event": ev})
        events.sort(key=lambda e: e["time_s"])
        return events

    def _pg_plot(self, traj, evts, boundaries, y_suffix, event_type, cond):
        """Build a pyqtgraph plot widget."""
        pw = pg.PlotWidget(
            title=f"{y_suffix.replace('_', ' ').title()} Trajectory + {event_type} Events — {cond.upper()}"
        )
        pw.setBackground(C_SURFACE)
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setLabel("left", y_suffix.replace("_", " ") + " (m)", color=C_MUTED)
        pw.setLabel("bottom", "Time (s)", color=C_MUTED)

        times = traj["time_s"].values

        # Plot trajectories for both sides
        for side, col_hex in [("left", C_LEFT_FOOT), ("right", C_RIGHT_FOOT)]:
            col_name = f"{side}_{y_suffix}"
            if col_name not in traj.columns:
                continue
            yvals = traj[col_name].values
            col_rgb = QColor(col_hex)
            pen = pg.mkPen(color=col_rgb, width=1.5)
            pw.plot(times, yvals, pen=pen, name=f"{side} {y_suffix}")

        # Overlay event markers
        if evts is not None and not evts.empty:
            for side, col_hex, sym in [("left", C_LEFT_FOOT, "t1"), ("right", C_RIGHT_FOOT, "t")]:
                col_name = f"{side}_{y_suffix}"
                if col_name not in traj.columns:
                    continue
                side_evts = evts[(evts["foot"] == side) & (evts["event_type"] == event_type)]
                if side_evts.empty:
                    continue
                evt_times = side_evts["time_s"].values
                # Find closest traj Y value for each event
                traj_times = traj["time_s"].values
                traj_y = traj[col_name].values
                evt_y = []
                for et in evt_times:
                    idx = np.argmin(np.abs(traj_times - et))
                    evt_y.append(traj_y[idx])
                evt_y = np.array(evt_y)

                scatter = pg.ScatterPlotItem(
                    x=evt_times, y=evt_y,
                    brush=pg.mkBrush(QColor(col_hex)),
                    pen=pg.mkPen(None),
                    size=8,
                    symbol=sym,
                    name=f"{side} {event_type} ({len(evt_times)})",
                )
                pw.addItem(scatter)

        # Boundary vertical lines
        for bev in boundaries:
            t = bev["time_s"]
            ev_type = bev["event"]
            color = "#5cb85c" if ev_type == "enter" else "#e05c5c"
            line = pg.InfiniteLine(
                pos=t, angle=90,
                pen=pg.mkPen(color=color, width=1.5, style=Qt.PenStyle.DashLine),
                label=ev_type,
                labelOpts={"color": color, "position": 0.95, "rotateAxis": (1, 0)},
            )
            pw.addItem(line)

        pw.addLegend(offset=(10, 10))
        self._plot_layout.addWidget(pw)

    def _mpl_plot(self, traj, evts, boundaries, y_suffix, event_type, cond):
        """Build a matplotlib fallback plot."""
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(C_SURFACE)
        ax.set_facecolor(C_BG)
        ax.tick_params(colors=C_MUTED)
        ax.spines[:].set_color(C_MUTED)
        ax.set_title(
            f"{y_suffix.replace('_', ' ').title()} + {event_type} Events — {cond.upper()}",
            color=C_TEXT, fontsize=11,
        )
        ax.set_xlabel("Time (s)", color=C_MUTED, fontsize=9)
        ax.set_ylabel(f"{y_suffix.replace('_', ' ')} (m)", color=C_MUTED, fontsize=9)

        times = traj["time_s"].values

        for side, col in [("left", C_LEFT_FOOT), ("right", C_RIGHT_FOOT)]:
            col_name = f"{side}_{y_suffix}"
            if col_name not in traj.columns:
                continue
            ax.plot(times, traj[col_name].values, color=col, linewidth=1,
                    label=f"{side} {y_suffix}", alpha=0.8)

        if evts is not None and not evts.empty:
            for side, col, marker in [("left", C_LEFT_FOOT, "v"), ("right", C_RIGHT_FOOT, "^")]:
                col_name = f"{side}_{y_suffix}"
                if col_name not in traj.columns:
                    continue
                side_evts = evts[(evts["foot"] == side) & (evts["event_type"] == event_type)]
                if side_evts.empty:
                    continue
                evt_times = side_evts["time_s"].values
                traj_times = traj["time_s"].values
                traj_y = traj[col_name].values
                evt_y = [traj_y[np.argmin(np.abs(traj_times - et))] for et in evt_times]
                ax.scatter(evt_times, evt_y, color=col, marker=marker, s=40, zorder=5,
                           label=f"{side} {event_type} ({len(evt_times)})")

        for bev in boundaries:
            color = "#5cb85c" if bev["event"] == "enter" else "#e05c5c"
            ax.axvline(bev["time_s"], color=color, linestyle="--", linewidth=1, alpha=0.7)

        ax.legend(facecolor=C_SURFACE, labelcolor=C_TEXT, fontsize=8)
        fig.tight_layout(pad=0.5)
        canvas = FigureCanvas(fig)
        plt.close(fig)
        self._plot_layout.addWidget(canvas)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Analysis — Sports2D + DUO-GAIT")
        self.resize(1280, 820)
        self._runner: Optional[PipelineRunner] = None
        self._batch_runner = None
        self._results: dict = {}

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left panel
        self._input_panel = InputPanel()
        self._input_panel.run_requested.connect(self._on_run_requested)
        self._input_panel.batch_run_requested.connect(self._on_batch_run_requested)
        root.addWidget(self._input_panel)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.VLine)
        div.setStyleSheet("color: #444;")
        root.addWidget(div)

        # Right area — results tabs
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        # Participant selector (for future batch mode)
        self._sub_selector = QComboBox()
        self._sub_selector.setFixedWidth(180)

        self._video_tab    = VideoTab()
        self._diag_tab     = DiagnosticsTab()
        self._st_tab       = ParameterTab("Single-Task Gait Parameters")
        self._dt_tab       = ParameterTab("Dual-Task Gait Parameters")
        self._dtc_tab      = DTCTab()
        self._raw_tab      = RawDataTab()

        self._tabs.addTab(self._video_tab, "📽  Annotated Video")
        self._tabs.addTab(self._diag_tab,  "🔬 Diagnostics")
        self._tabs.addTab(self._st_tab,    "🦶 ST Parameters")
        self._tabs.addTab(self._dt_tab,    "🧮 DT Parameters")
        self._tabs.addTab(self._dtc_tab,   "📊 Dual-Task Cost")
        self._tabs.addTab(self._raw_tab,   "📋 Raw Data")

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------

    @Slot(dict)
    def _on_run_requested(self, config: dict):
        self._runner = PipelineRunner(
            participant_id = config["participant_id"],
            st_input       = config["st_input"],
            dt_input       = config["dt_input"],
            st_is_video    = config["st_is_video"],
            dt_is_video    = config["dt_is_video"],
            height_m       = config["height_m"],
            fps            = config["fps"],
            output_dir     = config["output_dir"],
            st_boundaries_csv = config.get("st_boundaries_csv", ""),
            dt_boundaries_csv = config.get("dt_boundaries_csv", ""),
            speed_factor   = config.get("speed_factor", 1.0),
            save_video     = config.get("save_video", False),
            segment_mode   = config.get("segment_mode", False),
        )
        self._runner.progress.connect(self._on_progress)
        self._runner.finished.connect(self._on_finished)
        self._runner.error.connect(self._on_error)
        self._runner.sports2d_progress.connect(self._on_s2d_progress)
        self._runner.start()

    @Slot(str, str, int)
    def _on_progress(self, stage_name: str, status: str, pct: int):
        self._input_panel.update_stage(stage_name, status, pct)

    @Slot(str, int, float, str)
    def _on_s2d_progress(self, cond: str, pct: int, fps: float, eta: str):
        self._tabs.setCurrentWidget(self._video_tab)   # keep video tab in view
        self._video_tab.show_s2d_progress(cond, pct, fps, eta)

    @Slot(dict)
    def _on_finished(self, results: dict):
        self._results = results
        self._input_panel.on_finished()
        self._video_tab.hide_s2d_progress()

        # Populate result tabs
        try:
            self._st_tab.load_data(results.get("remove_outliers_st"))
            self._dt_tab.load_data(results.get("remove_outliers_dt"))

            self._diag_tab.load_data(
                traj_st=results.get("preprocess_st"),
                traj_dt=results.get("preprocess_dt"),
                events_st=results.get("detect_events_st"),
                events_dt=results.get("detect_events_dt"),
                raw_strides_st=results.get("calc_params_st"),
                raw_strides_dt=results.get("calc_params_dt"),
                clean_strides_st=results.get("remove_outliers_st"),
                clean_strides_dt=results.get("remove_outliers_dt"),
                boundaries_csv_st=self._runner.st_boundaries_csv if self._runner else "",
                boundaries_csv_dt=self._runner.dt_boundaries_csv if self._runner else "",
                speed_factor=self._runner.speed_factor if self._runner else 1.0,
            )

            dtc_payload = results.get("dtc", {})
            if isinstance(dtc_payload, dict):
                self._dtc_tab.load_data(
                    dtc_payload.get("dtc"),
                    dtc_payload.get("dtc_summary"),
                )

            self._raw_tab.load_data(
                results.get("calc_params_st"),
                results.get("calc_params_dt"),
            )

            # Load annotated videos if available
            if self._runner:
                st_vid = self._runner.get_annotated_video("st")
                dt_vid = self._runner.get_annotated_video("dt")
                if st_vid or dt_vid:
                    self._video_tab.load_videos(st_path=st_vid, dt_path=dt_vid)
                    self._tabs.setCurrentIndex(0)  # jump to video tab
                else:
                    self._tabs.setCurrentIndex(1)  # jump to diagnostics tab
            else:
                self._tabs.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.warning(self, "Display Error", str(e))

    @Slot(str)
    def _on_error(self, message: str):
        self._input_panel.on_error(message)

    # ------------------------------------------------------------------
    # Batch pipeline control
    # ------------------------------------------------------------------

    @Slot(dict)
    def _on_batch_run_requested(self, config: dict):
        from runners.batch_runner import BatchPipelineRunner
        self._batch_runner = BatchPipelineRunner(
            input_dir=config["dataset_dir"],
            output_dir=config["output_dir"],
            save_video=config.get("save_video", False),
            segment_mode=config.get("segment_mode", False),
        )
        self._batch_runner.batch_progress.connect(self._on_batch_progress)
        self._batch_runner.participant_complete.connect(self._on_participant_complete)
        self._batch_runner.file_error.connect(self._on_batch_file_error)
        self._batch_runner.batch_finished.connect(self._on_batch_finished)
        self._batch_runner.batch_error.connect(self._on_batch_error)
        self._batch_runner.sports2d_progress.connect(self._on_s2d_progress)
        self._batch_runner.start()

    @Slot(str, int, int, str, str, int)
    def _on_batch_progress(self, pid, idx, total, stage, status, pct):
        self._input_panel.update_batch_stage(pid, idx, total, stage, status, pct)

    @Slot(str, int, int)
    def _on_participant_complete(self, pid, idx, total):
        self._input_panel.stage_log.append(f"\u2713 {pid} complete ({idx}/{total})")

    @Slot(str, str, str)
    def _on_batch_file_error(self, pid, folder_path, missing_files):
        from runners.batch_runner import ErrorAction
        msg = (f"Participant {pid} is missing files:\n{missing_files}\n\n"
               f"Folder: {folder_path}\n\n"
               f"Skip this participant, Retry with a different folder, or Cancel?")
        box = QMessageBox(self)
        box.setWindowTitle("Missing Files")
        box.setText(msg)
        box.setIcon(QMessageBox.Icon.Warning)
        btn_skip   = box.addButton("Skip",   QMessageBox.ButtonRole.AcceptRole)
        btn_retry  = box.addButton("Retry",  QMessageBox.ButtonRole.ActionRole)
        btn_cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()

        clicked = box.clickedButton()
        if clicked == btn_cancel:
            self._batch_runner.set_error_response(ErrorAction.CANCEL)
        elif clicked == btn_retry:
            new_path = QFileDialog.getExistingDirectory(
                self, f"Select replacement folder for {pid}", folder_path
            )
            self._batch_runner.set_error_response(
                ErrorAction.RETRY, retry_folder=new_path or folder_path
            )
        else:
            self._batch_runner.set_error_response(ErrorAction.SKIP)

    @Slot(object)
    def _on_batch_finished(self, master_df):
        self._input_panel.on_finished()
        self._input_panel.stage_log.append("\n\u2713 Batch complete!")
        if master_df is not None and not master_df.empty:
            out_dir = self._input_panel.out_edit.text().strip()
            QMessageBox.information(
                self, "Batch Complete",
                f"All participants processed.\n\n"
                f"Master output saved to:\n{out_dir}/master/"
            )

    @Slot(str)
    def _on_batch_error(self, message: str):
        self._input_panel.on_error(message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch():
    app = QApplication(sys.argv)
    _apply_dark_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
