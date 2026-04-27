import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView,
                                QLabel, QPushButton, QMessageBox)
    from PyQt6.QtCore import Qt
except ImportError:
    try:
        from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                    QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView,
                                    QLabel, QPushButton, QMessageBox)
        from PySide6.QtCore import Qt
    except ImportError:
        print("Please install PyQt6 or PySide6")
        sys.exit(1)

import pyqtgraph as pg
import pyqtgraph.exporters

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUR_RESULTS_DIR = r"E:\Ziad\BioMechanics\test_3"
PAPER_RESULTS_DIR = r"E:\Ziad\BioMechanics\processed_DUO-gait_dataset\processed"

COMMON_PARAMS = [
    "stride_lengths_avg",
    "stride_times_avg",
    "swing_times_avg",
    "stance_times_avg",
    "stance_ratios_avg",
    "cadence_avg",
    "speed_avg"
]

class ComparisonApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DUO-Gait Paper Comparison")
        self.resize(1000, 800)
        
        self.our_data = {}
        self.paper_data = {}
        self.comparison_df = None
        
        self.load_data()
        self.build_ui()
        self.save_pngs()
        
    def load_data(self):
        # 1. Load Our Data
        our_dir = Path(OUR_RESULTS_DIR)
        if our_dir.exists():
            for sub_dir in our_dir.iterdir():
                if sub_dir.is_dir() and sub_dir.name.startswith("sub_"):
                    sub = sub_dir.name
                    st_csv = sub_dir / "04_strides_cleaned_st.csv"
                    dt_csv = sub_dir / "04_strides_cleaned_dt.csv"
                    
                    if st_csv.exists() and dt_csv.exists():
                        try:
                            df_st = pd.read_csv(st_csv)
                            df_dt = pd.read_csv(dt_csv)
                            
                            # Filter outliers if the column exists and it doesn't leave us with an empty dataset
                            if "is_outlier" in df_st.columns:
                                st_valid = df_st[df_st["is_outlier"] == False]
                                if not st_valid.empty:
                                    df_st = st_valid
                            if "is_outlier" in df_dt.columns:
                                dt_valid = df_dt[df_dt["is_outlier"] == False]
                                if not dt_valid.empty:
                                    df_dt = dt_valid
                                
                            self.our_data[sub] = {}
                            
                            params_map = {
                                "stride_lengths_avg": "stride_lengths",
                                "stride_times_avg": "stride_times",
                                "swing_times_avg": "swing_times",
                                "stance_times_avg": "stance_times",
                                "stance_ratios_avg": "stance_ratios"
                            }
                            
                            for avg_param, raw_col in params_map.items():
                                if raw_col in df_st.columns and raw_col in df_dt.columns:
                                    val_st = df_st[raw_col].mean()
                                    val_dt = df_dt[raw_col].mean()
                                    if pd.notna(val_st) and pd.notna(val_dt) and val_st != 0:
                                        self.our_data[sub][avg_param] = (val_st - val_dt) / val_st * 100.0
                                        
                            # Speed: stride_length / stride_time
                            if "stride_lengths" in df_st.columns and "stride_times" in df_st.columns:
                                speed_st = (df_st["stride_lengths"] / df_st["stride_times"]).mean()
                                speed_dt = (df_dt["stride_lengths"] / df_dt["stride_times"]).mean()
                                if pd.notna(speed_st) and pd.notna(speed_dt) and speed_st != 0:
                                    self.our_data[sub]["speed_avg"] = (speed_st - speed_dt) / speed_st * 100.0
                                    
                            # Cadence: 120 / stride_time
                            if "stride_times" in df_st.columns:
                                cad_st = (120.0 / df_st["stride_times"]).mean()
                                cad_dt = (120.0 / df_dt["stride_times"]).mean()
                                if pd.notna(cad_st) and pd.notna(cad_dt) and cad_st != 0:
                                    self.our_data[sub]["cadence_avg"] = (cad_st - cad_dt) / cad_st * 100.0
                                    
                        except Exception as e:
                            print(f"Error reading our raw files for {sub}: {e}")
        else:
            print(f"Warning: Our results not found at {OUR_RESULTS_DIR}")

        # 2. Load Paper Data
        paper_dir = Path(PAPER_RESULTS_DIR)
        st_dir = paper_dir / "OG_st_control"
        dt_dir = paper_dir / "OG_dt_control"
        
        if st_dir.exists() and dt_dir.exists():
            st_subs = [d.name for d in st_dir.iterdir() if d.is_dir()]
            dt_subs = [d.name for d in dt_dir.iterdir() if d.is_dir()]
            common_subs = set(st_subs).intersection(dt_subs)
            
            for sub in common_subs:
                st_csv = st_dir / sub / "aggregate_params.csv"
                dt_csv = dt_dir / sub / "aggregate_params.csv"
                
                if st_csv.exists() and dt_csv.exists():
                    try:
                        df_st = pd.read_csv(st_csv)
                        df_dt = pd.read_csv(dt_csv)
                        
                        if not df_st.empty and not df_dt.empty:
                            row_st = df_st.iloc[0]
                            row_dt = df_dt.iloc[0]
                            
                            self.paper_data[sub] = {}
                            for param in COMMON_PARAMS:
                                if param in row_st and param in row_dt:
                                    val_st = float(row_st[param])
                                    val_dt = float(row_dt[param])
                                    if val_st != 0 and pd.notna(val_st) and pd.notna(val_dt):
                                        # DTC = (ST - DT) / ST * 100
                                        dtc = (val_st - val_dt) / val_st * 100.0
                                        self.paper_data[sub][param] = dtc
                    except Exception as e:
                        print(f"Error processing paper sub {sub}: {e}")
        else:
            print(f"Warning: Paper results not found at {PAPER_RESULTS_DIR}")

        # 3. Align and Compare Data
        # We will compare the GROUP AVERAGES for all subjects available in each method
        # Alternatively, we could only compare subjects present in BOTH. We will compute the mean for each method over its available subjects.
        
        our_means = {}
        paper_means = {}
        
        for param in COMMON_PARAMS:
            # Our mean
            our_vals = [self.our_data[s][param] for s in self.our_data if param in self.our_data[s]]
            our_means[param] = np.mean(our_vals) if our_vals else 0.0
            
            # Paper mean
            paper_vals = [self.paper_data[s][param] for s in self.paper_data if param in self.paper_data[s]]
            paper_means[param] = np.mean(paper_vals) if paper_vals else 0.0

        rows = []
        for param in COMMON_PARAMS:
            our_val = our_means[param]
            paper_val = paper_means[param]
            # Absolute Error in % points
            abs_err = abs(our_val - paper_val)
            rows.append({
                "Parameter": param.replace("_avg", "").replace("_", " ").title(),
                "Our_DTC": our_val,
                "Paper_DTC": paper_val,
                "Abs_Error_Pt": abs_err
            })
            
        self.comparison_df = pd.DataFrame(rows)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # --- Graph ---
        self.plot_widget = pg.PlotWidget(title="Dual-Task Cost Comparison (Our Pipeline vs. Paper)")
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('left', 'DTC (%)', color='w')
        self.plot_widget.showGrid(y=True, alpha=0.3)
        
        params = self.comparison_df["Parameter"].tolist()
        our_vals = self.comparison_df["Our_DTC"].tolist()
        paper_vals = self.comparison_df["Paper_DTC"].tolist()
        
        x = np.arange(len(params))
        width = 0.35
        
        bar_our = pg.BarGraphItem(x=x - width/2, height=our_vals, width=width, brush='#4a90d9', name='Our Pipeline')
        bar_paper = pg.BarGraphItem(x=x + width/2, height=paper_vals, width=width, brush='#f0a030', name='Original Paper')
        
        self.plot_widget.addItem(bar_our)
        self.plot_widget.addItem(bar_paper)
        
        # Set x-ticks
        ticks = [list(zip(x, params))]
        self.plot_widget.getAxis('bottom').setTicks(ticks)
        
        layout.addWidget(self.plot_widget, stretch=2)
        
        # --- Table ---
        label = QLabel("Comparison Metrics (Error in Percentage Points):")
        label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(label)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Parameter", "Our DTC (%)", "Paper DTC (%)", "Abs Error (%)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setRowCount(len(self.comparison_df))
        
        for i, row in self.comparison_df.iterrows():
            self.table.setItem(i, 0, QTableWidgetItem(str(row["Parameter"])))
            self.table.setItem(i, 1, QTableWidgetItem(f"{row['Our_DTC']:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{row['Paper_DTC']:.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{row['Abs_Error_Pt']:.2f}"))
            
        layout.addWidget(self.table, stretch=1)
        
    def save_pngs(self):
        try:
            # Force UI to process events to render the plot correctly before export
            QApplication.processEvents()
            
            output_path = Path(__file__).parent / "DTC_Comparison_Chart.png"
            exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
            exporter.parameters()['width'] = 1200
            exporter.export(str(output_path))
            print(f"Saved comparison chart to {output_path}")
            
            # Also export a CSV for record keeping
            csv_path = Path(__file__).parent / "DTC_Comparison_Metrics.csv"
            self.comparison_df.to_csv(csv_path, index=False)
            print(f"Saved comparison metrics to {csv_path}")
            
        except Exception as e:
            print(f"Failed to export PNG: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Dark mode setup
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, Qt.GlobalColor.darkGray)
    palette.setColor(palette.ColorRole.WindowText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    window = ComparisonApp()
    window.show()
    sys.exit(app.exec())
