#!/usr/bin/env python3
"""
GUI.py — Sanger sequencing processor front-end (PyQt5)
Version: BLAST-Optimised + Robust Matrix Estimation + NCBI BLAST Integration

Channel–colour–letter contract (must match ud_processor.py)
----------------------------------------------------------------
  signals[0]  →  G  →  black
  signals[1]  →  A  →  green
  signals[2]  →  T  →  Red
  signals[3]  →  C  →  blue

BLAST-optimisation changes (this version)
------------------------------------------
  GUI-BLAST-1  Baseline method selector: rolling / als / morph.
  GUI-BLAST-2  "Prefer raw AB1 channels" checkbox.
  GUI-BLAST-3  Adaptive peak detection checkbox (default ON).
  GUI-BLAST-4  "Export BLAST-Ready FASTA" button.
  GUI-BLAST-5  "Export Reverse Complement FASTA" button.
  GUI-BLAST-6  Quality statistics panel extended.
  GUI-BLAST-7  BLAST readiness traffic light.

NEW: Robust matrix estimation for CSV / SRD files
--------------------------------------------------
  GUI-MATRIX-1  Method selector combo: robust / dominant_channel / nmf
  GUI-MATRIX-2  Dominance-ratio spinbox (default 1.5).
  GUI-MATRIX-3  Rich diagnostic report after estimation.

NEW: NCBI BLAST Integration
----------------------------
  GUI-NCBI-1   "BLAST via NCBI (online)" button in Export panel.
               Opens a non-modal dialog; GUI stays fully usable while waiting.
               Uses blast_ncbi.py (stdlib only, no extra pip installs).
"""

import sys
import os
import inspect
import traceback
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox,
    QTableWidget, QTableWidgetItem, QSpinBox, QDoubleSpinBox, QGroupBox,
    QGridLayout, QCheckBox, QTabWidget, QTextEdit, QScrollArea,
    QListWidget, QListWidgetItem, QLineEdit, QProgressBar, QAbstractItemView,
    QSlider,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import ud_processor as P

# ── NCBI BLAST integration (optional — graceful fallback if file is missing) ─
try:
    from blast_ncbi import open_blast_dialog
    _BLAST_AVAILABLE = True
except ImportError:
    _BLAST_AVAILABLE = False

CHANNEL_LETTERS = ('G', 'A', 'T', 'C')
CHANNEL_COLORS  = ('black', 'green', 'red', 'blue')


# ─────────────────────────────────────────────────────────────────────────────
# Worker threads
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, time, raw, matrix, params):
        super().__init__()
        self.time   = time
        self.raw    = raw
        self.matrix = matrix
        self.params = params

    def run(self):
        try:
            result = P.run_pipeline(self.time, self.raw,
                                    influence_matrix=self.matrix,
                                    **self.params)
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


class BatchWorker(QThread):
    progress  = pyqtSignal(int, int, str)
    file_done = pyqtSignal(dict)
    finished  = pyqtSignal()
    error     = pyqtSignal(str, str)

    def __init__(self, file_entries, matrix, params):
        super().__init__()
        self.file_entries = file_entries
        self.matrix       = matrix
        self.params       = params
        self._abort       = False

    def abort(self):
        self._abort = True

    def run(self):
        total = len(self.file_entries)
        for idx, (path, time, raw, meta) in enumerate(self.file_entries):
            if self._abort:
                break
            self.progress.emit(idx + 1, total, os.path.basename(path))
            try:
                result = P.run_pipeline(time, raw,
                                        influence_matrix=self.matrix,
                                        **self.params)
                result['_path'] = path
                result['_meta'] = meta
                self.file_done.emit(result)
            except Exception:
                self.error.emit(path, traceback.format_exc())
        self.finished.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class SangerGUI(QMainWindow):
    _PIPELINE_PARAMS: set = (
        set(inspect.signature(P.run_pipeline).parameters.keys())
        - {'time', 'signals', 'influence_matrix'}
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sanger Processor — BLAST-Optimised Pipeline")
        self.resize(1680, 1080)

        self.time           = None
        self.raw            = None
        self.ab1_meta       = None
        self.matrix         = None
        self.matrix_path    = None
        self.result         = None
        self.current_stage  = 'raw'
        self.matrix_origin  = None
        self._worker        = None
        self._batch_worker  = None
        self._batch_results = []
        self._compat_warned = False

        self.allow_uncertain = QCheckBox("Replace uncertain bases (N) with best guess")
        self.allow_uncertain.setChecked(True)

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        tabs = QTabWidget()
        root.addWidget(tabs)
        main_tab   = QWidget(); tabs.addTab(main_tab,   "Processing")
        matrix_tab = QWidget(); tabs.addTab(matrix_tab, "Matrix Tools")
        batch_tab  = QWidget(); tabs.addTab(batch_tab,  "Batch Processing")
        self._build_main_tab(main_tab)
        self._build_matrix_tab(matrix_tab)
        self._build_batch_tab(batch_tab)

    # ── Processing tab ────────────────────────────────────────────────────────

    def _build_main_tab(self, parent):
        layout = QHBoxLayout(parent)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(340)
        left_panel = QWidget()
        ll = QVBoxLayout(left_panel)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(4)
        scroll.setWidget(left_panel)
        layout.addWidget(scroll)

        # ── File operations ───────────────────────────────────────────────────
        fg = QGroupBox("File Operations")
        fl = QVBoxLayout(fg)
        self._add_btn(fl, "Load SRD File",            self.load_srd)
        self._add_btn(fl, "Load Signal CSV",           self.load_signal)
        self._add_btn(fl, "Load AB1 / ABI File",       self.load_ab1)

        self.cb_prefer_raw = QCheckBox("Prefer raw AB1 channels (DATA 1-4)")
        self.cb_prefer_raw.setChecked(True)
        self.cb_prefer_raw.setToolTip(
            "When ticked (recommended), the AB1 loader reads DATA 1-4 (raw\n"
            "fluorescence) instead of DATA 9-12 (KB-analysed/smoothed).\n\n"
            "Our pipeline does its own baseline, smoothing, and crosstalk\n"
            "correction.  Using KB-processed data as input double-processes\n"
            "the signal and distorts peak shapes.\n\n"
            "Only untick if DATA 1-4 appears blank in your AB1 file (rare\n"
            "on some older instruments that only write analysed channels).")
        fl.addWidget(self.cb_prefer_raw)

        self.lbl_csv_info = QLabel("No file loaded")
        self.lbl_csv_info.setStyleSheet("font-style:italic;color:#666")
        fl.addWidget(self.lbl_csv_info)

        self._add_btn(fl, "Load Crosstalk Matrix",     self.load_matrix)
        self._add_btn(fl, "Save Current Matrix",       self.save_matrix)

        # ── Matrix estimation controls ────────────────────────────────────────
        fl.addWidget(self._section_label("Matrix Estimation from Data"))

        mat_method_row = QHBoxLayout()
        mat_method_row.addWidget(QLabel("Method:"))
        self.combo_matrix_method = QComboBox()
        self.combo_matrix_method.addItems(['robust', 'dominant_channel', 'nmf'])
        self.combo_matrix_method.setCurrentText('robust')
        self.combo_matrix_method.setToolTip(
            "Matrix estimation algorithm for CSV / SRD files:\n\n"
            "  robust           — tries all methods, scores each for\n"
            "                     physical plausibility, returns the best.\n"
            "                     Recommended starting point.\n\n"
            "  dominant_channel — finds timepoints where one channel is\n"
            "                     the unambiguous maximum and averages\n"
            "                     their normalised 4-vectors.  Fast, no\n"
            "                     extra dependencies, reliable for CSV/SRD.\n\n"
            "  nmf              — Non-negative Matrix Factorisation;\n"
            "                     best when channels overlap heavily.\n"
            "                     Requires scikit-learn.")
        mat_method_row.addWidget(self.combo_matrix_method)
        fl.addLayout(mat_method_row)

        dom_ratio_row = QHBoxLayout()
        dom_ratio_row.addWidget(QLabel("Dominance ratio:"))
        self.spin_dom_ratio = QDoubleSpinBox()
        self.spin_dom_ratio.setRange(1.1, 5.0)
        self.spin_dom_ratio.setValue(1.5)
        self.spin_dom_ratio.setSingleStep(0.1)
        self.spin_dom_ratio.setDecimals(2)
        self.spin_dom_ratio.setToolTip(
            "Minimum ratio of the dominant channel to the second-highest\n"
            "channel for a timepoint to be used in matrix estimation.\n\n"
            "1.5 (default) — good balance of sample count vs purity\n"
            "1.2           — more samples, slightly noisier estimates\n"
            "2.0           — fewer but cleaner samples\n\n"
            "Only used by 'dominant_channel' and 'robust' methods.\n"
            "If you get fewer than ~50 samples per channel, lower this.")
        dom_ratio_row.addWidget(self.spin_dom_ratio)
        fl.addLayout(dom_ratio_row)

        self._add_btn(fl, "Estimate Matrix From Data", self.estimate_matrix_from_data)

        self.lbl_mat = QLabel("Matrix: None")
        self.lbl_mat.setStyleSheet("font-style:italic;color:#666")
        fl.addWidget(self.lbl_mat)
        ll.addWidget(fg)

        # ── Processing parameters ─────────────────────────────────────────────
        pg_box = QGroupBox("Processing Parameters")
        pg = QGridLayout(pg_box)
        row = 0

        pg.addWidget(self._section_label("Baseline Correction"), row, 0, 1, 2); row += 1
        pg.addWidget(QLabel("Baseline method:"), row, 0)
        self.combo_baseline = QComboBox()
        self.combo_baseline.addItems(['rolling', 'morph', 'als'])
        self.combo_baseline.setCurrentText('morph')
        self.combo_baseline.setToolTip(
            "Baseline correction algorithm:\n\n"
            "  rolling  — rolling median (original; fast, adequate for clean traces)\n"
            "  morph    — morphological opening; ~10× faster than ALS, better\n"
            "             than rolling for broad fluorescence humps (recommended\n"
            "             starting point for most traces)\n"
            "  als      — Asymmetric Least Squares (Eilers & Boelens 2005);\n"
            "             best accuracy on sloping baselines and large humps;\n"
            "             adds 2–5 s processing time per file\n\n"
            "Try 'morph' first.  Switch to 'als' if the baseline panel still\n"
            "shows a hump or the mean Phred Q is below 20.")
        pg.addWidget(self.combo_baseline, row, 1); row += 1

        self.spin_bg = self._spinbox(pg, row, "Baseline window:", 3, 1001, 101,
            tip="Rolling-median / morphological window size.\n"
                "Ignored for ALS (which uses its own smoothness penalty).\n"
                "Larger = smoother background; should be >> peak width."); row += 1

        pg.addWidget(self._section_label("Savitzky-Golay Smoothing"), row, 0, 1, 2); row += 1

        pg.addWidget(QLabel("SG window:"), row, 0)
        self.spin_sg = QSpinBox()
        self.spin_sg.setRange(3, 501); self.spin_sg.setValue(19); self.spin_sg.setSingleStep(2)
        self.spin_sg.setToolTip("Window length for SG filter (must be odd, > polyorder).\n"
                                "Larger = more smoothing. Typical: 11–31.")
        pg.addWidget(self.spin_sg, row, 1); row += 1

        self.slider_sg = QSlider(Qt.Horizontal)
        self.slider_sg.setRange(3, 101); self.slider_sg.setValue(19)
        self.slider_sg.setTickInterval(10); self.slider_sg.setTickPosition(QSlider.TicksBelow)
        self.slider_sg.setToolTip("Drag to adjust SG window and see live preview")
        pg.addWidget(self.slider_sg, row, 0, 1, 2); row += 1

        self.slider_sg.valueChanged.connect(self._on_sg_slider_changed)
        self.spin_sg.valueChanged.connect(self._on_sg_spin_changed)

        self.spin_sg_poly = self._spinbox(pg, row, "SG polyorder:", 1, 7, 3,
            tip="Polynomial order for SG. 2–3 is typical.\n"
                "Must be less than window length."); row += 1

        self.spin_sg_passes = self._spinbox(pg, row, "SG passes:", 1, 5, 1,
            tip="Number of successive SG filter applications.\n"
                "1=standard. 2–3 gives stronger smoothing without increasing\n"
                "window length, preserving peak shape better."); row += 1

        pg.addWidget(QLabel("SG boundary mode:"), row, 0)
        self.combo_sg_mode = QComboBox()
        self.combo_sg_mode.addItems(['interp', 'mirror', 'nearest', 'wrap', 'constant'])
        self.combo_sg_mode.setCurrentText('interp')
        self.combo_sg_mode.setToolTip(
            "How the SG filter handles trace edges:\n"
            "  interp  — polynomial extrapolation (default)\n"
            "  mirror  — reflect signal at ends (reduces edge ringing)\n"
            "  nearest — extend with edge value\n"
            "  wrap    — periodic/circular\n"
            "  constant — pad with zero\n"
            "Use 'mirror' if you see false peaks at the start or end of the trace.")
        pg.addWidget(self.combo_sg_mode, row, 1); row += 1

        preview_row = QHBoxLayout()
        self.btn_preview_smooth = QPushButton("⟳  Preview Smoothing")
        self.btn_preview_smooth.clicked.connect(self.preview_smoothing)
        preview_row.addWidget(self.btn_preview_smooth)
        pg.addLayout(preview_row, row, 0, 1, 2); row += 1

        self.lbl_smooth_info = QLabel("Smoothing preview: not run yet")
        self.lbl_smooth_info.setStyleSheet("font-style:italic;color:#666;font-size:9px;")
        pg.addWidget(self.lbl_smooth_info, row, 0, 1, 2); row += 1

        self.spin_sigma  = self._dspinbox(pg, row, "Sharpen sigma:", 0.1, 5.0, 0.4, 0.1); row += 1
        self.spin_amount = self._dspinbox(pg, row, "Sharpen amount:", 0.0, 2.0, 0.0, 0.1,
            tip="0 = disabled (recommended for BLAST)"); row += 1

        self.cb_mobility = QCheckBox("Enable mobility correction")
        self.cb_mobility.setChecked(True)
        self.cb_mobility.setToolTip(
            "Cross-correlation pre-alignment + cubic-polynomial inter-channel\n"
            "drift correction.  Corrects systematic timing offsets between\n"
            "dye channels that shift peaks away from their true position.")
        pg.addWidget(self.cb_mobility, row, 0, 1, 2); row += 1

        pg.addWidget(self._section_label("Running Noise Floor (SNR)"), row, 0, 1, 2); row += 1
        self.spin_noise_floor_window = self._spinbox(
            pg, row, "Noise floor window:", 11, 501, 51); row += 1
        self.spin_noise_floor_pct = self._dspinbox(
            pg, row, "Noise floor percentile:", 1.0, 49.0, 5.0, 1.0); row += 1

        pg.addWidget(self._section_label("Peak Detection"), row, 0, 1, 2); row += 1

        self.cb_adaptive_peaks = QCheckBox("Adaptive peak detection  (BLAST+)")
        self.cb_adaptive_peaks.setChecked(True)
        self.cb_adaptive_peaks.setToolTip(
            "Normalise each channel by its local amplitude envelope before\n"
            "peak finding.  This gives equal sensitivity in the read core\n"
            "(high amplitude) and read tail (low amplitude), recovering more\n"
            "bases without increasing false positives in the core.\n\n"
            "Highly recommended for BLAST: longer called regions → better\n"
            "e-value even if per-base quality is similar.")
        pg.addWidget(self.cb_adaptive_peaks, row, 0, 1, 2); row += 1

        self.spin_prom = self._dspinbox(pg, row, "Prominence:", 0.0001, 0.5, 0.004, 0.001,
            tip="Fraction of channel max (adaptive mode) or absolute fraction.\n"
                "0.004 is a good starting point for adaptive detection.\n"
                "Lower (0.002) to recover more tail peaks; raise to reduce noise.",
            decimals=4); row += 1
        self.spin_dist      = self._spinbox(pg, row, "Peak distance:", 1, 50, 4); row += 1
        self.spin_merge_tol = self._spinbox(pg, row, "Merge tolerance:", 1, 10, 3); row += 1

        pg.addWidget(self._section_label("Quality Filters"), row, 0, 1, 2); row += 1
        pg.addWidget(self.allow_uncertain, row, 0, 1, 2); row += 1
        self.spin_min_snr = self._dspinbox(pg, row, "Min SNR:", 0.5, 10.0, 1.5, 0.5); row += 1
        self.spin_min_iso = self._dspinbox(pg, row, "Min isolation:", 1.0, 5.0, 1.2, 0.1); row += 1

        pg.addWidget(self._section_label("Quality Trimming (Phred / BLAST)"), row, 0, 1, 2); row += 1
        self.cb_use_window_trim = QCheckBox("Use sliding-window trim (Kadane)")
        self.cb_use_window_trim.setChecked(True)
        pg.addWidget(self.cb_use_window_trim, row, 0, 1, 2); row += 1
        self.spin_window_trim_min_q = self._dspinbox(
            pg, row, "  Window min Q:", 1.0, 40.0, 20.0, 1.0,
            tip="Q20 = ≤1% error rate (recommended for BLAST)"); row += 1
        self.spin_window_trim_size = self._spinbox(
            pg, row, "  Window size (bases):", 5, 100, 20); row += 1

        pg.addWidget(self._section_label("Heterozygote Detection"), row, 0, 1, 2); row += 1

        self.cb_detect_het = QCheckBox("Flag het positions  (■ markers)")
        self.cb_detect_het.setChecked(False)
        self.cb_detect_het.setToolTip(
            "Enable heterozygote detection.\n\n"
            "OFF by default: BigDye v3.1 spectral bleed between dye channels\n"
            "is 25–45% of the dominant channel even after crosstalk correction.\n"
            "At het_threshold=0.45 this flags almost every peak as heterozygous\n"
            "— the ■ markers visible on nearly every call are false positives\n"
            "from normal BigDye physics, not genuine mixed-template SNPs.\n\n"
            "Tick only for diploid organism sequencing, known mixed-template /\n"
            "pooled samples, or SNP discovery runs.")
        pg.addWidget(self.cb_detect_het, row, 0, 1, 2); row += 1

        self.spin_het_threshold = self._dspinbox(
            pg, row, "  Het ratio threshold:", 0.1, 0.9, 0.45, 0.05,
            tip="Only active when 'Flag het positions' is checked."); row += 1

        self.spin_het_threshold.setEnabled(self.cb_detect_het.isChecked())
        self.cb_detect_het.toggled.connect(self.spin_het_threshold.setEnabled)

        pg.addWidget(self._section_label("Primer Trimming"), row, 0, 1, 2); row += 1
        pg.addWidget(QLabel("5′ primer sequence:"), row, 0); row += 1
        self.edit_primer = QLineEdit()
        self.edit_primer.setPlaceholderText("e.g. GTAAAACGACGGCCAGT  (blank = skip)")
        pg.addWidget(self.edit_primer, row, 0, 1, 2); row += 1

        pg.addWidget(self._section_label("Advanced Diagnostics"), row, 0, 1, 2); row += 1

        self.cb_auto_trim_blob = QCheckBox("Auto-trim dye blob")
        self.cb_auto_trim_blob.setChecked(True)
        pg.addWidget(self.cb_auto_trim_blob, row, 0, 1, 2); row += 1

        self.cb_suppress_n1 = QCheckBox("Suppress N-1 shadow peaks")
        self.cb_suppress_n1.setChecked(True)
        pg.addWidget(self.cb_suppress_n1, row, 0, 1, 2); row += 1

        self.cb_spacing_model = QCheckBox("Use spacing model")
        self.cb_spacing_model.setChecked(True)
        pg.addWidget(self.cb_spacing_model, row, 0, 1, 2); row += 1

        self.cb_detect_missing = QCheckBox("Detect / impute missing peaks")
        self.cb_detect_missing.setChecked(True)
        pg.addWidget(self.cb_detect_missing, row, 0, 1, 2); row += 1

        pg.addWidget(self._section_label("Channel Timing Shifts"), row, 0, 1, 2); row += 1
        pg.addWidget(QLabel("  Positive = delay  |  Negative = advance"), row, 0, 1, 2); row += 1

        self.spin_shift_G = self._spinbox(pg, row, "  G shift (black):", -20, 20, 0); row += 1
        self.spin_shift_A = self._spinbox(pg, row, "  A shift (green):",   -20, 20, 0); row += 1
        self.spin_shift_T = self._spinbox(pg, row, "  T shift (red):",  -20, 20, 0); row += 1
        self.spin_shift_C = self._spinbox(pg, row, "  C shift (blue):", -20, 20, 0); row += 1

        ll.addWidget(pg_box)

        # ── Run button ────────────────────────────────────────────────────────
        run_box = QGroupBox("Processing Controls")
        run_lay = QVBoxLayout(run_box)
        self.btn_run = QPushButton("▶  Run Pipeline")
        self.btn_run.clicked.connect(self.run_pipeline)
        self.btn_run.setStyleSheet(
            "font-weight:bold;background-color:#4CAF50;color:white;padding:6px;")
        run_lay.addWidget(self.btn_run)
        ll.addWidget(run_box)

        # ── Visualisation ─────────────────────────────────────────────────────
        vis_box = QGroupBox("Visualisation")
        vis_lay = QVBoxLayout(vis_box)
        vis_lay.addWidget(QLabel("Stage to View:"))
        self.combo_stage = QComboBox()
        self.combo_stage.addItems([
            "raw", "crosstalk", "baseline", "smooth",
            "mobility", "sharpened", "final"
        ])
        self.combo_stage.currentTextChanged.connect(self.on_stage_change)
        vis_lay.addWidget(self.combo_stage)
        self._add_btn(vis_lay, "Save Current Plot", self.save_displayed_plot)
        ll.addWidget(vis_box)

        # ── Export ────────────────────────────────────────────────────────────
        exp_box = QGroupBox("Export")
        exp_lay = QVBoxLayout(exp_box)

        btn_blast = QPushButton("🔬  Export BLAST-Ready FASTA  (fwd + RC)")
        btn_blast.clicked.connect(self.export_blast_ready)
        btn_blast.setStyleSheet(
            "font-weight:bold;background-color:#1565C0;color:white;padding:5px;")
        btn_blast.setToolTip(
            "Automatically selects the best quality window in the sequence\n"
            "and writes TWO files:\n"
            "  • yourfile.fasta        — forward strand (best window)\n"
            "  • yourfile_rc.fasta     — reverse complement of same window\n\n"
            "BLAST both.  One of them is the strand your primer sequenced.\n"
            "The one with hits is the correct orientation for your sample.")
        exp_lay.addWidget(btn_blast)

        # ── GUI-NCBI-1: NCBI BLAST button ─────────────────────────────────────
        btn_ncbi_blast = QPushButton("🔍  BLAST via NCBI (online)")
        btn_ncbi_blast.clicked.connect(self.blast_current)
        btn_ncbi_blast.setStyleSheet(
            "font-weight:bold;background-color:#0277BD;color:white;padding:5px;")
        btn_ncbi_blast.setToolTip(
            "Submit the best quality window directly to NCBI BLAST.\n\n"
            "• Requires an internet connection\n"
            "• Opens a non-modal dialog — GUI stays usable while waiting\n"
            "• Typical wait: 20–90 seconds for blastn / nt\n"
            "• Results appear in a colour-coded hit table inside the dialog\n\n"
            + ("blast_ncbi.py loaded ✓" if _BLAST_AVAILABLE
               else "⚠ blast_ncbi.py not found — place it alongside GUI.py"))
        exp_lay.addWidget(btn_ncbi_blast)

        self._add_btn(exp_lay, "↩  Export Reverse Complement FASTA", self.export_rc_fasta)
        self._add_btn(exp_lay, "Export FASTA (N-trimmed)",      self.export_fasta)
        self._add_btn(exp_lay, "Export FASTQ (quality scores)", self.export_fastq)
        self._add_btn(exp_lay, "Export Window-Trimmed FASTA",   self.export_window_fasta)
        self._add_btn(exp_lay, "Export Peaks CSV",              self.export_peaks_csv)
        ll.addWidget(exp_box)
        ll.addStretch()

        # ── Right panel ───────────────────────────────────────────────────────
        right = QWidget()
        rl    = QVBoxLayout(right)

        self.fig_signal    = Figure(figsize=(10, 4))
        self.canvas_signal = FigureCanvas(self.fig_signal)
        rl.addWidget(self.canvas_signal)
        rl.addWidget(NavigationToolbar(self.canvas_signal, self))

        self.fig_matrix    = Figure(figsize=(8, 2.5))
        self.canvas_matrix = FigureCanvas(self.fig_matrix)
        rl.addWidget(self.canvas_matrix)

        self.ab1_group = QGroupBox("AB1 KB Basecall Comparison")
        ab1_lay = QVBoxLayout(self.ab1_group)
        self.lbl_ab1 = QLabel("Load an AB1 file to see KB basecalls here.")
        self.lbl_ab1.setWordWrap(True)
        self.lbl_ab1.setStyleSheet("font-family:monospace;font-size:10px;")
        ab1_lay.addWidget(self.lbl_ab1)
        self.ab1_group.setVisible(False)
        rl.addWidget(self.ab1_group)

        qual_box = QGroupBox("Quality Statistics")
        qual_lay = QVBoxLayout(qual_box)
        self.lbl_qual = QLabel("Run the pipeline to see quality statistics.")
        self.lbl_qual.setWordWrap(True)
        self.lbl_qual.setStyleSheet("font-family:monospace;font-size:11px;")
        qual_lay.addWidget(self.lbl_qual)
        rl.addWidget(qual_box)

        seq_box = QGroupBox("Called Sequence (full, before export trimming)")
        seq_lay = QVBoxLayout(seq_box)
        self.lbl_seq = QLabel("Sequence: None")
        self.lbl_seq.setWordWrap(True)
        seq_lay.addWidget(self.lbl_seq)
        rl.addWidget(seq_box)

        self.table = QTableWidget(0, 14)
        self.table.setHorizontalHeaderLabels([
            "Pos", "Time", "Base", "Ch", "Amp", "SNR", "Isolation",
            "Shape", "Spacing", "Score", "Phred", "Tail", "Het", "Imputed"
        ])
        for col, w in enumerate([50, 65, 35, 55, 65, 55, 70, 55, 60, 55, 50, 35, 35, 55]):
            self.table.setColumnWidth(col, w)
        rl.addWidget(self.table)

        layout.addWidget(right, 1)

    # ── Matrix tab ────────────────────────────────────────────────────────────

    def _build_matrix_tab(self, parent):
        layout = QVBoxLayout(parent)
        btn_row = QHBoxLayout()
        self._add_btn_to(btn_row, "Compare with Another Matrix", self.compare_matrix)
        self._add_btn_to(btn_row, "Validate Current Matrix",     self.validate_matrix)
        self._add_btn_to(btn_row, "Use Identity Matrix",         self.use_identity_matrix)
        layout.addLayout(btn_row)
        self.matrix_info = QTextEdit()
        self.matrix_info.setReadOnly(True)
        self.matrix_info.setMaximumHeight(100)
        layout.addWidget(self.matrix_info)
        self.fig_matrix_tools    = Figure(figsize=(12, 4))
        self.canvas_matrix_tools = FigureCanvas(self.fig_matrix_tools)
        layout.addWidget(self.canvas_matrix_tools)

    # ── Batch tab ─────────────────────────────────────────────────────────────

    def _build_batch_tab(self, parent):
        layout = QVBoxLayout(parent)
        info = QLabel(
            "Add AB1 or CSV files below. Pipeline parameters from the "
            "<b>Processing</b> tab apply to every file. "
            "A crosstalk matrix must be loaded first."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        list_row = QHBoxLayout()
        self._add_btn_to(list_row, "Add AB1 Files…",  self.batch_add_ab1)
        self._add_btn_to(list_row, "Add CSV Files…",  self.batch_add_csv)
        self._add_btn_to(list_row, "Remove Selected", self.batch_remove_selected)
        self._add_btn_to(list_row, "Clear All",       self.batch_clear)
        layout.addLayout(list_row)

        self.batch_list = QListWidget()
        self.batch_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.batch_list.setMaximumHeight(160)
        layout.addWidget(self.batch_list)

        prog_row = QHBoxLayout()
        self.batch_progress   = QProgressBar()
        self.batch_progress.setFormat("%v / %m  —  %p%")
        self.batch_lbl_status = QLabel("Ready")
        prog_row.addWidget(self.batch_progress)
        prog_row.addWidget(self.batch_lbl_status)
        layout.addLayout(prog_row)

        run_row = QHBoxLayout()
        self.btn_batch_run = QPushButton("▶  Run Batch")
        self.btn_batch_run.setStyleSheet(
            "font-weight:bold;background-color:#2196F3;color:white;padding:6px;")
        self.btn_batch_run.clicked.connect(self.batch_run)
        self.btn_batch_abort = QPushButton("■  Abort")
        self.btn_batch_abort.setEnabled(False)
        self.btn_batch_abort.setStyleSheet(
            "font-weight:bold;background-color:#f44336;color:white;padding:6px;")
        self.btn_batch_abort.clicked.connect(self.batch_abort)
        run_row.addWidget(self.btn_batch_run)
        run_row.addWidget(self.btn_batch_abort)
        layout.addLayout(run_row)

        layout.addWidget(QLabel("Batch Results:"))
        self.batch_table = QTableWidget(0, 12)
        self.batch_table.setHorizontalHeaderLabels([
            "File", "Bases", "Confirmed", "Win-Trim", "BLAST Win",
            "Est ID%", "Imputed", "Het", "Mean Q", "Q≥20 %", "Q≥30 %", "Status"
        ])
        for col, w in enumerate([200, 55, 70, 65, 65, 60, 55, 45, 60, 65, 65, 70]):
            self.batch_table.setColumnWidth(col, w)
        layout.addWidget(self.batch_table, 1)

        export_row = QHBoxLayout()
        self._add_btn_to(export_row, "Export Batch Summary CSV",  self.batch_export_summary)
        self._add_btn_to(export_row, "Export All as Multi-FASTA", self.batch_export_fasta)
        layout.addLayout(export_row)

    # ─────────────────────────────────────────────────────────────────────────
    # Widget helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _section_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight:bold;margin-top:4px;")
        return lbl

    @staticmethod
    def _add_btn(layout, text, slot):
        b = QPushButton(text); b.clicked.connect(slot); layout.addWidget(b)

    @staticmethod
    def _add_btn_to(layout, text, slot):
        b = QPushButton(text); b.clicked.connect(slot); layout.addWidget(b)

    @staticmethod
    def _spinbox(layout, row, label, lo, hi, val, tip=""):
        layout.addWidget(QLabel(label), row, 0)
        sb = QSpinBox(); sb.setRange(lo, hi); sb.setValue(val)
        if tip: sb.setToolTip(tip)
        layout.addWidget(sb, row, 1)
        return sb

    @staticmethod
    def _dspinbox(layout, row, label, lo, hi, val, step, tip="", decimals=2):
        layout.addWidget(QLabel(label), row, 0)
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi); sb.setValue(val); sb.setSingleStep(step)
        sb.setDecimals(decimals)
        if tip: sb.setToolTip(tip)
        layout.addWidget(sb, row, 1)
        return sb

    # ─────────────────────────────────────────────────────────────────────────
    # SG slider ↔ spinbox
    # ─────────────────────────────────────────────────────────────────────────

    def _on_sg_slider_changed(self, val):
        if val % 2 == 0: val += 1
        self.spin_sg.blockSignals(True)
        self.spin_sg.setValue(val)
        self.spin_sg.blockSignals(False)
        if self.result is not None:
            self.preview_smoothing(silent=True)

    def _on_sg_spin_changed(self, val):
        if val % 2 == 0:
            val = max(val - 1, 3)
            self.spin_sg.blockSignals(True)
            self.spin_sg.setValue(val)
            self.spin_sg.blockSignals(False)
        self.slider_sg.blockSignals(True)
        self.slider_sg.setValue(min(val, self.slider_sg.maximum()))
        self.slider_sg.blockSignals(False)

    # ─────────────────────────────────────────────────────────────────────────
    # Parameter collection
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_shifts(self) -> list:
        return [
            self.spin_shift_G.value(),
            self.spin_shift_A.value(),
            self.spin_shift_T.value(),
            self.spin_shift_C.value(),
        ]

    def _collect_params(self) -> dict:
        all_params = dict(
            baseline_win           = self.spin_bg.value(),
            baseline_method        = self.combo_baseline.currentText(),
            sg_win                 = self.spin_sg.value(),
            sg_poly                = self.spin_sg_poly.value(),
            sg_passes              = self.spin_sg_passes.value(),
            sg_mode                = self.combo_sg_mode.currentText(),
            sigma                  = self.spin_sigma.value(),
            amount                 = self.spin_amount.value(),
            prom                   = self.spin_prom.value(),
            dist                   = self.spin_dist.value(),
            merge_tol              = self.spin_merge_tol.value(),
            mobility_enabled       = self.cb_mobility.isChecked(),
            channel_letters        = CHANNEL_LETTERS,
            channel_shifts         = self._collect_shifts(),
            min_snr                = self.spin_min_snr.value(),
            min_isolation          = self.spin_min_iso.value(),
            allow_uncertain        = self.allow_uncertain.isChecked(),
            primer_sequence        = self.edit_primer.text().strip(),
            auto_trim_blob         = self.cb_auto_trim_blob.isChecked(),
            suppress_n1            = self.cb_suppress_n1.isChecked(),
            use_spacing_model      = self.cb_spacing_model.isChecked(),
            detect_missing         = self.cb_detect_missing.isChecked(),
            het_threshold          = self.spin_het_threshold.value(),
            detect_het             = self.cb_detect_het.isChecked(),
            use_window_trim        = self.cb_use_window_trim.isChecked(),
            window_trim_min_q      = self.spin_window_trim_min_q.value(),
            window_trim_size       = self.spin_window_trim_size.value(),
            noise_floor_window     = self.spin_noise_floor_window.value(),
            noise_floor_percentile = self.spin_noise_floor_pct.value(),
            use_adaptive_peaks     = self.cb_adaptive_peaks.isChecked(),
        )
        filtered    = {k: v for k, v in all_params.items()
                       if k in self._PIPELINE_PARAMS}
        unsupported = set(all_params) - set(filtered)
        if unsupported and not self._compat_warned:
            self._compat_warned = True
            QMessageBox.warning(
                self, "Processor Version Mismatch",
                "Your ud_processor.py is older than this GUI.\n"
                "The following parameters are not supported and will be ignored:\n\n"
                + "\n".join(f"  • {k}" for k in sorted(unsupported))
                + "\n\nUpdate ud_processor.py to enable these features."
            )
        return filtered

    # ─────────────────────────────────────────────────────────────────────────
    # File loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_srd(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SRD File", os.getcwd(), "SRD files (*.srd);;All files (*.*)")
        if not path: return
        try:
            tree = ET.parse(path); root = tree.getroot()
            time_step_elem = root.find('TimeStep')
            time_step = float(time_step_elem.text) / 1000.0 if time_step_elem else 0.01
            data_list = []
            for point in root.findall('.//Point'):
                data_block = point.find('Data')
                if data_block is not None:
                    ints = [int(i.text) for i in data_block.findall('int')]
                    if len(ints) >= 4:
                        data_list.append(ints[:4])
            if not data_list:
                raise ValueError("No data points found in SRD file")
            signals = np.array(data_list, dtype=float).T
            time    = np.arange(signals.shape[1]) * time_step
        except Exception as e:
            QMessageBox.critical(self, "SRD Load Error", f"Failed to parse SRD:\n{e}"); return
        self._set_signal(time, signals, None, f"SRD: {os.path.basename(path)}", path)

    def load_signal(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Signal CSV", os.getcwd(), "CSV files (*.csv);;All files (*.*)")
        if not path: return
        try:
            time, signals = P.load_data_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return
        self._set_signal(time, signals, None, f"CSV: {os.path.basename(path)}", path)
        QMessageBox.information(self, "Loaded",
            f"Signal loaded.\nShape: {signals.shape}\nTime points: {len(time)}")

    def load_ab1(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open AB1 File", os.getcwd(),
            "AB1 files (*.ab1 *.abi *.ABI);;All files (*.*)")
        if not path: return
        try:
            prefer_raw = self.cb_prefer_raw.isChecked()
            time, signals, meta = P.load_ab1(path, prefer_raw_channels=prefer_raw)
        except Exception as e:
            QMessageBox.critical(self, "AB1 Load Error", str(e)); return
        name  = meta.get('sample_name', '') or os.path.basename(path)
        n_pts = meta.get('n_samples', len(time))
        src   = meta.get('data_source', '?')
        label = (f"AB1: {name}  ({n_pts} pts, 4 ch)  "
                 f"dye: {meta.get('dye_order','?')}  src: {src}")
        self._set_signal(time, signals, meta, label, path)
        self._show_ab1_metadata(meta)
        QMessageBox.information(self, "AB1 Loaded",
            f"Sample: {name}\nDye: {meta.get('dye_order','?')}\n"
            f"Data source: {src}\n"
            f"Points: {n_pts}  KB bases: {len(meta.get('kb_bases',''))}\n\n"
            "Run the pipeline to compare basecalls with KB.\n\n"
            f"Tip: data source '{src}' — "
            + ("raw channels preferred (recommended)."
               if '1-4' in src else
               "analysed channels used (consider ticking 'Prefer raw channels')."))

    def _set_signal(self, time, signals, meta, label, path):
        self.time     = time
        self.raw      = signals
        self.ab1_meta = meta
        self.result   = None
        self.lbl_csv_info.setText(label)
        if meta is None:
            self.ab1_group.setVisible(False)
        self.auto_load_matrix(path)
        self.plot_stage("raw")
        if self.matrix is not None:
            self.plot_matrix_heatmap(self.matrix, "Current Matrix")

    def _show_ab1_metadata(self, meta):
        kb = meta.get('kb_bases', '')
        kq = meta.get('kb_quality', np.array([]))
        pk = meta.get('peak_locs', np.array([]))
        lines = [f"KB basecalls ({len(kb)} bases):"]
        for i in range(0, min(len(kb), 200), 60):
            lines.append(f"  {i+1:>4}: {kb[i:i+60]}")
        if len(kb) > 200:
            lines.append(f"  … {len(kb)-200} more bases …")
        if len(kq) > 0:
            lines.append(f"\nKB quality — Mean: {float(np.mean(kq)):.1f}  "
                         f"Q≥20: {int(np.sum(kq >= 20))}  Q≥30: {int(np.sum(kq >= 30))}")
        if len(pk) > 1:
            lines.append(f"KB peak locations: {pk[0]} … {pk[-1]}  "
                         f"(spacing {int(np.min(np.diff(pk)))}–{int(np.max(np.diff(pk)))} pts)")
        self.lbl_ab1.setText("\n".join(lines))
        self.ab1_group.setVisible(True)

    def auto_load_matrix(self, data_path):
        base_dir = os.path.dirname(data_path)
        for name in ["matrix.csv", "crosstalk.csv",
                     "crosstalk_matrix.csv", "influence_matrix.csv"]:
            fp = os.path.join(base_dir, name)
            if os.path.exists(fp):
                try:
                    m = P.load_matrix_any(fp)
                    self._set_matrix(m, fp, 'loaded', f"{name} (auto-loaded)")
                    return True
                except Exception:
                    continue
        return False

    def load_matrix(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Matrix File", os.getcwd(), "All files (*.*)")
        if not path: return
        try:
            m = P.load_matrix_any(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return
        self._set_matrix(m, path, 'loaded', os.path.basename(path))
        QMessageBox.information(self, "Loaded", f"Matrix loaded.\nShape: {m.shape}")

    def _set_matrix(self, matrix, path, origin, label):
        self.matrix        = matrix
        self.matrix_path   = path
        self.matrix_origin = origin
        self.lbl_mat.setText(f"Matrix: {label}")
        self.plot_matrix_heatmap(matrix, label)
        self.update_matrix_info(matrix)

    def save_matrix(self):
        if self.matrix is None:
            QMessageBox.warning(self, "No Matrix", "Load or estimate a matrix first."); return
        default = os.path.basename(self.matrix_path) if self.matrix_path else "matrix.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Matrix", default, "CSV files (*.csv);;All files (*.*)")
        if not path: return
        try:
            P.save_matrix(self.matrix, path)
            QMessageBox.information(self, "Saved", f"Matrix saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Matrix operations
    # ─────────────────────────────────────────────────────────────────────────

    def estimate_matrix_from_data(self):
        if self.raw is None:
            QMessageBox.warning(self, "No Data", "Load signal data first.")
            return

        method       = self.combo_matrix_method.currentText()
        dom_ratio    = self.spin_dom_ratio.value()
        baseline_win = self.spin_bg.value()
        blob_end     = self._get_blob_end()

        shifts  = self._collect_shifts()
        shifted = P.apply_channel_shifts(self.raw, shifts)

        try:
            if method == 'dominant_channel':
                raw_m, diag = P.estimate_matrix_dominant_channel(
                    shifted,
                    baseline_win=baseline_win,
                    min_dominance_ratio=dom_ratio,
                    blob_end=blob_end,
                )
                new_m       = P.constrain_matrix_physical(raw_m)
                method_used = 'dominant_channel'
                all_diag    = {'dominant_channel': diag}

            elif method == 'nmf':
                raw_m, diag = P.estimate_matrix_nmf(
                    shifted,
                    baseline_win=baseline_win,
                    blob_end=blob_end,
                )
                new_m       = P.constrain_matrix_physical(raw_m)
                method_used = 'nmf'
                all_diag    = {'nmf': diag}

            else:
                new_m, method_used, all_diag = P.estimate_matrix_robust(
                    shifted,
                    baseline_win=baseline_win,
                    min_dominance_ratio=dom_ratio,
                    blob_end=blob_end,
                )

        except ImportError as e:
            QMessageBox.critical(self, "Missing Dependency",
                f"Method '{method}' requires an optional library:\n\n{e}")
            return
        except Exception as e:
            QMessageBox.critical(self, "Estimation Error",
                f"Matrix estimation failed (method={method}):\n\n{e}\n\n"
                + traceback.format_exc())
            return

        score          = P.score_matrix_plausibility(new_m)
        identity_score = P.score_matrix_plausibility(np.eye(4))
        lines = [
            f"Method requested  : {method}",
            f"Method used       : {method_used}",
            f"Plausibility score: {score:.2f}  "
            f"(identity matrix = {identity_score:.2f}; higher is better)",
            f"Blob end skipped  : {blob_end} samples",
            f"Dominance ratio   : {dom_ratio:.2f}",
            f"Baseline window   : {baseline_win}",
            "",
        ]

        diag_dom = all_diag.get('dominant_channel', {})
        if diag_dom:
            lines.append("Dominant-channel samples per channel:")
            for i, ch in enumerate(CHANNEL_LETTERS):
                n       = diag_dom.get(f'ch{i}_samples', '?')
                warn    = diag_dom.get(f'ch{i}_warn', '')
                not_max = diag_dom.get(f'ch{i}_skipped_not_max', '')
                dom_sk  = diag_dom.get(f'ch{i}_skipped_dominance', '')
                row_txt = f"  {ch}: {n} samples used"
                if not_max or dom_sk:
                    row_txt += f"  (skipped: not_max={not_max}, ratio={dom_sk})"
                if warn:
                    row_txt += f"\n      ⚠ {warn}"
                lines.append(row_txt)
            lines.append("")

        diag_nmf = all_diag.get('nmf', {})
        if diag_nmf and 'reconstruction_error' in diag_nmf:
            lines.append(f"NMF reconstruction error : {diag_nmf['reconstruction_error']:.4f}")
            lines.append(f"NMF iterations           : {diag_nmf.get('n_iter_done', '?')}")
            assign = diag_nmf.get('component_assignment', {})
            corrs  = diag_nmf.get('max_corr_per_comp', {})
            if assign:
                lines.append("NMF component → channel assignment:")
                for k, ch_idx in assign.items():
                    lines.append(f"  comp {k} → {CHANNEL_LETTERS[ch_idx]}  "
                                 f"(corr={corrs.get(k, 0):.3f})")
            lines.append("")

        if 'winner' in all_diag:
            lines.append(f"Robust winner : {all_diag['winner']}  "
                         f"(score {all_diag.get('winner_score', 0):.2f})")
            for mname in ('dominant_channel', 'nmf', 'peaks'):
                s = all_diag.get(mname, {}).get('plausibility_score')
                if s is not None:
                    lines.append(f"  {mname:<22} plausibility={s:.2f}")
            lines.append("")

        if self.matrix is not None:
            diff = self.matrix - new_m
            lines.append(
                f"Vs previous matrix — "
                f"L2 norm: {np.linalg.norm(diff):.4f}  "
                f"Max |diff|: {float(np.max(np.abs(diff))):.4f}"
            )
            lines.append("")

        bad_diag = []
        for i, ch in enumerate(CHANNEL_LETTERS):
            d    = float(new_m[i, i])
            cols = [float(new_m[r, i]) for r in range(4) if r != i]
            if d < 1.01:
                bad_diag.append(
                    f"  {ch}: diagonal={d:.3f} — identity fallback "
                    "(no clean dominant-channel samples found)")
            elif cols and d < max(cols):
                bad_diag.append(
                    f"  {ch}: diagonal ({d:.3f}) < max off-diagonal "
                    f"({max(cols):.3f}) — column may be unreliable")

        self._set_matrix(new_m, None,
                         f'estimated ({method_used})',
                         f"estimated [{method_used}] from data")

        report = "\n".join(lines)

        if bad_diag:
            report += (
                "\n\nWARNING — physically suspect columns:\n"
                + "\n".join(bad_diag)
                + "\n\nTroubleshooting tips:\n"
                "  • Lower 'Dominance ratio' to 1.2 to collect more samples\n"
                "  • Run the pipeline once first so blob_end is set correctly\n"
                "  • Try method='robust' to let the pipeline choose\n"
                "  • Try method='nmf' if channels overlap heavily\n"
                "  • A plausibility score > 10 is generally trustworthy\n"
                "  • If all else fails, load a matrix from a reference AB1 file"
            )
            QMessageBox.warning(self, "Matrix Estimated (with warnings)", report)
        else:
            QMessageBox.information(self, "Matrix Estimated", report)

    def compare_matrix(self):
        if self.matrix is None:
            QMessageBox.warning(self, "No Matrix", "Load or estimate a matrix first."); return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Matrix to Compare", os.getcwd(), "All files (*.*)")
        if not path: return
        try:
            other = P.load_matrix_any(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return
        diff = self.matrix - other
        self.fig_matrix_tools.clear()
        ax1, ax2, ax3 = self.fig_matrix_tools.subplots(1, 3)
        for ax, mat, title in [(ax1, self.matrix, "Current"),
                                (ax2, other, os.path.basename(path)),
                                (ax3, diff,  "Difference")]:
            vabs = max(float(np.max(np.abs(mat))), 1e-9)
            im = ax.imshow(mat, cmap='coolwarm', vmin=-vabs, vmax=vabs)
            ax.set_title(title)
            ax.set_xticks(range(4)); ax.set_xticklabels(CHANNEL_LETTERS)
            ax.set_yticks(range(4)); ax.set_yticklabels(CHANNEL_LETTERS)
            self.fig_matrix_tools.colorbar(im, ax=ax)
        self.fig_matrix_tools.tight_layout()
        self.canvas_matrix_tools.draw()
        QMessageBox.information(self, "Matrix Comparison",
            f"L2 Norm: {np.linalg.norm(diff):.6f}\n"
            f"Max |diff|: {float(np.max(np.abs(diff))):.6f}\n"
            f"Mean |diff|: {float(np.mean(np.abs(diff))):.6f}")

    def validate_matrix(self):
        if self.matrix is None:
            QMessageBox.warning(self, "No Matrix", "Load or estimate a matrix first."); return
        issues = []
        if self.matrix.shape != (4, 4):
            issues.append(f"Shape {self.matrix.shape} — expected (4,4)")
        for i, ch in enumerate(CHANNEL_LETTERS):
            d   = self.matrix[i, i]
            off = [abs(self.matrix[i, j]) for j in range(4) if j != i]
            if d <= 0: issues.append(f"{ch}: non-positive diagonal")
            elif d < 1.05: issues.append(f"{ch}: diagonal {d:.3f} near 1")
            if off and d < max(off):
                issues.append(f"{ch}: diagonal ({d:.3f}) < max off-diagonal ({max(off):.3f})")
        cond = float(np.linalg.cond(self.matrix))
        if cond > 1000: issues.append(f"Condition number high: {cond:.1f}")
        score = P.score_matrix_plausibility(self.matrix)
        if issues:
            QMessageBox.warning(self, "Validation Issues",
                "\n".join(f"  {s}" for s in issues)
                + f"\n\nPlausibility score: {score:.2f}")
        else:
            QMessageBox.information(self, "Validation OK",
                f"Matrix valid.\nCondition: {cond:.1f}\nPlausibility score: {score:.2f}")

    def use_identity_matrix(self):
        self._set_matrix(np.eye(4), None, 'identity', "Identity (no correction)")
        QMessageBox.information(self, "Identity Matrix",
            "Using identity matrix — no crosstalk correction.\n\n"
            "WARNING: BigDye spectral bleed (25–45%) will not be corrected.\n"
            "This is likely to reduce BLAST identity significantly.\n"
            "Load a real crosstalk matrix for best results.")

    def update_matrix_info(self, matrix):
        if matrix is None:
            self.matrix_info.setText("No matrix loaded"); return
        d     = [f"{matrix[i,i]:.3f}" for i in range(4)]
        score = P.score_matrix_plausibility(matrix) if hasattr(P, 'score_matrix_plausibility') else None
        score_txt = f"    Plausibility: {score:.2f}" if score is not None else ""
        self.matrix_info.setText(
            f"Shape: {matrix.shape}    Diagonal: [{', '.join(d)}]\n"
            f"Condition: {np.linalg.cond(matrix):.1f}    "
            f"Det: {np.linalg.det(matrix):.6f}{score_txt}")

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def preview_smoothing(self, silent: bool = False):
        if self.result is None:
            if not silent:
                QMessageBox.warning(self, "No Data", "Run the pipeline first.")
            return
        baseline = self.result['stages'].get('baseline')
        if baseline is None: return
        try:
            sg_win    = self.spin_sg.value()
            sg_poly   = self.spin_sg_poly.value()
            sg_passes = self.spin_sg_passes.value()
            sg_mode   = self.combo_sg_mode.currentText()
            smoothed  = P.smooth_savgol(baseline, sg_win, sg_poly, sg_passes, sg_mode)
        except Exception as e:
            if not silent:
                QMessageBox.critical(self, "Smoothing Error", str(e))
            return

        self.lbl_smooth_info.setText(
            f"Preview: win={sg_win}  poly={sg_poly}  passes={sg_passes}  mode={sg_mode}")

        self.fig_signal.clear()
        ax = self.fig_signal.add_subplot(111)

        for i in range(min(4, baseline.shape[0])):
            color = CHANNEL_COLORS[i]; ltr = CHANNEL_LETTERS[i]
            ax.plot(self.time, baseline[i],
                    color=color, linewidth=0.7, alpha=0.30, label=f"{ltr} baseline")
            ax.plot(self.time, smoothed[i],
                    color=color, linewidth=1.0, alpha=0.90, label=f"{ltr} smoothed")

        blob_end = self._get_blob_end()
        self._shade_blob(ax, blob_end)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=7)
        ax.set_title(
            f"SG Smoothing Preview  win={sg_win}  poly={sg_poly}  "
            f"passes={sg_passes}  mode={sg_mode}\n"
            "faint = baseline-corrected,  solid = smoothed")
        ax.set_xlabel("Sample index"); ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.20)
        self.fig_signal.tight_layout()
        self.canvas_signal.draw()

    def run_pipeline(self):
        if self.raw is None:
            QMessageBox.warning(self, "No Data", "Load signal data first."); return
        if self.matrix is None:
            reply = QMessageBox.question(self, "No Matrix",
                "No crosstalk matrix loaded.\n\n"
                "Using the identity matrix means BigDye spectral bleed (25–45%)\n"
                "will not be corrected, which typically reduces BLAST identity.\n\n"
                "Use identity matrix anyway?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.use_identity_matrix()
            else:
                return
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Processing…")
        self._worker = PipelineWorker(
            self.time.copy(), self.raw.copy(),
            self.matrix.copy(), self._collect_params())
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.start()

    def _on_pipeline_finished(self, result):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run Pipeline")
        self.result = result

        peaks              = result.get('peaks', [])
        seq                = result.get('sequence', '')
        seq_rc             = result.get('sequence_rc', P.reverse_complement(seq))
        window_trimmed_seq = result.get('window_trimmed_seq', '')
        qs                 = result.get('quality_stats', {})
        shifts             = self._collect_shifts()
        ptrim              = result.get('primer_trim_pos', 0)
        n_het              = result.get('n_heterozygotes', 0)
        n_imp              = result.get('n_imputed', 0)
        trimmed_preview    = P.trim_N_ends(seq)
        blast_method       = result.get('baseline_method', '?')

        blast_win_len = qs.get('blast_window_len', 0)
        blast_est_id  = qs.get('blast_est_identity', 0.0)

        seq_label = (
            (f"Primer-trimmed: {len(seq)} bases  |  " if ptrim else
             f"Full: {len(seq)} bases  |  ")
            + f"After N-trim: {len(trimmed_preview)} bases  |  "
            + f"Window-trimmed: {len(window_trimmed_seq)} bases  |  "
            + f"BLAST window: {blast_win_len} bases"
            + (f"  |  Primer at pos {ptrim}" if ptrim else "")
        )
        self.lbl_seq.setText(
            seq_label
            + f"\nFWD: {seq[:120]}{'…' if len(seq) > 120 else ''}"
            + f"\n RC: {seq_rc[:120]}{'…' if len(seq_rc) > 120 else ''}")

        if self.ab1_meta and self.ab1_meta.get('kb_bases'):
            kb   = self.ab1_meta['kb_bases']
            comp = min(len(kb), len(seq))
            matches = sum(a == b for a, b in zip(kb, seq))
            pct = 100.0 * matches / comp if comp else 0.0
            self.lbl_ab1.setText(
                self.lbl_ab1.text() +
                f"\n\n— Pipeline vs KB ({comp} bases compared) —\n"
                f"Matches: {matches}  Identity: {pct:.1f}%\n"
                f"Our: {seq[:80]}{'…' if len(seq) > 80 else ''}\n"
                f"KB:  {kb[:80]}{'…' if len(kb) > 80 else ''}")

        if qs:
            mean_snr = qs.get('mean_snr', 0)
            snr_note = ""
            if mean_snr > 1e6:
                snr_note = "\n⚠ SNR very high — noise floor may be near zero."
            elif mean_snr < 1:
                snr_note = "\n⚠ SNR very low — check signal quality or try ALS baseline."
            het_note = (
                "\n⚠ Het detection OFF — ■ markers suppressed (BigDye bleed)."
                if not self.cb_detect_het.isChecked() else ""
            )

            if blast_est_id >= 95.0 and blast_win_len >= 200:
                blast_light = f"🟢 BLAST-ready  est. {blast_est_id:.1f}% identity  {blast_win_len} bp"
            elif blast_est_id >= 85.0 or blast_win_len >= 100:
                blast_light = f"🟡 Marginal  est. {blast_est_id:.1f}% identity  {blast_win_len} bp"
            else:
                blast_light = (f"🔴 Below threshold  est. {blast_est_id:.1f}%  {blast_win_len} bp\n"
                               "   → try ALS baseline, check matrix, or export reverse complement")

            self.lbl_qual.setText(
                f"{blast_light}\n"
                f"Baseline method: {blast_method}\n\n"
                f"Total bases:      {qs.get('total_bases',0):>5}   "
                f"Confirmed: {qs.get('confirmed_bases',0):>5}   "
                f"N (tail): {qs.get('tail_rescued',0):>4}\n"
                f"Window-trimmed:   {qs.get('window_trimmed_len',0):>5}   "
                f"BLAST window: {blast_win_len:>5}   "
                f"Imputed: {qs.get('imputed_bases',0):>4}\n\n"
                f"Mean Phred Q: {qs.get('mean_phred',0):>5.1f}   "
                f"Median: {qs.get('median_phred',0):>5.1f}   "
                f"Min: {qs.get('min_phred',0):>3}   Max: {qs.get('max_phred',0):>3}\n"
                f"Q≥20 (≤1% err):  {qs.get('q20_count',0):>5} bases  ({qs.get('pct_q20',0):>5.1f}%)\n"
                f"Q≥30 (≤0.1% err):{qs.get('q30_count',0):>5} bases  ({qs.get('pct_q30',0):>5.1f}%)\n"
                f"Q≥35 (high qual):{qs.get('q35_count',0):>5} bases  ({qs.get('pct_q35',0):>5.1f}%)\n\n"
                f"Mean SNR: {mean_snr:>10.2f}   "
                f"Mean isolation: {qs.get('mean_isolation',0):>6.2f}   "
                f"Mean shape: {qs.get('mean_shape_score',0):>5.3f}"
                f"{snr_note}{het_note}\n\n"
                f"Het: {n_het}  Imputed: {n_imp}"
            )

        self.populate_table()
        self.combo_stage.setCurrentText('final')
        self.plot_stage('final')

        QMessageBox.information(self, "Processing Complete",
            f"Peaks: {len(peaks)}   Seq: {len(seq)} bases\n"
            f"N-trim: {len(trimmed_preview)}   Win-trim: {len(window_trimmed_seq)}\n"
            f"BLAST window: {blast_win_len} bases   Est. identity: {blast_est_id:.1f}%\n"
            + (f"Primer trim: {ptrim}\n" if ptrim else "")
            + f"Het: {n_het}   Imputed: {n_imp}\n"
            f"Mean Phred Q: {qs.get('mean_phred',0):.1f}   "
            f"Q≥20: {qs.get('pct_q20',0):.1f}%\n"
            f"Baseline: {blast_method}   "
            f"Matrix: {self.matrix_origin or 'unknown'}\n"
            f"Shifts G={shifts[0]} A={shifts[1]} T={shifts[2]} C={shifts[3]}\n\n"
            "Tip: use '🔍 BLAST via NCBI' to search the sequence online,\n"
            "or '🔬 Export BLAST-Ready FASTA' to export both strands.")

    def _on_pipeline_error(self, message):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Run Pipeline")
        QMessageBox.critical(self, "Processing Error", f"Pipeline failed:\n{message}")

    # ─────────────────────────────────────────────────────────────────────────
    # NCBI BLAST  [GUI-NCBI-1]
    # ─────────────────────────────────────────────────────────────────────────

    def blast_current(self):
        """
        Submit the best BLAST window to NCBI via the blast_ncbi dialog.
        Falls back gracefully if blast_ncbi.py is not installed.
        """
        if not _BLAST_AVAILABLE:
            QMessageBox.warning(self, "BLAST Module Not Found",
                "blast_ncbi.py was not found in the same folder as GUI.py.\n\n"
                "To enable online BLAST:\n"
                "  1. Save blast_ncbi.py alongside GUI.py and ud_processor.py\n"
                "  2. Restart the application\n\n"
                "No additional pip installs are needed (stdlib only).")
            return

        if not self.result:
            QMessageBox.warning(self, "No Data",
                "Run the pipeline first to generate a sequence.")
            return

        # Use the best quality window — same region as "Export BLAST-Ready FASTA"
        blast_win, _, _ = P.find_best_blast_window(self.result.get('peaks', []))
        if blast_win:
            seq = "".join(r['letter'] for r in blast_win)
        else:
            seq = P.trim_N_ends(self.result.get('sequence', ''))

        if not seq:
            QMessageBox.warning(self, "Empty Sequence",
                "No sequence available after trimming.\n"
                "Check the pipeline output and quality statistics.")
            return

        # Offer RC if the forward window is very short
        qs = self.result.get('quality_stats', {})
        if len(seq) < 100:
            reply = QMessageBox.question(
                self, "Short Forward Window",
                f"The forward BLAST window is only {len(seq)} bases.\n\n"
                "BLAST the reverse complement instead?\n\n"
                "(You can also edit the sequence manually in the dialog that opens.)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Yes:
                seq = P.reverse_complement(seq)

        open_blast_dialog(self, seq, qs)

    # ─────────────────────────────────────────────────────────────────────────
    # Batch processing
    # ─────────────────────────────────────────────────────────────────────────

    def _batch_file_entries(self):
        entries = []
        for i in range(self.batch_list.count()):
            item = self.batch_list.item(i)
            entries.append((item.data(Qt.UserRole),   item.data(Qt.UserRole+1),
                            item.data(Qt.UserRole+2), item.data(Qt.UserRole+3)))
        return entries

    def batch_add_ab1(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add AB1 Files", os.getcwd(),
            "AB1 files (*.ab1 *.abi *.ABI);;All files (*.*)")
        for p in paths: self._batch_load_file(p, True)

    def batch_add_csv(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add CSV Files", os.getcwd(), "CSV files (*.csv);;All files (*.*)")
        for p in paths: self._batch_load_file(p, False)

    def _batch_load_file(self, path, is_ab1):
        try:
            if is_ab1:
                prefer_raw = self.cb_prefer_raw.isChecked()
                time, signals, meta = P.load_ab1(path, prefer_raw_channels=prefer_raw)
            else:
                time, signals = P.load_data_csv(path); meta = None
        except Exception as e:
            QMessageBox.warning(self, "Skipped", f"Could not load:\n{path}\n\n{e}"); return
        item = QListWidgetItem(("🧬 " if is_ab1 else "📄 ") + os.path.basename(path))
        item.setData(Qt.UserRole,   path)
        item.setData(Qt.UserRole+1, time)
        item.setData(Qt.UserRole+2, signals)
        item.setData(Qt.UserRole+3, meta)
        self.batch_list.addItem(item)

    def batch_remove_selected(self):
        for item in self.batch_list.selectedItems():
            self.batch_list.takeItem(self.batch_list.row(item))

    def batch_clear(self):
        self.batch_list.clear(); self.batch_table.setRowCount(0)
        self._batch_results.clear()

    def batch_run(self):
        if self.batch_list.count() == 0:
            QMessageBox.warning(self, "No Files", "Add files first."); return
        if self.matrix is None:
            reply = QMessageBox.question(self, "No Matrix",
                "Use identity matrix for batch?\n\n"
                "Warning: this skips crosstalk correction and may reduce BLAST identity.",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes: self.use_identity_matrix()
            else: return
        self._batch_results.clear(); self.batch_table.setRowCount(0)
        self.batch_progress.setRange(0, self.batch_list.count())
        self.batch_progress.setValue(0)
        self.btn_batch_run.setEnabled(False); self.btn_batch_abort.setEnabled(True)
        self._batch_worker = BatchWorker(
            self._batch_file_entries(), self.matrix.copy(), self._collect_params())
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.file_done.connect(self._on_batch_file_done)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.start()

    def batch_abort(self):
        if self._batch_worker: self._batch_worker.abort()
        self.batch_lbl_status.setText("Aborting…")

    def _on_batch_progress(self, current, total, filename):
        self.batch_progress.setValue(current)
        self.batch_lbl_status.setText(f"[{current}/{total}] {filename}")

    def _on_batch_file_done(self, result):
        self._batch_results.append(result)
        path = result.get('_path', '?'); qs = result.get('quality_stats', {})
        row  = self.batch_table.rowCount(); self.batch_table.insertRow(row)
        for col, val in enumerate([
            os.path.basename(path), str(qs.get('total_bases',0)),
            str(qs.get('confirmed_bases',0)), str(qs.get('window_trimmed_len',0)),
            str(qs.get('blast_window_len',0)),
            f"{qs.get('blast_est_identity',0):.1f}%",
            str(qs.get('imputed_bases',0)), str(qs.get('heterozygote_bases',0)),
            f"{qs.get('mean_phred',0):.1f}", f"{qs.get('pct_q20',0):.1f}%",
            f"{qs.get('pct_q30',0):.1f}%", "✓ OK"
        ]):
            self.batch_table.setItem(row, col, QTableWidgetItem(val))

    def _on_batch_error(self, path, tb):
        row = self.batch_table.rowCount(); self.batch_table.insertRow(row)
        self.batch_table.setItem(row, 0, QTableWidgetItem(os.path.basename(path)))
        for col in range(1, 11): self.batch_table.setItem(row, col, QTableWidgetItem("—"))
        err = QTableWidgetItem("⚠ ERROR"); err.setToolTip(tb)
        self.batch_table.setItem(row, 11, err)
        # Fix: advance progress bar even on error so it doesn't stall
        self.batch_progress.setValue(self.batch_progress.value() + 1)

    def _on_batch_finished(self):
        self.btn_batch_run.setEnabled(True); self.btn_batch_abort.setEnabled(False)
        n_ok = sum(1 for r in self._batch_results if 'quality_stats' in r)
        self.batch_lbl_status.setText(f"Done — {n_ok}/{self.batch_list.count()} succeeded")
        QMessageBox.information(self, "Batch Complete",
            f"{n_ok} of {self.batch_list.count()} files processed successfully.")

    def batch_export_summary(self):
        if not self._batch_results:
            QMessageBox.warning(self, "No Results", "Run a batch first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Summary", "batch_summary.csv", "CSV files (*.csv);;All files (*.*)")
        if not path: return
        rows = []
        for r in self._batch_results:
            qs = r.get('quality_stats', {}); meta = r.get('_meta') or {}
            rows.append({
                'file':               os.path.basename(r.get('_path','')),
                'sample_name':        meta.get('sample_name',''),
                **{k: qs.get(k, 0) for k in [
                    'total_bases','confirmed_bases','tail_rescued','window_trimmed_len',
                    'blast_window_len','blast_est_identity',
                    'imputed_bases','heterozygote_bases','mean_phred','median_phred',
                    'pct_q20','pct_q30','pct_q35','mean_snr','mean_isolation','mean_shape_score']},
                'primer_trim_pos': r.get('primer_trim_pos',0),
                'n_heterozygotes': r.get('n_heterozygotes',0),
                'n_imputed':       r.get('n_imputed',0),
                'baseline_method': r.get('baseline_method','?'),
            })
        try:
            pd.DataFrame(rows).to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Batch summary saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def batch_export_fasta(self):
        if not self._batch_results:
            QMessageBox.warning(self, "No Results", "Run a batch first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Multi-FASTA", "batch_sequences.fasta",
            "FASTA files (*.fasta *.fa);;All files (*.*)")
        if not path: return
        written = 0
        try:
            with open(path, 'w') as fh:
                for r in self._batch_results:
                    seq = r.get('sequence', '')
                    if not seq: continue
                    blast_win, _, _ = P.find_best_blast_window(r.get('peaks', []))
                    seq  = "".join(x['letter'] for x in blast_win) if blast_win else P.trim_N_ends(seq)
                    qs   = r.get('quality_stats', {}); meta = r.get('_meta') or {}
                    name = (meta.get('sample_name','') or
                            os.path.splitext(os.path.basename(r.get('_path','sample')))[0])
                    hdr  = (f">{name} length={len(seq)} "
                            f"mean_Q={qs.get('mean_phred',0):.1f} "
                            f"pct_Q20={qs.get('pct_q20',0):.1f} "
                            f"est_id={qs.get('blast_est_identity',0):.1f}%")
                    fh.write(hdr + "\n")
                    for i in range(0, len(seq), 80): fh.write(seq[i:i+80]+"\n")
                    written += 1
            QMessageBox.information(self, "Saved",
                f"Multi-FASTA saved:\n{path}\n({written} sequences, BLAST windows)")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Export — single file
    # ─────────────────────────────────────────────────────────────────────────

    def export_blast_ready(self):
        if not self.result or not self.result.get('peaks'):
            QMessageBox.warning(self, "No Data", "Run pipeline first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save BLAST-Ready FASTA", "blast_ready.fasta",
            "FASTA files (*.fasta *.fa);;All files (*.*)")
        if not path: return
        try:
            meta = self.ab1_meta or {}
            sample = meta.get('sample_name', '') or os.path.splitext(os.path.basename(path))[0]
            fwd_seq, rc_seq = P.export_blast_ready(
                self.result['peaks'],
                self.result['sequence'],
                path,
                sample_name=sample,
                quality_stats=self.result.get('quality_stats'),
                also_export_rc=True,
            )
            rc_path = path.replace('.fasta', '_rc.fasta').replace('.fa', '_rc.fa')
            if rc_path == path: rc_path = path + '_rc.fasta'
            qs = self.result.get('quality_stats', {})
            QMessageBox.information(self, "BLAST-Ready Export",
                f"Forward FASTA:  {path}\n"
                f"  {len(fwd_seq)} bases\n\n"
                f"Reverse complement:  {rc_path}\n"
                f"  {len(rc_seq)} bases\n\n"
                f"Est. BLAST identity: {qs.get('blast_est_identity',0):.1f}%\n\n"
                "BLAST BOTH files at https://blast.ncbi.nlm.nih.gov/\n"
                "The one with hits is your correct strand orientation.\n\n"
                "Tip: use '🔍 BLAST via NCBI' to search directly from this app.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_rc_fasta(self):
        if not self.result or not self.result.get('sequence'):
            QMessageBox.warning(self, "No Sequence", "Run pipeline first."); return
        seq    = self.result['sequence']
        seq_rc = self.result.get('sequence_rc') or P.reverse_complement(seq)
        qs     = self.result.get('quality_stats', {})
        shifts = self._collect_shifts()
        header = (f">sanger_reverse_complement length={len(P.trim_N_ends(seq_rc))} "
                  f"mean_Q={qs.get('mean_phred',0):.1f} "
                  f"pct_Q20={qs.get('pct_q20',0):.1f} "
                  f"shifts=G{shifts[0]}_A{shifts[1]}_T{shifts[2]}_C{shifts[3]}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save RC FASTA", "sequence_rc.fasta",
            "FASTA files (*.fasta *.fa);;All files (*.*)")
        if not path: return
        try:
            written = P.save_fasta(seq_rc, path, header, trim_n=True)
            QMessageBox.information(self, "Saved",
                f"Reverse complement saved: {len(written)} bases → {path}\n\n"
                "If your forward sequence gave no BLAST hits, try this file.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_fasta(self):
        if not self.result or not self.result.get('sequence'):
            QMessageBox.warning(self, "No Sequence", "Run pipeline first."); return
        seq = self.result['sequence']; qs = self.result.get('quality_stats', {})
        shifts = self._collect_shifts(); trimmed = P.trim_N_ends(seq)
        ptrim = self.result.get('primer_trim_pos', 0)
        n_het = self.result.get('n_heterozygotes', 0)
        n_imp = self.result.get('n_imputed', 0)
        if len(trimmed) < len(seq):
            reply = QMessageBox.question(self, "N-Trimming",
                f"Terminal N bases will be removed:\n"
                f"  Original: {len(seq)}  Trimmed: {len(trimmed)}\nProceed?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No: return
        header = (f">sanger_sequence length={len(trimmed)} "
                  f"mean_Q={qs.get('mean_phred',0):.1f} pct_Q20={qs.get('pct_q20',0):.1f} "
                  f"shifts=G{shifts[0]}_A{shifts[1]}_T{shifts[2]}_C{shifts[3]}"
                  + (f" primer_trim={ptrim}" if ptrim else "")
                  + (f" het={n_het}" if n_het else "")
                  + (f" imputed={n_imp}" if n_imp else ""))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save FASTA", "sequence.fasta", "FASTA files (*.fasta *.fa);;All files (*.*)")
        if not path: return
        try:
            written = P.save_fasta(trimmed, path, header, trim_n=False)
            QMessageBox.information(self, "Saved", f"FASTA saved: {len(written)} bases → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_window_fasta(self):
        if not self.result or not self.result.get('window_trimmed_seq'):
            QMessageBox.warning(self, "No Data", "Run pipeline first."); return
        seq = self.result['window_trimmed_seq']; qs = self.result.get('quality_stats', {})
        shifts = self._collect_shifts()
        ptrim = self.result.get('primer_trim_pos', 0)
        header = (f">sanger_sequence_window_trimmed length={len(seq)} "
                  f"mean_Q={qs.get('mean_phred',0):.1f} pct_Q20={qs.get('pct_q20',0):.1f} "
                  f"window_size={self.spin_window_trim_size.value()} "
                  f"window_min_q={self.spin_window_trim_min_q.value():.0f} "
                  f"shifts=G{shifts[0]}_A{shifts[1]}_T{shifts[2]}_C{shifts[3]}"
                  + (f" primer_trim={ptrim}" if ptrim else ""))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Window-Trimmed FASTA", "sequence_window_trimmed.fasta",
            "FASTA files (*.fasta *.fa);;All files (*.*)")
        if not path: return
        try:
            written = P.save_fasta(seq, path, header, trim_n=False)
            QMessageBox.information(self, "Saved",
                f"Window-trimmed FASTA saved: {len(written)} bases → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_fastq(self):
        if not self.result or not self.result.get('peaks'):
            QMessageBox.warning(self, "No Data", "Run pipeline first."); return
        shifts = self._collect_shifts(); qs = self.result.get('quality_stats', {})
        header = (f"sanger_sequence mean_Q={qs.get('mean_phred',0):.1f} "
                  f"shifts=G{shifts[0]}_A{shifts[1]}_T{shifts[2]}_C{shifts[3]}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save FASTQ", "sequence.fastq",
            "FASTQ files (*.fastq *.fq);;All files (*.*)")
        if not path: return
        try:
            trimmed = P.save_fastq(
                self.result['peaks'], path, header,
                use_window_trim   = self.cb_use_window_trim.isChecked(),
                window_trim_min_q = self.spin_window_trim_min_q.value(),
                window_trim_size  = self.spin_window_trim_size.value())
            QMessageBox.information(self, "Saved",
                f"FASTQ saved: {len(trimmed)} bases → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def export_peaks_csv(self):
        if not self.result or not self.result.get('peaks'):
            QMessageBox.warning(self, "No Peaks", "Run pipeline first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Peaks CSV", "peaks.csv",
            "CSV files (*.csv);;All files (*.*)")
        if not path: return
        try:
            data = [{'position':      p['pos'],
                     'time':           p.get('time', p['pos']),
                     'base':           p['letter'],
                     'channel':        p['channel'],
                     'channel_name':   CHANNEL_LETTERS[p['channel']],
                     'channel_color':  CHANNEL_COLORS[p['channel']],
                     'amplitude':      p['amplitude'],
                     'snr':            p['snr'],
                     'isolation':      p['isolation'],
                     'shape_score':    p.get('shape_score',''),
                     'spacing_score':  p.get('spacing_score',''),
                     'quality_score':  p.get('score',0),
                     'phred':          p.get('phred',0),
                     'noise_level':    p.get('noise',0),
                     'tail_rescued':   p.get('tail_rescued',False),
                     'imputed':        p.get('imputed',False),
                     'heterozygote':   p.get('heterozygote',False),
                     'het_channel':    p.get('het_channel',-1),
                     'het_ratio':      p.get('het_ratio',0.0)}
                    for p in self.result['peaks']]
            pd.DataFrame(data).to_csv(path, index=False)
            QMessageBox.information(self, "Saved",
                f"Exported {len(data)} peaks → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def save_displayed_plot(self):
        stage = self.combo_stage.currentText()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", f"plot_{stage}.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path: return
        try:
            self.fig_signal.savefig(path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Visualisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_blob_end(self) -> int:
        if self.result:
            v = self.result['stages'].get('_blob_end', 0)
            return int(v) if isinstance(v, (int, float, np.integer)) else 0
        return 0

    def _shade_blob(self, ax, blob_end: int):
        if blob_end <= 0 or self.time is None: return
        t_end = self.time[min(blob_end, len(self.time) - 1)]
        ax.axvspan(self.time[0], t_end,
                   alpha=0.08, color='red', zorder=0,
                   label=f'Dye-blob (0–{blob_end})')

    def plot_stage(self, stage: str):
        self.fig_signal.clear()
        ax = self.fig_signal.add_subplot(111)

        if self.time is None:
            ax.text(0.5, 0.5, "No data loaded",
                    ha='center', va='center', transform=ax.transAxes)
            self.canvas_signal.draw(); return

        blob_end = self._get_blob_end()

        if stage == 'baseline' and self.result:
            self._plot_baseline_with_noise_floor(ax, blob_end)
        else:
            signal = (self.raw if stage == 'raw'
                      else (self.result['stages'].get(stage) if self.result else None))
            if signal is None:
                ax.text(0.5, 0.5, f"No data for stage '{stage}' — run pipeline first",
                        ha='center', va='center', transform=ax.transAxes)
                self.canvas_signal.draw(); return

            for i in range(min(4, signal.shape[0])):
                ax.plot(self.time, signal[i],
                        color=CHANNEL_COLORS[i], label=CHANNEL_LETTERS[i],
                        linewidth=1.0, alpha=0.85)

            self._shade_blob(ax, blob_end)

            if stage == 'final' and self.result and self.result.get('peaks'):
                self._annotate_peaks(ax, signal)

            if self.ab1_meta and stage in ('raw', 'final'):
                pk_locs = self.ab1_meta.get('peak_locs', np.array([]))
                if len(pk_locs) > 0:
                    valid = pk_locs[pk_locs < signal.shape[1]]
                    ax.vlines(self.time[valid], 0,
                              float(np.max(signal)) * 0.10,
                              color='gray', linewidth=0.5, alpha=0.35,
                              label='KB peaks')

            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', fontsize=8)
            ax.set_title(f"Stage: {stage}  ({signal.shape[1]} pts)  "
                         "G=black  A=green  T=red  C=blue  (◆=imputed  ■=het)")
            ax.set_xlabel("Sample index"); ax.set_ylabel("Intensity")
            ax.grid(True, alpha=0.22)

        self.fig_signal.tight_layout()
        self.canvas_signal.draw()

    def _plot_baseline_with_noise_floor(self, ax, blob_end: int):
        raw_sig = self.result['stages'].get('baseline_raw')
        if raw_sig is None:
            raw_sig = self.result['stages'].get('baseline')
        noise_floor = self.result['stages'].get('noise_floor')
        running_nf  = self.result['stages'].get('running_noise')

        if raw_sig is None:
            ax.text(0.5, 0.5, "baseline_raw stage not available.",
                    ha='center', va='center', transform=ax.transAxes)
            return

        sig_max = float(np.max(raw_sig)) if np.any(raw_sig > 0) else 1.0

        for i in range(min(4, raw_sig.shape[0])):
            color = CHANNEL_COLORS[i]
            ltr   = CHANNEL_LETTERS[i]

            ax.plot(self.time, raw_sig[i],
                    color=color, linewidth=1.0, alpha=0.90, label=ltr, zorder=3)

            if noise_floor is not None:
                floor_trace = noise_floor[i]
            elif running_nf is not None:
                floor_trace = running_nf[i]
            else:
                floor_trace = None

            if floor_trace is not None:
                ax.fill_between(self.time, 0, floor_trace,
                                color=color, alpha=0.18, linewidth=0, zorder=1)
                ax.plot(self.time, floor_trace,
                        color=color, linewidth=0.6, alpha=0.45,
                        linestyle='-', zorder=2, label='_nolegend_')

            if running_nf is not None and noise_floor is not None:
                ax.plot(self.time, running_nf[i],
                        color=color, linewidth=0.5, alpha=0.30,
                        linestyle=':', zorder=2, label='_nolegend_')

        self._shade_blob(ax, blob_end)
        ax.set_ylim(bottom=0, top=sig_max * 1.08)
        ax.set_xlim(left=float(self.time[0]), right=float(self.time[-1]))
        ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
        bl_label = self.combo_baseline.currentText()
        ax.set_title(
            f"Baseline stage [{bl_label}]  —  "
            "Solid = corrected signal,  Filled band = noise floor (raw baseline)")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.20)

    def _annotate_peaks(self, ax, signal):
        if not self.result or not self.result.get('peaks'): return
        sig_max        = float(np.max(signal)) if np.any(signal > 0) else 1.0
        y_floor        = sig_max * 0.01
        y_offset       = sig_max * 0.025
        min_label_gap  = 3
        peaks_sorted   = sorted(self.result['peaks'], key=lambda p: p['pos'])
        last_label_pos = -999
        for pk in peaks_sorted:
            pos = pk['pos']; ch = pk['channel']; ltr = pk['letter']
            if pos >= signal.shape[1]: continue
            if abs(pos - last_label_pos) < min_label_gap: continue
            y_apex   = max(float(np.max(signal[:, pos])), y_floor)
            y_marker = max(float(signal[ch, pos]), y_floor)
            marker = ('D' if pk.get('imputed') else
                      's' if pk.get('heterozygote') else 'o')
            ax.plot(self.time[pos], y_marker,
                    color='black', marker=marker, markersize=4,
                    markerfacecolor='none' if pk.get('imputed') else 'black',
                    zorder=5, linestyle='none')
            ax.text(self.time[pos], y_apex + y_offset, ltr,
                    fontsize=8, ha='center', va='bottom', fontweight='bold',
                    color=CHANNEL_COLORS[ch], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor='white', alpha=0.82))
            last_label_pos = pos

    def plot_matrix_heatmap(self, matrix: np.ndarray, title: str = "Matrix"):
        self.fig_matrix.clear()
        ax = self.fig_matrix.add_subplot(111)
        if matrix is None or matrix.size == 0:
            ax.text(0.5, 0.5, "No matrix", ha='center', va='center',
                    transform=ax.transAxes)
            self.canvas_matrix.draw(); return
        vmax = float(np.max(np.abs(matrix))) or 1.0
        im   = ax.imshow(matrix, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='equal')
        for i in range(4):
            for j in range(4):
                c = 'white' if abs(matrix[i, j]) > vmax * 0.5 else 'black'
                ax.text(j, i, f"{matrix[i,j]:.3f}",
                        ha='center', va='center', color=c, fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels(CHANNEL_LETTERS)
        ax.set_yticks(range(4)); ax.set_yticklabels(CHANNEL_LETTERS)
        ax.set_title(title)
        self.fig_matrix.colorbar(im, ax=ax)
        self.canvas_matrix.draw()

    def on_stage_change(self, stage: str):
        self.current_stage = stage
        self.plot_stage(stage)

    def populate_table(self):
        peaks = self.result.get('peaks', []) if self.result else []
        self.table.setRowCount(len(peaks))
        for row, pk in enumerate(peaks):
            ch = pk.get('channel', 0)
            vals = [
                str(pk.get('pos', '')),
                f"{pk.get('time', pk.get('pos', 0)):.1f}",
                pk.get('letter', ''),
                f"{ch} ({CHANNEL_COLORS[ch][:3]})" if ch < 4 else str(ch),
                f"{pk.get('amplitude', 0):.3f}",
                f"{pk.get('snr', 0):.2f}",
                f"{pk.get('isolation', 0):.2f}",
                f"{pk.get('shape_score', 0):.3f}" if 'shape_score' in pk else "",
                f"{pk.get('spacing_score', 0):.3f}" if 'spacing_score' in pk else "",
                f"{pk.get('score', 0):.1f}",
                str(pk.get('phred', 0)),
                "Y" if pk.get("tail_rescued") else "",
                f"{pk.get('het_ratio', 0):.2f}" if pk.get('heterozygote') else "",
                "Y" if pk.get("imputed") else "",
            ]
            for col, v in enumerate(vals):
                self.table.setItem(row, col, QTableWidgetItem(v))


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SangerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()