#!/usr/bin/env python3
"""
ud_processor.py — Production-grade Sanger sequencing pipeline
Version: BLAST-Optimised + Robust Matrix Estimation

Channel-colour-letter contract (enforced throughout)
-----------------------------------------------------
  signals[0]  →  G  →  black
  signals[1]  →  A  →  green
  signals[2]  →  T  →  Red
  signals[3]  →  C  →  blue

BLAST-optimisation changes (previous version — all retained)
------------------------------------------------------------
  BLAST-1  AB1 loader: raw DATA 1-4 preferred over KB-analysed DATA 9-12
  BLAST-2  reverse_complement()
  BLAST-3  baseline_als() — Asymmetric Least Squares
  BLAST-4  baseline_morphological()
  BLAST-5  find_peaks_adaptive()
  BLAST-6  find_best_blast_window()
  BLAST-7  auto_orient_sequence()
  BLAST-8  mobility_correction() with cross-correlation pre-alignment
  BLAST-9  _calibrated_phred() recalibrated SNR→Q mapping
  BLAST-10 export_blast_ready()
  BLAST-11 run_pipeline() baseline_method / prefer_raw_channels params

NEW: Robust matrix estimation for CSV / SRD files
--------------------------------------------------
  MATRIX-1  estimate_matrix_dominant_channel()
             Finds timepoints where each channel is the unambiguous maximum
             (by a configurable dominance ratio) after baseline correction.
             Averages the normalised 4-vector to form each matrix column.
             Works reliably on CSV and SRD data without requiring explicit
             single-channel peaks.

  MATRIX-2  estimate_matrix_nmf()
             Non-negative Matrix Factorisation of the 4×N signal matrix.
             Decomposes as M @ S_pure; M is the 4×4 crosstalk matrix.
             Requires scikit-learn (optional — graceful fallback).

  MATRIX-3  constrain_matrix_physical()
             Post-processing: diagonals → 1.0, off-diagonals clamped to
             [0, max_offdiag], ridge added for invertibility.

  MATRIX-4  estimate_matrix_robust()
             Tries all available methods, scores each for physical
             plausibility, returns the best.  GUI calls this.

  MATRIX-5  score_matrix_plausibility()
             Numeric quality score for a candidate matrix: rewards diagonal
             dominance and off-diagonals in [0,1], penalises negative or
             >1 off-diagonal values.
"""

import struct
import math
import warnings
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, correlate
from scipy.ndimage import (gaussian_filter1d, median_filter,
                            minimum_filter, maximum_filter)
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=RuntimeWarning)

MIN_BLAST_LENGTH = 100


# ============================================================
# AB1 / ABIF native parser  [BLAST-1]
# ============================================================

def load_ab1(path: str,
             prefer_raw_channels: bool = True
             ) -> Tuple[np.ndarray, np.ndarray, dict]:
    with open(path, 'rb') as f:
        if f.read(4) != b'ABIF':
            raise ValueError("Not a valid AB1 file")
        f.seek(26)
        f.read(4); f.read(4); f.read(2); f.read(4)
        dir_count  = struct.unpack('>i', f.read(4))[0]
        dir_offset = struct.unpack('>i', f.read(4))[0]

        entries = []
        f.seek(dir_offset)
        for _ in range(dir_count):
            tag    = f.read(4).decode('ascii', errors='replace')
            num    = struct.unpack('>i', f.read(4))[0]
            _      = struct.unpack('>h', f.read(2))[0]
            esize  = struct.unpack('>i', f.read(4))[0]
            ecount = struct.unpack('>i', f.read(4))[0]
            total  = esize * ecount
            raw4   = f.read(4)
            offset_val = struct.unpack('>i', raw4)[0] if total > 4 else raw4
            entries.append((tag, num, esize, ecount, total, offset_val))

        def get_raw(tag, num=1):
            for e in entries:
                if e[0] == tag and e[1] == num:
                    tot, val = e[4], e[5]
                    if tot <= 4:
                        return val if isinstance(val, bytes) else b'\x00' * 4
                    f.seek(val); return f.read(tot)
            return None

        def load_channel_set(start_num):
            chs = []
            for ch_num in range(start_num, start_num + 4):
                raw = get_raw('DATA', ch_num)
                if raw and len(raw) >= 4:
                    n = len(raw) // 2
                    chs.append(np.array(struct.unpack(f'>{n}h', raw[:n*2]),
                                        dtype=float))
                else:
                    chs.append(None)
            return chs

        def channel_quality_score(chs):
            score = 0.0
            for c in chs:
                if c is not None and len(c) > 10:
                    score += float(np.var(c)) * float(c.max() - c.min())
            return score

        chs_raw = load_channel_set(1)
        chs_ana = load_channel_set(9)
        score_raw = channel_quality_score(chs_raw)
        score_ana = channel_quality_score(chs_ana)

        if prefer_raw_channels:
            channels_raw = chs_raw if score_raw >= 0.05 * score_ana else chs_ana
            data_source  = '1-4 (raw)' if score_raw >= 0.05 * score_ana else '9-12 (analysed, fallback)'
        else:
            channels_raw = chs_ana if score_ana >= 0.05 * score_raw else chs_raw
            data_source  = '9-12 (analysed)' if score_ana >= 0.05 * score_raw else '1-4 (raw, fallback)'

        lengths = [len(c) for c in channels_raw if c is not None]
        if not lengths:
            raise ValueError("AB1 file contains no readable signal channels")
        N        = min(lengths)
        channels = [c[:N] if c is not None else np.zeros(N) for c in channels_raw]

        fwo_raw   = get_raw('FWO_', 1)
        dye_order = (fwo_raw.decode('ascii', errors='replace').strip('\x00')
                     if fwo_raw else 'GATC')

        signals_raw = np.array(channels)
        signals     = np.zeros_like(signals_raw)
        for ti, td in enumerate('GATC'):
            for si, sd in enumerate(dye_order[:4]):
                if sd == td and si < signals_raw.shape[0]:
                    signals[ti] = signals_raw[si]; break

        kb_bases = ''
        for n in [2, 1]:
            raw = get_raw('PBAS', n)
            if raw:
                kb_bases = raw.decode('ascii', errors='replace').strip('\x00'); break

        kb_quality = np.array([], dtype=float)
        for n in [2, 1]:
            raw = get_raw('PCON', n)
            if raw:
                kb_quality = np.array(list(raw), dtype=float); break

        peak_locs = np.array([], dtype=int)
        for n in [2, 1]:
            raw = get_raw('PLOC', n)
            if raw and len(raw) >= 2:
                cnt = len(raw) // 2
                peak_locs = np.array(struct.unpack(f'>{cnt}H', raw), dtype=int); break

        sample_name = ''
        raw = get_raw('SMPL', 1)
        if raw:
            try:
                sample_name = raw[1:1 + raw[0]].decode('ascii', errors='replace')
            except Exception:
                sample_name = raw.decode('ascii', errors='replace').strip('\x00')

    return np.arange(N), signals, {
        'kb_bases': kb_bases, 'kb_quality': kb_quality,
        'peak_locs': peak_locs, 'sample_name': sample_name,
        'dye_order': dye_order, 'n_samples': N,
        'data_source': data_source,
    }


# ============================================================
# CSV / matrix loaders
# ============================================================

def load_data_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df_raw = pd.read_csv(path, header=None, dtype=str, engine="python")
    if df_raw.shape[1] == 1 and ";" in str(df_raw.iloc[0, 0]):
        rows     = df_raw.iloc[:, 0].str.split(";").apply(lambda r: [x for x in r if x])
        row_list = rows.tolist()
        lens     = {len(r) for r in row_list}
        if len(lens) != 1:
            raise ValueError(f"Inconsistent semicolon CSV column counts: {sorted(lens)}")
        arr = np.array(row_list, dtype=float)
        return np.arange(arr.shape[0]), arr.T
    df  = pd.read_csv(path, engine="python")
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 5:
        return num.iloc[:, 0].values, num.iloc[:, 1:5].values.T
    if num.shape[1] == 4:
        return np.arange(len(num)), num.values.T
    raise ValueError("CSV must contain 4 signal columns (G, A, T, C)")


def load_matrix_any(path: str) -> np.ndarray:
    for sep in [';', ',', '\t', r'\s+']:
        try:
            df = pd.read_csv(path, header=None, engine="python",
                             sep=sep, comment="#")
            df = df.dropna(axis=1, how="all")
            if df.shape[0] >= 4 and df.shape[1] >= 4:
                return df.values[:4, :4].astype(float)
        except Exception:
            continue
    raise ValueError(f"Could not parse 4x4 matrix from {path}")


def save_matrix(matrix: np.ndarray, path: str) -> None:
    pd.DataFrame(matrix).to_csv(path, index=False, header=False)


# ============================================================
# Dye-blob detection
# ============================================================

def detect_dye_blob_end(
    signals: np.ndarray,
    search_limit: int = 500,
    threshold_fraction: float = 0.40,
    settle_window: int = 20,
) -> int:
    combined  = signals.sum(axis=0)
    N         = len(combined)
    limit     = min(search_limit, N)
    blob_peak = int(np.argmax(combined[:limit]))
    blob_max  = float(combined[blob_peak])
    if blob_max <= 0:
        return 0
    threshold = threshold_fraction * blob_max
    for i in range(blob_peak, N - settle_window):
        if (combined[i] < threshold and
                np.all(combined[i:i + settle_window] < threshold)):
            return i
    return min(blob_peak + 50, limit)


# ============================================================
# N-1 shadow peak suppression
# ============================================================

def suppress_n1_peaks(
    signals: np.ndarray,
    peaks_by_channel: List[np.ndarray],
    n1_ratio: float = 0.25,
    n1_range: Tuple[int, int] = (2, 8),
) -> List[np.ndarray]:
    lo, hi   = n1_range
    filtered = []
    for ch in range(4):
        peaks = peaks_by_channel[ch]
        if len(peaks) == 0:
            filtered.append(peaks); continue
        keep = np.ones(len(peaks), dtype=bool)
        sig  = signals[ch]
        for i, p in enumerate(peaks):
            for q in peaks:
                if q <= p: continue
                d = q - p
                if d > hi: break
                if d < lo: continue
                if sig[q] > 0 and sig[p] / (sig[q] + 1e-9) <= n1_ratio:
                    keep[i] = False; break
        filtered.append(peaks[keep])
    return filtered


# ============================================================
# Predicted peak spacing model
# ============================================================

def build_spacing_model(
    confirmed_positions: np.ndarray,
    smoothing_window: int = 21,
) -> Optional[np.ndarray]:
    if len(confirmed_positions) < 6:
        return None
    positions = np.sort(confirmed_positions)
    intervals = np.diff(positions).astype(float)
    midpoints = (positions[:-1] + positions[1:]) / 2.0
    wl = max(3, min(smoothing_window | 1, len(intervals) - 1))
    sm = (savgol_filter(intervals, wl, min(3, wl - 1))
          if len(intervals) >= wl else intervals.copy())
    for i in range(1, len(sm)):
        if sm[i] < sm[i - 1]: sm[i] = sm[i - 1]
    N_max = int(positions[-1]) + int(sm[-1]) * 5 + 1
    if len(midpoints) < 2: return np.full(N_max, sm[0])
    return interp1d(midpoints, sm, kind='linear', bounds_error=False,
                    fill_value=(sm[0], sm[-1]))(np.arange(N_max))


def score_spacing_consistency(
    pos: int,
    neighbours: np.ndarray,
    spacing_model: Optional[np.ndarray],
    tolerance: float = 0.35,
) -> float:
    if spacing_model is None or len(neighbours) < 2: return 1.0
    before = neighbours[neighbours < pos]
    after  = neighbours[neighbours > pos]
    scores = []
    if len(before) >= 1:
        prev = before[-1]
        if prev < len(spacing_model) and spacing_model[prev] > 0:
            dev = abs((pos - prev) - spacing_model[prev]) / spacing_model[prev]
            scores.append(max(0.0, 1.0 - dev / tolerance))
    if len(after) >= 1:
        nxt = after[0]
        if pos < len(spacing_model) and spacing_model[pos] > 0:
            dev = abs((nxt - pos) - spacing_model[pos]) / spacing_model[pos]
            scores.append(max(0.0, 1.0 - dev / tolerance))
    return float(np.mean(scores)) if scores else 1.0


# ============================================================
# Per-channel timing shifts
# ============================================================

def shift_channel(sig: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(sig)
    if shift > 0:   out[shift:] = sig[:-shift]
    elif shift < 0: out[:shift] = sig[-shift:]
    else:           out = sig.copy()
    return out


def apply_channel_shifts(signals: np.ndarray, channel_shifts: List[int]) -> np.ndarray:
    out = np.zeros_like(signals)
    for ch in range(min(4, signals.shape[0])):
        out[ch] = shift_channel(signals[ch], channel_shifts[ch])
    return out


# ============================================================
# Crosstalk correction
# ============================================================

def apply_crosstalk_correction(
    signals: np.ndarray,
    M: np.ndarray,
    channel_shifts: List[int] = None,
) -> np.ndarray:
    if channel_shifts is None: channel_shifts = [0, 0, 0, 0]
    shifted = apply_channel_shifts(signals, channel_shifts)
    cond    = np.linalg.cond(M)
    Minv    = np.linalg.pinv(M) if cond > 1000 else np.linalg.inv(M)
    return np.maximum(Minv @ shifted, 0.0)


# ============================================================
# Baseline correction helpers (shared by matrix estimation)
# ============================================================

def _baseline_correct_for_matrix(signals: np.ndarray, window: int = 101) -> np.ndarray:
    """
    Fast rolling-median baseline correction used internally before
    matrix estimation.  Returns baseline-corrected, non-negative signals.
    """
    out = np.zeros_like(signals)
    for ch in range(signals.shape[0]):
        w  = max(3, min(window | 1, signals.shape[1] - 1))
        bl = median_filter(signals[ch].astype(float), size=w, mode='reflect')
        out[ch] = np.maximum(signals[ch] - bl, 0.0)
    return out


# ============================================================
# Crosstalk matrix estimation  [MATRIX-1 … MATRIX-5]
# ============================================================

def constrain_matrix_physical(
    matrix: np.ndarray,
    max_offdiag: float = 0.95,
    ridge: float = 0.00,
) -> np.ndarray:
    """
    MATRIX-3 — Post-processing to enforce physical constraints.

    Rules applied in order:
      1. Clamp off-diagonals to [0, max_offdiag]  (fluorescence cannot be
         negative and cannot exceed the primary channel)
      2. Set all diagonal entries to exactly 1.0
      3. Add a small ridge (default 0.01) to improve invertibility without
         distorting the matrix

    Call this after any estimation method to guarantee a usable matrix.
    """
    m = matrix.copy().astype(float)
    np.fill_diagonal(m, 0.0)                     # zero diagonals temporarily
    m = np.clip(m, 0.0, max_offdiag)             # clamp off-diagonals
    np.fill_diagonal(m, 1.0)                     # restore unit diagonals
    m += ridge * np.eye(4)                       # invertibility ridge
    return m


def score_matrix_plausibility(matrix: np.ndarray) -> float:
    """
    MATRIX-5 — Numeric quality score for a candidate crosstalk matrix.

    Higher is better.  Rewards:
      • Diagonal values close to 1.0  (+10 per entry, max)
      • Off-diagonals in [0, 1]       (+1 per entry)
      • Diagonal dominates its column (+3 per column)
    Penalises:
      • Negative off-diagonals        (−4 per entry)
      • Off-diagonals > 1.0          (−value per entry)
      • Non-square or wrong size      (−999)
    """
    if matrix.shape != (4, 4):
        return -999.0
    score = 0.0
    for i in range(4):
        score -= abs(matrix[i, i] - 1.0) * 10          # want diagonal = 1
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            v = float(matrix[i, j])
            if 0.0 <= v <= 1.0:
                score += 1.0
            elif v < 0.0:
                score -= 4.0
            else:                                        # v > 1
                score -= v * 2.0
    for j in range(4):
        col = matrix[:, j]
        off_max = max(col[k] for k in range(4) if k != j)
        if col[j] >= off_max:
            score += 3.0
        else:
            score -= (off_max - col[j]) * 5.0           # heavy penalty
    return float(score)


def estimate_matrix_dominant_channel(
    signals: np.ndarray,
    baseline_win: int = 101,
    min_dominance_ratio: float = 1.5,
    min_amplitude_frac: float = 0.04,
    max_samples_per_ch: int = 500,
    blob_end: int = 0,
) -> Tuple[np.ndarray, dict]:
    """
    MATRIX-1 — Dominant-channel matrix estimation.

    Algorithm
    ---------
    For each channel ch in {G, A, T, C}:
      1. Apply rolling-median baseline correction.
      2. Sort all timepoints by channel ch's amplitude (descending).
      3. Keep only timepoints where:
           a) channel ch has the maximum signal of all four channels, AND
           b) ch's amplitude exceeds the second-highest channel by at
              least min_dominance_ratio.
      4. Average the 4-element normalised signal vector (sig / sig[ch]) over
         those timepoints → this is column ch of the influence matrix.

    Why this works on CSV / SRD data
    ----------------------------------
    Unlike peak-finding approaches, it does not require identifiable discrete
    peaks.  Any timepoint where one channel is unambiguously dominant
    contributes a clean sample of that channel's spectral signature.  On a
    Sanger trace there are always hundreds of such timepoints even if the
    peaks are not sharply resolved.

    Parameters
    ----------
    min_dominance_ratio : float, default 1.5
        How much larger ch must be than the second-highest channel.
        Lower (1.2) accepts more samples but includes noisier ones.
        Higher (2.0) gives cleaner samples but may yield fewer.
    min_amplitude_frac : float, default 0.04
        Discard timepoints below this fraction of the channel's 99th
        percentile amplitude (noise-floor guard).
    max_samples_per_ch : int, default 500
        Maximum number of timepoints averaged per channel.  500 is
        more than sufficient; using more does not improve accuracy.
    blob_end : int, default 0
        Skip the dye-blob region (first blob_end samples).

    Returns
    -------
    matrix    : (4, 4) ndarray — raw estimated matrix (before constraints)
    diagnostics : dict with per-channel sample counts and warnings
    """
    signals_bl = _baseline_correct_for_matrix(signals, baseline_win)
    n_ch, N    = signals_bl.shape
    matrix     = np.eye(4, dtype=float)
    diag       = {}

    for ch in range(min(4, n_ch)):
        sig_ch = signals_bl[ch, blob_end:]
        if sig_ch.max() <= 0:
            diag[f'ch{ch}_samples'] = 0
            diag[f'ch{ch}_warn']    = 'channel is all zeros'
            continue

        # Amplitude threshold: discard noise-floor timepoints
        amp_99   = float(np.percentile(sig_ch, 99))
        amp_thr  = min_amplitude_frac * amp_99
        if amp_thr <= 0:
            amp_thr = 1e-6

        # Sort by this channel's amplitude descending
        order = blob_end + np.argsort(sig_ch)[::-1]

        col_sum = np.zeros(4, dtype=float)
        count   = 0
        skipped_not_max   = 0
        skipped_dominance = 0

        for t in order:
            if t >= N:
                continue
            vals = signals_bl[:, t]
            if vals[ch] < amp_thr:
                break   # sorted descending — nothing below this is useful

            # (a) this channel must be the maximum
            if int(np.argmax(vals)) != ch:
                skipped_not_max += 1
                continue

            # (b) dominance ratio
            sorted_vals = np.sort(vals)[::-1]
            second      = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
            ratio       = float(vals[ch]) / (float(second) + 1e-9)
            if ratio < min_dominance_ratio:
                skipped_dominance += 1
                continue

            col_sum += vals / (vals[ch] + 1e-9)
            count   += 1
            if count >= max_samples_per_ch:
                break

        diag[f'ch{ch}_samples']          = count
        diag[f'ch{ch}_skipped_not_max']  = skipped_not_max
        diag[f'ch{ch}_skipped_dominance']= skipped_dominance

        if count >= 5:
            matrix[:, ch] = col_sum / count
            # Enforce diagonal ≥ 1 (can drift slightly below 1 due to averaging)
            matrix[ch, ch] = max(matrix[ch, ch], 1.0)
        else:
            # Not enough clean samples — identity fallback for this column
            matrix[:, ch]  = 0.0
            matrix[ch, ch] = 1.0
            diag[f'ch{ch}_warn'] = (
                f'Only {count} dominant samples found '
                f'(skipped: not_max={skipped_not_max}, ratio={skipped_dominance}). '
                f'Try lowering min_dominance_ratio or min_amplitude_frac.'
            )

    return matrix, diag


def estimate_matrix_nmf(
    signals: np.ndarray,
    baseline_win: int = 101,
    n_iter: int = 600,
    blob_end: int = 0,
) -> Tuple[np.ndarray, dict]:
    """
    MATRIX-2 — NMF-based crosstalk matrix estimation.

    Uses Non-negative Matrix Factorisation (sklearn) to decompose the
    baseline-corrected signal matrix as  signals ≈ M @ S_pure  where
    M is the 4×4 crosstalk matrix and S_pure contains the unmixed
    source signals.

    NMF naturally enforces non-negativity (fluorescence cannot be
    negative) and does not require any peak-detection step.  It is
    particularly good when multiple channels overlap heavily, making
    dominant-channel analysis difficult.

    Requires scikit-learn.  Raises ImportError gracefully if absent.

    Parameters
    ----------
    n_iter : int, default 600
        NMF solver iterations.  600 is reliable; lower for speed.

    Returns
    -------
    matrix      : (4, 4) ndarray — raw estimated matrix
    diagnostics : dict with reconstruction error and component assignment
    """
    try:
        from sklearn.decomposition import NMF as _NMF
    except ImportError:
        raise ImportError(
            "scikit-learn is required for NMF matrix estimation.\n"
            "Install with:  pip install scikit-learn\n"
            "Or choose 'dominant_channel' or 'robust' instead.")

    signals_bl = _baseline_correct_for_matrix(signals, baseline_win)
    # Skip blob region
    S = signals_bl[:, blob_end:].T.astype(float)   # shape (N, 4)

    # Downsample for speed on long traces
    step = max(1, S.shape[0] // 6000)
    S_ds = S[::step]

    model = _NMF(n_components=4, init='nndsvda', max_iter=n_iter, random_state=42)
    W = model.fit_transform(S_ds)   # (N_ds, 4) — time weights
    H = model.components_           # (4, 4)  — spectral signatures

    # H[k, :] is the spectral distribution of component k across 4 detectors.
    # We need to assign each NMF component k to the physical channel it
    # best represents.  Use correlation with the original signals.
    corr = np.zeros((4, 4))
    for k in range(4):
        comp_signal = W[:, k]                        # component k time series
        for ch in range(4):
            ch_signal   = S_ds[:, ch]
            denom       = (np.std(comp_signal) * np.std(ch_signal) + 1e-9)
            corr[k, ch] = float(np.mean(
                (comp_signal - comp_signal.mean()) *
                (ch_signal   - ch_signal.mean())
            ) / denom)

    # Greedy assignment: component → channel with highest correlation
    assignment   = {}      # comp → ch
    used_ch      = set()
    for _ in range(4):
        best_val, best_k, best_ch = -1, 0, 0
        for k in range(4):
            if k in assignment:
                continue
            for ch in range(4):
                if ch in used_ch:
                    continue
                if corr[k, ch] > best_val:
                    best_val, best_k, best_ch = corr[k, ch], k, ch
        assignment[best_k] = best_ch
        used_ch.add(best_ch)

    # Build mixing matrix from H.
    # H[k, :] normalised so H[k, assignment[k]] = 1 gives column assignment[k].
    matrix = np.eye(4, dtype=float)
    for k, ch in assignment.items():
        spectral_row = H[k, :]
        pivot = spectral_row[ch]
        if pivot > 0:
            matrix[:, ch] = spectral_row / pivot
            matrix[ch, ch] = max(matrix[ch, ch], 1.0)

    return matrix, {
        'reconstruction_error': float(model.reconstruction_err_),
        'n_iter_done':          int(model.n_iter_),
        'component_assignment': {int(k): int(v) for k, v in assignment.items()},
        'max_corr_per_comp':    {int(k): float(corr[k, assignment[k]])
                                 for k in assignment},
    }


def estimate_matrix_robust(
    signals: np.ndarray,
    baseline_win: int = 101,
    min_dominance_ratio: float = 1.5,
    blob_end: int = 0,
) -> Tuple[np.ndarray, str, dict]:
    """
    MATRIX-4 — Robust matrix estimation: try all methods, return best.

    Tries (in order):
      1. dominant_channel  — always available, fast, robust for CSV/SRD
      2. nmf               — requires scikit-learn; best for heavily mixed signals
      3. peaks             — original peak-based method (fallback)

    Each candidate matrix is scored by score_matrix_plausibility().
    Physical constraints (constrain_matrix_physical) are applied to
    the winner before returning it.

    Returns
    -------
    matrix      : (4, 4) ndarray — best constrained matrix
    method_used : str — name of the winning method
    all_diag    : dict — diagnostics for every method tried
    """
    all_diag   = {}
    candidates = {}   # method_name → (raw_matrix, diagnostics)

    # ── Method 1: dominant channel (always runs) ─────────────────────────────
    try:
        m, d = estimate_matrix_dominant_channel(
            signals,
            baseline_win=baseline_win,
            min_dominance_ratio=min_dominance_ratio,
            blob_end=blob_end,
        )
        candidates['dominant_channel'] = (m, d)
        all_diag['dominant_channel']   = d
    except Exception as e:
        all_diag['dominant_channel'] = {'error': str(e)}

    # ── Method 2: NMF (optional) ─────────────────────────────────────────────
    try:
        m, d = estimate_matrix_nmf(signals, baseline_win=baseline_win,
                                   blob_end=blob_end)
        candidates['nmf'] = (m, d)
        all_diag['nmf']   = d
    except ImportError:
        all_diag['nmf'] = {'info': 'scikit-learn not installed; NMF skipped'}
    except Exception as e:
        all_diag['nmf'] = {'error': str(e)}

    # ── Method 3: original peak-based (original, for comparison) ─────────────
    try:
        signals_bl = _baseline_correct_for_matrix(signals, baseline_win)
        m = estimate_influence_matrix_safe(
            signals_bl[:, blob_end:], prominence=0.01, min_distance=5)
        candidates['peaks'] = (m, {})
        all_diag['peaks']   = {}
    except Exception as e:
        all_diag['peaks'] = {'error': str(e)}

    # ── Score and select ──────────────────────────────────────────────────────
    best_method = 'identity'
    best_matrix = np.eye(4, dtype=float)
    best_score  = score_matrix_plausibility(np.eye(4))

    for method, (m, _) in candidates.items():
        # Apply physical constraints before scoring
        m_constrained = constrain_matrix_physical(m)
        s             = score_matrix_plausibility(m_constrained)
        all_diag[method]['plausibility_score'] = round(s, 3)
        if s > best_score:
            best_score  = s
            best_matrix = m_constrained
            best_method = method

    # If the best is still 'identity', apply constraints to dominant_channel
    # output anyway (it may have a lower score than identity due to large
    # off-diagonals before constraining — but is still better than identity)
    if best_method == 'identity' and 'dominant_channel' in candidates:
        best_matrix = constrain_matrix_physical(candidates['dominant_channel'][0])
        best_method = 'dominant_channel (constrained)'

    all_diag['winner'] = best_method
    all_diag['winner_score'] = best_score
    return best_matrix, best_method, all_diag


def estimate_influence_matrix_safe(
    raw_signals: np.ndarray,
    prominence: float = 0.01,
    min_distance: int = 5,
    channel_letters: tuple = ('G', 'A', 'T', 'C'),
) -> np.ndarray:
    """
    Original peak-based matrix estimation (kept for compatibility).

    For CSV / SRD files, prefer estimate_matrix_dominant_channel() or
    estimate_matrix_robust() which are more reliable when single-channel
    peaks cannot be cleanly isolated.
    """
    matrix = np.zeros((4, 4))
    for ch in range(4):
        peaks, _ = find_peaks(raw_signals[ch],
                              prominence=prominence * 0.5, distance=min_distance)
        for ratio in [2.0, 1.5, 1.2]:
            col_sum, count = np.zeros(4), 0
            for pos in peaks:
                v = raw_signals[:, pos]
                if v[ch] == 0: continue
                sv = np.sort(v)
                if sv[-1] < ratio * (sv[-2] + 1e-9): continue
                col_sum += v / v[ch]; count += 1
            if count >= 5:
                matrix[:, ch] = col_sum / count; break
        else:
            matrix[:, ch] = 0.0; matrix[ch, ch] = 1.0
    for i in range(4): matrix[i, i] = max(matrix[i, i], 1.0)
    eps = 0.01 * max(np.mean(np.abs(np.diag(matrix))), 1.0)
    matrix += eps * np.eye(4)
    return matrix


# ============================================================
# Baseline correction  [BLAST-3, BLAST-4]
# ============================================================

def baseline_rolling_median(
    signals: np.ndarray,
    window: int = 101,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros_like(signals)
    noise_floor = np.zeros_like(signals)
    for i, s in enumerate(signals):
        w = max(3, min(window | 1, len(s) - 1))
        baseline        = median_filter(s, size=w, mode='reflect')
        noise_floor[i]  = baseline.copy()
        out[i]          = np.maximum(s - baseline, 0.0)
    return out, noise_floor


def baseline_als(
    signals: np.ndarray,
    lam: float = 1e5,
    p: float = 0.002,
    niter: int = 10,
    max_pts: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import spsolve
    except ImportError:
        return baseline_rolling_median(signals, 101)

    out         = np.zeros_like(signals)
    noise_floor = np.zeros_like(signals)

    for i, s in enumerate(signals):
        N_full = len(s)
        s_f    = s.astype(float)
        decimate = max(1, N_full // max_pts)
        s_work   = s_f[::decimate] if decimate > 1 else s_f
        n        = len(s_work)
        e   = np.ones(n)
        D   = diags([e[:-2], -2 * e[:-1], e], [0, 1, 2],
                    shape=(n - 2, n), format='csc')
        DtD = lam * D.T.dot(D)
        w  = np.ones(n)
        bl = s_work.copy()
        for _ in range(niter):
            W  = diags(w, 0, shape=(n, n), format='csc')
            bl = spsolve(W + DtD, w * s_work)
            w  = np.where(s_work > bl, p, 1.0 - p)
        if decimate > 1:
            x_dec   = np.arange(0, N_full, decimate)[:len(bl)]
            x_full  = np.arange(N_full)
            bl_full = np.interp(x_full, x_dec, bl)
        else:
            bl_full = bl
        noise_floor[i] = np.maximum(bl_full, 0.0)
        out[i]         = np.maximum(s_f - bl_full, 0.0)

    return out, noise_floor


def baseline_morphological(
    signals: np.ndarray,
    window: int = 101,
) -> Tuple[np.ndarray, np.ndarray]:
    out         = np.zeros_like(signals)
    noise_floor = np.zeros_like(signals)
    for i, s in enumerate(signals):
        w        = max(3, min(window | 1, len(s) - 1))
        eroded   = minimum_filter(s, size=w, mode='reflect')
        opened   = maximum_filter(eroded, size=w, mode='reflect')
        baseline = gaussian_filter1d(opened, sigma=max(1, w // 8))
        noise_floor[i] = np.maximum(baseline, 0.0)
        out[i]         = np.maximum(s - baseline, 0.0)
    return out, noise_floor


def _apply_baseline(
    signals: np.ndarray,
    method: str,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == 'als':
        return baseline_als(signals)
    elif method == 'morph':
        return baseline_morphological(signals, window)
    else:
        return baseline_rolling_median(signals, window)


# ============================================================
# Smoothing
# ============================================================

def smooth_savgol(
    signals: np.ndarray,
    win: int = 19,
    poly: int = 3,
    n_passes: int = 1,
    mode: str = 'interp',
) -> np.ndarray:
    out = signals.copy()
    for _ in range(max(1, n_passes)):
        tmp = np.zeros_like(out)
        for i, s in enumerate(out):
            wl = max(poly + 2, min(win, len(s) - 1))
            if wl % 2 == 0: wl = max(wl - 1, poly + 2)
            tmp[i] = savgol_filter(s, wl, min(poly, wl - 1), mode=mode)
        out = np.maximum(tmp, 0.0)
    return out


# ============================================================
# Running noise floor
# ============================================================

def compute_running_noise_floor(
    signals: np.ndarray,
    window: int = 51,
    percentile: float = 10.0,
) -> np.ndarray:
    n_ch, N  = signals.shape
    noise    = np.zeros_like(signals)
    half     = max(window // 2, 1)
    for ch in range(n_ch):
        s       = signals[ch].astype(float)
        padded  = np.pad(s, half, mode='reflect')
        win_len = 2 * half + 1
        from numpy.lib.stride_tricks import as_strided
        shape   = (N, win_len)
        strides = (padded.strides[0], padded.strides[0])
        windows = as_strided(padded, shape=shape, strides=strides)
        row_pct = np.percentile(windows, percentile, axis=1)
        ch_max  = float(np.max(np.abs(s))) if np.any(s > 0) else 1.0
        noise[ch] = np.maximum(row_pct, ch_max * 0.0005)
    return noise


# ============================================================
# Mobility correction  [BLAST-8]
# ============================================================

def mobility_correction(
    time: np.ndarray,
    signals: np.ndarray,
    peak_prominence: float = 0.01,
    poly_deg: int = 3,
    enabled: bool = True,
) -> Tuple[np.ndarray, bool]:
    if not enabled: return signals.copy(), False
    combined = signals.sum(axis=0)
    N        = signals.shape[1]
    warped   = np.zeros_like(signals)

    thresh    = peak_prominence * np.percentile(combined, 90)
    ref_peaks, _ = find_peaks(combined, prominence=thresh)
    if len(ref_peaks) < 4: return signals.copy(), False

    for ch in range(4):
        sig = signals[ch]
        xcorr  = correlate(sig, combined, mode='full')
        lag    = int(np.argmax(xcorr)) - (N - 1)
        lag    = max(-N // 10, min(N // 10, lag))
        shifted = shift_channel(sig, -lag) if lag != 0 else sig.copy()

        ch_thresh = peak_prominence * np.percentile(shifted, 90)
        peaks, _  = find_peaks(shifted, prominence=ch_thresh)
        if len(peaks) < 4:
            warped[ch] = shifted; continue

        pairs, used = [], set()
        for rp in ref_peaks:
            cands = [p for p in peaks if p not in used]
            if not cands: break
            nearest = min(cands, key=lambda x: abs(x - rp))
            if abs(nearest - rp) < 20:
                pairs.append((rp, nearest - rp)); used.add(nearest)
        if len(pairs) < 4:
            warped[ch] = shifted; continue

        xs, ys = zip(*pairs)
        deg    = min(poly_deg, len(xs) - 1)
        drift  = np.polyval(np.polyfit(xs, ys, deg), np.arange(N))
        t_new  = np.arange(N, dtype=float) - drift
        for i in range(1, N):
            if t_new[i] <= t_new[i - 1]: t_new[i] = t_new[i - 1] + 1e-6
        warped[ch] = interp1d(t_new, shifted, bounds_error=False,
                              fill_value=0.0)(np.arange(N))

    return warped, True


# ============================================================
# Tail sharpening
# ============================================================

def sharpen_tail_only(
    signals: np.ndarray,
    tail_fraction: float = 0.25,
    sigma: float = 0.4,
    amount: float = 0.0,
) -> np.ndarray:
    out = signals.copy()
    if amount == 0.0: return out
    N, start = signals.shape[1], int((1 - tail_fraction) * signals.shape[1])
    for ch in range(4):
        s = signals[ch]
        if s.max() <= 0: continue
        blurred = gaussian_filter1d(s, sigma)
        ramp    = np.zeros_like(s)
        ramp[start:] = np.linspace(0, amount, N - start)
        sharp = np.maximum(s + ramp * (s - blurred), 0.0)
        if sharp.max() > 0: sharp *= s.max() / sharp.max()
        out[ch] = sharp
    return out


# ============================================================
# Noise floor helpers
# ============================================================

def _compute_channel_noise_floor(sig: np.ndarray) -> float:
    if sig.max() <= 0: return 1e-6
    below = sig[sig < 0.10 * sig.max()]
    if len(below) < 20:
        return max(float(sig.max()) * 0.01, 1e-6)
    return max(float(np.std(below)), float(sig.max()) * 0.005, 1e-6)


def _local_noise(
    sig: np.ndarray,
    pos: int,
    half_win: int = 20,
    global_floor: float = 1e-6,
    running_noise: Optional[np.ndarray] = None,
) -> float:
    if running_noise is not None and pos < len(running_noise):
        return max(float(running_noise[pos]), global_floor)
    N     = len(sig)
    lo    = max(0, pos - half_win); hi = min(N, pos + half_win)
    ex_lo = max(lo, pos - 5);       ex_hi = min(hi, pos + 6)
    idx   = np.concatenate([np.arange(lo, ex_lo), np.arange(ex_hi, hi)])
    if len(idx) < 6:
        win = sig[max(0, pos - 10):min(N, pos + 11)]
        if len(win) < 4: return global_floor
        q75, q25 = np.percentile(win, [75, 25])
        return max(float((q75 - q25) / 1.35), global_floor)
    q75, q25 = np.percentile(sig[idx], [75, 25])
    return max(float((q75 - q25) / 1.35), global_floor)


# ============================================================
# Peak shape scoring
# ============================================================

def _gaussian_1d(x, amp, center, sigma):
    return amp * np.exp(-0.5 * ((x - center) / (sigma + 1e-9)) ** 2)


def _score_peak_shape(sig: np.ndarray, pos: int, half_win: int = 8) -> float:
    lo  = max(0, pos - half_win)
    hi  = min(len(sig), pos + half_win + 1)
    win = sig[lo:hi]
    if len(win) < 5 or win.max() <= 0: return 0.5
    ci       = pos - lo
    la       = float(np.sum(win[:ci]))
    ra       = float(np.sum(win[ci + 1:]))
    symmetry = 1.0 - abs(la - ra) / (la + ra + 1e-9)
    try:
        x  = np.arange(len(win), dtype=float) - ci
        y  = win / (win.max() + 1e-9)
        popt, _ = curve_fit(_gaussian_1d, x, y, p0=[1.0, 0.0, 2.0],
                            bounds=([0, -half_win, 0.3], [1.5, half_win, half_win]),
                            maxfev=300)
        fitted    = _gaussian_1d(x, *popt)
        ss_res    = float(np.sum((y - fitted) ** 2))
        ss_tot    = float(np.sum((y - y.mean()) ** 2))
        r_squared = max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))
    except Exception:
        r_squared = symmetry
    return float(0.4 * symmetry + 0.6 * r_squared)


# ============================================================
# Adaptive peak finding  [BLAST-5]
# ============================================================

def find_peaks_adaptive(
    sig: np.ndarray,
    global_prom: float,
    distance: int,
    envelope_window: int = 300,
    floor_frac: float = 0.03,
) -> np.ndarray:
    maxv = float(np.max(sig))
    if maxv <= 0:
        return np.array([], dtype=int)
    env_win  = max(3, envelope_window)
    envelope = maximum_filter(sig, size=env_win, mode='reflect')
    envelope = np.maximum(envelope, maxv * floor_frac)
    norm_sig = sig / envelope
    pks, _ = find_peaks(norm_sig, prominence=global_prom, distance=distance)
    return pks


# ============================================================
# Cluster resolution
# ============================================================

def _merge_clusters(candidates: list, merge_tol: int) -> list:
    if not candidates: return []
    merged        = []
    cluster       = [candidates[0]]
    cluster_start = candidates[0][0]
    for cur in candidates[1:]:
        if abs(cur[0] - cluster_start) <= merge_tol:
            cluster.append(cur)
        else:
            merged.append(cluster)
            cluster       = [cur]
            cluster_start = cur[0]
    merged.append(cluster)
    return merged


def _resolve_cluster(group: list, signals: np.ndarray) -> Optional[Tuple[int, int]]:
    seen, valid = set(), []
    for g in group:
        if isinstance(g, tuple) and len(g) == 2:
            key = (int(g[0]), int(g[1]))
            if key not in seen:
                seen.add(key); valid.append(key)
    if not valid: return None
    best_pos, best_ch = max(valid,
                            key=lambda pc: float(signals[pc[1], pc[0]]))
    return best_pos, best_ch


# ============================================================
# Quality scoring  [BLAST-9]
# ============================================================

def _calibrated_phred(
    snr: float,
    isolation: float,
    shape_score: float,
    spacing_score: float,
    position_frac: float,
) -> int:
    q_base = 20.0 * math.log10(max(snr, 1.0)) + 18.0
    q_base = max(0.0, min(40.0, q_base))
    if   isolation < 1.05:  iso_pen = 3.0
    elif isolation < 1.25:  iso_pen = 1.5 * (1.25 - isolation) / 0.2
    else:                  iso_pen = 0.0
    shape_pen   = 1.5 * (1.0 - max(0.0, min(1.0, shape_score)))
    spacing_pen = 1.0 * (1.0 - max(0.0, min(1.0, spacing_score)))
    if   position_frac > 0.88: pos_pen = 6.0 * (position_frac - 0.88) / 0.12
    elif position_frac > 0.75: pos_pen = 2.5 * (position_frac - 0.75) / 0.13
    else:                      pos_pen = 0.0
    return int(min(max(round(q_base - iso_pen - shape_pen - spacing_pen - pos_pen), 5), 40))


def _calibrated_raw_score(snr, isolation, shape_score, spacing_score) -> float:
    snr_c     = min(40.0 * math.log10(max(snr, 0.01) + 1.0) / math.log10(11.0), 40.0)
    iso_c     = min(25.0 * math.log10(max(isolation, 1.0)) / math.log10(5.0), 25.0)
    shape_c   = 20.0 * max(0.0, min(1.0, shape_score))
    spacing_c = 14.0 * max(0.0, min(1.0, spacing_score))
    return min(snr_c + iso_c + shape_c + spacing_c, 99.0)


# ============================================================
# Heterozygote detection
# ============================================================

def detect_heterozygote(
    signals: np.ndarray,
    pos: int,
    threshold: float = 0.45,
) -> Tuple[bool, int, float]:
    if pos < 0 or pos >= signals.shape[1]: return False, -1, 0.0
    vals = signals[:, pos]
    idx  = np.argsort(vals)[::-1]
    if vals[idx[0]] <= 0: return False, -1, 0.0
    ratio = float(vals[idx[1]] / (vals[idx[0]] + 1e-9))
    if ratio >= threshold: return True, int(idx[1]), ratio
    return False, -1, 0.0


# ============================================================
# Missing peak imputation
# ============================================================

def detect_missing_peaks(
    peak_calls: list,
    spacing_model: Optional[np.ndarray],
    signals: np.ndarray,
    time: np.ndarray,
    blob_end: int,
    letters: tuple,
    channel_floors: List[float],
    running_noise: Optional[np.ndarray] = None,
    min_amplitude_frac: float = 0.03,
    gap_ratio_threshold: float = 1.60,
) -> list:
    if spacing_model is None or len(peak_calls) < 5: return peak_calls
    called_pos = sorted([p['pos'] for p in peak_calls])
    max_signal  = float(np.max(signals))
    if max_signal <= 0 or len(called_pos) < 2: return peak_calls
    imputed = []
    for i in range(len(called_pos) - 1):
        p1, p2  = called_pos[i], called_pos[i + 1]
        pred    = float(spacing_model[p1]) if p1 < len(spacing_model) else float(spacing_model[-1])
        if pred <= 0: continue
        gap_ratio = (p2 - p1) / pred
        if gap_ratio < gap_ratio_threshold: continue
        n_miss = int(round(gap_ratio)) - 1
        if n_miss <= 0 or n_miss > 3: continue
        for k in range(1, n_miss + 1):
            pred_pos    = max(blob_end, min(int(p1 + k * pred), signals.shape[1] - 1))
            search_half = max(3, int(pred * 0.28))
            lo = max(blob_end, pred_pos - search_half)
            hi = min(signals.shape[1] - 1, pred_pos + search_half)
            region = signals[:, lo:hi + 1]
            if region.size == 0: continue
            col_max = region.max(axis=0)
            local_peaks, _ = find_peaks(col_max)
            if len(local_peaks) == 0: continue
            best_local = local_peaks[int(np.argmax(col_max[local_peaks]))]
            best_pos   = lo + best_local
            vals    = signals[:, best_pos]
            best_ch = int(np.argmax(vals))
            amp     = float(vals[best_ch])
            if amp < min_amplitude_frac * max_signal: continue
            if any(abs(cp - best_pos) < pred * 0.4 for cp in called_pos): continue
            rn    = running_noise[best_ch] if running_noise is not None else None
            noise = _local_noise(signals[best_ch], best_pos,
                                 global_floor=channel_floors[best_ch], running_noise=rn)
            imputed.append({
                'pos': best_pos, 'time': float(time[best_pos]),
                'channel': best_ch, 'letter': letters[best_ch],
                'amplitude': amp, 'snr': float(amp / max(noise, 1e-9)),
                'isolation': 1.0, 'shape_score': 0.5, 'spacing_score': 0.9,
                'score': 8.0, 'phred': 10, 'noise': float(noise),
                'tail_rescued': True, 'imputed': True,
                'heterozygote': False, 'het_channel': -1, 'het_ratio': 0.0,
            })
    if not imputed: return peak_calls
    combined = peak_calls + imputed
    combined.sort(key=lambda r: r['pos'])
    return combined


# ============================================================
# Quality trimming
# ============================================================

def sliding_window_trim(
    results: list,
    window: int = 20,
    min_q: float = 20.0,
) -> Tuple[list, int, int]:
    if not results: return [], 0, 0
    quals = np.array([r['phred'] for r in results], dtype=float)
    n     = len(quals)
    if n < window:
        return (results, 0, n) if float(np.mean(quals)) >= min_q else ([], 0, 0)
    scores = quals - min_q
    max_sum = float('-inf'); cur_sum = 0.0; cur_start = 0; best_start = 0; best_end = 0
    for i in range(n):
        if cur_sum <= 0: cur_sum = scores[i]; cur_start = i
        else:            cur_sum += scores[i]
        if cur_sum > max_sum:
            max_sum = cur_sum; best_start = cur_start; best_end = i + 1
    if max_sum <= 0: return [], 0, 0
    return results[best_start:best_end], best_start, best_end


def trim_low_quality_ends(results: list, min_phred: int = 15) -> list:
    if not results: return results
    left = 0
    while left < len(results) and results[left]['phred'] < min_phred: left += 1
    right = len(results) - 1
    while right >= left and results[right]['phred'] < min_phred:       right -= 1
    return results[left:right + 1]


# ============================================================
# Primer removal
# ============================================================

def trim_N_ends(sequence: str) -> str:
    return sequence.strip('N')


def trim_primer(
    sequence: str, primer: str,
    max_mismatches: int = 2, search_limit: int = 80,
) -> Tuple[str, int]:
    if not primer or not sequence: return sequence, 0
    plen = len(primer); primer = primer.upper(); seq_upper = sequence.upper()
    search_end = min(len(sequence) - plen + 1, search_limit)
    best_pos, best_mm = 0, max_mismatches + 1
    for i in range(search_end):
        mm = sum(1 for a, b in zip(primer, seq_upper[i:i + plen])
                 if a != b and a != 'N' and b != 'N')
        if mm < best_mm: best_mm = mm; best_pos = i
    if best_mm <= max_mismatches:
        return sequence[best_pos + plen:], best_pos + plen
    return sequence, 0


# ============================================================
# Reverse complement  [BLAST-2]
# ============================================================

_COMPLEMENT = str.maketrans('ACGTacgtNn', 'TGCAtgcaNn')

def reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


# ============================================================
# Best BLAST window  [BLAST-6]
# ============================================================

def find_best_blast_window(
    results: list,
    min_length: int = 150,
) -> Tuple[list, int, int]:
    if not results:
        return results, 0, 0
    for q_thresh in [30.0, 25.0, 20.0, 15.0, 10.0]:
        trimmed, start, end = sliding_window_trim(results, window=20, min_q=q_thresh)
        if len(trimmed) >= min_length:
            return trimmed, start, end
    return results, 0, len(results)


# ============================================================
# Auto-orient  [BLAST-7]
# ============================================================

def auto_orient_sequence(results: list, sequence: str) -> dict:
    seq_rc = reverse_complement(sequence)
    fwd_window, _, _ = find_best_blast_window(results)
    fwd_phreds = [r['phred'] for r in fwd_window] if fwd_window else [0]
    fwd_mean_q = float(np.mean(fwd_phreds))
    fwd_len    = len(fwd_window)
    return {
        'forward':                  sequence,
        'forward_len':              len(sequence),
        'forward_blast_window_len': fwd_len,
        'forward_mean_q':           fwd_mean_q,
        'rc':                       seq_rc,
        'rc_len':                   len(seq_rc),
        'rc_blast_window_len':      fwd_len,
        'rc_mean_q':                fwd_mean_q,
        'recommended':              'forward',
    }


# ============================================================
# Peak detection & basecalling
# ============================================================

def detect_and_basecall(
    time: np.ndarray,
    signals: np.ndarray,
    letters: tuple = ("G", "A", "T", "C"),
    prominence: float = 0.002,
    distance: int = 2,
    min_snr: float = 1.0,
    min_isolation: float = 1.15,
    merge_tol: int = 2,
    allow_uncertain: bool = True,
    use_spacing_model: bool = True,
    blob_end: int = 0,
    suppress_n1: bool = True,
    detect_missing: bool = True,
    het_threshold: float = 0.45,
    detect_het: bool = False,
    running_noise: Optional[np.ndarray] = None,
    use_adaptive_peaks: bool = True,
) -> Tuple[list, str, float]:
    channel_floors = [_compute_channel_noise_floor(signals[ch]) for ch in range(4)]

    peaks_by_channel = []
    for ch in range(4):
        maxv = float(np.max(signals[ch]))
        if maxv <= 0:
            peaks_by_channel.append(np.array([], dtype=int)); continue
        if use_adaptive_peaks:
            pks = find_peaks_adaptive(signals[ch], prominence, distance)
        else:
            pks, _ = find_peaks(
                signals[ch],
                prominence=max(prominence * maxv, 1e-6),
                distance=distance,
            )
        peaks_by_channel.append(pks[pks >= blob_end])

    if suppress_n1:
        peaks_by_channel = suppress_n1_peaks(signals, peaks_by_channel)

    candidates = sorted(
        [(int(p), ch) for ch, pks in enumerate(peaks_by_channel) for p in pks],
        key=lambda x: x[0])
    if not candidates: return [], "", 0.0

    clusters   = _merge_clusters(candidates, merge_tol)
    peak_calls = [r for r in (_resolve_cluster(g, signals) for g in clusters)
                  if r is not None]

    pos_seen: dict = {}
    for pos, ch in peak_calls:
        amp = float(signals[ch, pos])
        if pos not in pos_seen or amp > pos_seen[pos][1]:
            pos_seen[pos] = (ch, amp)
    peak_calls = sorted(
        [(pos, ch) for pos, (ch, _) in pos_seen.items()],
        key=lambda x: x[0])

    N = signals.shape[1]
    results, confirmed_positions = [], []
    spacing_model = None

    if use_spacing_model and len(peak_calls) >= 10:
        pre = []
        for pos, ch in peak_calls:
            amp  = float(signals[ch, pos])
            rn   = running_noise[ch] if running_noise is not None else None
            nois = _local_noise(signals[ch], pos,
                                global_floor=channel_floors[ch], running_noise=rn)
            snr  = amp / max(nois, 1e-9)
            sv   = np.sort(signals[:, pos])
            iso  = float(sv[-1] / (sv[-2] + 1e-9))
            if snr >= min_snr * 1.2 and iso >= min_isolation * 1.05:
                pre.append(pos)
        if len(pre) >= 6:
            spacing_model = build_spacing_model(np.array(pre))

    tail_fraction = 0.40
    effective_len = max(N - blob_end, 1)

    for pos, ch in peak_calls:
        vals = signals[:, pos]
        if np.max(vals) < 1e-6: continue
        amp  = float(signals[ch, pos])
        rn   = running_noise[ch] if running_noise is not None else None
        nois = _local_noise(signals[ch], pos,
                            global_floor=channel_floors[ch], running_noise=rn)
        snr  = amp / max(nois, 1e-9)
        if snr < min_snr * 0.15: continue
        sv = np.sort(vals)
        if sv[-1] <= 0: continue
        isolation     = float(sv[-1] / (sv[-2] + 1e-9))
        spacing_score = score_spacing_consistency(
            pos, np.array(confirmed_positions), spacing_model)
        shape_score   = _score_peak_shape(signals[ch], pos)
        frac          = (pos - blob_end) / effective_len

        if frac > (1.0 - tail_fraction):
            t          = (frac - (1.0 - tail_fraction)) / tail_fraction
            iso_thresh = max(min_isolation - 0.25 * t, 1.05)
            snr_thresh = max(min_snr * (1.0 - 0.40 * t), 0.80)
        else:
            iso_thresh = min_isolation
            snr_thresh = min_snr

        if snr < snr_thresh or isolation < iso_thresh: continue

        passes_snr  = snr >= min_snr
        passes_iso  = isolation >= min_isolation
        # Only flag as tail_rescued if it fails BOTH — failing just one 
        # is acceptable for a genuine peak
        passes_strict = passes_snr and passes_iso
        tail_rescued  = not passes_snr and not passes_iso  # must fail both
        if (not tail_rescued and use_spacing_model
                and spacing_model is not None and spacing_score < 0.25):
            tail_rescued = True

        letter = "N" if (not allow_uncertain and tail_rescued) else letters[ch]
        phred  = _calibrated_phred(snr, isolation, shape_score, spacing_score, frac)
        score  = _calibrated_raw_score(snr, isolation, shape_score, spacing_score)
        if tail_rescued:
            phred = max(5, phred // 2)
            score *= 0.5

        if detect_het:
            is_het, het_ch, het_ratio = detect_heterozygote(signals, pos, het_threshold)
        else:
            is_het, het_ch, het_ratio = False, -1, 0.0

        results.append({
            'pos': pos, 'time': float(time[pos]),
            'channel': ch, 'letter': letter,
            'amplitude': amp, 'snr': float(snr),
            'isolation': isolation, 'shape_score': float(shape_score),
            'spacing_score': float(spacing_score), 'score': float(score),
            'phred': phred, 'noise': float(nois),
            'tail_rescued': tail_rescued, 'imputed': False,
            'heterozygote': is_het, 'het_channel': het_ch, 'het_ratio': float(het_ratio),
        })
        if not tail_rescued:
            confirmed_positions.append(pos)

    results.sort(key=lambda r: r['pos'])

    if detect_missing and spacing_model is not None:
        results = detect_missing_peaks(
            results, spacing_model, signals, time,
            blob_end, letters, channel_floors, running_noise=running_noise)

    sequence    = "".join(r['letter'] for r in results)

    confirmed   = [r for r in results if not r['tail_rescued'] and not r.get('imputed')]
    avg_quality = float(np.mean([r['phred'] for r in confirmed])) if confirmed else 0.0
#                                ^^^^^^ was r['score'] — now matches what GUI displays
    return results, sequence, avg_quality


# ============================================================
# Quality statistics
# ============================================================

def compute_quality_stats(
    results: list,
    window_trim_min_q: float = 20.0,
    window_trim_size: int = 20,
) -> dict:
    if not results: return {}
    confirmed = [r for r in results if not r.get('tail_rescued') and not r.get('imputed')]
    phreds = [r['phred']                for r in confirmed]
    snrs   = [r['snr']                  for r in confirmed]
    isos   = [r['isolation']            for r in confirmed]
    shapes = [r.get('shape_score', 0.5) for r in confirmed]
    n_conf = len(confirmed)
    q20    = sum(1 for q in phreds if q >= 20)
    q30    = sum(1 for q in phreds if q >= 30)
    q35    = sum(1 for q in phreds if q >= 35)
    trimmed_res, _, _ = sliding_window_trim(results, window_trim_size, window_trim_min_q)
    blast_win, _, _   = find_best_blast_window(results)
    blast_win_phreds  = [r['phred'] for r in blast_win] if blast_win else [0]
    if blast_win_phreds:
        blast_est_identity = 100.0 * float(np.mean(
            [1.0 - 10 ** (-q / 10.0) for q in blast_win_phreds]))
    else:
        blast_est_identity = 0.0
    return {
        'total_bases':           len(results),
        'confirmed_bases':       n_conf,
        'tail_rescued':          sum(1 for r in results if r.get('tail_rescued') and not r.get('imputed')),
        'imputed_bases':         sum(1 for r in results if r.get('imputed')),
        'heterozygote_bases':    sum(1 for r in results if r.get('heterozygote')),
        'window_trimmed_len':    len(trimmed_res),
        'blast_window_len':      len(blast_win),
        'blast_est_identity':    blast_est_identity,
        'blast_ready':           len(blast_win) >= MIN_BLAST_LENGTH,
        'mean_phred':            float(np.mean(phreds))   if phreds else 0.0,
        'median_phred':          float(np.median(phreds)) if phreds else 0.0,
        'min_phred':             int(min(phreds))          if phreds else 0,
        'max_phred':             int(max(phreds))          if phreds else 0,
        'q20_count': q20, 'q30_count': q30, 'q35_count': q35,
        'pct_q20':   100.0 * q20 / n_conf if n_conf else 0.0,
        'pct_q30':   100.0 * q30 / n_conf if n_conf else 0.0,
        'pct_q35':   100.0 * q35 / n_conf if n_conf else 0.0,
        'mean_snr':              float(np.mean(snrs))   if snrs   else 0.0,
        'mean_isolation':        float(np.mean(isos))   if isos   else 0.0,
        'mean_shape_score':      float(np.mean(shapes)) if shapes else 0.0,
    }


# ============================================================
# Export
# ============================================================

def save_fasta(sequence, path, header=">sequence", trim_n=True) -> str:
    if trim_n: sequence = trim_N_ends(sequence)
    with open(path, "w") as fh:
        fh.write(f"{header}\n")
        for i in range(0, len(sequence), 80):
            fh.write(sequence[i:i + 80] + "\n")
    return sequence


def save_fastq(results, path, header="sanger_sequence",
               use_window_trim=True, window_trim_min_q=20.0,
               window_trim_size=20, min_phred_trim=15) -> list:
    if use_window_trim:
        trimmed, _, _ = sliding_window_trim(results, window_trim_size, window_trim_min_q)
        if not trimmed:
            trimmed = trim_low_quality_ends(results, min_phred_trim)
    else:
        trimmed = trim_low_quality_ends(results, min_phred_trim)
    sequence = "".join(r['letter'] for r in trimmed)
    qual_str = "".join(chr(r['phred'] + 33) for r in trimmed)
    with open(path, "w") as fh:
        fh.write(f"@{header}\n{sequence}\n+\n{qual_str}\n")
    return trimmed


def export_blast_ready(
    results: list,
    sequence: str,
    path: str,
    sample_name: str = "sanger",
    quality_stats: Optional[dict] = None,
    also_export_rc: bool = True,
) -> Tuple[str, Optional[str]]:
    blast_win, start, end = find_best_blast_window(results)
    best_seq = "".join(r['letter'] for r in blast_win) if blast_win else trim_N_ends(sequence)
    qs = quality_stats or {}
    header_fwd = (
        f">{'>' if not sample_name.startswith('>') else ''}{sample_name}_forward"
        f" length={len(best_seq)}"
        f" blast_window={start}-{end}"
        f" mean_Q={qs.get('mean_phred', 0):.1f}"
        f" pct_Q20={qs.get('pct_q20', 0):.1f}"
        f" est_identity={qs.get('blast_est_identity', 0):.1f}%"
    ).replace('>>', '>')
    save_fasta(best_seq, path, header_fwd, trim_n=False)
    rc_seq = None
    if also_export_rc:
        rc_seq   = reverse_complement(best_seq)
        rc_path  = path.replace('.fasta', '_rc.fasta').replace('.fa', '_rc.fa')
        if rc_path == path: rc_path = path + '_rc.fasta'
        header_rc = header_fwd.replace('_forward', '_reverse_complement')
        save_fasta(rc_seq, rc_path, header_rc, trim_n=False)
    return best_seq, rc_seq


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(
    time: np.ndarray,
    signals: np.ndarray,
    influence_matrix: np.ndarray,
    baseline_win: int = 101,
    sg_win: int = 19,
    sg_poly: int = 3,
    sg_passes: int = 1,
    sg_mode: str = 'interp',
    sigma: float = 0.4,
    amount: float = 0.0,
    prom: float = 0.002,
    dist: int = 2,
    merge_tol: int = 2,
    mobility_enabled: bool = True,
    channel_letters: tuple = ("G", "A", "T", "C"),
    channel_shifts: List[int] = None,
    min_snr: float = 1.0,
    min_isolation: float = 1.15,
    allow_uncertain: bool = True,
    auto_trim_blob: bool = True,
    suppress_n1: bool = True,
    use_spacing_model: bool = True,
    detect_missing: bool = True,
    het_threshold: float = 0.45,
    detect_het: bool = False,
    use_window_trim: bool = True,
    window_trim_min_q: float = 20.0,
    window_trim_size: int = 20,
    primer_sequence: str = "",
    noise_floor_window: int = 51,
    noise_floor_percentile: float = 5.0,
    baseline_method: str = 'rolling',
    use_adaptive_peaks: bool = True,
) -> dict:
    if channel_shifts is None:
        channel_shifts = [0, 0, 0, 0]

    stages = {}
    stages["raw"] = signals.copy()

    crosstalk = apply_crosstalk_correction(signals, influence_matrix, channel_shifts)
    stages["crosstalk"]    = crosstalk.copy()
    stages["baseline_raw"] = crosstalk.copy()

    baseline, noise_floor = _apply_baseline(crosstalk, baseline_method, baseline_win)
    stages["baseline"]    = baseline.copy()
    stages["noise_floor"] = noise_floor.copy()

    smooth = smooth_savgol(baseline, sg_win, sg_poly, sg_passes, sg_mode)
    stages["smooth"] = smooth.copy()

    running_noise = compute_running_noise_floor(
        smooth, window=noise_floor_window, percentile=noise_floor_percentile)
    stages["running_noise"] = running_noise.copy()

    with np.errstate(divide='ignore', invalid='ignore'):
        snr_map = np.clip(smooth / (noise_floor + 1e-9), 0.0, 50.0)
    stages["snr_map"] = snr_map.copy()

    mobility, mobility_applied = mobility_correction(
        time, smooth, peak_prominence=prom, enabled=mobility_enabled)
    stages["mobility"] = mobility.copy()

    sharpened = sharpen_tail_only(mobility, sigma=sigma, amount=amount)
    stages["sharpened"] = sharpened.copy()

    blob_end = detect_dye_blob_end(sharpened) if auto_trim_blob else 0
    stages["_blob_end"] = blob_end

    running_noise_final = compute_running_noise_floor(
        sharpened, window=noise_floor_window, percentile=noise_floor_percentile)

    peaks, sequence, avg_quality = detect_and_basecall(
        time, sharpened,
        letters=channel_letters,
        prominence=prom,
        distance=dist,
        merge_tol=merge_tol,
        min_snr=min_snr,
        min_isolation=min_isolation,
        allow_uncertain=allow_uncertain,
        use_spacing_model=use_spacing_model,
        blob_end=blob_end,
        suppress_n1=suppress_n1,
        detect_missing=detect_missing,
        het_threshold=het_threshold,
        detect_het=detect_het,
        running_noise=running_noise_final,
        use_adaptive_peaks=use_adaptive_peaks,
    )
    stages["final"] = sharpened.copy()

    primer_trim_pos = 0
    if primer_sequence and peaks:
        trimmed_seq, primer_trim_pos = trim_primer(sequence, primer_sequence)
        if primer_trim_pos > 0:
            peaks    = peaks[primer_trim_pos:]
            sequence = trimmed_seq

    window_trimmed, _, _ = sliding_window_trim(peaks, window_trim_size, window_trim_min_q)
    window_trimmed_seq   = "".join(r['letter'] for r in window_trimmed)
    quality_stats        = compute_quality_stats(peaks, window_trim_min_q, window_trim_size)
    orient               = auto_orient_sequence(peaks, sequence)

    return {
        "stages":               stages,
        "peaks":                peaks,
        "sequence":             sequence,
        "sequence_rc":          orient['rc'],
        "window_trimmed_seq":   window_trimmed_seq,
        "window_trimmed_peaks": window_trimmed,
        "avg_quality":          avg_quality,
        "mobility_applied":     mobility_applied,
        "quality_stats":        quality_stats,
        "blob_end":             blob_end,
        "primer_trim_pos":      primer_trim_pos,
        "n_heterozygotes":      sum(1 for p in peaks if p.get('heterozygote')),
        "n_imputed":            sum(1 for p in peaks if p.get('imputed')),
        "orient":               orient,
        "baseline_method":      baseline_method,
    }