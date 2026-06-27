#!/usr/bin/env python3
"""
models_comparison_all_planes.py
Compare all 4 trained 18-channel models (Timestamp CNN/LSTM, PV CNN/LSTM)
against multiple different held-out CP test SUBJECTS, not a single fixed
saved segment.

For each held-out CP test file, this script:
  - rebuilds that subject's own scaled rolling windows (per-model feature
    representation), so windows never cross a subject/file boundary
  - runs teacher-forced next-tick prediction across the whole file
  - computes per-channel metrics (MAE, RMSE, R2)

It then reports:
  - a per-subject x per-model summary table (mean MAE / mean R2 over the
    18 channels), so you can see how consistent each model is across
    different real patients
  - optional full plots (pred vs gt, phase-binned error) for one chosen
    subject across all 4 models

Run from project root:
  python models_comparison_all_planes.py --list-subjects
  python models_comparison_all_planes.py                      # table over all held-out subjects
  python models_comparison_all_planes.py --subject 3 --plot    # plots for one subject
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# NOTE: tensorflow is imported lazily inside load_all_models() so that the
# --demo preview (and the plotting code) can run without TensorFlow installed.

# ============================================================
# SETTINGS (must match the *_all_planes.py training scripts)
# ============================================================
STRIDE_LEN = 51
WINDOW = 51

CP_FOLDER = "Data_CP/"
CP_SHEET = "Data"
CP_SKIPROWS = [1, 2]
MAX_CP_FILES = 500
TRAIN_FRAC, VAL_FRAC, SPLIT_SEED = 0.70, 0.15, 42

# Per-subject healthy trial files live alongside the bulk training file
# (Data_Normal/dynamics_total_augmented.xlsx) used to train the typical-gait
# branch of every model; these per-subject files share the CP files' sheet
# layout, so they're read the same way. NOT a held-out set -- every one of
# these subjects' strides was part of the typical-gait training data.
NORMAL_FOLDER = "Data_Normal/"
NORMAL_EXCLUDE_FILES = {
    "dynamics_total_augmented.xlsx",
    "randomized_data_healthy.xlsx",
    "Healthy_Data_1_Person.xlsx",
}

SAVE_DIR = "Neural_Networks_Output/Saved_Models"
SCALER_DIR = "Neural_Networks_Output/Scaler"
PLOT_DIR = "Neural_Networks_Output/Subjects/SUBJECT_5"

ANGLE_COLS = [
    "LHipAngles (1)", "LHipAngles (2)", "LHipAngles (3)",
    "LKneeAngles (1)", "LKneeAngles (2)", "LKneeAngles (3)",
    "LAnkleAngles (1)", "LAnkleAngles (2)", "LAnkleAngles (3)",
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]
RIGHT_ANGLE_COLS = [
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]
LHIP_COL, RHIP_COL = "LHipAngles (1)", "RHipAngles (1)"
LFO_COL, RFO_COL = "Left Foot Off", "Right Foot Off"
PV_COLS = ["PhaseVariable_Left", "PhaseVariable_Right"]

SIDES, JOINTS, PLANES = ["Left", "Right"], ["Hip", "Knee", "Ankle"], ["Sagittal (1)", "Frontal (2)", "Transverse (3)"]
SIDE_BASE = [0, 9]
GT_COLOR, PRED_COLOR = "#1f77b4", "#d62728"

MODEL_NAMES = ["timestamp_cnn", "timestamp_lstm", "pv_cnn", "pv_lstm"]
MODEL_LABELS = {
    "timestamp_cnn": "Timestamp CNN",
    "timestamp_lstm": "Timestamp LSTM",
    "pv_cnn": "PV CNN",
    "pv_lstm": "PV LSTM",
}


# ============================================================
# Helpers shared with the training scripts
# ============================================================
def half_stride_shift(stride_len: int) -> int:
    return int(stride_len // 2)


def roll_stridewise_1d(x: np.ndarray, stride_len: int, shift: int) -> np.ndarray:
    x = np.asarray(x)
    n_strides = len(x) // stride_len
    x = x[:n_strides * stride_len].copy()
    out = x.copy()
    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len
        out[a:b] = np.roll(x[a:b], shift)
    return out


def apply_right_leg_half_stride_offset(df: pd.DataFrame, stride_len: int, right_angle_cols, pv_right_col=None) -> pd.DataFrame:
    shift = half_stride_shift(stride_len)
    out = df.copy()
    for col in right_angle_cols:
        out[col] = roll_stridewise_1d(out[col].to_numpy(), stride_len, shift)
    if pv_right_col is not None and pv_right_col in out.columns:
        out[pv_right_col] = roll_stridewise_1d(out[pv_right_col].to_numpy(), stride_len, shift)
    return out


def compute_pv_stride(q: np.ndarray, c: float, enforce_monotonic: bool = True) -> np.ndarray:
    q = q.astype(np.float64)
    N = q.shape[0]
    c = float(np.clip(c, 0.05, 0.95))
    q0 = float(q[0])
    idx_min = int(np.argmin(q))
    qmin = float(q[idx_min])
    denom_stance = q0 - qmin
    s = np.zeros(N, dtype=np.float64)
    if abs(denom_stance) < 1e-6:
        s[:idx_min + 1] = np.linspace(0.0, c, idx_min + 1)
    else:
        s[:idx_min + 1] = ((q0 - q[:idx_min + 1]) / denom_stance) * c
    sm = s[idx_min]
    denom_swing = q0 - qmin
    if abs(denom_swing) < 1e-6:
        s[idx_min:] = np.linspace(sm, 1.0, N - idx_min)
    else:
        s[idx_min:] = 1.0 + ((1.0 - sm) / denom_swing) * (q[idx_min:] - q0)
    s = np.clip(s, 0.0, 1.0)
    if enforce_monotonic:
        for i in range(1, N):
            if s[i] < s[i - 1]:
                s[i] = s[i - 1]
    return s


def compute_phase_variables(df: pd.DataFrame, stride_len: int) -> pd.DataFrame:
    n_strides = len(df) // stride_len
    out = df.iloc[:n_strides * stride_len].copy()
    pvL = np.zeros(len(out), dtype=np.float32)
    pvR = np.zeros(len(out), dtype=np.float32)
    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len
        cL = float(out[LFO_COL].iloc[a]) / 100.0
        cR = float(out[RFO_COL].iloc[a]) / 100.0
        pvL[a:b] = compute_pv_stride(out[LHIP_COL].values[a:b], c=cL)
        pvR[a:b] = compute_pv_stride(out[RHIP_COL].values[a:b], c=cR)
    out["PhaseVariable_Left"] = pvL
    out["PhaseVariable_Right"] = pvR
    return out


def make_rolling_windows(features: np.ndarray, targets: np.ndarray, window: int):
    T = features.shape[0]
    if T <= window:
        return None, None
    D, K = features.shape[1], targets.shape[1]
    X = np.zeros((T - window, window, D), dtype=np.float32)
    y = np.zeros((T - window, K), dtype=np.float32)
    for i in range(T - window):
        X[i] = features[i:i + window]
        y[i] = targets[i + window]
    return X, y


# ============================================================
# Recover the held-out CP test file list (identical split logic
# to the 4 *_all_planes.py training scripts)
# ============================================================
def get_cp_test_subjects():
    """
    Returns {subject_id: [trial files...]} for the held-out test subjects,
    using the identical patient-level split as the 4 training scripts.
    Each CP file is a single 51-row stride; each real patient has ~5 trial
    files, so a subject's trials must be concatenated to get enough rows
    to form even one rolling window.
    """
    cp_files = [fn for fn in sorted(os.listdir(CP_FOLDER)) if fn.endswith(".xlsx")][:MAX_CP_FILES]
    subj_to_files = {}
    for fn in cp_files:
        subj_to_files.setdefault(fn.split("-")[0], []).append(fn)
    subj_ids = sorted(subj_to_files.keys())

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(subj_ids))
    n_train = int(TRAIN_FRAC * len(subj_ids))
    n_val = int(VAL_FRAC * len(subj_ids))
    test_subj = sorted(subj_ids[i] for i in perm[n_train + n_val:])
    return {sid: subj_to_files[sid] for sid in test_subj}


def mean_toe_off(df: pd.DataFrame):
    """Mean measured toe-off (% of stride) per side, from the Foot Off columns.
    These anchor the gait-phase shading to this subject's own data. Falls back
    to the textbook 60% if the columns are missing or empty."""
    def _m(col, default=60.0):
        if col in df.columns and df[col].notna().any():
            v = float(df[col].mean())
            return v if 5.0 <= v <= 95.0 else default
        return default
    return _m(LFO_COL), _m(RFO_COL)


def load_subject_trials(files, folder: str) -> pd.DataFrame:
    """Concatenate one subject's trial files (each a single 51-row stride)."""
    frames = []
    for fn in files:
        fp = os.path.join(folder, fn)
        df = pd.read_excel(
            fp, sheet_name=CP_SHEET,
            usecols=ANGLE_COLS + [LFO_COL, RFO_COL],
            skiprows=CP_SKIPROWS,
        ).fillna(0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_cp_subject(files) -> pd.DataFrame:
    return load_subject_trials(files, CP_FOLDER)


def load_healthy_subject(files) -> pd.DataFrame:
    return load_subject_trials(files, NORMAL_FOLDER)


def get_healthy_subjects():
    """Returns {subject_id: [trial files...]} for every per-subject healthy
    trial file in Data_Normal/ (excludes the bulk training/augmentation
    files). Mirrors get_cp_test_subjects() but is NOT a held-out split."""
    files = [
        fn for fn in sorted(os.listdir(NORMAL_FOLDER))
        if fn.endswith(".xlsx") and fn not in NORMAL_EXCLUDE_FILES
    ]
    subj_to_files = {}
    for fn in files:
        subj_to_files.setdefault(fn.split("-")[0], []).append(fn)
    return subj_to_files


# ============================================================
# Model + scaler loading
# ============================================================
def load_all_models():
    from tensorflow.keras.models import load_model
    models = {}

    models["timestamp_cnn"] = dict(
        model=load_model(os.path.join(SAVE_DIR, "Timestamp_cnn_next_tick_model_18.keras")),
        scaler=joblib.load(os.path.join(SCALER_DIR, "standard_scaler_typical_cnn_next_tick_18.save")),
        kind="timestamp",
    )
    models["timestamp_lstm"] = dict(
        model=load_model(os.path.join(SAVE_DIR, "Timestamp_lstm_next_tick_model_18.keras")),
        scaler=joblib.load(os.path.join(SCALER_DIR, "standard_scaler_typical_lstm_next_tick_18.save")),
        kind="timestamp",
    )
    models["pv_cnn"] = dict(
        model=load_model(os.path.join(SAVE_DIR, "PV_cnn_next_tick_18ch_final.keras")),
        scaler_pv=joblib.load(os.path.join(SCALER_DIR, "scaler_pv_cnn_18ch.save")),
        scaler_ang=joblib.load(os.path.join(SCALER_DIR, "scaler_angles_cnn_18ch.save")),
        kind="pv",
    )
    models["pv_lstm"] = dict(
        model=load_model(os.path.join(SAVE_DIR, "PV_lstm_next_tick_18ch_final.keras")),
        scaler_pv=joblib.load(os.path.join(SCALER_DIR, "scaler_pv_lstm_18ch.save")),
        scaler_ang=joblib.load(os.path.join(SCALER_DIR, "scaler_angles_lstm_18ch.save")),
        kind="pv",
    )
    return models


def predict_next_tick(model_entry, df_subject: pd.DataFrame):
    """
    Builds this subject's own rolling windows (no cross-file leakage) for the
    given model's feature representation, runs next-tick prediction, and
    returns (pred_deg, gt_deg) both (N,18), or (None, None) if too short.
    """
    df = df_subject.copy()

    if model_entry["kind"] == "timestamp":
        df = apply_right_leg_half_stride_offset(df, STRIDE_LEN, RIGHT_ANGLE_COLS)
        scaler = model_entry["scaler"]
        ang_sc = scaler.transform(df[ANGLE_COLS].to_numpy()).astype(np.float32)
        X, y = make_rolling_windows(ang_sc, ang_sc, WINDOW)
        if X is None:
            return None, None
        pred_sc = model_entry["model"].predict(X, verbose=0)
        pred_sc = np.asarray(pred_sc)
        if pred_sc.ndim == 3:        # multi-step head -> take immediate next-tick
            pred_sc = pred_sc[:, 0, :]
        pred_deg = scaler.inverse_transform(pred_sc)
        gt_deg = scaler.inverse_transform(y)
        return pred_deg, gt_deg

    else:  # PV models
        # PV must be computed on the raw, unshifted data (anchored to the
        # start of each stride) before the right leg is shifted -- same
        # order as the *_all_planes.py training scripts.
        df = compute_phase_variables(df, STRIDE_LEN)
        df = apply_right_leg_half_stride_offset(df, STRIDE_LEN, RIGHT_ANGLE_COLS, pv_right_col="PhaseVariable_Right")
        scaler_pv, scaler_ang = model_entry["scaler_pv"], model_entry["scaler_ang"]
        pv_sc = scaler_pv.transform(df[PV_COLS].to_numpy())
        ang_sc = scaler_ang.transform(df[ANGLE_COLS].to_numpy())
        feats = np.concatenate([pv_sc, ang_sc], axis=1)
        X, y = make_rolling_windows(feats, ang_sc, WINDOW)
        if X is None:
            return None, None
        pred_sc = model_entry["model"].predict(X, verbose=0)
        pred_deg = scaler_ang.inverse_transform(pred_sc)
        gt_deg = scaler_ang.inverse_transform(y)
        return pred_deg, gt_deg


# ============================================================
# Metrics
# ============================================================
def compute_metrics_deg(pred: np.ndarray, gt: np.ndarray):
    eps = 1e-12
    err = pred - gt
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((gt - gt.mean(axis=0)) ** 2, axis=0) + eps
    r2 = 1.0 - ss_res / ss_tot
    return mae, rmse, r2


def overall_stats_text(preds: dict, gt_ref: np.ndarray, chans) -> str:
    """One line per model: overall MAE/RMSE/R2 averaged over `chans`,
    computed on the full (unsegmented) prediction vs ground truth."""
    lines = ["Overall (this subject):"]
    for name in MODEL_NAMES:
        pred = preds.get(name)
        if pred is None:
            continue
        n = min(len(pred), len(gt_ref))
        mae, rmse, r2 = compute_metrics_deg(pred[:n][:, chans], gt_ref[:n][:, chans])
        lines.append(f"{MODEL_LABELS[name]:<15s} MAE={mae.mean():5.2f}  "
                     f"RMSE={rmse.mean():5.2f}  R2={r2.mean():5.2f}")
    return "\n".join(lines)


# ============================================================
# Plots (one subject, all 4 models)
# ============================================================
def _grid_fig(side, title, ylabel):
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    fig.suptitle(f"{title} — {side} Side", fontsize=12)
    for row, joint in enumerate(JOINTS):
        for col, plane in enumerate(PLANES):
            ax = axes[row, col]
            ax.set_title(f"{side} {joint} – {plane}", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.35)
    return fig, axes


def plot_subject_all_models(preds: dict, gt_ref: np.ndarray, subject_name: str, save_dir=None):
    T = min(300, gt_ref.shape[0])
    t = np.arange(T)
    colors = {"timestamp_cnn": "#d62728", "timestamp_lstm": "#2ca02c", "pv_cnn": "#9467bd", "pv_lstm": "#ff7f0e"}

    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = _grid_fig(side, f"Subject {subject_name}: Predicted vs Ground Truth", "Angle (deg)")
        for row in range(3):
            for col in range(3):
                idx = base + row * 3 + col
                ax = axes[row, col]
                ax.plot(t, gt_ref[:T, idx], color=GT_COLOR, lw=1.8, label="Ground truth")
                for name, pred in preds.items():
                    if pred is None:
                        continue
                    ax.plot(t[:min(T, len(pred))], pred[:T, idx], color=colors[name], lw=1.1, ls="--", label=MODEL_LABELS[name])
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)
        for col in range(3):
            axes[2, col].set_xlabel("Sample", fontsize=8)
        plt.tight_layout()
        if save_dir:
            fname = os.path.join(save_dir, f"compare4_{side.lower()}_subject_{subject_name}.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"Saved: {fname}")
        plt.show()


def plot_phase_error_all_models(preds: dict, gt_ref: np.ndarray, subject_name: str, save_dir=None):
    n_strides = gt_ref.shape[0] // STRIDE_LEN
    T_use = n_strides * STRIDE_LEN
    x = np.linspace(0, 100, STRIDE_LEN)
    colors = {"timestamp_cnn": "#d62728", "timestamp_lstm": "#2ca02c", "pv_cnn": "#9467bd", "pv_lstm": "#ff7f0e"}

    gt_s = gt_ref[:T_use].reshape(n_strides, STRIDE_LEN, 18)

    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = _grid_fig(side, f"Subject {subject_name}: Mean |Pred-GT| vs Gait Phase", "|Error| (deg)")
        for row in range(3):
            for col in range(3):
                idx = base + row * 3 + col
                ax = axes[row, col]
                for name, pred in preds.items():
                    if pred is None:
                        continue
                    n2 = min(n_strides, len(pred) // STRIDE_LEN)
                    pred_s = pred[:n2 * STRIDE_LEN].reshape(n2, STRIDE_LEN, 18)
                    err = np.abs(pred_s[:, :, idx] - gt_s[:n2, :, idx]).mean(axis=0)
                    ax.plot(x, err, color=colors[name], lw=1.6, label=MODEL_LABELS[name])
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)
        for col in range(3):
            axes[2, col].set_xlabel("Gait cycle (%)", fontsize=8)
        plt.tight_layout()
        if save_dir:
            fname = os.path.join(save_dir, f"compare4_phase_error_{side.lower()}_subject_{subject_name}.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"Saved: {fname}")
        plt.show()


# ===== PHASE_PLOT_SECTION_START =====
# Constants for the phase plots. SIDES, JOINTS, PLANES, STRIDE_LEN and
# MODEL_LABELS are reused from the main script's constants above.
GT_COLOR_DARK = "#1f2d3d"
_PHASE_SIDE_BASE = {"Left": 0, "Right": 9}
MODEL_COLORS = {
    "timestamp_cnn": "#d62728",
    "timestamp_lstm": "#2ca02c",
    "pv_cnn": "#9467bd",
    "pv_lstm": "#ff7f0e",
}

# ------------------------------------------------------------------
# Gait-phase definition
# ------------------------------------------------------------------
# Each phase is (full name, abbreviation, fill colour, group).
# The (start, end) boundaries are computed at draw time from the toe-off %,
# using the standard relative fractions below.
PHASE_ORDER = [
    "Loading Response",
    "Mid Stance",
    "Terminal Stance",
    "Pre-Swing",
    "Initial Swing",
    "Mid Swing",
    "Terminal Swing",
]
PHASE_ABBR = {
    "Loading Response": "LR",
    "Mid Stance": "MSt",
    "Terminal Stance": "TSt",
    "Pre-Swing": "PSw",
    "Initial Swing": "ISw",
    "Mid Swing": "MSw",
    "Terminal Swing": "TSw",
}
# Warm = stance group, cool = swing group.
PHASE_COLORS = {
    "Loading Response": "#FBE3D3",
    "Mid Stance":       "#F6CFB0",
    "Terminal Stance":  "#F0B488",
    "Pre-Swing":        "#EBD27A",   # amber: push-off / pre-swing
    "Initial Swing":    "#D2EAE1",
    "Mid Swing":        "#AAD2E3",
    "Terminal Swing":   "#82B6D8",
}
# Relative position of each phase *within* its parent (stance or swing),
# derived from the standard cycle (toe-off ~60%): LR 0-10, MSt 10-30,
# TSt 30-50, PSw 50-60 (of stance); ISw 60-73, MSw 73-87, TSw 87-100 (of swing).
_STANCE_FRAC = {
    "Loading Response": (0.000, 0.167),
    "Mid Stance":       (0.167, 0.500),
    "Terminal Stance":  (0.500, 0.833),
    "Pre-Swing":        (0.833, 1.000),
}
_SWING_FRAC = {
    "Initial Swing":  (0.000, 0.325),
    "Mid Swing":      (0.325, 0.675),
    "Terminal Swing": (0.675, 1.000),
}


def phase_boundaries(toe_off_pct: float = 60.0):
    """Return {phase: (start_pct, end_pct)} for a given toe-off percentage."""
    toe = float(np.clip(toe_off_pct, 5.0, 95.0))
    bounds = {}
    for ph, (a, b) in _STANCE_FRAC.items():
        bounds[ph] = (a * toe, b * toe)
    for ph, (a, b) in _SWING_FRAC.items():
        bounds[ph] = (toe + a * (100.0 - toe), toe + b * (100.0 - toe))
    return bounds


def add_gait_phase_background(ax, toe_off_pct: float = 60.0,
                              alpha: float = 0.55, label_top: bool = False,
                              mark_toe_off: bool = True):
    """Shade the 8 gait phases behind whatever is plotted on ``ax``.

    Assumes the x-axis is the gait cycle in percent (0-100).
    """
    bounds = phase_boundaries(toe_off_pct)
    for ph in PHASE_ORDER:
        a, b = bounds[ph]
        ax.axvspan(a, b, color=PHASE_COLORS[ph], alpha=alpha, lw=0, zorder=0)
    if mark_toe_off:
        ax.axvline(toe_off_pct, color="#333333", lw=1.1, ls=(0, (4, 3)),
                   alpha=0.8, zorder=1)
    if label_top:
        ymin, ymax = ax.get_ylim()
        ytxt = ymax - 0.02 * (ymax - ymin)
        for ph in PHASE_ORDER:
            a, b = bounds[ph]
            ax.text((a + b) / 2.0, ytxt, PHASE_ABBR[ph], ha="center", va="top",
                    fontsize=6.5, color="#444444", zorder=3, clip_on=True)
    ax.set_xlim(0, 100)


def gait_phase_legend_handles(toe_off_pct: float = 60.0):
    """Patch handles mapping each colour -> full phase name (for a figure legend)."""
    handles = []
    for ph in PHASE_ORDER:
        handles.append(Patch(facecolor=PHASE_COLORS[ph], alpha=0.85,
                             label=f"{PHASE_ABBR[ph]} – {ph}"))
    return handles


# ------------------------------------------------------------------
# Stride segmentation
# ------------------------------------------------------------------
def segment_strides(arr: np.ndarray, stride_len: int = STRIDE_LEN,
                    unshift_right: bool = True) -> np.ndarray:
    """Reshape an (N,18) trace into (n_strides, stride_len, 18).

    If ``unshift_right`` is True, the right-leg channels (9..17) are rolled
    back by half a stride to undo the training-time half-stride offset, so
    that sample 0 corresponds to *that leg's* heel strike on both sides.
    """
    arr = np.asarray(arr)
    n = arr.shape[0] // stride_len
    if n == 0:
        return None
    seg = arr[:n * stride_len].reshape(n, stride_len, arr.shape[1]).copy()
    if unshift_right:
        shift = stride_len // 2
        seg[:, :, 9:18] = np.roll(seg[:, :, 9:18], -shift, axis=1)
    return seg


def _channel_index(side: str, joint_row: int, plane_col: int) -> int:
    return _PHASE_SIDE_BASE[side] + joint_row * 3 + plane_col


def _new_grid(side: str, title: str, ylabel: str):
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.6), sharex=True)
    fig.suptitle(f"{title}  —  {side} Side", fontsize=14, fontweight="bold")
    for row, joint in enumerate(JOINTS):
        for col, plane in enumerate(PLANES):
            ax = axes[row, col]
            ax.set_title(f"{side} {joint} – {plane}", fontsize=9.5)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25, zorder=0.5)
    return fig, axes


def _finish_and_save(fig, axes, save_path, toe_off_pct, x_label,
                     show, line_handles=None, line_labels=None):
    for col in range(3):
        axes[2, col].set_xlabel(x_label, fontsize=8)

    phase_handles = gait_phase_legend_handles(toe_off_pct)
    handles = list(phase_handles)
    labels = [h.get_label() for h in phase_handles]
    if line_handles:
        handles += line_handles
        labels += line_labels
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ==================================================================
# 1) Stride-averaged predicted vs ground truth over the normalised cycle
# ==================================================================
def plot_normalized_cycle_with_phases(preds: dict, gt_ref: np.ndarray,
                                       subject_name: str,
                                       toe_off_left: float = 60.0,
                                       toe_off_right: float = 60.0,
                                       stride_len: int = STRIDE_LEN,
                                       unshift_right: bool = True,
                                       save_dir: str = None, show: bool = True):
    """Mean predicted vs mean ground-truth trajectory over ONE normalised gait
    cycle (0-100%), averaged across all of the subject's strides, with a
    ground-truth ±1 SD band and the 8 gait phases shaded behind."""
    gt_seg = segment_strides(gt_ref, stride_len, unshift_right)
    if gt_seg is None:
        print("Not enough data for a full stride; skipping normalised-cycle plot.")
        return
    x = np.linspace(0, 100, stride_len)
    gt_mean = gt_seg.mean(axis=0)
    gt_std = gt_seg.std(axis=0)

    pred_means = {}
    for name, pred in preds.items():
        if pred is None:
            continue
        ps = segment_strides(pred, stride_len, unshift_right)
        if ps is not None:
            pred_means[name] = ps.mean(axis=0)

    for side in SIDES:
        toe = toe_off_left if side == "Left" else toe_off_right
        fig, axes = _new_grid(
            side, f"Subject {subject_name}: Predicted vs Ground Truth over the Gait Cycle",
            "Angle (deg)")
        line_handles, line_labels = [], []
        for row in range(3):
            for col in range(3):
                idx = _channel_index(side, row, col)
                ax = axes[row, col]
                add_gait_phase_background(ax, toe, label_top=(row == 0))
                ax.fill_between(x, gt_mean[:, idx] - gt_std[:, idx],
                                gt_mean[:, idx] + gt_std[:, idx],
                                color=GT_COLOR_DARK, alpha=0.12, lw=0, zorder=2)
                hl, = ax.plot(x, gt_mean[:, idx], color=GT_COLOR_DARK, lw=2.4,
                              zorder=5)
                if row == 0 and col == 0:
                    line_handles.append(hl); line_labels.append("Ground truth (mean ±1 SD)")
                for name, pmean in pred_means.items():
                    hl, = ax.plot(x, pmean[:, idx], color=MODEL_COLORS[name],
                                  lw=1.6, ls="--", zorder=4)
                    if row == 0 and col == 0:
                        line_handles.append(hl); line_labels.append(MODEL_LABELS[name])
        chans = [_channel_index(side, r, c) for r in range(3) for c in range(3)]
        stats_text = overall_stats_text(preds, gt_ref, chans)
        fig.text(0.99, 0.965, stats_text, ha="right", va="top", fontsize=8,
                 family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="#888888", alpha=0.85))
        save_path = (os.path.join(save_dir,
                     f"cycle_phases_{side.lower()}_subject_{subject_name}.png")
                     if save_dir else None)
        _finish_and_save(fig, axes, save_path, toe, "Gait cycle (%)", show,
                         line_handles, line_labels)


# ==================================================================
# 2) Per-phase mean |error| curve, with phases shaded
# ==================================================================
def plot_phase_error_with_phases(preds: dict, gt_ref: np.ndarray,
                                  subject_name: str,
                                  toe_off_left: float = 60.0,
                                  toe_off_right: float = 60.0,
                                  stride_len: int = STRIDE_LEN,
                                  unshift_right: bool = True,
                                  save_dir: str = None, show: bool = True):
    """Mean |pred - gt| as a function of position in the gait cycle, with the
    8 phases shaded -- shows *where* in the stride each model struggles."""
    gt_seg = segment_strides(gt_ref, stride_len, unshift_right)
    if gt_seg is None:
        return
    x = np.linspace(0, 100, stride_len)
    seg_preds = {n: segment_strides(p, stride_len, unshift_right)
                 for n, p in preds.items() if p is not None}

    for side in SIDES:
        toe = toe_off_left if side == "Left" else toe_off_right
        fig, axes = _new_grid(
            side, f"Subject {subject_name}: Mean |Prediction Error| across the Gait Cycle",
            "|Error| (deg)")
        line_handles, line_labels = [], []
        for row in range(3):
            for col in range(3):
                idx = _channel_index(side, row, col)
                ax = axes[row, col]
                add_gait_phase_background(ax, toe, alpha=0.5,
                                          label_top=(row == 0))
                for name, ps in seg_preds.items():
                    n2 = min(ps.shape[0], gt_seg.shape[0])
                    err = np.abs(ps[:n2, :, idx] - gt_seg[:n2, :, idx]).mean(axis=0)
                    hl, = ax.plot(x, err, color=MODEL_COLORS[name], lw=1.9,
                                  zorder=4)
                    if row == 0 and col == 0:
                        line_handles.append(hl); line_labels.append(MODEL_LABELS[name])
        save_path = (os.path.join(save_dir,
                     f"phase_error_phases_{side.lower()}_subject_{subject_name}.png")
                     if save_dir else None)
        _finish_and_save(fig, axes, save_path, toe, "Gait cycle (%)", show,
                         line_handles, line_labels)


# ==================================================================
# 3) Mean error broken down by gait phase -- grouped bar chart
# ==================================================================
def plot_per_phase_error_bars(preds: dict, gt_ref: np.ndarray,
                              subject_name: str,
                              toe_off_left: float = 60.0,
                              toe_off_right: float = 60.0,
                              stride_len: int = STRIDE_LEN,
                              unshift_right: bool = True,
                              save_dir: str = None, show: bool = True):
    """For each side, a grouped bar chart of mean |error| (deg) inside each of
    the 8 gait phases, one bar group per model -- a compact, quantitative
    summary of *which phase* each model handles best."""
    gt_seg = segment_strides(gt_ref, stride_len, unshift_right)
    if gt_seg is None:
        return
    x_pct = np.linspace(0, 100, stride_len)
    seg_preds = {n: segment_strides(p, stride_len, unshift_right)
                 for n, p in preds.items() if p is not None}
    model_names = list(seg_preds.keys())

    for side in SIDES:
        toe = toe_off_left if side == "Left" else toe_off_right
        bounds = phase_boundaries(toe)
        chans = [_PHASE_SIDE_BASE[side] + k for k in range(9)]   # this side's 9 channels

        # mean |error| per (model, phase), averaged over channels + strides + frames-in-phase
        table = np.full((len(model_names), len(PHASE_ORDER)), np.nan)
        for mi, name in enumerate(model_names):
            ps = seg_preds[name]
            n2 = min(ps.shape[0], gt_seg.shape[0])
            abserr = np.abs(ps[:n2, :, chans] - gt_seg[:n2, :, chans])  # (n,stride,9)
            for pj, ph in enumerate(PHASE_ORDER):
                a, b = bounds[ph]
                mask = (x_pct >= a) & (x_pct < b if pj < len(PHASE_ORDER) - 1 else x_pct <= b)
                if mask.any():
                    table[mi, pj] = abserr[:, mask, :].mean()

        fig, ax = plt.subplots(figsize=(13, 5.6))
        ax.set_title(f"Subject {subject_name}: Mean |Error| per Gait Phase "
                     f"— {side} Side", fontsize=13, fontweight="bold")
        n_models = len(model_names)
        group_w = 0.8
        bar_w = group_w / max(n_models, 1)
        x = np.arange(len(PHASE_ORDER))
        # subtle phase-colour band behind each group
        for pj, ph in enumerate(PHASE_ORDER):
            ax.axvspan(pj - 0.5, pj + 0.5, color=PHASE_COLORS[ph], alpha=0.45,
                       lw=0, zorder=0)
        for mi, name in enumerate(model_names):
            offset = -group_w / 2 + bar_w * (mi + 0.5)
            ax.bar(x + offset, table[mi], width=bar_w * 0.92,
                   color=MODEL_COLORS[name], label=MODEL_LABELS[name],
                   edgecolor="white", lw=0.5, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{PHASE_ABBR[p]}\n{p}" for p in PHASE_ORDER],
                           fontsize=8)
        ax.set_ylabel("Mean |Error| (deg)", fontsize=10)
        ax.set_xlim(-0.5, len(PHASE_ORDER) - 0.5)
        ax.grid(True, axis="y", alpha=0.3, zorder=0.5)
        ax.legend(fontsize=9, ncol=n_models, loc="upper left", frameon=False)
        fig.tight_layout()
        save_path = (os.path.join(save_dir,
                     f"per_phase_bars_{side.lower()}_subject_{subject_name}.png")
                     if save_dir else None)
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# ==================================================================
# 4) Single-joint "hero" figure for a clean presentation slide
# ==================================================================
def plot_single_joint_hero(preds: dict, gt_ref: np.ndarray, subject_name: str,
                           side: str = "Left", joint: str = "Knee",
                           plane_col: int = 0, toe_off_pct: float = 60.0,
                           stride_len: int = STRIDE_LEN,
                           unshift_right: bool = True,
                           save_dir: str = None, show: bool = True):
    """One large, slide-ready panel for a single joint/plane: ground-truth band,
    all model means, fully labelled gait phases.  Great as a 'hero' figure."""
    gt_seg = segment_strides(gt_ref, stride_len, unshift_right)
    if gt_seg is None:
        return
    joint_row = JOINTS.index(joint)
    idx = _channel_index(side, joint_row, plane_col)
    x = np.linspace(0, 100, stride_len)

    fig, ax = plt.subplots(figsize=(12, 6.2))
    add_gait_phase_background(ax, toe_off_pct, alpha=0.6, label_top=True)

    gt_mean = gt_seg[:, :, idx].mean(axis=0)
    gt_std = gt_seg[:, :, idx].std(axis=0)
    ax.fill_between(x, gt_mean - gt_std, gt_mean + gt_std, color=GT_COLOR_DARK,
                    alpha=0.12, lw=0, zorder=2)
    ax.plot(x, gt_mean, color=GT_COLOR_DARK, lw=3.0, label="Ground truth (mean ±1 SD)",
            zorder=5)
    for name, pred in preds.items():
        if pred is None:
            continue
        ps = segment_strides(pred, stride_len, unshift_right)
        if ps is None:
            continue
        ax.plot(x, ps[:, :, idx].mean(axis=0), color=MODEL_COLORS[name],
                lw=2.0, ls="--", label=MODEL_LABELS[name], zorder=4)

    ax.set_title(f"Subject {subject_name}: {side} {joint} – {PLANES[plane_col]}",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Gait cycle (%)", fontsize=11)
    ax.set_ylabel("Angle (deg)", fontsize=11)
    ax.grid(True, alpha=0.25, zorder=0.5)

    line_leg = ax.legend(loc="upper right", fontsize=9, frameon=True,
                         framealpha=0.9)
    phase_handles = gait_phase_legend_handles(toe_off_pct)
    ax.add_artist(line_leg)
    fig.legend(handles=phase_handles, labels=[h.get_label() for h in phase_handles],
               loc="lower center", ncol=7, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_path = (os.path.join(save_dir,
                 f"hero_{side.lower()}_{joint.lower()}_p{plane_col+1}_subject_{subject_name}.png")
                 if save_dir else None)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ==================================================================
# Convenience wrapper
# ==================================================================
def make_all_phase_plots(preds, gt_ref, subject_name,
                         toe_off_left=60.0, toe_off_right=60.0,
                         stride_len=STRIDE_LEN, unshift_right=True,
                         save_dir=None, show=False):
    """Produce the full set of phase-shaded figures in one call."""
    plot_normalized_cycle_with_phases(preds, gt_ref, subject_name,
                                      toe_off_left, toe_off_right, stride_len,
                                      unshift_right, save_dir, show)
    plot_phase_error_with_phases(preds, gt_ref, subject_name,
                                 toe_off_left, toe_off_right, stride_len,
                                 unshift_right, save_dir, show)
    plot_per_phase_error_bars(preds, gt_ref, subject_name,
                              toe_off_left, toe_off_right, stride_len,
                              unshift_right, save_dir, show)
    plot_single_joint_hero(preds, gt_ref, subject_name, side="Left",
                           joint="Knee", plane_col=0,
                           toe_off_pct=toe_off_left, stride_len=stride_len,
                           unshift_right=unshift_right, save_dir=save_dir,
                           show=show)



# ============================================================
# Built-in synthetic preview (no models or patient data needed)
# ============================================================
def run_demo(save_dir="Plots"):
    """Render every phase-shaded figure from synthetic but plausible gait data,
    so the visual design can be checked without the trained models. Run with:
        python models_comparison_all_planes.py --demo
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.RandomState(7)
    STRIDE, N = STRIDE_LEN, 14
    t = np.linspace(0, 1, STRIDE, endpoint=False)
    g = lambda c, w: np.exp(-((t - c) ** 2) / (2 * w ** 2))

    hip_sag = 20 * np.cos(2 * np.pi * t) + 10
    hip_fro = 6 * np.sin(2 * np.pi * t) + 2
    hip_tra = 5 * np.cos(2 * np.pi * t + 0.6)
    knee_sag = 8 + 10 * g(0.14, 0.06) + 55 * g(0.73, 0.10)
    knee_fro = 5 * np.sin(2 * np.pi * t + 0.4) - 2
    knee_tra = 6 * np.cos(2 * np.pi * t + 1.0)
    ank_sag = -4 * np.cos(2 * np.pi * t) + 3 - 12 * g(0.62, 0.05)
    ank_fro = 4 * np.sin(2 * np.pi * t + 0.8)
    ank_tra = 5 * np.cos(2 * np.pi * t + 0.2)
    base9 = np.stack([hip_sag, hip_fro, hip_tra, knee_sag, knee_fro, knee_tra,
                      ank_sag, ank_fro, ank_tra], axis=1)
    base18 = np.concatenate([base9, base9 * 0.95 + 1.5], axis=1)

    gt_nat = np.zeros((N * STRIDE, 18), np.float32)
    for s in range(N):
        gt_nat[s*STRIDE:(s+1)*STRIDE] = base18 * (1 + 0.04*rng.randn(18)) + 0.8*rng.randn(STRIDE, 18)
    shift = STRIDE // 2
    gt_pipe = gt_nat.copy()
    for s in range(N):
        a, b = s*STRIDE, (s+1)*STRIDE
        gt_pipe[a:b, 9:18] = np.roll(gt_nat[a:b, 9:18], shift, axis=0)

    profs = {"timestamp_cnn": 1.6*g(0.73, 0.12), "timestamp_lstm": 1.1*g(0.50, 0.18),
             "pv_cnn": 0.9*g(0.15, 0.10), "pv_lstm": 0.7*np.ones(STRIDE)}
    preds = {}
    for name, prof in profs.items():
        pf = np.tile(prof, N)[:, None]
        preds[name] = (gt_pipe + pf * (2.0 + rng.randn(*gt_pipe.shape) * 0.5)).astype(np.float32)

    print("Rendering synthetic preview into", save_dir)
    make_all_phase_plots(preds, gt_pipe, "DEMO", toe_off_left=58.0,
                         toe_off_right=60.0, save_dir=save_dir, show=False)
    print("Demo done.")



# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Compare 4 all-planes models across multiple held-out CP subjects")
    parser.add_argument("--demo", action="store_true", help="Render synthetic preview plots (no models/data needed) and exit")
    parser.add_argument("--list-subjects", action="store_true", help="List held-out CP test files with their index and exit")
    parser.add_argument("--subject", type=int, default=None, help="Index (from --list-subjects) of one CP subject to inspect")
    parser.add_argument("--list-healthy-subjects", action="store_true", help="List healthy (Data_Normal) subjects with their index and exit")
    parser.add_argument("--healthy_subject", type=int, default=0, help="Index (from --list-healthy-subjects) of the healthy subject to compare against. NOTE: these subjects were part of typical-gait training data, not a held-out set.")
    parser.add_argument("--plot", action="store_true", help="Show the complete-stride plots for the chosen --subject (CP) and --healthy_subject")
    parser.add_argument("--no_save", action="store_true", help="Don't save PNGs when plotting")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    test_subjects = get_cp_test_subjects()
    subject_ids = list(test_subjects.keys())

    if args.list_subjects:
        for i, sid in enumerate(subject_ids):
            print(f"[{i:3d}] {sid}  ({len(test_subjects[sid])} trials)")
        print(f"\n{len(subject_ids)} held-out CP test subjects.")
        return

    healthy_subjects = get_healthy_subjects()
    healthy_ids = list(healthy_subjects.keys())

    if args.list_healthy_subjects:
        for i, sid in enumerate(healthy_ids):
            print(f"[{i:3d}] {sid}  ({len(healthy_subjects[sid])} trials)")
        print(f"\n{len(healthy_ids)} healthy subjects "
              f"(NOTE: part of typical-gait training data, not held out).")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)
    save_dir = None if args.no_save else PLOT_DIR

    print(f"Loading 4 models...")
    models = load_all_models()

    if args.subject is not None:
        targets = [subject_ids[args.subject]]
    else:
        targets = subject_ids

    rows = []
    last_preds, last_gt, last_df = None, None, None
    for sid in targets:
        df_subject = load_cp_subject(test_subjects[sid])
        preds, gts = {}, {}
        for name in MODEL_NAMES:
            pred_deg, gt_deg = predict_next_tick(models[name], df_subject)
            preds[name] = pred_deg
            gts[name] = gt_deg
            if pred_deg is None:
                continue
            mae, rmse, r2 = compute_metrics_deg(pred_deg, gt_deg)
            rows.append({
                "subject": sid, "model": MODEL_LABELS[name],
                "mean_MAE": mae.mean(), "mean_RMSE": rmse.mean(), "mean_R2": r2.mean(),
            })

        if args.subject is not None:
            last_df = df_subject
            last_preds, last_gt = preds, gts["timestamp_cnn"] if gts["timestamp_cnn"] is not None else next(g for g in gts.values() if g is not None)

    summary = pd.DataFrame(rows)
    print("\n=== Per-subject metrics (mean over 18 channels) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    print("\n=== Cross-subject summary per model (mean ± std across held-out subjects) ===")
    agg = summary.groupby("model")[["mean_MAE", "mean_RMSE", "mean_R2"]].agg(["mean", "std"])
    print(agg.to_string(float_format=lambda v: f"{v:7.3f}"))

    if args.subject is not None and args.plot and last_preds is not None:
        subject_name = targets[0]
        plot_subject_all_models(last_preds, last_gt, subject_name, save_dir=save_dir)
        plot_phase_error_all_models(last_preds, last_gt, subject_name, save_dir=save_dir)

        # --- phase-shaded presentation plots (gait_phase_plots.py) ---
        to_left, to_right = mean_toe_off(last_df)
        print(f"Measured toe-off: Left {to_left:.1f}%  Right {to_right:.1f}%")
        make_all_phase_plots(
            last_preds, last_gt, subject_name,
            toe_off_left=to_left, toe_off_right=to_right,
            save_dir=save_dir, show=not args.no_save,
        )

    if args.plot and healthy_ids:
        hsid = healthy_ids[args.healthy_subject]
        print(f"\nHealthy subject: {hsid} "
              f"(NOTE: part of typical-gait training data, not held out)")
        df_healthy = load_healthy_subject(healthy_subjects[hsid])
        h_preds, h_gt = {}, None
        for name in MODEL_NAMES:
            pred_deg, gt_deg = predict_next_tick(models[name], df_healthy)
            h_preds[name] = pred_deg
            if h_gt is None and gt_deg is not None:
                h_gt = gt_deg
        to_left_h, to_right_h = mean_toe_off(df_healthy)
        print(f"Measured toe-off: Left {to_left_h:.1f}%  Right {to_right_h:.1f}%")
        plot_normalized_cycle_with_phases(
            h_preds, h_gt, f"Healthy_{hsid}",
            toe_off_left=to_left_h, toe_off_right=to_right_h,
            save_dir=save_dir, show=not args.no_save,
        )


if __name__ == "__main__":
    main()