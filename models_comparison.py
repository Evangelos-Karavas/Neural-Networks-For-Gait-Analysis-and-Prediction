#!/usr/bin/env python3
"""
models_comparison.py  (Timestamp vs Phase-Variable, 1-to-1)

What it does:
- Loads BOTH trained models + their scalers:
    * Timestamp rolling next-tick model (angles-only)
    * PV rolling next-tick model (pv+angles)
- Builds a CP dataset the SAME way as your PV script:
    * reads CP per file
    * computes PV per stride from hip angle + foot-off %
    * applies right-leg half-stride shift stridewise (angles + PV_Right)
- Builds rolling windows for BOTH networks from the SAME cp_df stream:
    * Timestamp: (51,6)->(6)
    * PV:        (51,8)->(6)
- Produces:
    1) Single-stride, one-step-ahead figure (like your screenshot) for Timestamp
    2) Single-stride, one-step-ahead figure (like your screenshot) for PV
    3) Segment plots: GT vs Timestamp vs PV (6 joints)
    4) Segment plots: |error| Timestamp vs PV (6 joints)
    5) Segment plots: residual histograms for both
    6) Segment plots: mean |error| vs phase (binned by PV_Left)
- Saves Excel outputs in Predictions/

Edit paths at the top if your filenames differ.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model


# ============================================================
# SETTINGS (edit paths if needed)
# ============================================================
STRIDE_LEN = 51
WINDOW = 51
MAX_CP_FILES = 500

CP_FOLDER = "Data_CP/"
CP_SHEET = "Data"
CP_SKIPROWS = [1, 2]

# --- Timestamp artifacts ---
TIMESTAMP_MODEL_PATH = "Saved_Models/Timestamp_lstm_model.keras"
TIMESTAMP_SCALER_PATH = "Scaler/standard_scaler_typical_lstm.save"  # angles scaler (fit on typical)

# --- PV artifacts ---
PV_MODEL_PATH = "Saved_Models/PV_rolling_next_tick_lstm_final.keras"
PV_SCALER_PV_PATH = "Scaler/scaler_pv.save"
PV_SCALER_ANGLES_PATH = "Scaler/scaler_angles.save"

PRED_DIR = "Predictions"
os.makedirs(PRED_DIR, exist_ok=True)

# --- columns ---
ANGLE_COLS = [
    "LHipAngles (1)", "LKneeAngles (1)", "LAnkleAngles (1)",
    "RHipAngles (1)", "RKneeAngles (1)", "RAnkleAngles (1)"
]
RIGHT_ANGLE_COLS = ["RHipAngles (1)", "RKneeAngles (1)", "RAnkleAngles (1)"]

LHIP_COL = "LHipAngles (1)"
RHIP_COL = "RHipAngles (1)"
LFO_COL = "Left Foot Off"
RFO_COL = "Right Foot Off"

PV_COLS = ["PhaseVariable_Left", "PhaseVariable_Right"]

JOINT_LABELS = ["L Hip", "L Knee", "L Ankle", "R Hip", "R Knee", "R Ankle"]
JOINT_TITLES = [
    "Hips Flexion-Extension Left",
    "Knees Flexion-Extension Left",
    "Ankles Dorsiflexion-Plantarflexion Left",
    "Hips Flexion-Extension Right",
    "Knees Flexion-Extension Right",
    "Ankles Dorsiflexion-Plantarflexion Right",
]


# ============================================================
# Helpers: stridewise right-leg half-stride shift
# ============================================================
def half_stride_shift(stride_len: int) -> int:
    return int(stride_len // 2)  # 51 -> 25

def roll_stridewise_1d(x: np.ndarray, stride_len: int, shift: int) -> np.ndarray:
    """
    Circularly roll a 1D array within each stride block (no mixing across strides).
    """
    x = np.asarray(x)
    n_strides = len(x) // stride_len
    x = x[:n_strides * stride_len].copy()
    out = x.copy()
    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len
        out[a:b] = np.roll(x[a:b], shift)
    return out

def apply_right_leg_half_stride_offset(df: pd.DataFrame, stride_len: int, right_angle_cols, pv_right_col: str = None) -> pd.DataFrame:
    shift = half_stride_shift(stride_len)
    out = df.copy()
    for col in right_angle_cols:
        out[col] = roll_stridewise_1d(out[col].to_numpy(), stride_len, shift)
    if pv_right_col is not None and pv_right_col in out.columns:
        out[pv_right_col] = roll_stridewise_1d(out[pv_right_col].to_numpy(), stride_len, shift)
    return out


# ============================================================
# Phase variable computation (same as your PV script)
# ============================================================
def compute_pv_stride(q: np.ndarray, c: float, enforce_monotonic: bool = True) -> np.ndarray:
    """
    q: hip angle over one stride, shape (N,)
    c: stance fraction in [0,1], from Foot-Off% / 100
    Returns PV s in [0,1).
    """
    q = q.astype(np.float64)
    N = q.shape[0]

    c = float(np.clip(c, 0.05, 0.95))
    q0 = float(q[0])
    idx_min = int(np.argmin(q))
    qmin = float(q[idx_min])

    denom = (q0 - qmin)
    if abs(denom) < 1e-9:
        s = np.linspace(0.0, 1.0, N, endpoint=False, dtype=np.float32)
        return np.clip(s, 0.0, 1.0 - 1e-6)

    s = np.zeros(N, dtype=np.float64)

    # stance portion
    s[:idx_min + 1] = ((q0 - q[:idx_min + 1]) / denom) * c

    # swing portion
    s[idx_min:] = 1.0 + ((1.0 - c) / denom) * (q[idx_min:] - q0)

    s = np.clip(s, 0.0, 1.0)
    if enforce_monotonic:
        s = np.maximum.accumulate(s)

    return np.clip(s.astype(np.float32), 0.0, 1.0 - 1e-6)

def compute_phase_variables(df: pd.DataFrame, stride_len: int,
                           lhip_col: str, rhip_col: str,
                           lfo_col: str, rfo_col: str,
                           enforce_monotonic: bool = True) -> pd.DataFrame:
    n_strides = len(df) // stride_len
    out = df.iloc[:n_strides * stride_len].copy()

    pvL = np.zeros(len(out), dtype=np.float32)
    pvR = np.zeros(len(out), dtype=np.float32)

    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len

        cL = float(out[lfo_col].iloc[a]) / 100.0
        cR = float(out[rfo_col].iloc[a]) / 100.0

        qL = out[lhip_col].values[a:b]
        qR = out[rhip_col].values[a:b]

        pvL[a:b] = compute_pv_stride(qL, c=cL, enforce_monotonic=enforce_monotonic)
        pvR[a:b] = compute_pv_stride(qR, c=cR, enforce_monotonic=enforce_monotonic)

    out["PhaseVariable_Left"] = pvL
    out["PhaseVariable_Right"] = pvR
    return out


# ============================================================
# Rolling windows (next-tick)
# ============================================================
def make_rolling_windows(features: np.ndarray, targets: np.ndarray, window: int):
    """
    features: (T,D)
    targets:  (T,K)
    returns:
      X: (T-window, window, D)
      y: (T-window, K) where y[i] corresponds to targets[i+window]
    """
    T = features.shape[0]
    if T <= window:
        raise ValueError(f"Not enough timesteps ({T}) for window={window}")

    D = features.shape[1]
    K = targets.shape[1]

    X = np.zeros((T - window, window, D), dtype=np.float32)
    y = np.zeros((T - window, K), dtype=np.float32)

    for i in range(T - window):
        X[i] = features[i:i + window]
        y[i] = targets[i + window]

    return X, y


def rollout_from_dataset_next_tick(model, X_full, y_full, start_i, horizon):
    """
    Autoregressive rollout in the SAME scaled space as X_full/y_full.

    X_full: (N, WINDOW, D)  dataset windows
    y_full: (N, K)          dataset targets (next tick)
    start_i: starting window index
    horizon: number of predicted steps H

    Returns:
      pred_scaled: (H, K)
      gt_scaled:   (H, K)
      input_window_scaled: (WINDOW, D)  (the initial input window)
    """
    N = len(X_full)
    if start_i < 0 or start_i + horizon >= N:
        raise ValueError(f"start_i={start_i} too close to end for horizon={horizon}. N={N}")

    window = X_full[start_i].copy()   # (WINDOW,D)
    D = window.shape[1]
    K = y_full.shape[1]

    preds = np.zeros((horizon, K), dtype=np.float32)
    gts   = np.zeros((horizon, K), dtype=np.float32)

    for h in range(horizon):
        # predict next tick
        yhat = model.predict(window.reshape(1, window.shape[0], D), verbose=0)[0]  # (K,)
        preds[h] = yhat
        gts[h] = y_full[start_i + h]  # ground truth for this step (aligned)

        # build next window:
        # shift up by 1, append "next features" from dataset where possible
        # For Timestamp (D=6), next features == predicted angles.
        # For PV (D=8), next features are [PV_L, PV_R, angles...].
        # We DO NOT predict PV; we take PV from dataset for correct conditioning.
        if D == 6:
            next_feat = yhat  # angles-only
        else:
            # D==8: first 2 are PV, last 6 are angles.
            # Take PV from the dataset at the NEXT window's last timestep:
            # Equivalent to dataset's "true PV at time (start_i+h+WINDOW)".
            next_pv = X_full[start_i + h, -1, :2]  # already scaled PV
            next_feat = np.concatenate([next_pv, yhat], axis=0)  # (8,)

        window = np.vstack([window[1:], next_feat])

    return preds, gts, X_full[start_i]



# ============================================================
# Plot helpers
# ============================================================
def plot_one_stride_one_step(stride_in_deg_51x6: np.ndarray,
                             pred_next_deg_6: np.ndarray,
                             gt_next_deg_6: np.ndarray,
                             title_prefix: str = ""):
    """
    Your screenshot style:
      - blue line: input stride (51 samples)
      - green x: one-step prediction at sample 51
      - red x: actual at sample 51
    """
    stride = np.asarray(stride_in_deg_51x6)
    pred = np.asarray(pred_next_deg_6).reshape(-1)
    gt   = np.asarray(gt_next_deg_6).reshape(-1)

    if stride.shape != (51, 6):
        raise ValueError(f"stride shape must be (51,6), got {stride.shape}")
    if pred.shape != (6,) or gt.shape != (6,):
        raise ValueError(f"pred/gt must be (6,), got pred={pred.shape} gt={gt.shape}")

    x_in = np.arange(51)
    x_next = 51

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=False)
    axes = axes.flatten()

    # Layout: left column (L hip/knee/ankle), right column (R hip/knee/ankle)
    placement = [0, 3, 1, 4, 2, 5]  # (LHip,RHip,LKnee,RKnee,LAnk,RAnk) displayed row-wise

    for plot_i, joint_i in enumerate(placement):
        ax = axes[plot_i]
        ax.plot(x_in, stride[:, joint_i], linewidth=2, label="input")
        ax.plot([x_next], [pred[joint_i]], marker="x", markersize=8, linestyle="None",
                label="one-step-ahead prediction")
        ax.plot([x_next], [gt[joint_i]], marker="x", markersize=8, linestyle="None",
                label="actual")

        ax.set_title(JOINT_TITLES[joint_i], fontsize=10)
        ax.set_xlabel("Time-step")
        ax.set_ylabel("Angle (degrees)")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)

    fig.suptitle(f"{title_prefix} (1 stride input → 1-step ahead)", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()

def plot_rollout_one_step_ahead(
    input_window_deg: np.ndarray,     # (WINDOW,6)
    pred_series_deg: np.ndarray,      # (H,6)
    gt_series_deg: np.ndarray,        # (H,6)
    title_prefix: str = "",
):
    """
    Reproduces the style:
      - Blue line: input window (t=0..WINDOW-1)
      - Green x: one-step-ahead predictions for the rollout horizon (t=WINDOW..WINDOW+H-1)
      - Red line: actual (ground truth) over the same horizon

    All arrays are in degrees.
    """
    inp = np.asarray(input_window_deg)
    pred = np.asarray(pred_series_deg)
    gt = np.asarray(gt_series_deg)

    if inp.shape[1] != 6 or pred.shape[1] != 6 or gt.shape[1] != 6:
        raise ValueError("Expected 6 joints in last dimension.")
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have same shape. pred={pred.shape}, gt={gt.shape}")

    W = inp.shape[0]
    H = pred.shape[0]

    x_in = np.arange(W)                 # 0..W-1
    x_out = np.arange(W, W + H)         # W..W+H-1

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=False)
    axes = axes.flatten()

    # Layout: left column = left joints, right column = right joints, row-wise
    placement = [0, 3, 1, 4, 2, 5]

    for plot_i, joint_i in enumerate(placement):
        ax = axes[plot_i]

        # Input window (blue)
        ax.plot(x_in, inp[:, joint_i], linewidth=2, label="input")

        # Actual over horizon (red line)
        ax.plot(x_out, gt[:, joint_i], linewidth=2, label="actual")

        # One-step predictions over horizon (green x)
        ax.plot(x_out, pred[:, joint_i], linestyle="None", marker="x", markersize=6,
                label="one-step-ahead predictions")

        ax.set_title(JOINT_TITLES[joint_i], fontsize=10)
        ax.set_xlabel("Time-step")
        ax.set_ylabel("Angle (degrees)")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)

    fig.suptitle(f"{title_prefix} (rolling one-step-ahead rollout)", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()

def plot_gt_ts_pv(gt_deg_2d, ts_deg_2d, pv_deg_2d, title="GT vs Timestamp vs PV"):
    gt = np.asarray(gt_deg_2d)
    ts = np.asarray(ts_deg_2d)
    pv = np.asarray(pv_deg_2d)
    if gt.ndim != 2 or ts.ndim != 2 or pv.ndim != 2:
        raise ValueError(f"Segment plots require 2D arrays (T,6). Got gt={gt.shape}, ts={ts.shape}, pv={pv.shape}")

    t = np.arange(gt.shape[0])
    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, gt[:, j], label="GT")
        ax.plot(t, ts[:, j], label="Timestamp")
        ax.plot(t, pv[:, j], label="PV")
        ax.set_ylabel(JOINT_LABELS[j])
        ax.grid(True)
        if j == 0:
            ax.set_title(title)
            ax.legend()
    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.show()

def plot_abs_error(gt_deg_2d, ts_deg_2d, pv_deg_2d, title="|Error| Timestamp vs PV"):
    gt = np.asarray(gt_deg_2d)
    ts = np.asarray(ts_deg_2d)
    pv = np.asarray(pv_deg_2d)
    if gt.ndim != 2 or ts.ndim != 2 or pv.ndim != 2:
        raise ValueError(f"Abs-error plots require 2D arrays (T,6). Got gt={gt.shape}, ts={ts.shape}, pv={pv.shape}")

    t = np.arange(gt.shape[0])
    err_ts = np.abs(ts - gt)
    err_pv = np.abs(pv - gt)

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, err_ts[:, j], label="|Timestamp - GT|")
        ax.plot(t, err_pv[:, j], label="|PV - GT|")
        ax.set_ylabel(f"{JOINT_LABELS[j]}\n|err| (deg)")
        ax.grid(True)
        if j == 0:
            ax.set_title(title)
            ax.legend()
    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.show()

def plot_residual_hist(ts_deg_2d, pv_deg_2d, gt_deg_2d, title="Residual histograms (Pred - GT)"):
    gt = np.asarray(gt_deg_2d)
    ts = np.asarray(ts_deg_2d)
    pv = np.asarray(pv_deg_2d)
    if gt.ndim != 2 or ts.ndim != 2 or pv.ndim != 2:
        raise ValueError(f"Residual plots require 2D arrays (T,6). Got gt={gt.shape}, ts={ts.shape}, pv={pv.shape}")

    ts_err = ts - gt
    pv_err = pv - gt

    fig, axes = plt.subplots(6, 2, figsize=(14, 14))
    for j in range(6):
        axes[j, 0].hist(ts_err[:, j], bins=60)
        axes[j, 0].set_title(f"{JOINT_LABELS[j]}: Timestamp")
        axes[j, 0].grid(True)

        axes[j, 1].hist(pv_err[:, j], bins=60)
        axes[j, 1].set_title(f"{JOINT_LABELS[j]}: PV")
        axes[j, 1].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def phase_binned_mean_abs_err(pv_0to1, abs_err_2d, nbins=20):
    pv = np.clip(np.asarray(pv_0to1).reshape(-1), 0.0, 0.999999)
    abs_err = np.asarray(abs_err_2d)
    out = np.zeros((nbins, abs_err.shape[1]), dtype=np.float64)
    cnt = np.zeros((nbins,), dtype=np.int64)

    bins = np.floor(pv * nbins).astype(int)
    for i, b in enumerate(bins):
        out[b] += abs_err[i]
        cnt[b] += 1

    for b in range(nbins):
        out[b] = out[b] / cnt[b] if cnt[b] > 0 else np.nan
    return out

def plot_error_vs_phase(pv_phase_1d, gt_deg_2d, ts_deg_2d, pv_deg_2d, nbins=20, title="Mean |error| vs phase"):
    gt = np.asarray(gt_deg_2d)
    ts = np.asarray(ts_deg_2d)
    pv = np.asarray(pv_deg_2d)
    if gt.ndim != 2 or ts.ndim != 2 or pv.ndim != 2:
        raise ValueError(f"Phase plots require 2D arrays (T,6). Got gt={gt.shape}, ts={ts.shape}, pv={pv.shape}")

    err_ts = np.abs(ts - gt)
    err_pv = np.abs(pv - gt)

    b_ts = phase_binned_mean_abs_err(pv_phase_1d, err_ts, nbins=nbins)
    b_pv = phase_binned_mean_abs_err(pv_phase_1d, err_pv, nbins=nbins)
    x = (np.arange(nbins) + 0.5) / nbins

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(x, b_ts[:, j], label="Timestamp")
        ax.plot(x, b_pv[:, j], label="PV")
        ax.set_ylabel(f"{JOINT_LABELS[j]}\n|err| (deg)")
        ax.grid(True)
        if j == 0:
            ax.set_title(title)
            ax.legend()
    axes[-1].set_xlabel("Phase (0 → 1)")
    plt.tight_layout()
    plt.show()

def print_metrics(gt_deg_2d, pred_deg_2d, name):
    gt = np.asarray(gt_deg_2d)
    pr = np.asarray(pred_deg_2d)
    err = pr - gt
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    print(f"\n=== {name} metrics (deg) ===")
    for j in range(6):
        print(f"{JOINT_LABELS[j]:6s}  MAE={mae[j]:6.2f}  RMSE={rmse[j]:6.2f}")
    print("===========================")


# ============================================================
# Load CP dataset (same structure as PV pipeline, per file)
# ============================================================
def load_cp_dataframe_with_pv(cp_folder: str, max_files: int):
    if not os.path.isdir(cp_folder):
        raise FileNotFoundError(f"CP folder not found: {cp_folder}")

    frames = []
    count = 0

    for fn in sorted(os.listdir(cp_folder)):
        if not fn.endswith(".xlsx"):
            continue
        fp = os.path.join(cp_folder, fn)
        try:
            df = pd.read_excel(
                fp, sheet_name=CP_SHEET,
                usecols=ANGLE_COLS + [LFO_COL, RFO_COL],
                skiprows=CP_SKIPROWS
            ).fillna(0)
        except Exception as e:
            print(f"Skipping {fp} (read error): {e}")
            continue

        df = compute_phase_variables(df, STRIDE_LEN, LHIP_COL, RHIP_COL, LFO_COL, RFO_COL, enforce_monotonic=True)
        df = apply_right_leg_half_stride_offset(df, STRIDE_LEN, RIGHT_ANGLE_COLS, pv_right_col="PhaseVariable_Right")

        frames.append(df)
        count += 1
        if count >= max_files:
            break

    if not frames:
        raise RuntimeError("No CP files loaded.")
    return pd.concat(frames, ignore_index=True).fillna(0)


# ============================================================
# MAIN
# ============================================================
def main():
    # ---- Load models + scalers ----
    if not os.path.exists(TIMESTAMP_MODEL_PATH):
        raise FileNotFoundError(f"Timestamp model not found: {TIMESTAMP_MODEL_PATH}")
    if not os.path.exists(PV_MODEL_PATH):
        raise FileNotFoundError(f"PV model not found: {PV_MODEL_PATH}")

    ts_model = load_model(TIMESTAMP_MODEL_PATH)
    pv_model = load_model(PV_MODEL_PATH)

    ts_scaler = joblib.load(TIMESTAMP_SCALER_PATH)
    pv_scaler = joblib.load(PV_SCALER_PV_PATH)
    pv_ang_scaler = joblib.load(PV_SCALER_ANGLES_PATH)

    # ---- Build CP dataframe with PV + right-shift applied ----
    cp_df = load_cp_dataframe_with_pv(CP_FOLDER, MAX_CP_FILES)

    # ---- Prepare timestamp windows (angles only) ----
    cp_angles_deg = cp_df[ANGLE_COLS].to_numpy()  # in degrees
    cp_angles_scaled_ts = ts_scaler.transform(cp_angles_deg)

    X_ts, y_ts = make_rolling_windows(cp_angles_scaled_ts, cp_angles_scaled_ts, window=WINDOW)

    # ---- Prepare PV windows (pv + angles) ----
    cp_pv = cp_df[PV_COLS].to_numpy()
    cp_pv_scaled = pv_scaler.transform(cp_pv)

    cp_angles_scaled_pv = pv_ang_scaler.transform(cp_angles_deg)
    cp_feat_scaled_pv = np.concatenate([cp_pv_scaled, cp_angles_scaled_pv], axis=1)  # (T,8)

    X_pv, y_pv = make_rolling_windows(cp_feat_scaled_pv, cp_angles_scaled_pv, window=WINDOW)

    # ---- Align lengths ----
    N = min(len(X_ts), len(X_pv))
    X_ts, y_ts = X_ts[:N], y_ts[:N]
    X_pv, y_pv = X_pv[:N], y_pv[:N]

    # ============================================================
    # (SEGMENT EVALUATION + PLOTS
    # ============================================================
    T_plot = 600
    start = max(0, N - T_plot)

    X_ts_seg, y_ts_seg = X_ts[start:start + T_plot], y_ts[start:start + T_plot]
    X_pv_seg, y_pv_seg = X_pv[start:start + T_plot], y_pv[start:start + T_plot]

    pred_ts_seg_scaled = ts_model.predict(X_ts_seg, verbose=0)   # (T_plot,6)
    pred_pv_seg_scaled = pv_model.predict(X_pv_seg, verbose=0)   # (T_plot,6)

    gt_seg_deg = pv_ang_scaler.inverse_transform(y_pv_seg)       # (T_plot,6) degrees
    pred_ts_seg_deg = ts_scaler.inverse_transform(pred_ts_seg_scaled)
    pred_pv_seg_deg = pv_ang_scaler.inverse_transform(pred_pv_seg_scaled)

    print_metrics(gt_seg_deg, pred_ts_seg_deg, "Timestamp")
    print_metrics(gt_seg_deg, pred_pv_seg_deg, "PV")

    plot_gt_ts_pv(gt_seg_deg, pred_ts_seg_deg, pred_pv_seg_deg, title="GT vs Timestamp vs PV (CP segment)")
    plot_abs_error(gt_seg_deg, pred_ts_seg_deg, pred_pv_seg_deg, title="Absolute error comparison (CP segment)")
    plot_residual_hist(pred_ts_seg_deg, pred_pv_seg_deg, gt_seg_deg, title="Residual histograms (Pred - GT)")

    # Phase for each tick corresponds to last timestep of each PV window in this segment
    pv_last_scaled = X_pv_seg[:, -1, :2]  # (T_plot,2)
    pv_last = pv_scaler.inverse_transform(pv_last_scaled)
    pv_left = pv_last[:, 0]

    plot_error_vs_phase(pv_left, gt_seg_deg, pred_ts_seg_deg, pred_pv_seg_deg, nbins=20, title="Mean |error| vs phase (binned by PV_Left)")

    # Save segment table
    seg_out = pd.DataFrame(np.hstack([gt_seg_deg, pred_ts_seg_deg, pred_pv_seg_deg]),columns=[f"GT_{c}" for c in ANGLE_COLS] + [f"TS_{c}" for c in ANGLE_COLS] + [f"PV_{c}" for c in ANGLE_COLS])
    seg_file = os.path.join(PRED_DIR, "compare_ts_vs_pv_cp_segment.xlsx")
    seg_out.to_excel(seg_file, index=False)
    print(f"\nSaved segment comparison to: {seg_file}")

    # ============================================================
    # SINGLE-STRIDE ONE-STEP-AHEAD
    # ============================================================
    k = 5  # stride index (change as you like)
    a = k * STRIDE_LEN
    b = a + STRIDE_LEN

    if b >= len(cp_df):
        raise ValueError(f"Stride k={k} too close to end of cp_df (b={b} >= {len(cp_df)})")
    if a >= N:
        raise ValueError(f"Stride k={k} too large for rolling window arrays (a={a} >= N={N})")

    # Input stride (degrees)
    stride_in_deg = cp_angles_deg[a:b]             # (51,6)
    gt_next_deg = cp_angles_deg[b]                # (6,) the next sample after the stride

    i = a  # window starts at a, predicts at a+51 = b

    # Timestamp one-step prediction
    pred_ts_one_scaled = ts_model.predict(X_ts[i:i+1], verbose=0)[0]
    pred_ts_one_deg = ts_scaler.inverse_transform(pred_ts_one_scaled.reshape(1, -1))[0]

    # PV one-step prediction
    pred_pv_one_scaled = pv_model.predict(X_pv[i:i+1], verbose=0)[0]
    pred_pv_one_deg = pv_ang_scaler.inverse_transform(pred_pv_one_scaled.reshape(1, -1))[0]

    plot_one_stride_one_step(stride_in_deg, pred_ts_one_deg, gt_next_deg, title_prefix="Timestamp model")
    plot_one_stride_one_step(stride_in_deg, pred_pv_one_deg, gt_next_deg, title_prefix="Phase-variable model")

    # Save the single-stride point for documentation
    one_df = pd.DataFrame( np.vstack([gt_next_deg, pred_ts_one_deg, pred_pv_one_deg]), index=["GT_next", "Timestamp_next", "PV_next"], columns=ANGLE_COLS)
    one_file = os.path.join(PRED_DIR, f"compare_single_stride_k{k}_next_tick.xlsx")
    one_df.to_excel(one_file)
    print(f"Saved single-stride next-tick table to: {one_file}")


    # Choose a rollout horizon (e.g., 250 samples ~ about 5 strides)
    H = 250

    # Pick a stride-aligned start to make the plot nice
    k = 10
    start_time = k * STRIDE_LEN
    start_i = start_time  # because WINDOW==51 and windows are built starting at each timestep

    # --- Timestamp rollout (scaled) ---
    pred_ts_scaled, gt_ts_scaled, input_ts_scaled = rollout_from_dataset_next_tick(
        ts_model, X_ts, y_ts, start_i=start_i, horizon=H
    )

    # Convert to degrees for plotting
    input_ts_deg = ts_scaler.inverse_transform(input_ts_scaled[:, :6])  # (51,6)
    pred_ts_deg = ts_scaler.inverse_transform(pred_ts_scaled)
    gt_ts_deg   = ts_scaler.inverse_transform(gt_ts_scaled)

    plot_rollout_one_step_ahead(
        input_window_deg=input_ts_deg,
        pred_series_deg=pred_ts_deg,
        gt_series_deg=gt_ts_deg,
        title_prefix="Timestamp model"
    )

    # --- PV rollout (scaled) ---
    pred_pv_scaled, gt_pv_scaled, input_pv_scaled = rollout_from_dataset_next_tick(
        pv_model, X_pv, y_pv, start_i=start_i, horizon=H
    )

    # For PV input window: last 6 dims are angles
    input_pv_deg = pv_ang_scaler.inverse_transform(input_pv_scaled[:, 2:])  # (51,6)
    pred_pv_deg  = pv_ang_scaler.inverse_transform(pred_pv_scaled)
    gt_pv_deg    = pv_ang_scaler.inverse_transform(gt_pv_scaled)

    plot_rollout_one_step_ahead(
        input_window_deg=input_pv_deg,
        pred_series_deg=pred_pv_deg,
        gt_series_deg=gt_pv_deg,
        title_prefix="Phase-variable model"
    )

if __name__ == "__main__":
    main()
