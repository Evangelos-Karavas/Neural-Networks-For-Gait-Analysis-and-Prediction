#!/usr/bin/env python3
"""
models_comparison_4models.py
(Timestamp-LSTM vs PV-LSTM vs Timestamp-CNN vs PV-CNN) using a CP stream.

All comparisons are:
  input windows Timestamp: (51 x 6) | PV: (51 x 8) -> predict next angles (6)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model


# ============================================================
# SETTINGS
# ============================================================
STRIDE_LEN = 51
WINDOW = 51
MAX_CP_FILES = 500

CP_FOLDER = "Data_CP/"
CP_SHEET = "Data"
CP_SKIPROWS = [1, 2]

PRED_DIR = "Predictions"
os.makedirs(PRED_DIR, exist_ok=True)

# -----------------------------
# LSTM artifacts
# -----------------------------
TIMESTAMP_LSTM_MODEL_PATH = "Saved_Models/Timestamp_lstm_model.keras"
TIMESTAMP_LSTM_SCALER_PATH = "Scaler/standard_scaler_typical_lstm.save"

TIMESTAMP_CNN_MODEL_PATH = "Saved_Models/Timestamp_cnn_model.keras"
TIMESTAMP_CNN_SCALER_PATH = "Scaler/standard_scaler_typical_cnn.save"

PV_LSTM_MODEL_PATH = "Saved_Models\PV_rolling_next_tick_lstm.keras"
PV_LSTM_SCALER_PV_PATH = "Scaler/scaler_pv.save"
PV_LSTM_SCALER_ANGLES_PATH = "Scaler/scaler_angles.save"


PV_CNN_MODEL_PATH = "Saved_Models/PV_rolling_next_tick_cnn.keras"
PV_CNN_SCALER_PV_PATH = "Scaler/scaler_pv_cnn.save"
PV_CNN_SCALER_ANGLES_PATH = "Scaler/scaler_angles_cnn.save"

# -----------------------------
# columns
# -----------------------------
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
# Phase variable computation
# ============================================================
def compute_pv_stride(q: np.ndarray, c: float, enforce_monotonic: bool = True) -> np.ndarray:
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

    # stance
    s[:idx_min + 1] = ((q0 - q[:idx_min + 1]) / denom) * c
    # swing
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
    X[i] = features[i:i+window]
    y[i] = targets[i+window]
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


# ============================================================
# Model output adapter: return NEXT-TICK prediction (N,6) in scaled space
# ============================================================
def predict_next_tick_scaled(model, X):
    """
    Handles:
      - next-tick models: (N,6)
      - stride->stride models: (N,51,6)  -> take [:,0,:] as next tick
    """
    pred = model.predict(X, verbose=0)
    pred = np.asarray(pred)

    if pred.ndim == 2 and pred.shape[1] == 6:
        return pred.astype(np.float32)

    if pred.ndim == 3 and pred.shape[1] == WINDOW and pred.shape[2] == 6:
        # stride->next-stride: next tick corresponds to first sample of predicted next stride
        return pred[:, 0, :].astype(np.float32)

    raise ValueError(f"Unsupported model output shape: {pred.shape}. Expected (N,6) or (N,{WINDOW},6).")


def rollout_from_dataset_next_tick_any(model, X_full, y_full, start_i, horizon):
    """
    Autoregressive rollout.
      - next-tick models (output 6)
      - stride->stride models (output 51x6, we take first sample)
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
        yhat = predict_next_tick_scaled(model, window.reshape(1, WINDOW, D))[0]  # (6,)
        preds[h] = yhat
        gts[h]   = y_full[start_i + h]

        if D == 6:
            next_feat = yhat
        else:
            next_pv = X_full[start_i + h, -1, :2]
            next_feat = np.concatenate([next_pv, yhat], axis=0)

        window = np.vstack([window[1:], next_feat])

    return preds, gts, X_full[start_i]


# ============================================================
# Plot helpers (existing + 4-model variants)
# ============================================================
def plot_one_stride_one_step(stride_in_deg_51x6, pred_next_deg_6, gt_next_deg_6, title_prefix=""):
    stride = np.asarray(stride_in_deg_51x6)
    pred = np.asarray(pred_next_deg_6).reshape(-1)
    gt   = np.asarray(gt_next_deg_6).reshape(-1)

    x_in = np.arange(51)
    x_next = 51

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=False)
    axes = axes.flatten()
    placement = [0, 3, 1, 4, 2, 5]

    for plot_i, joint_i in enumerate(placement):
        ax = axes[plot_i]
        ax.plot(x_in, stride[:, joint_i], linewidth=2, label="input")
        ax.plot([x_next], [pred[joint_i]], marker="x", markersize=8, linestyle="None", label="prediction")
        ax.plot([x_next], [gt[joint_i]], marker="x", markersize=8, linestyle="None", label="actual")
        ax.set_title(JOINT_TITLES[joint_i], fontsize=10)
        ax.set_xlabel("Time-step")
        ax.set_ylabel("Angle (deg)")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.suptitle(f"{title_prefix} (1 stride input → 1-step ahead)", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()

def plot_rollout_one_step_ahead(input_window_deg, pred_series_deg, gt_series_deg, title_prefix=""):
    inp = np.asarray(input_window_deg)
    pred = np.asarray(pred_series_deg)
    gt = np.asarray(gt_series_deg)

    W = inp.shape[0]
    H = pred.shape[0]
    x_in = np.arange(W)
    x_out = np.arange(W, W + H)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=False)
    axes = axes.flatten()
    placement = [0, 3, 1, 4, 2, 5]

    for plot_i, joint_i in enumerate(placement):
        ax = axes[plot_i]
        ax.plot(x_in, inp[:, joint_i], linewidth=2, label="input")
        ax.plot(x_out, gt[:, joint_i], linewidth=2, label="actual")
        ax.plot(x_out, pred[:, joint_i], linestyle="None", marker="x", markersize=6, label="predictions")
        ax.set_title(JOINT_TITLES[joint_i], fontsize=10)
        ax.set_xlabel("Time-step")
        ax.set_ylabel("Angle (deg)")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.suptitle(f"{title_prefix} (rolling one-step-ahead rollout)", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()

def plot_gt_multi(gt_deg_2d, preds_dict_deg_2d, title="Ground Truth vs Models"):
    gt = np.asarray(gt_deg_2d)
    t = np.arange(gt.shape[0])

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, gt[:, j], label="Ground Truth")
        for name, pr in preds_dict_deg_2d.items():
            ax.plot(t, np.asarray(pr)[:, j], label=name)
        ax.set_ylabel(JOINT_LABELS[j])
        ax.grid(True)
        if j == 0:
            ax.set_title(title)
            ax.legend()
    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.show()

def plot_abs_error_multi(gt_deg_2d, preds_dict_deg_2d, title="|Error| vs time"):
    gt = np.asarray(gt_deg_2d)
    t = np.arange(gt.shape[0])

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    for j, ax in enumerate(axes):
        for name, pr in preds_dict_deg_2d.items():
            pr = np.asarray(pr)
            ax.plot(t, np.abs(pr[:, j] - gt[:, j]), label=name)
        ax.set_ylabel(f"{JOINT_LABELS[j]}\n|err| (deg)")
        ax.grid(True)
        if j == 0:
            ax.set_title(title)
            ax.legend()
    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.show()

def plot_residual_hist_multi(gt_deg_2d, preds_dict_deg_2d, title="Residual histograms (Pred - Ground Truth)"):
    gt = np.asarray(gt_deg_2d)
    names = list(preds_dict_deg_2d.keys())
    n_models = len(names)

    fig, axes = plt.subplots(6, n_models, figsize=(4*n_models + 2, 14))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=1)

    for j in range(6):
        for m, name in enumerate(names):
            pr = np.asarray(preds_dict_deg_2d[name])
            resid = pr[:, j] - gt[:, j]
            ax = axes[j, m]
            ax.hist(resid, bins=60)
            ax.set_title(f"{JOINT_LABELS[j]}: {name}")
            ax.grid(True)

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

def plot_error_vs_phase_multi(pv_phase_1d, gt_deg_2d, preds_dict_deg_2d, nbins=20, title="Mean |error| vs phase"):
    gt = np.asarray(gt_deg_2d)
    x = (np.arange(nbins) + 0.5) / nbins

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    for j, ax in enumerate(axes):
        for name, pr in preds_dict_deg_2d.items():
            pr = np.asarray(pr)
            err = np.abs(pr - gt)
            b = phase_binned_mean_abs_err(pv_phase_1d, err, nbins=nbins)
            ax.plot(x, b[:, j], label=name)
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
# Load CP dataset (PV computed per file)
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
    # ---- Check files exist ----
    paths = [
        TIMESTAMP_LSTM_MODEL_PATH, PV_LSTM_MODEL_PATH,
        TIMESTAMP_LSTM_SCALER_PATH, PV_LSTM_SCALER_PV_PATH, PV_LSTM_SCALER_ANGLES_PATH,
        TIMESTAMP_CNN_MODEL_PATH, TIMESTAMP_CNN_SCALER_PATH,
        PV_CNN_MODEL_PATH, PV_CNN_SCALER_PV_PATH, PV_CNN_SCALER_ANGLES_PATH
    ]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    # ---- Load models ----
    ts_lstm = load_model(TIMESTAMP_LSTM_MODEL_PATH)
    pv_lstm = load_model(PV_LSTM_MODEL_PATH)
    ts_cnn  = load_model(TIMESTAMP_CNN_MODEL_PATH)
    pv_cnn  = load_model(PV_CNN_MODEL_PATH)

    # ---- Load scalers ----
    ts_lstm_scaler = joblib.load(TIMESTAMP_LSTM_SCALER_PATH)

    pv_lstm_scaler_pv    = joblib.load(PV_LSTM_SCALER_PV_PATH)
    pv_lstm_scaler_angles= joblib.load(PV_LSTM_SCALER_ANGLES_PATH)

    ts_cnn_scaler = joblib.load(TIMESTAMP_CNN_SCALER_PATH)

    pv_cnn_scaler_pv     = joblib.load(PV_CNN_SCALER_PV_PATH)
    pv_cnn_scaler_angles = joblib.load(PV_CNN_SCALER_ANGLES_PATH)

    # ---- Build CP dataframe with PV + right-shift applied ----
    cp_df = load_cp_dataframe_with_pv(CP_FOLDER, MAX_CP_FILES)

    # ---- Base angles in degrees ----
    cp_angles_deg = cp_df[ANGLE_COLS].to_numpy()

    # ---- Build windows for TS-LSTM (angles-only, scaled with ts_lstm_scaler) ----
    angles_scaled_ts_lstm = ts_lstm_scaler.transform(cp_angles_deg)
    X_ts_lstm, y_ts_lstm  = make_rolling_windows(angles_scaled_ts_lstm, angles_scaled_ts_lstm, window=WINDOW)

    # ---- Build windows for PV-LSTM (pv+angles, scaled with pv_lstm scalers) ----
    cp_pv = cp_df[PV_COLS].to_numpy()
    pv_scaled_lstm = pv_lstm_scaler_pv.transform(cp_pv)
    angles_scaled_pv_lstm = pv_lstm_scaler_angles.transform(cp_angles_deg)
    feat_scaled_pv_lstm = np.concatenate([pv_scaled_lstm, angles_scaled_pv_lstm], axis=1)  # (T,8)
    X_pv_lstm, y_pv_lstm = make_rolling_windows(feat_scaled_pv_lstm, angles_scaled_pv_lstm, window=WINDOW)

    # ---- Build windows for TS-CNN (angles-only, scaled with ts_cnn_scaler) ----
    angles_scaled_ts_cnn = ts_cnn_scaler.transform(cp_angles_deg)
    X_ts_cnn, y_ts_cnn   = make_rolling_windows(angles_scaled_ts_cnn, angles_scaled_ts_cnn, window=WINDOW)

    # ---- Build windows for PV-CNN (pv+angles, scaled with pv_cnn scalers) ----
    pv_scaled_cnn = pv_cnn_scaler_pv.transform(cp_pv)
    angles_scaled_pv_cnn = pv_cnn_scaler_angles.transform(cp_angles_deg)
    feat_scaled_pv_cnn = np.concatenate([pv_scaled_cnn, angles_scaled_pv_cnn], axis=1)  # (T,8)
    X_pv_cnn, y_pv_cnn = make_rolling_windows(feat_scaled_pv_cnn, angles_scaled_pv_cnn, window=WINDOW)

    # ---- Align lengths across all four ----
    N = min(len(X_ts_lstm), len(X_pv_lstm), len(X_ts_cnn), len(X_pv_cnn))
    X_ts_lstm, y_ts_lstm = X_ts_lstm[:N], y_ts_lstm[:N]
    X_pv_lstm, y_pv_lstm = X_pv_lstm[:N], y_pv_lstm[:N]
    X_ts_cnn,  y_ts_cnn  = X_ts_cnn[:N],  y_ts_cnn[:N]
    X_pv_cnn,  y_pv_cnn  = X_pv_cnn[:N],  y_pv_cnn[:N]

    # ============================================================
    # SEGMENT EVALUATION + PLOTS
    # ============================================================
    T_plot = 300
    start = max(0, N - T_plot)

    X_ts_lstm_seg, y_ts_lstm_seg = X_ts_lstm[start:start + T_plot], y_ts_lstm[start:start + T_plot]
    X_pv_lstm_seg, y_pv_lstm_seg = X_pv_lstm[start:start + T_plot], y_pv_lstm[start:start + T_plot]
    X_ts_cnn_seg,  y_ts_cnn_seg  = X_ts_cnn[start:start + T_plot],  y_ts_cnn[start:start + T_plot]
    X_pv_cnn_seg,  y_pv_cnn_seg  = X_pv_cnn[start:start + T_plot],  y_pv_cnn[start:start + T_plot]

    # Predict next-tick (scaled) with adapters
    pred_ts_lstm_scaled = predict_next_tick_scaled(ts_lstm, X_ts_lstm_seg)   # (T,6)
    pred_pv_lstm_scaled = predict_next_tick_scaled(pv_lstm, X_pv_lstm_seg)   # (T,6)
    pred_ts_cnn_scaled  = predict_next_tick_scaled(ts_cnn,  X_ts_cnn_seg)    # (T,6) (adapter handles (T,51,6))
    pred_pv_cnn_scaled  = predict_next_tick_scaled(pv_cnn,  X_pv_cnn_seg)    # (T,6)

    # Ground-truth degrees: use PV angle scaler (LSTM) target space for consistency
    gt_seg_deg = pv_lstm_scaler_angles.inverse_transform(y_pv_lstm_seg)

    # Convert each model prediction to degrees using its own angle scaler
    pred_ts_lstm_deg = ts_lstm_scaler.inverse_transform(pred_ts_lstm_scaled)
    pred_pv_lstm_deg = pv_lstm_scaler_angles.inverse_transform(pred_pv_lstm_scaled)
    pred_ts_cnn_deg  = ts_cnn_scaler.inverse_transform(pred_ts_cnn_scaled)
    pred_pv_cnn_deg  = pv_cnn_scaler_angles.inverse_transform(pred_pv_cnn_scaled)

    # Print metrics (all 4)
    print_metrics(gt_seg_deg, pred_ts_lstm_deg, "TS-LSTM")
    print_metrics(gt_seg_deg, pred_pv_lstm_deg, "PV-LSTM")
    print_metrics(gt_seg_deg, pred_ts_cnn_deg,  "TS-CNN")
    print_metrics(gt_seg_deg, pred_pv_cnn_deg,  "PV-CNN")

    preds_seg = {
        "TS-LSTM": pred_ts_lstm_deg,
        "PV-LSTM": pred_pv_lstm_deg,
        "TS-CNN":  pred_ts_cnn_deg,
        "PV-CNN":  pred_pv_cnn_deg,
    }

    plot_gt_multi(gt_seg_deg, preds_seg, title="Ground Truth vs 4 models (CP segment)")
    plot_abs_error_multi(gt_seg_deg, preds_seg, title="Absolute error (CP segment)")

    plot_residual_hist_multi(gt_seg_deg, preds_seg, title="Residual histograms (Pred - Ground Truth)")

    # Phase for each tick corresponds to last timestep of each PV window in this segment
    pv_last_scaled = X_pv_lstm_seg[:, -1, :2]  # (T,2) in pv_lstm scaler space
    pv_last = pv_lstm_scaler_pv.inverse_transform(pv_last_scaled)
    pv_left = pv_last[:, 0]

    plot_error_vs_phase_multi(
        pv_left, gt_seg_deg, preds_seg, nbins=20,
        title="Mean |error| vs phase (binned by PV_Left) - 4 models"
    )

    # Save segment table
    seg_out = pd.DataFrame(
        np.hstack([gt_seg_deg,
                   pred_ts_lstm_deg, pred_pv_lstm_deg,
                   pred_ts_cnn_deg,  pred_pv_cnn_deg]),
        columns=([f"GT_{c}" for c in ANGLE_COLS] +
                 [f"TSLSTM_{c}" for c in ANGLE_COLS] +
                 [f"PVLSTM_{c}" for c in ANGLE_COLS] +
                 [f"TSCNN_{c}" for c in ANGLE_COLS] +
                 [f"PVCNN_{c}" for c in ANGLE_COLS])
    )
    seg_file = os.path.join(PRED_DIR, "compare_4models_cp_segment.xlsx")
    seg_out.to_excel(seg_file, index=False)
    print(f"\nSaved segment comparison to: {seg_file}")

    # ============================================================
    # SINGLE-STRIDE ONE-STEP-AHEAD (4 models)
    # ============================================================
    k = 5
    a = k * STRIDE_LEN
    b = a + STRIDE_LEN
    if b >= len(cp_df):
        raise ValueError(f"Stride k={k} too close to end of cp_df (b={b} >= {len(cp_df)})")
    if a >= N:
        raise ValueError(f"Stride k={k} too large for rolling window arrays (a={a} >= N={N})")

    stride_in_deg = cp_angles_deg[a:b]   # (51,6)
    gt_next_deg   = cp_angles_deg[b]     # (6,)
    i = a

    # TS-LSTM one-step
    pred_ts_lstm_one_scaled = predict_next_tick_scaled(ts_lstm, X_ts_lstm[i:i+1])[0]
    pred_ts_lstm_one_deg = ts_lstm_scaler.inverse_transform(pred_ts_lstm_one_scaled.reshape(1,-1))[0]

    # PV-LSTM one-step
    pred_pv_lstm_one_scaled = predict_next_tick_scaled(pv_lstm, X_pv_lstm[i:i+1])[0]
    pred_pv_lstm_one_deg = pv_lstm_scaler_angles.inverse_transform(pred_pv_lstm_one_scaled.reshape(1,-1))[0]

    # TS-CNN one-step (adapter handles 51x6 output)
    pred_ts_cnn_one_scaled = predict_next_tick_scaled(ts_cnn, X_ts_cnn[i:i+1])[0]
    pred_ts_cnn_one_deg = ts_cnn_scaler.inverse_transform(pred_ts_cnn_one_scaled.reshape(1,-1))[0]

    # PV-CNN one-step
    pred_pv_cnn_one_scaled = predict_next_tick_scaled(pv_cnn, X_pv_cnn[i:i+1])[0]
    pred_pv_cnn_one_deg = pv_cnn_scaler_angles.inverse_transform(pred_pv_cnn_one_scaled.reshape(1,-1))[0]

    plot_one_stride_one_step(stride_in_deg, pred_ts_lstm_one_deg, gt_next_deg, title_prefix="TS-LSTM")
    plot_one_stride_one_step(stride_in_deg, pred_pv_lstm_one_deg, gt_next_deg, title_prefix="PV-LSTM")
    plot_one_stride_one_step(stride_in_deg, pred_ts_cnn_one_deg,  gt_next_deg, title_prefix="TS-CNN")
    plot_one_stride_one_step(stride_in_deg, pred_pv_cnn_one_deg,  gt_next_deg, title_prefix="PV-CNN")

    one_df = pd.DataFrame(
        np.vstack([gt_next_deg,
                   pred_ts_lstm_one_deg, pred_pv_lstm_one_deg,
                   pred_ts_cnn_one_deg,  pred_pv_cnn_one_deg]),
        index=["GT_next", "TS_LSTM_next", "PV_LSTM_next", "TS_CNN_next", "PV_CNN_next"],
        columns=ANGLE_COLS
    )
    one_file = os.path.join(PRED_DIR, f"compare_single_stride_k{k}_next_tick_4models.xlsx")
    one_df.to_excel(one_file)
    print(f"Saved single-stride next-tick table to: {one_file}")

    # ============================================================
    # ROLLING ONE-STEP-AHEAD ROLLOUT (4 models)
    # ============================================================
    H = 250
    k = 10
    start_time = k * STRIDE_LEN
    start_i = start_time

    # --- TS-LSTM rollout ---
    pred_ts_lstm_scaled, gt_ts_lstm_scaled, input_ts_lstm_scaled = rollout_from_dataset_next_tick_any(
        ts_lstm, X_ts_lstm, y_ts_lstm, start_i=start_i, horizon=H
    )
    input_ts_lstm_deg = ts_lstm_scaler.inverse_transform(input_ts_lstm_scaled[:, :6])
    pred_ts_lstm_deg  = ts_lstm_scaler.inverse_transform(pred_ts_lstm_scaled)
    gt_ts_lstm_deg    = ts_lstm_scaler.inverse_transform(gt_ts_lstm_scaled)

    plot_rollout_one_step_ahead(input_ts_lstm_deg, pred_ts_lstm_deg, gt_ts_lstm_deg, title_prefix="TS-LSTM")

    # --- PV-LSTM rollout ---
    pred_pv_lstm_scaled, gt_pv_lstm_scaled, input_pv_lstm_scaled = rollout_from_dataset_next_tick_any(
        pv_lstm, X_pv_lstm, y_pv_lstm, start_i=start_i, horizon=H
    )
    input_pv_lstm_deg = pv_lstm_scaler_angles.inverse_transform(input_pv_lstm_scaled[:, 2:])
    pred_pv_lstm_deg  = pv_lstm_scaler_angles.inverse_transform(pred_pv_lstm_scaled)
    gt_pv_lstm_deg    = pv_lstm_scaler_angles.inverse_transform(gt_pv_lstm_scaled)

    plot_rollout_one_step_ahead(input_pv_lstm_deg, pred_pv_lstm_deg, gt_pv_lstm_deg, title_prefix="PV-LSTM")

    # --- TS-CNN rollout (adapter uses first sample of stride prediction) ---
    pred_ts_cnn_scaled, gt_ts_cnn_scaled, input_ts_cnn_scaled = rollout_from_dataset_next_tick_any(
        ts_cnn, X_ts_cnn, y_ts_cnn, start_i=start_i, horizon=H
    )
    input_ts_cnn_deg = ts_cnn_scaler.inverse_transform(input_ts_cnn_scaled[:, :6])
    pred_ts_cnn_deg  = ts_cnn_scaler.inverse_transform(pred_ts_cnn_scaled)
    gt_ts_cnn_deg    = ts_cnn_scaler.inverse_transform(gt_ts_cnn_scaled)

    plot_rollout_one_step_ahead(input_ts_cnn_deg, pred_ts_cnn_deg, gt_ts_cnn_deg, title_prefix="TS-CNN")

    # --- PV-CNN rollout ---
    pred_pv_cnn_scaled, gt_pv_cnn_scaled, input_pv_cnn_scaled = rollout_from_dataset_next_tick_any(
        pv_cnn, X_pv_cnn, y_pv_cnn, start_i=start_i, horizon=H
    )
    input_pv_cnn_deg = pv_cnn_scaler_angles.inverse_transform(input_pv_cnn_scaled[:, 2:])
    pred_pv_cnn_deg  = pv_cnn_scaler_angles.inverse_transform(pred_pv_cnn_scaled)
    gt_pv_cnn_deg    = pv_cnn_scaler_angles.inverse_transform(gt_pv_cnn_scaled)

    plot_rollout_one_step_ahead(input_pv_cnn_deg, pred_pv_cnn_deg, gt_pv_cnn_deg, title_prefix="PV-CNN")


if __name__ == "__main__":
    main()
