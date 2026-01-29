#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError


# ============================================================
# SETTINGS
# ============================================================
STRIDE_LEN = 51
WINDOW = 51

EPOCHS = 150
BATCH_SIZE = 512
MAX_CP_FILES = 500

TYPICAL_FILE = "Data_Normal/randomized_data_healthy.xlsx"
CP_FOLDER = "Data_CP/"
CP_SHEET = "Data"
CP_SKIPROWS = [1, 2]

SAVE_DIR = "Saved_Models"
PRED_DIR = "Predictions"
SCALER_DIR = "Scaler"

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
ALL_FEATURE_COLS = PV_COLS + ANGLE_COLS


# ============================================================
# Helpers: stridewise shift (half-stride from cadence)
# ============================================================
def half_stride_shift(stride_len: int) -> int:
    """
    Compute half-stride offset from stride length.
    For odd stride_len like 51, this returns 25.
    """
    return int(stride_len // 2)


def roll_stridewise_1d(x: np.ndarray, stride_len: int, shift: int) -> np.ndarray:
    """
    Circularly roll a 1D array within each stride block.
    Positive shift moves events later in the stride (towards larger indices).
    """
    x = np.asarray(x)
    n_strides = len(x) // stride_len
    x = x[:n_strides * stride_len].copy()
    out = x.copy()
    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len
        out[a:b] = np.roll(x[a:b], shift)
    return out


def apply_right_leg_half_stride_offset(df: pd.DataFrame, stride_len: int, right_angle_cols, pv_right_col: str = "PhaseVariable_Right") -> pd.DataFrame:

    shift = half_stride_shift(stride_len)
    out = df.copy()
    for col in right_angle_cols:
        out[col] = roll_stridewise_1d(out[col].to_numpy(), stride_len, shift)
    out[pv_right_col] = roll_stridewise_1d(out[pv_right_col].to_numpy(), stride_len, shift)
    return out


# ============================================================
# Phaser variable computation
# ============================================================
def compute_pv_stride(q: np.ndarray, c: float, enforce_monotonic: bool = True) -> np.ndarray:
    """
    q: hip angle over one stride, shape (N,)
    c: stance fraction in [0,1], from Foot-Off%/100
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

    # stance portion (down to min)
    s[:idx_min + 1] = ((q0 - q[:idx_min + 1]) / denom) * c

    # swing portion
    s[idx_min:] = 1.0 + ((1.0 - c) / denom) * (q[idx_min:] - q0)

    s = np.clip(s, 0.0, 1.0)
    if enforce_monotonic:
        s = np.maximum.accumulate(s)

    return np.clip(s.astype(np.float32), 0.0, 1.0 - 1e-6)

def compute_phase_variables(df: pd.DataFrame, stride_len: int, lhip_col: str, rhip_col: str, lfo_col: str, rfo_col: str, enforce_monotonic: bool = True) -> pd.DataFrame:
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
# Rolling windows LSTM Model
# ============================================================
def make_rolling_windows(features: np.ndarray, targets: np.ndarray, window: int):
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


def build_model(input_dim: int, window: int, output_dim: int):
    model = Sequential([
        LSTM(128, activation="tanh", return_sequences=True, input_shape=(window, input_dim)),
        Dropout(0.2),
        LSTM(128, activation="tanh", return_sequences=False),
        Dropout(0.2),
        Dense(128, activation="relu"),
        Dense(output_dim, activation="linear"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae", RootMeanSquaredError()],
    )
    return model


# ============================================================
# Plot helpers
# ============================================================
def plot_pv_sanity(df, stride_len=51, n_strides=3, title_prefix=""):
    total_strides = len(df) // stride_len
    end_stride = min(total_strides, n_strides)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for s in range(end_stride):
        a, b = s * stride_len, (s + 1) * stride_len
        x = np.arange(stride_len)
        axes[0].plot(x, df["PhaseVariable_Left"].values[a:b], label=f"stride {s}")
        axes[1].plot(x, df["PhaseVariable_Right"].values[a:b], label=f"stride {s}")

    axes[0].set_ylabel("PV Left")
    axes[1].set_ylabel("PV Right")
    axes[1].set_xlabel("Sample within stride")
    axes[0].grid(True); axes[1].grid(True)
    axes[0].set_title(f"{title_prefix}: Phase Variable")
    axes[0].legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_pv_left_right_overlay(df, title="Phase Variable Left & Right", max_samples=None):

    pvL = df["PhaseVariable_Left"].to_numpy()
    pvR = df["PhaseVariable_Right"].to_numpy()

    if max_samples is not None:
        pvL = pvL[:max_samples]
        pvR = pvR[:max_samples]
        t = np.arange(max_samples)
    else:
        t = np.arange(len(pvL))

    plt.figure(figsize=(12, 4))
    plt.plot(t, pvL, label="PV Left")
    plt.plot(t, pvR, label="PV Right")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Sample")
    plt.ylabel("Phase Variable")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pv_left_right_per_stride(df, stride_len=51, n_strides=4, title="PV Left & Right (per stride)"):
    """
    Plot PV_Left and PV_Right overlayed, but reset x-axis each stride (0..stride_len-1).
    This is often the clearest view.
    """
    total_strides = len(df) // stride_len
    end_stride = min(total_strides, n_strides)

    plt.figure(figsize=(12, 4))
    x = np.arange(stride_len)

    for s in range(end_stride):
        a, b = s * stride_len, (s + 1) * stride_len
        plt.plot(x, df["PhaseVariable_Left"].values[a:b], alpha=0.9, label="PV Left" if s == 0 else None)
        plt.plot(x, df["PhaseVariable_Right"].values[a:b], alpha=0.9, linestyle="--", label="PV Right" if s == 0 else None)

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Sample within stride")
    plt.ylabel("Phase Variable")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pred_vs_gt_series(pred_angles, gt_angles, title=""):
    t = np.arange(gt_angles.shape[0])
    labels = ["L Hip", "L Knee", "L Ankle", "R Hip", "R Knee", "R Ankle"]

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, gt_angles[:, i], label="Ground truth")
        ax.plot(t, pred_angles[:, i], label="Predicted")
        ax.set_ylabel(labels[i])
        ax.grid(True)
        if i == 0:
            ax.set_title(title)
            ax.legend()
        if i == 5:
            ax.set_xlabel("Sample")
    plt.tight_layout()
    plt.show()

def reshape_to_strides(arr_Tx6: np.ndarray, stride_len: int) -> np.ndarray:
    """(T,6) -> (N_strides, stride_len, 6) by truncation."""
    T = arr_Tx6.shape[0]
    n = T // stride_len
    return arr_Tx6[:n * stride_len].reshape(n, stride_len, 6)

def stride_mean_std(strides_NxSx6: np.ndarray):
    """Return (S,6) mean and std across strides."""
    mu = strides_NxSx6.mean(axis=0)
    sd = strides_NxSx6.std(axis=0)
    return mu, sd

def compute_metrics_deg(pred: np.ndarray, gt: np.ndarray):
    eps = 1e-12
    err = pred - gt
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    gt_range = np.ptp(gt, axis=0)
    nrmse = rmse / (gt_range + eps)

    r = np.zeros(6)
    for j in range(6):
        x = gt[:, j] - gt[:, j].mean()
        y = pred[:, j] - pred[:, j].mean()
        denom = (np.linalg.norm(x) * np.linalg.norm(y)) + eps
        r[j] = float(np.dot(x, y) / denom)

    ss_res = np.sum((gt - pred)**2, axis=0)
    ss_tot = np.sum((gt - np.mean(gt, axis=0))**2, axis=0) + eps
    r2 = 1.0 - ss_res / ss_tot

    return {"MAE_deg": mae, "RMSE_deg": rmse, "NRMSE": nrmse, "Pearson_r": r, "R2": r2}

def print_metrics_table(metrics: dict, labels=ANGLE_COLS):
    print("\n=== Model Error Metrics (degrees) ===")
    for j, name in enumerate(labels):
        print(
            f"{name:7s} | "
            f"MAE={metrics['MAE_deg'][j]:6.2f}  "
            f"RMSE={metrics['RMSE_deg'][j]:6.2f}  "
            f"NRMSE={metrics['NRMSE'][j]:6.3f}  "
            f"r={metrics['Pearson_r'][j]:6.3f}  "
            f"R2={metrics['R2'][j]:6.3f}"
        )
    print("====================================\n")

def plot_stride_mean_std_compare(typ_strides, cp_strides, pred_strides, title_prefix=""):
    S = typ_strides.shape[1]
    x = np.linspace(0, 100, S)

    typ_mu, typ_sd = stride_mean_std(typ_strides)
    cp_mu,  cp_sd  = stride_mean_std(cp_strides)
    pr_mu,  pr_sd  = stride_mean_std(pred_strides)

    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(x, typ_mu[:, j], label="Typical (mean)")
        ax.fill_between(x, typ_mu[:, j] - typ_sd[:, j], typ_mu[:, j] + typ_sd[:, j], alpha=0.2)

        ax.plot(x, cp_mu[:, j], label="CP (mean)")
        ax.fill_between(x, cp_mu[:, j] - cp_sd[:, j], cp_mu[:, j] + cp_sd[:, j], alpha=0.2)

        ax.plot(x, pr_mu[:, j], label="Pred (from CP, mean)")
        ax.fill_between(x, pr_mu[:, j] - pr_sd[:, j], pr_mu[:, j] + pr_sd[:, j], alpha=0.2)

        ax.set_ylabel(ANGLE_COLS[j])
        ax.grid(True)
        if j == 0:
            ax.set_title(f"{title_prefix} Mean ± Std over Stride")
            ax.legend(ncol=3, fontsize=9)

    axes[-1].set_xlabel("Gait cycle (%)")
    plt.tight_layout()
    plt.show()

def plot_overlay_strides(strides, title="", max_strides=30):
    N, S, _ = strides.shape
    x = np.linspace(0, 100, S)
    mu, _ = stride_mean_std(strides)

    useN = min(N, max_strides)
    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
    for j, ax in enumerate(axes):
        for i in range(useN):
            ax.plot(x, strides[i, :, j], alpha=0.15)
        ax.plot(x, mu[:, j], linewidth=2.5, label="Mean")
        ax.set_ylabel(ANGLE_COLS[j])
        ax.grid(True)
        if j == 0:
            ax.set_title(title)
            ax.legend()
    axes[-1].set_xlabel("Gait cycle (%)")
    plt.tight_layout()
    plt.show()

def plot_phase_binned_abs_error(gt_strides, pred_strides, title="Abs Error vs Gait Phase"):
    S = gt_strides.shape[1]
    x = np.linspace(0, 100, S)
    abs_err = np.abs(pred_strides - gt_strides)
    mean_abs_err = abs_err.mean(axis=0)

    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(x, mean_abs_err[:, j])
        ax.set_ylabel(f"{ANGLE_COLS[j]}\n|err| (deg)")
        ax.grid(True)
        if j == 0:
            ax.set_title(title)
    axes[-1].set_xlabel("Gait cycle (%)")
    plt.tight_layout()
    plt.show()

def plot_residual_hist(pred: np.ndarray, gt: np.ndarray, title="Residual histograms (Pred - GT)"):
    err = pred - gt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    for j, ax in enumerate(axes):
        ax.hist(err[:, j], bins=60)
        ax.set_title(ANGLE_COLS[j])
        ax.grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)

    shift = half_stride_shift(STRIDE_LEN)

    # -------------------------
    # Load Typical
    # -------------------------
    if not os.path.exists(TYPICAL_FILE):
        raise FileNotFoundError(f"Typical file not found: {TYPICAL_FILE}")

    typ_df = pd.read_excel(TYPICAL_FILE).copy()
    needed = set(ANGLE_COLS + [LFO_COL, RFO_COL])
    missing = needed.difference(typ_df.columns)
    if missing:
        raise KeyError(f"Typical file missing columns: {sorted(missing)}")

    typ_df = typ_df[ANGLE_COLS + [LFO_COL, RFO_COL]].fillna(0)

    # Compute PV on original unshifted signals (both legs start stance at index 0 in your stored data)
    typ_df = compute_phase_variables(typ_df, STRIDE_LEN, LHIP_COL, RHIP_COL, LFO_COL, RFO_COL, enforce_monotonic=True)

    # Shift RIGHT angles + PV_Right by half stride to correct timing
    typ_df = apply_right_leg_half_stride_offset(typ_df, STRIDE_LEN, RIGHT_ANGLE_COLS, pv_right_col="PhaseVariable_Right")

    plot_pv_sanity(typ_df, stride_len=STRIDE_LEN, n_strides=3, title_prefix="Typical")
    plot_pv_left_right_overlay(typ_df, title="PV Left & Right", max_samples=51*10)
    plot_pv_left_right_per_stride(typ_df, stride_len=STRIDE_LEN, n_strides=6, title="Typical: PV Left & Right (first strides)")

    # -------------------------
    # Load CP (per file, then concat)
    # -------------------------
    if not os.path.isdir(CP_FOLDER):
        raise FileNotFoundError(f"CP folder not found: {CP_FOLDER}")

    cp_frames = []
    file_counter = 0

    for fn in sorted(os.listdir(CP_FOLDER)):
        if not fn.endswith(".xlsx"):
            continue
        fp = os.path.join(CP_FOLDER, fn)

        try:
            df = pd.read_excel(
                fp, sheet_name=CP_SHEET,
                usecols=ANGLE_COLS + [LFO_COL, RFO_COL],
                skiprows=CP_SKIPROWS
            ).fillna(0)
        except Exception as e:
            print(f"Skipping {fp} (read error): {e}")
            continue

        # Compute PV on unshifted
        df = compute_phase_variables(df, STRIDE_LEN, LHIP_COL, RHIP_COL, LFO_COL, RFO_COL, enforce_monotonic=True)

        # Shift right angles + PV_Right for THIS patient/file
        df = apply_right_leg_half_stride_offset(df, STRIDE_LEN, RIGHT_ANGLE_COLS, pv_right_col="PhaseVariable_Right")

        cp_frames.append(df)
        file_counter += 1
        if file_counter >= MAX_CP_FILES:
            break

    if not cp_frames:
        raise RuntimeError("No CP files loaded. Check CP_FOLDER / sheet / columns.")

    cp_df = pd.concat(cp_frames, ignore_index=True).fillna(0)

    plot_pv_sanity(cp_df, stride_len=STRIDE_LEN, n_strides=3, title_prefix="CP")
    plot_pv_left_right_overlay(cp_df, title="CP: PV Left & Right (full, overlay)", max_samples=51*10)
    plot_pv_left_right_per_stride(cp_df, stride_len=STRIDE_LEN, n_strides=6, title="CP: PV Left & Right (first strides)")

    # -------------------------
    # Scale PV and angles separately
    # -------------------------
    scaler_pv = StandardScaler()
    scaler_ang = StandardScaler()

    typ_pv_scaled = scaler_pv.fit_transform(typ_df[PV_COLS].to_numpy())
    typ_ang_scaled = scaler_ang.fit_transform(typ_df[ANGLE_COLS].to_numpy())

    cp_pv_scaled = scaler_pv.transform(cp_df[PV_COLS].to_numpy())
    cp_ang_scaled = scaler_ang.transform(cp_df[ANGLE_COLS].to_numpy())

    joblib.dump(scaler_pv, os.path.join(SCALER_DIR, "scaler_pv.save"))
    joblib.dump(scaler_ang, os.path.join(SCALER_DIR, "scaler_angles.save"))

    typ_features_scaled = np.concatenate([typ_pv_scaled, typ_ang_scaled], axis=1)  # (T,8)
    cp_features_scaled  = np.concatenate([cp_pv_scaled,  cp_ang_scaled],  axis=1)

    typ_targets_scaled = typ_ang_scaled  # (T,6)
    cp_targets_scaled  = cp_ang_scaled

    # -------------------------
    # Rolling windows
    # -------------------------
    X_typ, y_typ = make_rolling_windows(typ_features_scaled, typ_targets_scaled, window=WINDOW)
    X_cp,  y_cp  = make_rolling_windows(cp_features_scaled,  cp_targets_scaled,  window=WINDOW)

    split_typ = max(1, int(0.2 * len(X_typ)))
    split_cp  = max(1, int(0.2 * len(X_cp)))

    X_train = X_typ[split_typ:]
    y_train = y_typ[split_typ:]

    X_val = np.vstack([X_typ[:split_typ], X_cp[:split_cp]])
    y_val = np.vstack([y_typ[:split_typ], y_cp[:split_cp]])

    X_test = X_cp
    y_test = y_cp

    print("Shapes:")
    print("X_train", X_train.shape, "y_train", y_train.shape)
    print("X_val  ", X_val.shape,   "y_val  ", y_val.shape)
    print("X_test ", X_test.shape,  "y_test ", y_test.shape)

    # -------------------------
    # Train model
    # -------------------------
    model = build_model(input_dim=typ_features_scaled.shape[1], window=WINDOW, output_dim=6)
    model.summary()

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1
    )

    model_path = os.path.join(SAVE_DIR, "PV_rolling_next_tick_lstm.keras")
    model.save(model_path, include_optimizer=True)
    print("Saved model to:", model_path)

    loss, mae, rmse = model.evaluate(X_test, y_test, verbose=0)
    print(f"CP Test: MSE={loss:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # -------------------------
    # Plot prediction on CP (next-tick)
    # -------------------------
    T_plot = 400
    start = max(0, len(X_test) - T_plot - 1)

    pred_scaled = model.predict(X_test[start:start + T_plot], verbose=0)
    gt_scaled = y_test[start:start + T_plot]

    pred_deg = scaler_ang.inverse_transform(pred_scaled)
    gt_deg = scaler_ang.inverse_transform(gt_scaled)


    # -------------------------
    # Build stride-aligned views
    # -------------------------
    stride_len = STRIDE_LEN

    # Typical & CP from original dataframes
    typ_angles = typ_df[ANGLE_COLS].to_numpy()
    cp_angles  = cp_df[ANGLE_COLS].to_numpy()

    typ_strides = reshape_to_strides(typ_angles, stride_len)
    cp_strides  = reshape_to_strides(cp_angles, stride_len)

    # Pred/GT: use the same segment length for both, then stride-chop
    T_common = min(len(pred_deg), len(gt_deg))
    pred_seg = pred_deg[:T_common]
    gt_seg   = gt_deg[:T_common]

    pred_strides = reshape_to_strides(pred_seg, stride_len)
    gt_strides   = reshape_to_strides(gt_seg, stride_len)

    # Optionally: match number of strides across sets for fair mean±std comparison
    Nmin = min(len(typ_strides), len(cp_strides), len(pred_strides), len(gt_strides))
    typ_strides = typ_strides[:Nmin]
    cp_strides  = cp_strides[:Nmin]
    pred_strides = pred_strides[:Nmin]
    gt_strides   = gt_strides[:Nmin]

    # -------------------------
    # Plots: Typical vs CP vs Pred
    # -------------------------
    plot_stride_mean_std_compare(
        typ_strides, cp_strides, pred_strides,
        title_prefix="Typical vs CP vs Predicted (from CP)"
    )

    plot_overlay_strides(typ_strides, title="Typical: many strides + mean", max_strides=40)
    plot_overlay_strides(cp_strides,  title="CP: many strides + mean", max_strides=40)
    plot_overlay_strides(pred_strides, title="Predicted (from CP): many strides + mean", max_strides=40)

    # -------------------------
    # Error outputs
    # -------------------------
    metrics = compute_metrics_deg(pred_seg, gt_seg)
    print_metrics_table(metrics)

    plot_phase_binned_abs_error(gt_strides, pred_strides, title="Mean |Pred-GT| vs gait phase (stride bins)")
    plot_residual_hist(pred_seg, gt_seg, title="Residual histograms (Pred - GT)")



    # PV for each predicted tick corresponds to last timestep of window
    pv_segment_scaled = X_test[start:start + T_plot, -1, :2]
    pv_segment = scaler_pv.inverse_transform(pv_segment_scaled)

    out_cols = PV_COLS + ANGLE_COLS
    pred_out = np.concatenate([pv_segment, pred_deg], axis=1)
    gt_out   = np.concatenate([pv_segment, gt_deg], axis=1)

    pred_df = pd.DataFrame(pred_out, columns=out_cols)
    gt_df   = pd.DataFrame(gt_out,   columns=out_cols)

    pred_file = os.path.join(PRED_DIR, "rolling_pred_next_tick_with_pv.xlsx")
    gt_file   = os.path.join(PRED_DIR, "rolling_gt_next_tick_with_pv.xlsx")
    pred_df.to_excel(pred_file, index=False)
    gt_df.to_excel(gt_file, index=False)

    print("Saved rolling predictions to:", pred_file)
    print("Saved rolling ground truth to:", gt_file)

    plot_pred_vs_gt_series(pred_deg, gt_deg, title=f"CP next-tick prediction (WINDOW={WINDOW}, right shift={shift})")


if __name__ == "__main__":
    main()
