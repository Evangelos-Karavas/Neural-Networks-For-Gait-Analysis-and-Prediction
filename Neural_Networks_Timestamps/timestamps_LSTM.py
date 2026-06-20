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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


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

SAVE_DIR = "Neural_Networks_Output/Saved_Models"
PRED_DIR = "Neural_Networks_Output/Predictions"
SCALER_DIR = "Neural_Networks_Output/Scaler"
PLOT_DIR = "Neural_Networks_Output/Plots"

ANGLE_COLS = [
    "LHipAngles (1)", "LKneeAngles (1)", "LAnkleAngles (1)",
    "RHipAngles (1)", "RKneeAngles (1)", "RAnkleAngles (1)"
]
RIGHT_ANGLE_COLS = ["RHipAngles (1)", "RKneeAngles (1)", "RAnkleAngles (1)"]


# ============================================================
# Helpers: stridewise right-leg half-stride shift
# ============================================================
def half_stride_shift(stride_len: int) -> int:
    return int(stride_len // 2)  # 51 -> 25

def roll_stridewise_1d(x: np.ndarray, stride_len: int, shift: int) -> np.ndarray:
    """
    Circularly roll a 1D array within each stride block.
    This avoids mixing across stride boundaries.
    """
    x = np.asarray(x)
    n_strides = len(x) // stride_len
    x = x[:n_strides * stride_len].copy()
    out = x.copy()
    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len
        out[a:b] = np.roll(x[a:b], shift)
    return out

def apply_right_leg_half_stride_offset_angles(df: pd.DataFrame, stride_len: int, right_angle_cols) -> pd.DataFrame:
    shift = half_stride_shift(stride_len)
    out = df.copy()
    for col in right_angle_cols:
        out[col] = roll_stridewise_1d(out[col].to_numpy(), stride_len, shift)
    return out


# ============================================================
# Rolling windows: (window,6) -> next tick (6,)
# ============================================================
def make_rolling_windows(features: np.ndarray, targets: np.ndarray, window: int):
    """
    features: (T,6)
    targets:  (T,6)
    returns:
      X: (T-window, window, 6)
      y: (T-window, 6)
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
# Model
# ============================================================
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
# Plot + metrics helpers (same spirit as PV script)
# ============================================================
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
    T = arr_Tx6.shape[0]
    n = T // stride_len
    return arr_Tx6[:n * stride_len].reshape(n, stride_len, 6)

def stride_mean_std(strides_NxSx6: np.ndarray):
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
            f"{name:16s} | "
            f"MAE={metrics['MAE_deg'][j]:6.2f}  "
            f"RMSE={metrics['RMSE_deg'][j]:6.2f}  "
            f"NRMSE={metrics['NRMSE'][j]:6.3f}  "
            f"r={metrics['Pearson_r'][j]:6.3f}  "
            f"R2={metrics['R2'][j]:6.3f}"
        )
    print("====================================\n")

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
    np.random.seed(42)
    tf.random.set_seed(42)

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
    missing = set(ANGLE_COLS).difference(typ_df.columns)
    if missing:
        raise KeyError(f"Typical file missing columns: {sorted(missing)}")

    # Right-leg half-stride shift is already baked into TYPICAL_FILE by the
    # data augmentation step (Data_Augmentation/data_randomize_kinematics.py).
    typ_df = typ_df[ANGLE_COLS].fillna(0)

    # -------------------------
    # Load CP (per file, shift per file, then concat)
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
                usecols=ANGLE_COLS,
                skiprows=CP_SKIPROWS
            ).fillna(0)
        except Exception as e:
            print(f"Skipping {fp} (read error): {e}")
            continue

        # IMPORTANT: shift right leg within each file (prevents boundary artifacts)
        df = apply_right_leg_half_stride_offset_angles(df, STRIDE_LEN, RIGHT_ANGLE_COLS)

        cp_frames.append(df)
        file_counter += 1
        if file_counter >= MAX_CP_FILES:
            break

    if not cp_frames:
        raise RuntimeError("No CP files loaded. Check CP_FOLDER / sheet / columns.")

    cp_df = pd.concat(cp_frames, ignore_index=True).fillna(0)

    # -------------------------
    # Scale angles (single scaler fit on Typical)
    # -------------------------
    scaler = StandardScaler()
    typ_scaled = scaler.fit_transform(typ_df[ANGLE_COLS].to_numpy())
    cp_scaled  = scaler.transform(cp_df[ANGLE_COLS].to_numpy())

    # Save scalers like your old timestamp script
    joblib.dump(scaler, os.path.join(SCALER_DIR, "standard_scaler_typical_lstm.save"))
    joblib.dump(scaler, os.path.join(SCALER_DIR, "standard_scaler_cp_lstm.save"))

    # For rolling next-tick: features=angles, targets=next angles
    typ_features_scaled = typ_scaled
    typ_targets_scaled  = typ_scaled
    cp_features_scaled  = cp_scaled
    cp_targets_scaled   = cp_scaled

    # -------------------------
    # Rolling windows
    # -------------------------
    X_typ, y_typ = make_rolling_windows(typ_features_scaled, typ_targets_scaled, window=WINDOW)
    X_cp,  y_cp  = make_rolling_windows(cp_features_scaled,  cp_targets_scaled,  window=WINDOW)

    split_typ = max(1, int(0.2 * len(X_typ)))
    split_cp  = max(1, int(0.2 * len(X_cp)))

    # Temporal gap of WINDOW samples to prevent leakage at val/train boundary
    X_train = X_typ[split_typ + WINDOW:]
    y_train = y_typ[split_typ + WINDOW:]

    X_val = np.vstack([X_typ[:split_typ], X_cp[:split_cp]])
    y_val = np.vstack([y_typ[:split_typ], y_cp[:split_cp]])

    # Test on CP data NOT used in validation (prevents data leakage)
    X_test = X_cp[split_cp:]
    y_test = y_cp[split_cp:]

    print("Shapes:")
    print("X_train", X_train.shape, "y_train", y_train.shape)
    print("X_val  ", X_val.shape,   "y_val  ", y_val.shape)
    print("X_test ", X_test.shape,  "y_test ", y_test.shape)
    print(f"(Right half-stride shift used: {shift})")

    # -------------------------
    # Train model
    # -------------------------
    model = build_model(input_dim=6, window=WINDOW, output_dim=6)
    model.summary()

    es_cb = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[es_cb, lr_cb],
        verbose=1
    )

    model_path = os.path.join(SAVE_DIR, "Timestamp_lstm_model.keras")
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
    gt_scaled   = y_test[start:start + T_plot]

    pred_deg = scaler.inverse_transform(pred_scaled)
    gt_deg   = scaler.inverse_transform(gt_scaled)

    # -------------------------
    # Stride-aligned views (like PV script)
    # -------------------------
    typ_angles = typ_df[ANGLE_COLS].to_numpy()
    cp_angles  = cp_df[ANGLE_COLS].to_numpy()

    typ_strides = reshape_to_strides(typ_angles, STRIDE_LEN)
    cp_strides  = reshape_to_strides(cp_angles,  STRIDE_LEN)

    T_common = min(len(pred_deg), len(gt_deg))
    pred_seg = pred_deg[:T_common]
    gt_seg   = gt_deg[:T_common]

    pred_strides = reshape_to_strides(pred_seg, STRIDE_LEN)
    gt_strides   = reshape_to_strides(gt_seg,   STRIDE_LEN)

    Nmin = min(len(typ_strides), len(cp_strides), len(pred_strides), len(gt_strides))
    typ_strides  = typ_strides[:Nmin]
    cp_strides   = cp_strides[:Nmin]
    pred_strides = pred_strides[:Nmin]
    gt_strides   = gt_strides[:Nmin]

    # Overlay plots
    plot_overlay_strides(typ_strides, title="Typical: many strides + mean", max_strides=40)
    plot_overlay_strides(cp_strides,  title="CP: many strides + mean",      max_strides=40)
    plot_overlay_strides(pred_strides, title="Predicted (from CP): many strides + mean", max_strides=40)

    # Error metrics and plots
    metrics = compute_metrics_deg(pred_seg, gt_seg)
    print_metrics_table(metrics)

    plot_phase_binned_abs_error(gt_strides, pred_strides, title="Mean |Pred-GT| vs gait phase (stride bins)")
    plot_residual_hist(pred_seg, gt_seg, title="Residual histograms (Pred - GT)")

    # Save predictions like PV (angles only)
    pred_df = pd.DataFrame(pred_deg, columns=ANGLE_COLS)
    gt_df   = pd.DataFrame(gt_deg,   columns=ANGLE_COLS)

    pred_file = os.path.join(PRED_DIR, "timestamp_rolling_pred_next_tick.xlsx")
    gt_file   = os.path.join(PRED_DIR, "timestamp_rolling_gt_next_tick.xlsx")
    pred_df.to_excel(pred_file, index=False)
    gt_df.to_excel(gt_file, index=False)

    print("Saved rolling predictions to:", pred_file)
    print("Saved rolling ground truth to:", gt_file)

    plot_pred_vs_gt_series(pred_deg, gt_deg, title=f"Timestamp next-tick prediction (WINDOW={WINDOW}, right shift={shift})")


if __name__ == "__main__":
    main()
