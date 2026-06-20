#!/usr/bin/env python3
"""
timestamps_CNN_next_tick.py
Timestamp CNN rolling next-tick prediction (51x6 -> 6)

This version is intentionally aligned with your Timestamp LSTM rolling next-tick pipeline:
  Fit scaler on Typical only (angles-only)
  Transform CP with the same scaler
  Build ROLLING windows:
        X[i] = angles[i : i+51]   (51x6)
        y[i] = angles[i+51]       (6,)
  Train on Typical
  Validate on Typical + a small slice of CP
  Test on CP
  Produce the SAME "LSTM-like" plots:
        - Single-stride one-step-ahead (blue line + green x + red x)
        - Rolling one-step-ahead rollout plot (blue input, red GT, green x preds)
  Save model + scaler into the same folders:
        - Saved_Models/Timestamp_cnn_next_tick_model.keras
        - Scaler/standard_scaler_typical_cnn_next_tick.save

Expected:
  - Data_Normal/randomized_data_healthy.xlsx
  - Data_CP/*.xlsx  (sheet "Data")
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


# =============================================================================
# Configuration
# =============================================================================
WINDOW = 51
STRIDE_LEN = 51
N_FEATURES = 6

COLUMNS = [
    "LHipAngles (1)", "LKneeAngles (1)", "LAnkleAngles (1)",
    "RHipAngles (1)", "RKneeAngles (1)", "RAnkleAngles (1)"
]

TYPICAL_XLSX = "Data_Normal/randomized_data_healthy.xlsx"
CP_FOLDER = "Data_CP/"
CP_SHEET = "Data"
CP_SKIPROWS = [1, 2]
MAX_CP_FILES = 500

# Optional alignment: right-leg cyclic shift INSIDE each stride block
RIGHT_LEG_SHIFT = 25  # 51 -> 25 is half-stride
RIGHT_LEG_COLS = ["RHipAngles (1)", "RKneeAngles (1)", "RAnkleAngles (1)"]

# Training hyperparameters (try to match your LSTM settings)
LR = 0.001
EPOCHS = 150
BATCH_SIZE = 512
DROPOUT = 0.2

# Evaluation plots
T_PLOT_SEG = 600    # segment length for metrics/plots
ROLL_H = 250        # rollout horizon
ROLL_START_STRIDE = 10  # stride-aligned start (start_i = stride * 51)

# Output folders/files
SAVE_DIR = "Neural_Networks_Output/Saved_Models"
SCALER_DIR = "Neural_Networks_Output/Scaler"
PRED_DIR = "Neural_Networks_Output/Predictions"
PLOT_DIR = "Neural_Networks_Output/Plots"

MODEL_OUT = os.path.join(SAVE_DIR, "Timestamp_cnn_model.keras")
SCALER_OUT = os.path.join(SCALER_DIR, "standard_scaler_typical_cnn.save")


# =============================================================================
# Plot labels (match your thesis plots)
# =============================================================================
JOINT_LABELS = ["L Hip", "L Knee", "L Ankle", "R Hip", "R Knee", "R Ankle"]
JOINT_TITLES = [
    "Hips Flexion-Extension Left",
    "Knees Flexion-Extension Left",
    "Ankles Dorsiflexion-Plantarflexion Left",
    "Hips Flexion-Extension Right",
    "Knees Flexion-Extension Right",
    "Ankles Dorsiflexion-Plantarflexion Right",
]


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def divergence_fix(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Light endpoint nudge to avoid extreme wrap mismatch.
    (kept because you used something similar)
    """
    if df.empty:
        return df
    for col in cols:
        last_value = df[col].values[-1]
        first_value = df[col].values[0]
        divergence = np.abs(last_value - first_value)
        if divergence > 5:
            mean_value = (last_value + first_value) / 2.0
            df.loc[df.index[-1], col] = mean_value
            df.loc[df.index[0], col] = mean_value
    return df

def stridewise_roll_right_leg(df: pd.DataFrame, stride_len: int, delay: int, right_cols) -> pd.DataFrame:
    """
    Roll the right-leg columns *inside each stride* so you don't mix stride boundaries.
    """
    if delay == 0 or df.empty:
        return df

    arr = df[right_cols].to_numpy()
    n = (len(arr) // stride_len) * stride_len
    if n == 0:
        return df

    arr = arr[:n].reshape(-1, stride_len, len(right_cols))
    arr = np.roll(arr, shift=delay, axis=1)
    rolled = arr.reshape(n, len(right_cols))

    df2 = df.copy()
    df2.loc[df2.index[:n], right_cols] = rolled
    return df2

def load_cp_concat(cp_folder: str, max_files: int) -> pd.DataFrame:
    if not os.path.isdir(cp_folder):
        raise FileNotFoundError(f"CP folder not found: {cp_folder}")

    frames = []
    count = 0
    for fn in sorted(os.listdir(cp_folder)):
        if not fn.endswith(".xlsx"):
            continue
        fp = os.path.join(cp_folder, fn)
        try:
            df = pd.read_excel(fp, sheet_name=CP_SHEET, usecols=COLUMNS, skiprows=CP_SKIPROWS).fillna(0)
        except Exception as e:
            print(f"Skipping {fp} (read error): {e}")
            continue

        frames.append(df)
        count += 1
        if count >= max_files:
            break

    if not frames:
        raise RuntimeError("No CP files loaded.")
    out = pd.concat(frames, ignore_index=True).fillna(0)
    return out

def make_rolling_next_tick_pairs(series_Tx6: np.ndarray, window: int = 51):
    """
    series_Tx6: (T,6) scaled
    X: (T-window, window, 6)
    y: (T-window, 6)
    """
    T = series_Tx6.shape[0]
    if T <= window:
        raise ValueError(f"Not enough timesteps: T={T} for window={window}")

    X = np.zeros((T - window, window, 6), dtype=np.float32)
    y = np.zeros((T - window, 6), dtype=np.float32)
    for i in range(T - window):
        X[i] = series_Tx6[i:i+window]
        y[i] = series_Tx6[i+window]
    return X, y

def print_metrics_deg(gt_deg_2d, pred_deg_2d, name):
    gt = np.asarray(gt_deg_2d)
    pr = np.asarray(pred_deg_2d)
    err = pr - gt
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    print(f"\n=== {name} metrics (deg) ===")
    for j in range(6):
        print(f"{JOINT_LABELS[j]:6s}  MAE={mae[j]:6.2f}  RMSE={rmse[j]:6.2f}")
    print("===========================")

def plot_one_stride_one_step(stride_in_deg_51x6, pred_next_deg_6, gt_next_deg_6, title_prefix=""):
    """
    Screenshot-style:
      - blue line: input stride (51)
      - green x: prediction at t=51
      - red x: GT at t=51
    """
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
    """
      - blue: input window (0..50)
      - red: GT over horizon (51..)
      - green x: predictions over horizon
    """
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
        ax.plot(x_out, pred[:, joint_i], linestyle="None", marker="x", markersize=6, label="one-step predictions")
        ax.set_title(JOINT_TITLES[joint_i], fontsize=10)
        ax.set_xlabel("Time-step")
        ax.set_ylabel("Angle (deg)")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.suptitle(f"{title_prefix} (rolling one-step-ahead rollout)", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.show()

def rollout_next_tick(model, X_full, y_full, start_i, horizon):
    """
    X_full: (N,51,6) scaled
    y_full: (N,6) scaled
    """
    N = len(X_full)
    if start_i < 0 or start_i + horizon >= N:
        raise ValueError(f"start_i={start_i} too close to end for horizon={horizon}. N={N}")

    window = X_full[start_i].copy()  # (51,6)
    preds = np.zeros((horizon, 6), dtype=np.float32)
    gts   = np.zeros((horizon, 6), dtype=np.float32)

    for h in range(horizon):
        yhat = model.predict(window.reshape(1, WINDOW, 6), verbose=0)[0]  # (6,)
        preds[h] = yhat
        gts[h]   = y_full[start_i + h]
        window = np.vstack([window[1:], yhat])

    return preds, gts, X_full[start_i]


# =============================================================================
# Build CNN model (51x6 -> 6)
# =============================================================================
def build_timestamp_cnn_next_tick(window=51, n_features=6, dropout=0.2):
    model = Sequential([
        # Block 1: 32 -> 48 -> pool
        Conv1D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu",
               input_shape=(window, n_features)),
        Conv1D(filters=48, kernel_size=3, strides=2, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(dropout),

        # Block 2: 256 -> 256 -> pool
        Conv1D(filters=256, kernel_size=3, strides=2, padding="same", activation="relu"),
        Conv1D(filters=256, kernel_size=3, strides=2, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(dropout),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(dropout),
        Dense(n_features, activation="linear"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="mse",
        metrics=["mae", RootMeanSquaredError()],
    )
    return model


# =============================================================================
# Main
# =============================================================================
def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    ensure_dir(PRED_DIR)
    ensure_dir(SAVE_DIR)
    ensure_dir(SCALER_DIR)
    ensure_dir(PLOT_DIR)

    # ----------------------------
    # Load Typical
    # ----------------------------
    if not os.path.exists(TYPICAL_XLSX):
        raise FileNotFoundError(f"Typical file not found: {TYPICAL_XLSX}")

    # Right-leg half-stride shift is already baked into TYPICAL_XLSX by the
    # data augmentation step (Data_Augmentation/data_randomize_kinematics.py).
    df_typ = pd.read_excel(TYPICAL_XLSX, usecols=COLUMNS).fillna(0)
    # ----------------------------
    # Load CP (concat)
    # ----------------------------
    df_cp = load_cp_concat(CP_FOLDER, MAX_CP_FILES)
    df_cp = divergence_fix(df_cp, COLUMNS)

    # Optional stridewise right-leg roll (keeps alignment consistent with your other work)
    if RIGHT_LEG_SHIFT != 0:
        df_cp = stridewise_roll_right_leg(df_cp, STRIDE_LEN, RIGHT_LEG_SHIFT, RIGHT_LEG_COLS)

    # ----------------------------
    # Scaling (fit on Typical, transform CP)
    # ----------------------------
    scaler = StandardScaler()
    typ_scaled = scaler.fit_transform(df_typ[COLUMNS]).astype(np.float32)
    cp_scaled  = scaler.transform(df_cp[COLUMNS]).astype(np.float32)

    joblib.dump(scaler, SCALER_OUT)
    print(f"Saved scaler to: {SCALER_OUT}")

    # ----------------------------
    # Build rolling next-tick datasets
    # ----------------------------
    X_typ, y_typ = make_rolling_next_tick_pairs(typ_scaled, window=WINDOW)
    X_cp,  y_cp  = make_rolling_next_tick_pairs(cp_scaled,  window=WINDOW)

    # Split typical for train/val
    n_typ = len(X_typ)
    val_n_typ = max(1, int(0.2 * n_typ))
    # Temporal gap of WINDOW samples to prevent leakage at val/train boundary
    X_train, y_train = X_typ[val_n_typ + WINDOW:], y_typ[val_n_typ + WINDOW:]
    X_val_typ, y_val_typ = X_typ[:val_n_typ], y_typ[:val_n_typ]

    n_cp = len(X_cp)
    val_n_cp = max(1, int(0.2 * n_cp))
    X_val = np.vstack([X_val_typ, X_cp[:val_n_cp]])
    y_val = np.vstack([y_val_typ, y_cp[:val_n_cp]])

    # Test on CP data NOT used in validation (prevents data leakage)
    X_test, y_test = X_cp[val_n_cp:], y_cp[val_n_cp:]

    print(f"Typical rolling samples: {len(X_typ)}")
    print(f"CP rolling samples:      {len(X_cp)}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test(CP): {len(X_test)}")

    # ----------------------------
    # Build + train CNN
    # ----------------------------
    model = build_timestamp_cnn_next_tick(window=WINDOW, n_features=N_FEATURES, dropout=DROPOUT)
    model.summary()

    es_cb = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[es_cb, lr_cb],
        verbose=1
    )

    # ----------------------------
    # Evaluate (scaled metrics)
    # ----------------------------
    print("\n=== Evaluate (scaled space) ===")
    tr = model.evaluate(X_train, y_train, verbose=0)
    va = model.evaluate(X_val, y_val, verbose=0)
    te = model.evaluate(X_test, y_test, verbose=0)
    print(f"Train: loss={tr[0]:.6f}  MAE={tr[1]:.6f}  RMSE={tr[2]:.6f}")
    print(f"Val:   loss={va[0]:.6f}  MAE={va[1]:.6f}  RMSE={va[2]:.6f}")
    print(f"Test:  loss={te[0]:.6f}  MAE={te[1]:.6f}  RMSE={te[2]:.6f}")

    # Save model
    model.save(MODEL_OUT, include_optimizer=True)
    print(f"Saved model to: {MODEL_OUT}")

    # ----------------------------
    # Segment evaluation in DEGREES (thesis metrics)
    # ----------------------------
    T_plot = min(T_PLOT_SEG, len(X_test))
    start = max(0, len(X_test) - T_plot)

    X_seg = X_test[start:start+T_plot]
    y_seg = y_test[start:start+T_plot]

    pred_seg_scaled = model.predict(X_seg, verbose=0)  # (T,6)

    gt_seg_deg   = scaler.inverse_transform(y_seg)
    pred_seg_deg = scaler.inverse_transform(pred_seg_scaled)

    print_metrics_deg(gt_seg_deg, pred_seg_deg, "Timestamp CNN (next-tick)")

    # Save segment predictions
    seg_out = pd.DataFrame(
        np.hstack([gt_seg_deg, pred_seg_deg]),
        columns=[f"GT_{c}" for c in COLUMNS] + [f"CNN_{c}" for c in COLUMNS]
    )
    seg_file = os.path.join(PRED_DIR, "timestamp_cnn_next_tick_cp_segment.xlsx")
    seg_out.to_excel(seg_file, index=False)
    print(f"Saved segment comparison to: {seg_file}")

    # ----------------------------
    # Single-stride one-step-ahead plot (LSTM-like)
    # ----------------------------
    # For the "single stride input", we need ORIGINAL degrees sequence from CP
    cp_deg = df_cp[COLUMNS].to_numpy()
    # pick stride index safely
    k = 5
    a = k * STRIDE_LEN
    b = a + STRIDE_LEN
    if b >= len(cp_deg):
        k = max(0, (len(cp_deg) // STRIDE_LEN) - 2)
        a = k * STRIDE_LEN
        b = a + STRIDE_LEN

    stride_in_deg = cp_deg[a:b]          # (51,6)
    gt_next_deg   = cp_deg[b]            # (6,)

    # build the corresponding scaled window from cp_scaled at the same location
    # NOTE: X_cp[i] corresponds to cp_scaled[i:i+51], y_cp[i] = cp_scaled[i+51]
    i = a
    i = int(np.clip(i, 0, len(X_cp)-1))
    pred_one_scaled = model.predict(X_cp[i:i+1], verbose=0)[0]
    pred_one_deg = scaler.inverse_transform(pred_one_scaled.reshape(1, -1))[0]

    plot_one_stride_one_step(
        stride_in_deg_51x6=stride_in_deg,
        pred_next_deg_6=pred_one_deg,
        gt_next_deg_6=gt_next_deg,
        title_prefix="Timestamp CNN (rolling next-tick)"
    )

    # ----------------------------
    # Rolling rollout plot (LSTM-like)
    # ----------------------------
    start_i = int(ROLL_START_STRIDE * STRIDE_LEN)
    start_i = int(np.clip(start_i, 0, len(X_cp) - (ROLL_H + 1)))

    pred_roll_scaled, gt_roll_scaled, input_win_scaled = rollout_next_tick(
        model, X_cp, y_cp, start_i=start_i, horizon=ROLL_H
    )

    input_win_deg = scaler.inverse_transform(input_win_scaled)     # (51,6)
    pred_roll_deg = scaler.inverse_transform(pred_roll_scaled)     # (H,6)
    gt_roll_deg   = scaler.inverse_transform(gt_roll_scaled)       # (H,6)

    plot_rollout_one_step_ahead(
        input_window_deg=input_win_deg,
        pred_series_deg=pred_roll_deg,
        gt_series_deg=gt_roll_deg,
        title_prefix="Timestamp CNN (rolling next-tick)"
    )

    # ----------------------------
    # Training curves (optional, but useful)
    # ----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train Loss (MSE)")
    plt.plot(history.history["val_loss"], label="Val Loss (MSE)")
    plt.title("Timestamp CNN (next-tick) Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "timestamp_cnn_next_tick_loss.png"), dpi=200)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Val MAE")
    plt.title("Timestamp CNN (next-tick) MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "timestamp_cnn_next_tick_mae.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
