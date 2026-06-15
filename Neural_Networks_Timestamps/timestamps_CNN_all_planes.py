#!/usr/bin/env python3
"""
timestamps_CNN_next_tick_18dof.py
Timestamp CNN rolling next-tick prediction (51x18 -> 18)

Aligned with your Timestamp LSTM rolling next-tick pipeline:
  - Fit scaler on Typical only (angles-only, now 18 columns)
  - Transform CP with the same scaler
  - Build ROLLING windows:
        X[i] = angles[i : i+51]   (51x18)
        y[i] = angles[i+51]       (18,)
  - Train on Typical
  - Validate on Typical + a small slice of CP
  - Test on CP
  - Produce the SAME "LSTM-like" plots but adapted to 18 outputs:
        - Single-stride one-step-ahead (blue line + green x + red x) for 18 channels
        - Rolling one-step-ahead rollout plot (blue input, red GT, green x preds) for 18 channels
  - Save model + scaler:
        - Saved_Models/Timestamp_cnn_next_tick_model_18.keras
        - Scaler/standard_scaler_typical_cnn_next_tick_18.save

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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, ZeroPadding1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

# --- Keras save-model version workaround (same pattern you used)
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__


# =============================================================================
# Configuration
# =============================================================================
WINDOW = 51
STRIDE_LEN = 51

# 18 columns: 6 joints (L/R hip,knee,ankle) x 3 planes (1,2,3)
COLUMNS = [
    # Left
    "LHipAngles (1)", "LHipAngles (2)", "LHipAngles (3)",
    "LKneeAngles (1)", "LKneeAngles (2)", "LKneeAngles (3)",
    "LAnkleAngles (1)", "LAnkleAngles (2)", "LAnkleAngles (3)",
    # Right
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]
N_FEATURES = len(COLUMNS)  # 18

TYPICAL_XLSX = "Data_Normal/dynamics_total_augmented.xlsx"
CP_FOLDER = "Data_CP/"
CP_SHEET = "Data"
CP_SKIPROWS = [1, 2]
MAX_CP_FILES = 500

# Optional alignment: right-leg cyclic shift INSIDE each stride block
RIGHT_LEG_SHIFT = 25  # 51 -> 25 is half-stride
RIGHT_LEG_COLS = [
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]

# Training hyperparameters
LR = 0.001
EPOCHS = 150
BATCH_SIZE = 512
DROPOUT = 0.2

# Evaluation plots
T_PLOT_SEG = 600    # segment length for metrics/plots
ROLL_H = 250        # rollout horizon
ROLL_START_STRIDE = 10  # stride-aligned start (start_i = stride * 51)

# Output folders/files
SAVE_DIR = "Saved_Models"
SCALER_DIR = "Scaler"
PRED_DIR = "Predictions"
PLOT_DIR = "Plots"

MODEL_OUT = os.path.join(SAVE_DIR, "Timestamp_cnn_next_tick_model_18.keras")
SCALER_OUT = os.path.join(SCALER_DIR, "standard_scaler_typical_cnn_next_tick_18.save")


# =============================================================================
# Plot labels (18 channels)
# Layout choice: 3 rows (planes) x 6 columns (joints: LHip LKnee LAnkle RHip RKnee RAnkle)
# =============================================================================
PLANE_NAMES = {
    1: "Sagittal (1)",
    2: "Frontal (2)",
    3: "Transverse (3)",
}

JOINT_ORDER_6 = ["L Hip", "L Knee", "L Ankle", "R Hip", "R Knee", "R Ankle"]
JOINT_TITLES_6 = [
    "Hip (Left)",
    "Knee (Left)",
    "Ankle (Left)",
    "Hip (Right)",
    "Knee (Right)",
    "Ankle (Right)",
]

# Index mapping from (plane,row) & (joint,col) -> column index in COLUMNS
# We defined COLUMNS in blocks: LHip(1,2,3), LKnee(1,2,3), LAnkle(1,2,3), RHip(1,2,3), RKnee(1,2,3), RAnkle(1,2,3)
# So: base per joint is [0..5], each has 3 planes.
JOINT_BASE_INDEX = {
    "L Hip": 0,
    "L Knee": 1,
    "L Ankle": 2,
    "R Hip": 3,
    "R Knee": 4,
    "R Ankle": 5,
}

def channel_index(joint_name: str, plane: int) -> int:
    """Return 0..17 index into COLUMNS for given joint and plane (1..3)."""
    if plane not in (1, 2, 3):
        raise ValueError("plane must be 1,2,3")
    j = JOINT_BASE_INDEX[joint_name]
    # each joint has 3 planes, stored consecutively
    return j * 3 + (plane - 1)


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def divergence_fix(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Light endpoint nudge to avoid extreme wrap mismatch.
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

def make_rolling_next_tick_pairs(series_TxD: np.ndarray, window: int = 51):
    """
    series_TxD: (T,D) scaled
    X: (T-window, window, D)
    y: (T-window, D)
    """
    series_TxD = np.asarray(series_TxD)
    T, D = series_TxD.shape
    if T <= window:
        raise ValueError(f"Not enough timesteps: T={T} for window={window}")

    X = np.zeros((T - window, window, D), dtype=np.float32)
    y = np.zeros((T - window, D), dtype=np.float32)
    for i in range(T - window):
        X[i] = series_TxD[i:i+window]
        y[i] = series_TxD[i+window]
    return X, y

def print_metrics_deg(gt_deg_2d, pred_deg_2d, name, columns):
    gt = np.asarray(gt_deg_2d)
    pr = np.asarray(pred_deg_2d)
    err = pr - gt
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    print(f"\n=== {name} metrics (deg) ===")
    for j, col in enumerate(columns):
        print(f"{col:18s}  MAE={mae[j]:7.2f}  RMSE={rmse[j]:7.2f}")
    print("===========================")

def _plot_grid_18(fig, axes, x_in, x_next_or_out, inp, pred, gt, mode: str, title_prefix=""):
    """
    mode:
      - "one_step": x_next_or_out is scalar (next index), pred/gt shape (18,)
      - "rollout": x_next_or_out is array (H,), pred/gt shape (H,18)
    axes is shape (3,6)
    """
    for r, plane in enumerate([1, 2, 3]):
        for c, jname in enumerate(JOINT_ORDER_6):
            ax = axes[r, c]
            idx = channel_index(jname, plane)

            if mode == "one_step":
                ax.plot(x_in, inp[:, idx], linewidth=2, label="input")
                ax.plot([x_next_or_out], [pred[idx]], marker="x", markersize=7, linestyle="None", label="prediction")
                ax.plot([x_next_or_out], [gt[idx]], marker="x", markersize=7, linestyle="None", label="actual")
            else:
                ax.plot(x_in, inp[:, idx], linewidth=2, label="input")
                ax.plot(x_next_or_out, gt[:, idx], linewidth=2, label="actual")
                ax.plot(x_next_or_out, pred[:, idx], linestyle="None", marker="x", markersize=4, label="one-step predictions")

            ax.set_title(f"{JOINT_TITLES_6[c]} — {PLANE_NAMES[plane]}", fontsize=9)
            ax.set_xlabel("Time-step")
            ax.set_ylabel("Angle (deg)")
            ax.grid(True)

    # Single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.suptitle(title_prefix, y=0.995, fontsize=12)

def plot_one_stride_one_step_18(stride_in_deg_51x18, pred_next_deg_18, gt_next_deg_18, title_prefix=""):
    """
    18-channel screenshot-style:
      - blue line: input stride (51)
      - green x: prediction at t=51
      - red x: GT at t=51
    Grid: 3 (planes) x 6 (joints)
    """
    stride = np.asarray(stride_in_deg_51x18)
    pred = np.asarray(pred_next_deg_18).reshape(-1)
    gt   = np.asarray(gt_next_deg_18).reshape(-1)

    x_in = np.arange(STRIDE_LEN)
    x_next = STRIDE_LEN

    fig, axes = plt.subplots(3, 6, figsize=(18, 9), sharex=False)
    _plot_grid_18(
        fig, axes,
        x_in=x_in,
        x_next_or_out=x_next,
        inp=stride,
        pred=pred,
        gt=gt,
        mode="one_step",
        title_prefix=f"{title_prefix} (1 stride input → 1-step ahead)"
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.965])
    plt.show()

def plot_rollout_one_step_ahead_18(input_window_deg, pred_series_deg, gt_series_deg, title_prefix=""):
    """
    18-channel rollout:
      - blue: input window (0..50)
      - red: GT over horizon (51..)
      - green x: predictions over horizon
    Grid: 3 (planes) x 6 (joints)
    """
    inp = np.asarray(input_window_deg)
    pred = np.asarray(pred_series_deg)
    gt = np.asarray(gt_series_deg)

    W = inp.shape[0]
    H = pred.shape[0]
    x_in = np.arange(W)
    x_out = np.arange(W, W + H)

    fig, axes = plt.subplots(3, 6, figsize=(18, 9), sharex=False)
    _plot_grid_18(
        fig, axes,
        x_in=x_in,
        x_next_or_out=x_out,
        inp=inp,
        pred=pred,
        gt=gt,
        mode="rollout",
        title_prefix=f"{title_prefix} (rolling one-step-ahead rollout)"
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.965])
    plt.show()

def rollout_next_tick(model, X_full, y_full, start_i, horizon):
    """
    Generic D-dim rollout:
      X_full: (N, WINDOW, D) scaled
      y_full: (N, D) scaled
    """
    N = len(X_full)
    if start_i < 0 or start_i + horizon >= N:
        raise ValueError(f"start_i={start_i} too close to end for horizon={horizon}. N={N}")

    window = X_full[start_i].copy()  # (WINDOW, D)
    D = window.shape[1]
    preds = np.zeros((horizon, D), dtype=np.float32)
    gts   = np.zeros((horizon, D), dtype=np.float32)

    for h in range(horizon):
        yhat = model.predict(window.reshape(1, WINDOW, D), verbose=0)[0]  # (D,)
        preds[h] = yhat
        gts[h]   = y_full[start_i + h]
        window = np.vstack([window[1:], yhat])

    return preds, gts, X_full[start_i]


# =============================================================================
# Build CNN model (51x18 -> 18)
# =============================================================================
def build_timestamp_cnn_next_tick(window=51, n_features=18, dropout=0.2):
    model = Sequential([
        # Block 1: 32 -> 48 -> pool
        ZeroPadding1D(padding=2, input_shape=(window, n_features)),
        Conv1D(filters=32, kernel_size=3, strides=2, dilation_rate=1, padding="same", activation="relu"),
        ZeroPadding1D(padding=2),
        Conv1D(filters=48, kernel_size=3, strides=2, dilation_rate=1, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(dropout),

        # Block 2: 256 -> 256 -> pool
        ZeroPadding1D(padding=2),
        Conv1D(filters=256, kernel_size=3, strides=2, dilation_rate=1, padding="same", activation="relu"),
        ZeroPadding1D(padding=2),
        Conv1D(filters=256, kernel_size=3, strides=2, dilation_rate=1, padding="same", activation="relu"),
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
    ensure_dir(PRED_DIR)
    ensure_dir(SAVE_DIR)
    ensure_dir(SCALER_DIR)
    ensure_dir(PLOT_DIR)

    # ----------------------------
    # Load Typical
    # ----------------------------
    if not os.path.exists(TYPICAL_XLSX):
        raise FileNotFoundError(f"Typical file not found: {TYPICAL_XLSX}")

    df_typ = pd.read_excel(TYPICAL_XLSX, usecols=COLUMNS).fillna(0)
    if RIGHT_LEG_SHIFT != 0:
        df_typ = stridewise_roll_right_leg(df_typ, STRIDE_LEN, RIGHT_LEG_SHIFT, RIGHT_LEG_COLS)

    # ----------------------------
    # Load CP (concat)
    # ----------------------------
    df_cp = load_cp_concat(CP_FOLDER, MAX_CP_FILES)
    df_cp = divergence_fix(df_cp, COLUMNS)

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
    X_train, y_train = X_typ[val_n_typ:], y_typ[val_n_typ:]
    X_val_typ, y_val_typ = X_typ[:val_n_typ], y_typ[:val_n_typ]

    # Small CP slice into validation
    n_cp = len(X_cp)
    val_n_cp = max(1, int(0.2 * n_cp))
    X_val = np.vstack([X_val_typ, X_cp[:val_n_cp]])
    y_val = np.vstack([y_val_typ, y_cp[:val_n_cp]])

    # Test on all CP
    X_test, y_test = X_cp, y_cp

    print(f"Typical rolling samples: {len(X_typ)}")
    print(f"CP rolling samples:      {len(X_cp)}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test(CP): {len(X_test)}")
    print(f"Feature dims: {N_FEATURES} (inputs) -> {N_FEATURES} (outputs)")

    # ----------------------------
    # Build + train CNN
    # ----------------------------
    model = build_timestamp_cnn_next_tick(window=WINDOW, n_features=N_FEATURES, dropout=DROPOUT)
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
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

    pred_seg_scaled = model.predict(X_seg, verbose=0)  # (T,18)

    gt_seg_deg   = scaler.inverse_transform(y_seg)
    pred_seg_deg = scaler.inverse_transform(pred_seg_scaled)

    print_metrics_deg(gt_seg_deg, pred_seg_deg, "Timestamp CNN (next-tick, 18ch)", COLUMNS)

    # Save segment predictions
    seg_out = pd.DataFrame(
        np.hstack([gt_seg_deg, pred_seg_deg]),
        columns=[f"GT_{c}" for c in COLUMNS] + [f"CNN_{c}" for c in COLUMNS]
    )
    seg_file = os.path.join(PRED_DIR, "timestamp_cnn_next_tick_cp_segment_18ch.xlsx")
    seg_out.to_excel(seg_file, index=False)
    print(f"Saved segment comparison to: {seg_file}")

    # ----------------------------
    # Single-stride one-step-ahead plot (18ch)
    # ----------------------------
    cp_deg = df_cp[COLUMNS].to_numpy()

    k = 5
    a = k * STRIDE_LEN
    b = a + STRIDE_LEN
    if b >= len(cp_deg):
        k = max(0, (len(cp_deg) // STRIDE_LEN) - 2)
        a = k * STRIDE_LEN
        b = a + STRIDE_LEN

    stride_in_deg = cp_deg[a:b]          # (51,18)
    gt_next_deg   = cp_deg[b]            # (18,)

    # X_cp[i] corresponds to cp_scaled[i:i+51], y_cp[i] = cp_scaled[i+51]
    i = int(np.clip(a, 0, len(X_cp) - 1))
    pred_one_scaled = model.predict(X_cp[i:i+1], verbose=0)[0]  # (18,)
    pred_one_deg = scaler.inverse_transform(pred_one_scaled.reshape(1, -1))[0]

    plot_one_stride_one_step_18(
        stride_in_deg_51x18=stride_in_deg,
        pred_next_deg_18=pred_one_deg,
        gt_next_deg_18=gt_next_deg,
        title_prefix="Timestamp CNN (rolling next-tick, 18ch)"
    )

    # ----------------------------
    # Rolling rollout plot (18ch)
    # ----------------------------
    start_i = int(ROLL_START_STRIDE * STRIDE_LEN)
    start_i = int(np.clip(start_i, 0, len(X_cp) - (ROLL_H + 1)))

    pred_roll_scaled, gt_roll_scaled, input_win_scaled = rollout_next_tick(
        model, X_cp, y_cp, start_i=start_i, horizon=ROLL_H
    )

    input_win_deg = scaler.inverse_transform(input_win_scaled)     # (51,18)
    pred_roll_deg = scaler.inverse_transform(pred_roll_scaled)     # (H,18)
    gt_roll_deg   = scaler.inverse_transform(gt_roll_scaled)       # (H,18)

    plot_rollout_one_step_ahead_18(
        input_window_deg=input_win_deg,
        pred_series_deg=pred_roll_deg,
        gt_series_deg=gt_roll_deg,
        title_prefix="Timestamp CNN (rolling next-tick, 18ch)"
    )

    # ----------------------------
    # Training curves
    # ----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train Loss (MSE)")
    plt.plot(history.history["val_loss"], label="Val Loss (MSE)")
    plt.title("Timestamp CNN (next-tick, 18ch) Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "timestamp_cnn_next_tick_loss_18ch.png"), dpi=200)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Val MAE")
    plt.title("Timestamp CNN (next-tick, 18ch) MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "timestamp_cnn_next_tick_mae_18ch.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
