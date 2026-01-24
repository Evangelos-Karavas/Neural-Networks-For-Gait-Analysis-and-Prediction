#!/usr/bin/env python3
"""
Rolling-window LSTM + Phase Variable (PV) features for gait prediction (next control tick)

You said:
- Left/Right Foot Off are PERCENTAGES per stride (0..100).
- Each stride is 51 samples and starts at stance.
- You want a rolling-window (streaming-friendly) model that predicts the NEXT timestamp.

This script:
1) Loads TD (typical) and CP data.
2) Builds per-timestep PV_L and PV_R for every stride using Foot-Off % (stance fraction).
   PV ramps 0->c during stance and c->1 during swing. (c = foot_off%/100)
3) Builds rolling windows:
      X: last W timesteps of [PV_L, PV_R, 6 angles]  => shape (W, 8)
      y: angles at next tick                           => shape (6,)
4) Trains an LSTM (seq->one) to predict next-tick angles.
5) Evaluates and plots a continuous rollout on CP test data.
6) Saves model + scalers + prediction Excel.

Later in ROS2:
- compute PV online the same way (using contact timing / stance fraction estimate),
  or in sim use contact sensors to update phase.
- feed a rolling buffer of size W into the trained model each control tick.

Author: (you)
"""

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
# SETTINGS (EDIT THESE)
# ============================================================
STRIDE_LEN = 51

# Rolling window length (timesteps). Start with 20–30.
WINDOW = 25

EPOCHS = 80
BATCH_SIZE = 256

MAX_CP_FILES = 500

TYPICAL_FILE = "Data_Normal/randomized_data_healthy.xlsx"
CP_FOLDER = "Data_CP/"
CP_SHEET = "Data"

SAVE_DIR = "Saved_Models"
PRED_DIR = "Predictions"
SCALER_DIR = "Scaler"

# Column names
ANGLE_COLS = [
    "LHipAngles (1)", "LKneeAngles (1)", "LAnkleAngles (1)",
    "RHipAngles (1)", "RKneeAngles (1)", "RAnkleAngles (1)"
]
LFO_COL = "Left Foot Off"
RFO_COL = "Right Foot Off"
PV_COLS = ["PhaseVariable_Left", "PhaseVariable_Right"]
ALL_FEATURE_COLS = PV_COLS + ANGLE_COLS  # model input features

# For CP files
CP_SKIPROWS = [1, 2]


# ============================================================
# PV COMPUTATION (per timestep, per stride) using Foot-Off %
# ============================================================
def pv_from_foot_off_percent(stride_len: int, foot_off_percent: float) -> np.ndarray:
    """
    Build PV over one stride (length stride_len) given Foot-Off as % stance.
    PV ramps 0->c during stance and c->1 during swing (linear in each phase).
    """
    c = float(np.clip(foot_off_percent / 100.0, 0.05, 0.95))  # stance fraction
    stance_len = int(round(stride_len * c))
    stance_len = int(np.clip(stance_len, 1, stride_len - 1))
    swing_len = stride_len - stance_len

    pv = np.zeros(stride_len, dtype=np.float32)

    # stance: [0, c)
    pv[:stance_len] = np.linspace(0.0, c, stance_len, endpoint=False, dtype=np.float32)

    # swing: [c, 1)
    pv[stance_len:] = np.linspace(c, 1.0, swing_len, endpoint=False, dtype=np.float32)

    # Ensure [0,1)
    return np.clip(pv, 0.0, 1.0 - 1e-6)


def add_pv_columns_from_fo(df: pd.DataFrame, stride_len: int,
                           lfo_col: str, rfo_col: str) -> pd.DataFrame:
    """
    Adds PhaseVariable_Left/Right per timestep by repeating a stride PV template
    computed from Foot-Off % at the start of each stride.
    Assumes df is contiguous strides of length stride_len.
    """
    n_strides = len(df) // stride_len
    out = df.iloc[:n_strides * stride_len].copy()

    pvL = np.zeros(len(out), dtype=np.float32)
    pvR = np.zeros(len(out), dtype=np.float32)

    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len
        foL = float(out[lfo_col].iloc[a])
        foR = float(out[rfo_col].iloc[a])

        pvL[a:b] = pv_from_foot_off_percent(stride_len, foL)
        pvR[a:b] = pv_from_foot_off_percent(stride_len, foR)

    out["PhaseVariable_Left"] = pvL
    out["PhaseVariable_Right"] = pvR
    return out


# ============================================================
# ROLLING WINDOWS
# ============================================================
def make_rolling_windows(features: np.ndarray,
                         targets: np.ndarray,
                         window: int):
    """
    features: (T, D)   e.g. PV+angles
    targets:  (T, K)   e.g. angles
    returns:
      X: (N, window, D)
      y: (N, K) where y corresponds to time t (next tick) for window ending at t-1.
    """
    T = features.shape[0]
    D = features.shape[1]
    K = targets.shape[1]

    if T <= window:
        raise ValueError(f"Not enough timesteps ({T}) for window={window}")

    X = np.zeros((T - window, window, D), dtype=np.float32)
    y = np.zeros((T - window, K), dtype=np.float32)

    for i in range(T - window):
        X[i] = features[i:i + window]
        y[i] = targets[i + window]  # next tick

    return X, y


def build_model(input_dim: int, window: int, output_dim: int):
    """
    seq->one LSTM: (window, input_dim) -> (output_dim)
    """
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
# PLOTTING / ROLLOUT
# ============================================================
def plot_pred_vs_gt_series(pred_angles, gt_angles, title=""):
    """
    pred_angles, gt_angles: (T, 6) in degrees
    """
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


def rollout_one_step(model, X_seq):
    """
    X_seq: (window, D) scaled
    returns predicted next angles (6,) scaled
    """
    return model.predict(X_seq.reshape(1, X_seq.shape[0], X_seq.shape[1]), verbose=0)[0]


def plot_pv_sanity(df, stride_len=51, n_strides=3, title_prefix=""):
    total_strides = len(df) // stride_len
    end_stride = min(total_strides, n_strides)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for s in range(0, end_stride):
        a, b = s * stride_len, (s + 1) * stride_len
        x = np.arange(stride_len)
        axes[0].plot(x, df["PhaseVariable_Left"].values[a:b], label=f"stride {s}")
        axes[1].plot(x, df["PhaseVariable_Right"].values[a:b], label=f"stride {s}")

    axes[0].set_ylabel("PV Left")
    axes[1].set_ylabel("PV Right")
    axes[1].set_xlabel("Sample within stride")
    axes[0].grid(True); axes[1].grid(True)
    axes[0].set_title(f"{title_prefix} PV sanity (linear stance->swing)")
    axes[0].legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)

    # -------------------------
    # Load Typical
    # -------------------------
    if not os.path.exists(TYPICAL_FILE):
        raise FileNotFoundError(f"Typical file not found: {TYPICAL_FILE}")

    # Typical file might already have Foot Off columns; if not, you must add them.
    typ_df = pd.read_excel(TYPICAL_FILE).copy()
    needed = set(ANGLE_COLS + [LFO_COL, RFO_COL])
    missing = needed.difference(typ_df.columns)
    if missing:
        raise KeyError(f"Typical file missing columns: {sorted(missing)}")

    typ_df = typ_df[ANGLE_COLS + [LFO_COL, RFO_COL]].fillna(0)
    typ_df = add_pv_columns_from_fo(typ_df, STRIDE_LEN, LFO_COL, RFO_COL)

    plot_pv_sanity(typ_df, stride_len=STRIDE_LEN, n_strides=3, title_prefix="Typical")

    # -------------------------
    # Load CP (many files)
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
            df = pd.read_excel(fp, sheet_name=CP_SHEET,
                               usecols=ANGLE_COLS + [LFO_COL, RFO_COL],
                               skiprows=CP_SKIPROWS).fillna(0)
        except Exception as e:
            print(f"Skipping {fp} (read error): {e}")
            continue

        cp_frames.append(df)
        file_counter += 1
        if file_counter >= MAX_CP_FILES:
            break

    if not cp_frames:
        raise RuntimeError("No CP files loaded. Check CP_FOLDER / sheet / columns.")

    cp_df = pd.concat(cp_frames, ignore_index=True).fillna(0)

    # Optional: global edge divergence fix (kept from your earlier pipeline)
    for col in ANGLE_COLS:
        last_value = float(cp_df[col].values[-1])
        first_value = float(cp_df[col].values[0])
        if abs(last_value - first_value) > 2:
            mean_value = (last_value + first_value) / 2.0
            cp_df.loc[cp_df.index[-1], col] = mean_value
            cp_df.loc[cp_df.index[0], col] = mean_value

    cp_df = add_pv_columns_from_fo(cp_df, STRIDE_LEN, LFO_COL, RFO_COL)

    plot_pv_sanity(cp_df, stride_len=STRIDE_LEN, n_strides=3, title_prefix="CP")

    # -------------------------
    # Build features/targets (continuous time series)
    # -------------------------
    typ_features = typ_df[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)  # (T, 8)
    typ_targets  = typ_df[ANGLE_COLS].to_numpy(dtype=np.float32)        # (T, 6)

    cp_features = cp_df[ALL_FEATURE_COLS].to_numpy(dtype=np.float32)
    cp_targets  = cp_df[ANGLE_COLS].to_numpy(dtype=np.float32)

    # -------------------------
    # Scale: PV and angles separately (recommended)
    # -------------------------
    scaler_pv = StandardScaler()
    scaler_ang = StandardScaler()

    # Fit scalers on Typical only (common choice), apply to both.
    typ_pv_scaled = scaler_pv.fit_transform(typ_df[PV_COLS].to_numpy())
    typ_ang_scaled = scaler_ang.fit_transform(typ_df[ANGLE_COLS].to_numpy())

    cp_pv_scaled = scaler_pv.transform(cp_df[PV_COLS].to_numpy())
    cp_ang_scaled = scaler_ang.transform(cp_df[ANGLE_COLS].to_numpy())

    joblib.dump(scaler_pv, os.path.join(SCALER_DIR, "scaler_pv.save"))
    joblib.dump(scaler_ang, os.path.join(SCALER_DIR, "scaler_angles.save"))

    typ_features_scaled = np.concatenate([typ_pv_scaled, typ_ang_scaled], axis=1)  # (T,8)
    cp_features_scaled  = np.concatenate([cp_pv_scaled,  cp_ang_scaled],  axis=1)

    # Targets are angles (scaled)
    typ_targets_scaled = typ_ang_scaled  # (T,6)
    cp_targets_scaled  = cp_ang_scaled

    # -------------------------
    # Rolling windows
    # -------------------------
    X_typ, y_typ = make_rolling_windows(typ_features_scaled, typ_targets_scaled, window=WINDOW)
    X_cp,  y_cp  = make_rolling_windows(cp_features_scaled,  cp_targets_scaled,  window=WINDOW)

    # Split: train mainly on typical, test on CP
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
    # Model
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

    # Evaluate
    loss, mae, rmse = model.evaluate(X_test, y_test, verbose=0)
    print(f"CP Test: MSE={loss:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # -------------------------
    # Continuous rollout plot on CP
    # -------------------------
    # Pick a segment in CP test
    T_plot = 400  # number of ticks to plot
    start = max(0, len(X_test) - T_plot - 1)

    pred_scaled = np.zeros((T_plot, 6), dtype=np.float32)
    gt_scaled = y_test[start:start + T_plot]  # (T_plot,6)

    for i in range(T_plot):
        pred_scaled[i] = model.predict(X_test[start + i].reshape(1, WINDOW, -1), verbose=0)[0]

    # Inverse to degrees
    pred_deg = scaler_ang.inverse_transform(pred_scaled)
    gt_deg = scaler_ang.inverse_transform(gt_scaled)

    # Save to Excel (with PV ground-truth for the plotted segment)
    # PV for each predicted tick corresponds to the last timestep in the window
    pv_segment_scaled = X_test[start:start + T_plot, -1, :2]  # (T_plot,2) scaled PV at window end
    pv_segment = scaler_pv.inverse_transform(pv_segment_scaled)  # back to ~[0,1)

    out_cols = ["PhaseVariable_Left", "PhaseVariable_Right"] + ANGLE_COLS
    pred_out = np.concatenate([pv_segment, pred_deg], axis=1)
    gt_out = np.concatenate([pv_segment, gt_deg], axis=1)

    pred_df = pd.DataFrame(pred_out, columns=out_cols)
    gt_df = pd.DataFrame(gt_out, columns=out_cols)

    pred_file = os.path.join(PRED_DIR, "rolling_pred_next_tick_with_pv.xlsx")
    gt_file   = os.path.join(PRED_DIR, "rolling_gt_next_tick_with_pv.xlsx")
    pred_df.to_excel(pred_file, index=False)
    gt_df.to_excel(gt_file, index=False)

    print("Saved rolling predictions to:", pred_file)
    print("Saved rolling ground truth to:", gt_file)

    plot_pred_vs_gt_series(pred_deg, gt_deg, title=f"CP rolling prediction (WINDOW={WINDOW})")

    # -------------------------
    # Export a tiny helper snippet for ROS2 usage (printed)
    # -------------------------
    print("\nROS2 usage idea:")
    print(f"- Keep a FIFO buffer of the last WINDOW={WINDOW} timesteps of features [PV_L, PV_R, 6 angles].")
    print("- Scale PV with scaler_pv and angles with scaler_ang (loaded from disk).")
    print("- Concatenate scaled PV+angles to shape (WINDOW, 8), call model -> next angles (scaled), inverse-scale angles.")
    print("- Use those angles as reference for your controller (impedance/PD/etc.).")


if __name__ == "__main__":
    main()
