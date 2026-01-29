#!/usr/bin/env python3
"""
Timestamp CNN stride-to-next-stride prediction (51x6 -> 51x6)

This script is intentionally aligned with the timestamp LSTM pipeline:
  - Fit scaler on Typical only, transform CP with the same scaler
  - Reshape into strides of length 51
  - Build (stride_i -> stride_{i+1}) pairs
  - Train on Typical, validate on Typical+CP, test on CP
  - Regression metrics: MAE + RMSE (no accuracy)
  - Autoregressive rollout for future stride prediction
  - Stride-aligned plotting utilities

Expected folders/files (same as your LSTM script):
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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape, ZeroPadding1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
# --- Keras save-model version workaround (same pattern as your other scripts)
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__


# =============================================================================
# Configuration
# =============================================================================
STRIDE_LEN = 51
N_FEATURES = 6

COLUMNS = [
    'LHipAngles (1)', 'LKneeAngles (1)', 'LAnkleAngles (1)',
    'RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)'
]

TYPICAL_XLSX = "Data_Normal/randomized_data_healthy.xlsx"
CP_FOLDER = "Data_CP/"

# Right-leg cyclic shift (per-stride), if you want to match your previous alignment
RIGHT_LEG_SHIFT = 25  # samples
RIGHT_LEG_COLS = ['RHipAngles (1)', 'RKneeAngles (1)', 'RAnkleAngles (1)']

# Training hyperparameters (choose to match your LSTM defaults)
LR = 0.001
EPOCHS = 150
BATCH_SIZE = 102
DROPOUT = 0.2

# Rollout / plotting
N_ROLLOUT_STRIDES = 4   # how many future strides to predict for comparison plots
START_STRIDE_TYPICAL = 1  # start index for rollout (can be negative, like Python indexing)
START_STRIDE_CP = 1


# =============================================================================
# Utilities
# =============================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def divergence_fix(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    Same idea as your LSTM script: if the first/last diverge too much, nudge them.
    This is a light "wraparound" fix to avoid extreme endpoint mismatch.
    """
    if df.empty:
        return df

    for col in cols:
        last_value = df[col].values[-1]
        first_value = df[col].values[0]
        divergence = np.abs(last_value - first_value)

        if divergence > 5 or divergence > 2:
            mean_value = (last_value + first_value) / 2
            df.loc[df.index[-1], col] = mean_value
            df.loc[df.index[0], col] = mean_value
    return df

def stridewise_roll_right_leg(df: pd.DataFrame, stride_len: int, delay: int, right_cols) -> pd.DataFrame:
    """
    Roll the right-leg columns *inside each stride* to avoid boundary mixing
    (matches the improved approach from your LSTM codebase).
    """
    if delay == 0:
        return df

    arr = df[right_cols].to_numpy()
    n = (len(arr) // stride_len) * stride_len
    if n == 0:
        return df

    arr = arr[:n].reshape(-1, stride_len, len(right_cols))
    arr = np.roll(arr, shift=delay, axis=1)  # roll within each stride
    rolled = arr.reshape(n, len(right_cols))

    df2 = df.copy()
    df2.loc[df2.index[:n], right_cols] = rolled
    return df2

def make_next_stride_pairs(strides_51x6: np.ndarray):
    """X: stride i, Y: stride i+1"""
    if len(strides_51x6) < 2:
        raise ValueError("Need at least 2 strides to build next-stride pairs.")
    return strides_51x6[:-1], strides_51x6[1:]

def evaluate_model(model, X, Y, label: str):
    results = model.evaluate(X, Y, verbose=0)
    # order: [loss, mae, rmse]
    loss, mae, rmse = results[0], results[1], results[2]
    print(f"🔹 {label} Evaluation:")
    print(f"Loss (MSE): {loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    print("=" * 90)

def predict_future_strides(model, start_stride_scaled_51x6: np.ndarray, num_strides: int, scaler: StandardScaler) -> np.ndarray:
    """
    Autoregressive rollout:
      start_stride_scaled_51x6: (51,6) in SCALED space
      returns: (num_strides*51, 6) in ORIGINAL scale
    """
    preds_scaled = []
    current = start_stride_scaled_51x6.reshape(1, STRIDE_LEN, N_FEATURES)

    for _ in range(num_strides):
        next_stride = model.predict(current, verbose=0)   # (1,51,6)
        preds_scaled.append(next_stride[0])              # (51,6)
        current = next_stride                            # feed prediction back in

    preds_scaled = np.vstack(preds_scaled)               # (num_strides*51, 6)
    return scaler.inverse_transform(preds_scaled)

def build_gt_future_strides_orig(df_orig: pd.DataFrame, i_start_stride: int, num_strides: int) -> np.ndarray:
    """
    Build ground-truth future strides (original scale) for plotting.
    Returns (num_strides*51, 6) corresponding to strides i_start+1 ... i_start+num_strides.
    """
    arr = df_orig[COLUMNS].to_numpy()
    n_strides = len(arr) // STRIDE_LEN
    arr = arr[:n_strides * STRIDE_LEN].reshape(n_strides, STRIDE_LEN, N_FEATURES)

    # Normalize negative indices
    if i_start_stride < 0:
        i_start_stride = n_strides + i_start_stride

    i1 = i_start_stride + 1
    i2 = i_start_stride + 1 + num_strides
    if i2 > n_strides:
        raise ValueError(f"Not enough strides for ground truth: start={i_start_stride}, need up to {i2-1}, have {n_strides-1}")

    gt = arr[i1:i2].reshape(num_strides * STRIDE_LEN, N_FEATURES)
    return gt

def plot_comparison(predicted: np.ndarray, actual: np.ndarray, title_prefix: str, save_path: str = None):
    """Plots actual vs predicted joint angles for all 6 channels."""
    time = np.arange(actual.shape[0])

    labels_left = ['LHipAngles', 'LKneeAngles', 'LAnkleAngles']
    labels_right = ['RHipAngles', 'RKneeAngles', 'RAnkleAngles']

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

    for i, name in enumerate(labels_left):
        axes[i].plot(time, actual[:, i], label=f"Actual {name}")
        axes[i].plot(time, predicted[:, i], label=f"Predicted {name}", linestyle='dashed')
        axes[i].set_ylabel("Angle")
        axes[i].legend()
        axes[i].set_title(f"{title_prefix}: {name}")

    for i, name in enumerate(labels_right):
        j = i + 3
        axes[j].plot(time, actual[:, j], label=f"Actual {name}")
        axes[j].plot(time, predicted[:, j], label=f"Predicted {name}", linestyle='dashed')
        axes[j].set_ylabel("Angle")
        axes[j].legend()
        axes[j].set_title(f"{title_prefix}: {name}")

    axes[-1].set_xlabel("Time (samples)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_multiple_knee_predictions(actual: np.ndarray, predicted_strides: list, label: str, save_path: str = None):
    """
    Knee-only plot for multiple predicted strides stacked in time.
    actual: (num_strides*51, 6)  original scale
    predicted_strides: list of (51,6) original-scale strides, length = num_strides
    """
    pred = np.vstack(predicted_strides)
    time = np.arange(actual.shape[0])

    # Left knee is col 1, right knee is col 4
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(time, actual[:, 1], label=f"Actual LKnee ({label})")
    axes[0].plot(time, pred[:, 1], label=f"Predicted LKnee ({label})", linestyle='dashed')
    axes[0].set_ylabel("Angle")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time, actual[:, 4], label=f"Actual RKnee ({label})")
    axes[1].plot(time, pred[:, 4], label=f"Predicted RKnee ({label})", linestyle='dashed')
    axes[1].set_ylabel("Angle")
    axes[1].set_xlabel("Time (samples)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


# =============================================================================
# Main
# =============================================================================
def main():
    # Create directories
    ensure_dir("Predictions")
    ensure_dir("Saved_Models")
    ensure_dir("Scaler")
    ensure_dir("Plots")

    # ----------------------------
    # Load Typical
    # ----------------------------
    if not os.path.exists(TYPICAL_XLSX):
        raise FileNotFoundError(f"Typical file not found: {TYPICAL_XLSX}")

    data_typical = pd.read_excel(TYPICAL_XLSX, usecols=COLUMNS)
    data_typical.fillna(0, inplace=True)

    # ----------------------------
    # Load CP (concatenate many xlsx files)
    # ----------------------------
    if not os.path.isdir(CP_FOLDER):
        raise FileNotFoundError(f"CP folder not found: {CP_FOLDER}")

    data_cp = pd.DataFrame()
    file_counter = 0
    for file_name in os.listdir(CP_FOLDER):
        if file_name.endswith(".xlsx"):
            file_counter += 1
            file_path = os.path.join(CP_FOLDER, file_name)
            df = pd.read_excel(file_path, "Data", usecols=COLUMNS, skiprows=[1, 2])
            df.fillna(0, inplace=True)
            data_cp = pd.concat([data_cp, df], ignore_index=True)

            if file_counter >= 500:
                break

    data_cp = divergence_fix(data_cp, COLUMNS)
    data_cp.fillna(0, inplace=True)

    # Optional: per-stride right-leg roll (useful if your CP data is phase-shifted)
    if RIGHT_LEG_SHIFT != 0 and not data_cp.empty:
        data_cp = stridewise_roll_right_leg(data_cp, STRIDE_LEN, RIGHT_LEG_SHIFT, RIGHT_LEG_COLS)

    # ----------------------------
    # Scaling (FIT on Typical, TRANSFORM CP)  ✅ matches LSTM pipeline
    # ----------------------------
    scaler = StandardScaler()
    typical_scaled = scaler.fit_transform(data_typical[COLUMNS])
    joblib.dump(scaler, "Scaler/standard_scaler_typical_cnn.save")

    cp_scaled = scaler.transform(data_cp[COLUMNS])
    joblib.dump(scaler, "Scaler/standard_scaler_cp_cnn.save")

    # ----------------------------
    # Reshape to strides
    # ----------------------------
    n_typ = len(typical_scaled) // STRIDE_LEN
    n_cp = len(cp_scaled) // STRIDE_LEN

    strides_typ = typical_scaled[:n_typ * STRIDE_LEN].reshape(n_typ, STRIDE_LEN, N_FEATURES)
    strides_cp = cp_scaled[:n_cp * STRIDE_LEN].reshape(n_cp, STRIDE_LEN, N_FEATURES)

    # Build next-stride pairs
    X_typ, Y_typ = make_next_stride_pairs(strides_typ)
    X_cp, Y_cp = make_next_stride_pairs(strides_cp)

    # Split indices on pairs (not raw strides)
    split_idx_typ = max(1, int(0.2 * len(X_typ)))
    split_idx_cp = max(1, int(0.2 * len(X_cp)))

    # Train on typical (holdout first chunk for val), validate on mix, test on CP
    X_train, Y_train = X_typ[split_idx_typ:], Y_typ[split_idx_typ:]
    X_val = np.vstack((X_typ[:split_idx_typ], X_cp[:split_idx_cp]))
    Y_val = np.vstack((Y_typ[:split_idx_typ], Y_cp[:split_idx_cp]))
    X_test, Y_test = X_cp, Y_cp

    print(f"Typical strides: {len(strides_typ)}  | pairs: {len(X_typ)}")
    print(f"CP strides:      {len(strides_cp)}   | pairs: {len(X_cp)}")
    print(f"Train pairs: {len(X_train)} | Val pairs: {len(X_val)} | Test pairs: {len(X_test)}")

    # ----------------------------
    # Build CNN model (51x6 -> 51x6)
    # ----------------------------
    model = Sequential([
        # Block 1: 32, 48 then pool
        ZeroPadding1D(padding=2, input_shape=(STRIDE_LEN, N_FEATURES)),
        Conv1D(filters=32, kernel_size=3, strides=2, dilation_rate=1,
            padding='valid', activation='relu'),
        ZeroPadding1D(padding=2),
        Conv1D(filters=48, kernel_size=3, strides=2, dilation_rate=1,
            padding='valid', activation='relu'),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(DROPOUT),

        # Block 2: 256, 256 then pool
        ZeroPadding1D(padding=2),
        Conv1D(filters=256, kernel_size=3, strides=2, dilation_rate=1,
            padding='valid', activation='relu'),
        ZeroPadding1D(padding=2),
        Conv1D(filters=256, kernel_size=3, strides=2, dilation_rate=1,
            padding='valid', activation='relu'),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(DROPOUT),

        # FC head -> full stride output
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(DROPOUT),

        Dense(STRIDE_LEN * N_FEATURES, activation='linear'),
        Reshape((STRIDE_LEN, N_FEATURES))
    ])

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='mse',
        metrics=['mae', RootMeanSquaredError()]
    )

    model.summary()

    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, Y_val),
        verbose=1
    )

    # ----------------------------
    # Evaluate + Save
    # ----------------------------
    evaluate_model(model, X_train, Y_train, "Training Data")
    evaluate_model(model, X_val, Y_val, "Validation Data")
    evaluate_model(model, X_test, Y_test, "Testing Data")

    model.save("Saved_Models/Timestamp_cnn_model.keras", include_optimizer=True)

    # ----------------------------
    # Autoregressive rollout plots (stride-aligned, like your LSTM plotting intent)
    # ----------------------------
    # Rebuild ORIGINAL strides for GT from the original dataframes
    # (we keep these in original units, not scaled)
    data_typical_orig = data_typical.copy()
    data_cp_orig = data_cp.copy()

    # Choose safe start indices
    def safe_start(n_strides, requested):
        idx = requested
        if idx < 0:
            idx = n_strides + idx
        # need idx + N_ROLLOUT_STRIDES <= n_strides - 1 (because we compare future strides)
        max_start = (n_strides - 1) - N_ROLLOUT_STRIDES
        return int(np.clip(idx, 0, max_start))

    n_strides_typ = len(strides_typ)
    n_strides_cp = len(strides_cp)

    start_typ = safe_start(n_strides_typ, START_STRIDE_TYPICAL)
    start_cp  = safe_start(n_strides_cp,  START_STRIDE_CP)

    # Predictions (orig scale)
    pred_typ = predict_future_strides(model, strides_typ[start_typ], N_ROLLOUT_STRIDES, scaler)
    pred_cp  = predict_future_strides(model, strides_cp[start_cp],  N_ROLLOUT_STRIDES, scaler)

    # Ground truth (orig scale)
    gt_typ = build_gt_future_strides_orig(data_typical_orig, start_typ, N_ROLLOUT_STRIDES)
    gt_cp  = build_gt_future_strides_orig(data_cp_orig,      start_cp,  N_ROLLOUT_STRIDES)

    # Save predictions
    pd.DataFrame(pred_typ, columns=COLUMNS).to_csv("Predictions/predicted_future_strides_typical_cnn.csv", index=False)
    pd.DataFrame(pred_cp,  columns=COLUMNS).to_csv("Predictions/predicted_future_strides_cp_cnn.csv", index=False)

    # Full 6-channel comparison
    plot_comparison(pred_typ, gt_typ, title_prefix="CNN Typical Rollout",
                    save_path="Plots/cnn_typical_rollout_all_joints.png")
    plot_comparison(pred_cp, gt_cp, title_prefix="CNN CP Rollout",
                    save_path="Plots/cnn_cp_rollout_all_joints.png")

    # Knee-only multi-stride plot (stacked)
    pred_strides_typ = [pred_typ[i*STRIDE_LEN:(i+1)*STRIDE_LEN, :] for i in range(N_ROLLOUT_STRIDES)]
    pred_strides_cp  = [pred_cp[i*STRIDE_LEN:(i+1)*STRIDE_LEN, :]  for i in range(N_ROLLOUT_STRIDES)]

    plot_multiple_knee_predictions(gt_typ, pred_strides_typ, label="Typical",
                                   save_path="Plots/cnn_typical_rollout_knees.png")
    plot_multiple_knee_predictions(gt_cp, pred_strides_cp, label="CP",
                                   save_path="Plots/cnn_cp_rollout_knees.png")

    # Optional: training curves
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Training Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title("CNN Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/cnn_loss_curve.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title("CNN MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/cnn_mae_curve.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
