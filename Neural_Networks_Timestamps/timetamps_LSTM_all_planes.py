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

TYPICAL_FILE = "Data_Normal/randomized_data_healthy_all_columns.xlsx"
CP_FOLDER = "Data_CP/"
CP_SHEET = "Data"
CP_SKIPROWS = [1, 2]

SAVE_DIR = "Saved_Models"
PRED_DIR = "Predictions"
SCALER_DIR = "Scaler"
PLOT_DIR = "Plots"

# 18 columns: 6 joints (L/R hip,knee,ankle) x 3 planes (1,2,3)
ANGLE_COLS = [
    # Left
    "LHipAngles (1)", "LHipAngles (2)", "LHipAngles (3)",
    "LKneeAngles (1)", "LKneeAngles (2)", "LKneeAngles (3)",
    "LAnkleAngles (1)", "LAnkleAngles (2)", "LAnkleAngles (3)",
    # Right
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]
N_FEATURES = len(ANGLE_COLS)  # 18

RIGHT_ANGLE_COLS = [
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]

MODEL_OUT = os.path.join(SAVE_DIR, "Timestamp_lstm_next_tick_model_18.keras")
SCALER_OUT = os.path.join(SCALER_DIR, "standard_scaler_typical_lstm_next_tick_18.save")


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
# Rolling windows: (window,D) -> next tick (D,)
# ============================================================
def make_rolling_windows(series_TxD: np.ndarray, window: int):
    """
    series_TxD: (T,D)
    returns:
      X: (T-window, window, D)
      y: (T-window, D)
    """
    series_TxD = np.asarray(series_TxD)
    T, D = series_TxD.shape
    if T <= window:
        raise ValueError(f"Not enough timesteps ({T}) for window={window}")

    X = np.zeros((T - window, window, D), dtype=np.float32)
    y = np.zeros((T - window, D), dtype=np.float32)

    for i in range(T - window):
        X[i] = series_TxD[i:i + window]
        y[i] = series_TxD[i + window]

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
# Plot + metrics helpers (18ch)
# ============================================================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def reshape_to_strides(arr_TxD: np.ndarray, stride_len: int) -> np.ndarray:
    T, D = arr_TxD.shape
    n = T // stride_len
    return arr_TxD[:n * stride_len].reshape(n, stride_len, D)

def stride_mean_std(strides_NxSxD: np.ndarray):
    mu = strides_NxSxD.mean(axis=0)
    sd = strides_NxSxD.std(axis=0)
    return mu, sd

def compute_metrics_deg(pred: np.ndarray, gt: np.ndarray):
    eps = 1e-12
    err = pred - gt
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    gt_range = np.ptp(gt, axis=0)
    nrmse = rmse / (gt_range + eps)

    r = np.zeros(pred.shape[1])
    for j in range(pred.shape[1]):
        x = gt[:, j] - gt[:, j].mean()
        y = pred[:, j] - pred[:, j].mean()
        denom = (np.linalg.norm(x) * np.linalg.norm(y)) + eps
        r[j] = float(np.dot(x, y) / denom)

    ss_res = np.sum((gt - pred) ** 2, axis=0)
    ss_tot = np.sum((gt - np.mean(gt, axis=0)) ** 2, axis=0) + eps
    r2 = 1.0 - ss_res / ss_tot

    return {"MAE_deg": mae, "RMSE_deg": rmse, "NRMSE": nrmse, "Pearson_r": r, "R2": r2}

def print_metrics_table(metrics: dict, labels):
    print("\n=== Model Error Metrics (degrees) ===")
    for j, name in enumerate(labels):
        print(
            f"{name:18s} | "
            f"MAE={metrics['MAE_deg'][j]:7.2f}  "
            f"RMSE={metrics['RMSE_deg'][j]:7.2f}  "
            f"NRMSE={metrics['NRMSE'][j]:7.3f}  "
            f"r={metrics['Pearson_r'][j]:7.3f}  "
            f"R2={metrics['R2'][j]:7.3f}"
        )
    print("====================================\n")

def plot_pred_vs_gt_series(pred_angles, gt_angles, labels, title="", max_points=600):
    pred_angles = np.asarray(pred_angles)
    gt_angles = np.asarray(gt_angles)
    T = min(max_points, gt_angles.shape[0])
    t = np.arange(T)

    n = len(labels)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(10, 0.45 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, gt_angles[:T, i], label="Ground truth")
        ax.plot(t, pred_angles[:T, i], label="Predicted")
        ax.set_ylabel(labels[i], fontsize=9)
        ax.grid(True)
        if i == 0:
            ax.set_title(title)
            ax.legend()
    axes[-1].set_xlabel("Sample")
    plt.tight_layout()
    plt.show()

def plot_overlay_strides(strides, labels, title="", max_strides=30):
    """
    strides: (N, S, D)
    """
    N, S, D = strides.shape
    x = np.linspace(0, 100, S)
    mu, _ = stride_mean_std(strides)

    useN = min(N, max_strides)
    fig, axes = plt.subplots(D, 1, figsize=(14, max(10, 0.45 * D)), sharex=True)
    if D == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        for i in range(useN):
            ax.plot(x, strides[i, :, j], alpha=0.15)
        ax.plot(x, mu[:, j], linewidth=2.5, label="Mean")
        ax.set_ylabel(labels[j], fontsize=9)
        ax.grid(True)
        if j == 0:
            ax.set_title(title)
            ax.legend()
    axes[-1].set_xlabel("Gait cycle (%)")
    plt.tight_layout()
    plt.show()

def plot_phase_binned_abs_error(gt_strides, pred_strides, labels, title="Mean |Pred-GT| vs gait phase"):
    """
    gt_strides, pred_strides: (N,S,D)
    """
    S = gt_strides.shape[1]
    x = np.linspace(0, 100, S)
    abs_err = np.abs(pred_strides - gt_strides)
    mean_abs_err = abs_err.mean(axis=0)  # (S,D)
    D = mean_abs_err.shape[1]

    fig, axes = plt.subplots(D, 1, figsize=(14, max(10, 0.45 * D)), sharex=True)
    if D == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        ax.plot(x, mean_abs_err[:, j])
        ax.set_ylabel(f"{labels[j]}\n|err| (deg)", fontsize=9)
        ax.grid(True)
        if j == 0:
            ax.set_title(title)
    axes[-1].set_xlabel("Gait cycle (%)")
    plt.tight_layout()
    plt.show()

def plot_residual_hist(pred: np.ndarray, gt: np.ndarray, labels, title="Residual histograms (Pred - GT)"):
    """
    Make multiple histogram figures to keep it readable:
      - 18ch -> 3 figures of 6 hists each.
    """
    err = pred - gt
    D = err.shape[1]
    per_fig = 6
    n_figs = int(np.ceil(D / per_fig))

    for f in range(n_figs):
        a = f * per_fig
        b = min(D, (f + 1) * per_fig)
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()
        for k, j in enumerate(range(a, b)):
            ax = axes[k]
            ax.hist(err[:, j], bins=60)
            ax.set_title(labels[j], fontsize=10)
            ax.grid(True)
        # hide unused axes
        for k in range(b - a, len(axes)):
            axes[k].axis("off")

        fig.suptitle(f"{title} (channels {a+1}-{b})")
        plt.tight_layout()
        plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(SAVE_DIR)
    ensure_dir(PRED_DIR)
    ensure_dir(SCALER_DIR)
    ensure_dir(PLOT_DIR)

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

    typ_df = typ_df[ANGLE_COLS].fillna(0)

    # Apply right-leg alignment (stridewise)
    typ_df = apply_right_leg_half_stride_offset_angles(typ_df, STRIDE_LEN, RIGHT_ANGLE_COLS)

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
                fp,
                sheet_name=CP_SHEET,
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
    typ_scaled = scaler.fit_transform(typ_df[ANGLE_COLS].to_numpy()).astype(np.float32)
    cp_scaled  = scaler.transform(cp_df[ANGLE_COLS].to_numpy()).astype(np.float32)

    joblib.dump(scaler, SCALER_OUT)
    print("Saved scaler to:", SCALER_OUT)

    # -------------------------
    # Rolling windows
    # -------------------------
    X_typ, y_typ = make_rolling_windows(typ_scaled, window=WINDOW)
    X_cp,  y_cp  = make_rolling_windows(cp_scaled,  window=WINDOW)

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
    print(f"(Right half-stride shift used: {shift})")
    print(f"(Dims: {N_FEATURES} inputs -> {N_FEATURES} outputs)")

    # -------------------------
    # Train model
    # -------------------------
    model = build_model(input_dim=N_FEATURES, window=WINDOW, output_dim=N_FEATURES)
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1
    )

    model.save(MODEL_OUT, include_optimizer=True)
    print("Saved model to:", MODEL_OUT)

    loss, mae, rmse = model.evaluate(X_test, y_test, verbose=0)
    print(f"CP Test (scaled): MSE={loss:.6f}  MAE={mae:.6f}  RMSE={rmse:.6f}")

    # -------------------------
    # Plot prediction on CP (next-tick) in degrees
    # -------------------------
    T_plot = 600
    start = max(0, len(X_test) - T_plot - 1)

    pred_scaled = model.predict(X_test[start:start + T_plot], verbose=0)  # (T,18)
    gt_scaled   = y_test[start:start + T_plot]                            # (T,18)

    pred_deg = scaler.inverse_transform(pred_scaled)
    gt_deg   = scaler.inverse_transform(gt_scaled)

    # Metrics in degrees
    metrics = compute_metrics_deg(pred_deg, gt_deg)
    print_metrics_table(metrics, labels=ANGLE_COLS)

    # -------------------------
    # Stride-aligned views
    # -------------------------
    cp_angles_deg = cp_df[ANGLE_COLS].to_numpy()
    cp_strides = reshape_to_strides(cp_angles_deg, STRIDE_LEN)

    pred_strides = reshape_to_strides(pred_deg, STRIDE_LEN)
    gt_strides   = reshape_to_strides(gt_deg,   STRIDE_LEN)

    Nmin = min(len(cp_strides), len(pred_strides), len(gt_strides))
    cp_strides   = cp_strides[:Nmin]
    pred_strides = pred_strides[:Nmin]
    gt_strides   = gt_strides[:Nmin]

    plot_overlay_strides(cp_strides,   labels=ANGLE_COLS, title="CP: many strides + mean", max_strides=40)
    plot_overlay_strides(pred_strides, labels=ANGLE_COLS, title="Predicted (from CP): many strides + mean", max_strides=40)

    plot_phase_binned_abs_error(gt_strides, pred_strides, labels=ANGLE_COLS, title="Mean |Pred-GT| vs gait phase (stride bins)")
    plot_residual_hist(pred_deg, gt_deg, labels=ANGLE_COLS, title="Residual histograms (Pred - GT)")

    plot_pred_vs_gt_series(
        pred_deg, gt_deg,
        labels=ANGLE_COLS,
        title=f"Timestamp LSTM next-tick prediction (WINDOW={WINDOW}, right shift={shift})",
        max_points=600
    )

    # -------------------------
    # Save predictions
    # -------------------------
    pred_df = pd.DataFrame(pred_deg, columns=ANGLE_COLS)
    gt_df   = pd.DataFrame(gt_deg,   columns=ANGLE_COLS)

    pred_file = os.path.join(PRED_DIR, "timestamp_lstm_rolling_pred_next_tick_18ch.xlsx")
    gt_file   = os.path.join(PRED_DIR, "timestamp_lstm_rolling_gt_next_tick_18ch.xlsx")
    pred_df.to_excel(pred_file, index=False)
    gt_df.to_excel(gt_file, index=False)

    print("Saved rolling predictions to:", pred_file)
    print("Saved rolling ground truth to:", gt_file)

    # -------------------------
    # Training curves
    # -------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="Train Loss (MSE)")
    plt.plot(history.history["val_loss"], label="Val Loss (MSE)")
    plt.title("Timestamp LSTM (next-tick, 18ch) Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "timestamp_lstm_next_tick_loss_18ch.png"), dpi=200)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Val MAE")
    plt.title("Timestamp LSTM (next-tick, 18ch) MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "timestamp_lstm_next_tick_mae_18ch.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
