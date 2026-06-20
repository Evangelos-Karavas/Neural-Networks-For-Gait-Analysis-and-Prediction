#!/usr/bin/env python3
"""
phase_variable_cnn_all_planes.py
Phase Variable CNN — all 18 angle channels (all 3 planes per joint).

Extends phase_variable_cnn.py from sagittal-only (6 channels) to
all planes (18 channels): sagittal (1), frontal (2), transverse (3)
for each of Left/Right Hip, Knee, Ankle.

Architecture:
  Input:  (window=51, 20)  — 2 PV channels + 18 angle channels
  Output: (18,)            — next-tick prediction for all 18 angles

The phase variable is still derived from the sagittal hip angles only.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================
STRIDE_LEN = 51
WINDOW     = 51
EPOCHS     = 150
BATCH_SIZE = 512
MAX_CP_FILES = 500

TYPICAL_FILE = "Data_Normal/dynamics_total_augmented.xlsx"
CP_FOLDER    = "Data_CP/"
CP_SHEET     = "Data"
CP_SKIPROWS  = [1, 2]

SAVE_DIR   = "Saved_Models"
PRED_DIR   = "Predictions"
SCALER_DIR = "Scaler"
PLOT_DIR   = "Plots"

# 18 columns: 6 joints × 3 planes
ANGLE_COLS = [
    "LHipAngles (1)", "LHipAngles (2)", "LHipAngles (3)",
    "LKneeAngles (1)", "LKneeAngles (2)", "LKneeAngles (3)",
    "LAnkleAngles (1)", "LAnkleAngles (2)", "LAnkleAngles (3)",
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]
N_ANGLES = len(ANGLE_COLS)  # 18

RIGHT_ANGLE_COLS = [
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]

LHIP_COL = "LHipAngles (1)"
RHIP_COL = "RHipAngles (1)"
LFO_COL  = "Left Foot Off"
RFO_COL  = "Right Foot Off"

PV_COLS          = ["PhaseVariable_Left", "PhaseVariable_Right"]
ALL_FEATURE_COLS = PV_COLS + ANGLE_COLS  # 20 input features

SIDES     = ["Left", "Right"]
JOINTS    = ["Hip",  "Knee", "Ankle"]
PLANES    = ["Sagittal (1)", "Frontal (2)", "Transverse (3)"]
SIDE_BASE = [0, 9]

GT_COLOR   = "#1f77b4"
PRED_COLOR = "#d62728"


# ============================================================
# Helpers: stridewise shift
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


def apply_right_leg_half_stride_offset(
    df: pd.DataFrame,
    stride_len: int,
    right_angle_cols,
    pv_right_col: str = "PhaseVariable_Right",
) -> pd.DataFrame:
    shift = half_stride_shift(stride_len)
    out   = df.copy()
    for col in right_angle_cols:
        out[col] = roll_stridewise_1d(out[col].to_numpy(), stride_len, shift)
    out[pv_right_col] = roll_stridewise_1d(out[pv_right_col].to_numpy(), stride_len, shift)
    return out


# ============================================================
# Phase variable computation
# ============================================================
def compute_pv_stride(q: np.ndarray, c: float, enforce_monotonic: bool = True):
    q  = q.astype(np.float64)
    N  = q.shape[0]
    c  = float(np.clip(c, 0.05, 0.95))
    q0 = float(q[0])

    idx_min = int(np.argmin(q))
    qmin    = float(q[idx_min])

    denom_stance = q0 - qmin
    s = np.zeros(N, dtype=np.float64)

    if abs(denom_stance) < 1e-6:
        # Degenerate: hip angle is flat → linear fallback 0 → c
        s[:idx_min + 1] = np.linspace(0.0, c, idx_min + 1)
    else:
        s[:idx_min + 1] = ((q0 - q[:idx_min + 1]) / denom_stance) * c

    sm          = s[idx_min]
    denom_swing = q0 - qmin

    if abs(denom_swing) < 1e-6:
        # Degenerate swing phase → linear fallback sm → 1
        s[idx_min:] = np.linspace(sm, 1.0, N - idx_min)
    else:
        s[idx_min:] = 1.0 + ((1.0 - sm) / denom_swing) * (q[idx_min:] - q0)

    s = np.clip(s, 0.0, 1.0)

    if enforce_monotonic:
        for i in range(1, N):
            if s[i] < s[i - 1]:
                s[i] = s[i - 1]
    return s


def compute_phase_variables(
    df: pd.DataFrame,
    stride_len: int,
    lhip_col: str,
    rhip_col: str,
    lfo_col: str,
    rfo_col: str,
    enforce_monotonic: bool = True,
) -> pd.DataFrame:
    n_strides = len(df) // stride_len
    out = df.iloc[:n_strides * stride_len].copy()

    pvL = np.zeros(len(out), dtype=np.float32)
    pvR = np.zeros(len(out), dtype=np.float32)

    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len
        cL = float(out[lfo_col].iloc[a]) / 100.0
        cR = float(out[rfo_col].iloc[a]) / 100.0
        pvL[a:b] = compute_pv_stride(out[lhip_col].values[a:b], c=cL, enforce_monotonic=enforce_monotonic)
        pvR[a:b] = compute_pv_stride(out[rhip_col].values[a:b], c=cR, enforce_monotonic=enforce_monotonic)

    out["PhaseVariable_Left"]  = pvL
    out["PhaseVariable_Right"] = pvR
    return out


# ============================================================
# Rolling windows
# ============================================================
def make_rolling_windows(features: np.ndarray, targets: np.ndarray, window: int):
    T = features.shape[0]
    if T <= window:
        raise ValueError(f"Not enough timesteps ({T}) for window={window}")
    D, K = features.shape[1], targets.shape[1]
    X = np.zeros((T - window, window, D), dtype=np.float32)
    y = np.zeros((T - window, K), dtype=np.float32)
    for i in range(T - window):
        X[i] = features[i:i + window]
        y[i] = targets[i + window]
    return X, y


# ============================================================
# Model
# ============================================================
def build_cnn_model(input_dim: int, window: int, output_dim: int,
                    lr: float = 1e-3, dropout: float = 0.2):
    model = Sequential([
        Conv1D(32,  kernel_size=3, strides=2, padding="same", activation="relu",
               input_shape=(window, input_dim)),
        Conv1D(48,  kernel_size=3, strides=2, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(dropout),

        Conv1D(384, kernel_size=3, strides=2, padding="same", activation="relu"),
        Conv1D(384, kernel_size=3, strides=2, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(dropout),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(dropout),
        Dense(output_dim, activation="linear"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss="mse",
        metrics=["mae", RootMeanSquaredError()],
    )
    return model


# ============================================================
# Metrics
# ============================================================
def compute_metrics_deg(pred: np.ndarray, gt: np.ndarray):
    eps = 1e-12
    D   = pred.shape[1]
    err = pred - gt

    mae   = np.mean(np.abs(err), axis=0)
    rmse  = np.sqrt(np.mean(err ** 2, axis=0))
    nrmse = rmse / (np.ptp(gt, axis=0) + eps)

    r = np.zeros(D)
    for j in range(D):
        x = gt[:, j] - gt[:, j].mean()
        y = pred[:, j] - pred[:, j].mean()
        denom = (np.linalg.norm(x) * np.linalg.norm(y)) + eps
        r[j] = float(np.dot(x, y) / denom)

    ss_res = np.sum((gt - pred) ** 2, axis=0)
    ss_tot = np.sum((gt - gt.mean(axis=0)) ** 2, axis=0) + eps
    r2 = 1.0 - ss_res / ss_tot

    return {"MAE_deg": mae, "RMSE_deg": rmse, "NRMSE": nrmse, "Pearson_r": r, "R2": r2}


def print_metrics_table(metrics: dict):
    print(f"\n{'Channel':22s}  {'MAE':>7}  {'RMSE':>7}  {'NRMSE':>7}  {'r':>7}  {'R2':>7}")
    print("=" * 70)
    for j, col in enumerate(ANGLE_COLS):
        print(
            f"{col:22s}  "
            f"{metrics['MAE_deg'][j]:7.3f}  "
            f"{metrics['RMSE_deg'][j]:7.3f}  "
            f"{metrics['NRMSE'][j]:7.3f}  "
            f"{metrics['Pearson_r'][j]:7.3f}  "
            f"{metrics['R2'][j]:7.3f}"
        )
    print("=" * 70 + "\n")


# ============================================================
# Reshape helpers
# ============================================================
def reshape_to_strides(arr: np.ndarray, stride_len: int) -> np.ndarray:
    T, D = arr.shape
    n    = T // stride_len
    return arr[:n * stride_len].reshape(n, stride_len, D)


def stride_mean_std(strides: np.ndarray):
    return strides.mean(axis=0), strides.std(axis=0)


# ============================================================
# Plot helpers (3x3 Left/Right grid for 18 channels)
# ============================================================
def _grid_fig(side: str, title: str, ylabel: str):
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


def plot_pred_vs_gt_18ch(pred, gt, title="", max_samples=400):
    T = min(max_samples, len(pred))
    t = np.arange(T)
    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = _grid_fig(side, title, "Angle (deg)")
        for row in range(3):
            for col in range(3):
                idx = base + row * 3 + col
                ax  = axes[row, col]
                ax.plot(t, gt[:T, idx],   color=GT_COLOR,   lw=1.6, label="Ground truth")
                ax.plot(t, pred[:T, idx], color=PRED_COLOR, lw=1.2, ls="--", label="Predicted")
                if row == 0 and col == 0:
                    ax.legend(fontsize=8)
        for col in range(3):
            axes[2, col].set_xlabel("Sample", fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_stride_mean_std_compare_18ch(typ_strides, cp_strides, pred_strides, title=""):
    x = np.linspace(0, 100, typ_strides.shape[1])
    typ_mu, typ_sd = stride_mean_std(typ_strides)
    cp_mu,  cp_sd  = stride_mean_std(cp_strides)
    pr_mu,  pr_sd  = stride_mean_std(pred_strides)

    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = _grid_fig(side, title, "Angle (deg)")
        for row in range(3):
            for col in range(3):
                idx = base + row * 3 + col
                ax  = axes[row, col]
                ax.plot(x, typ_mu[:, idx], label="Typical mean")
                ax.fill_between(x, typ_mu[:, idx] - typ_sd[:, idx], typ_mu[:, idx] + typ_sd[:, idx], alpha=0.18)
                ax.plot(x, cp_mu[:, idx],  label="CP mean")
                ax.fill_between(x, cp_mu[:, idx] - cp_sd[:, idx],   cp_mu[:, idx] + cp_sd[:, idx],  alpha=0.18)
                ax.plot(x, pr_mu[:, idx],  ls="--", label="Pred mean")
                ax.fill_between(x, pr_mu[:, idx] - pr_sd[:, idx],   pr_mu[:, idx] + pr_sd[:, idx],  alpha=0.18)
                if row == 0 and col == 0:
                    ax.legend(fontsize=7, ncol=3)
        for col in range(3):
            axes[2, col].set_xlabel("Gait cycle (%)", fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_overlay_strides_18ch(strides, title="", max_strides=30):
    N, S, _ = strides.shape
    x  = np.linspace(0, 100, S)
    mu, _ = stride_mean_std(strides)
    use_N = min(N, max_strides)

    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = _grid_fig(side, title, "Angle (deg)")
        for row in range(3):
            for col in range(3):
                idx = base + row * 3 + col
                ax  = axes[row, col]
                for i in range(use_N):
                    ax.plot(x, strides[i, :, idx], alpha=0.15, lw=0.7)
                ax.plot(x, mu[:, idx], lw=2.5, label="Mean")
                if row == 0 and col == 0:
                    ax.legend(fontsize=8)
        for col in range(3):
            axes[2, col].set_xlabel("Gait cycle (%)", fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_phase_abs_error_18ch(gt_strides, pred_strides, title="Mean |Pred−GT| vs gait phase"):
    x       = np.linspace(0, 100, gt_strides.shape[1])
    abs_err = np.abs(pred_strides - gt_strides).mean(axis=0)

    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = _grid_fig(side, title, "|Error| (deg)")
        for row in range(3):
            for col in range(3):
                idx = base + row * 3 + col
                axes[row, col].plot(x, abs_err[:, idx], color="#2ca02c", lw=1.8)
        for col in range(3):
            axes[2, col].set_xlabel("Gait cycle (%)", fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_residual_hist_18ch(pred, gt, title="Residual histograms (Pred − GT)"):
    err     = pred - gt
    per_fig = 6
    n_figs  = int(np.ceil(N_ANGLES / per_fig))
    for f in range(n_figs):
        a, b = f * per_fig, min(N_ANGLES, (f + 1) * per_fig)
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.flatten()
        for k, j in enumerate(range(a, b)):
            axes[k].hist(err[:, j], bins=60)
            axes[k].set_title(ANGLE_COLS[j], fontsize=9)
            axes[k].grid(True)
        for k in range(b - a, len(axes)):
            axes[k].axis("off")
        fig.suptitle(f"{title} (channels {a + 1}–{b})")
        plt.tight_layout()
        plt.show()


def plot_pv_sanity(df, stride_len=51, n_strides=3, title_prefix=""):
    end_stride = min(len(df) // stride_len, n_strides)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for s in range(end_stride):
        a, b = s * stride_len, (s + 1) * stride_len
        x    = np.arange(stride_len)
        axes[0].plot(x, df["PhaseVariable_Left"].values[a:b],  label=f"stride {s}")
        axes[1].plot(x, df["PhaseVariable_Right"].values[a:b], label=f"stride {s}")
    axes[0].set_ylabel("PV Left")
    axes[1].set_ylabel("PV Right")
    axes[1].set_xlabel("Sample within stride")
    axes[0].grid(True); axes[1].grid(True)
    axes[0].set_title(f"{title_prefix}: Phase Variable")
    axes[0].legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.show()


# ============================================================
# DTW + rollout checkpointing
# ============================================================
def dtw_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    N, M = len(a), len(b)
    dp   = np.full((N + 1, M + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            dp[i, j] = abs(a[i - 1] - b[j - 1]) + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[N, M])


def dtw_distance_multidim(A: np.ndarray, B: np.ndarray, per_dim_normalize: bool = True) -> float:
    A, B  = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    total = 0.0
    eps   = 1e-9
    for d in range(A.shape[1]):
        dist = dtw_distance_1d(A[:, d], B[:, d])
        if per_dim_normalize:
            dist /= (np.ptp(B[:, d]) + eps)
        total += dist
    return float(total)


def rollout_recursive_with_measured_pv(
    model,
    X_seq_scaled: np.ndarray,
    y_seq_scaled: np.ndarray,
    scaler_ang,
    horizon: int,
    pv_dim: int = 2,
):
    T, window, D = X_seq_scaled.shape
    n_ang = D - pv_dim
    H     = min(horizon, T)

    w            = X_seq_scaled[0].copy()
    gt_scaled    = y_seq_scaled[:H]
    gt_deg       = scaler_ang.inverse_transform(gt_scaled)
    preds_scaled = np.zeros((H, n_ang), dtype=np.float32)

    for t in range(H):
        pred_t          = model.predict(w[None], verbose=0)[0]
        preds_scaled[t] = pred_t.astype(np.float32)
        if t < H - 1:
            pv_next   = X_seq_scaled[t + 1, -1, :pv_dim]
            next_feat = np.concatenate([pv_next, preds_scaled[t]], axis=0)
            w = np.roll(w, shift=-1, axis=0)
            w[-1, :] = next_feat

    pred_deg = scaler_ang.inverse_transform(preds_scaled)
    return pred_deg, gt_deg


class DTWRolloutCheckpoint(Callback):
    def __init__(
        self,
        X_val_seq, y_val_seq, scaler_ang,
        save_path: str, horizon: int,
        pv_dim: int = 2, per_dim_normalize: bool = True,
        verbose: int = 1, every_n_epochs: int = 5,
    ):
        super().__init__()
        self.Xv, self.yv       = X_val_seq, y_val_seq
        self.scaler_ang        = scaler_ang
        self.save_path         = save_path
        self.horizon           = int(horizon)
        self.pv_dim            = int(pv_dim)
        self.per_dim_normalize = bool(per_dim_normalize)
        self.verbose           = int(verbose)
        self.every_n_epochs    = max(1, int(every_n_epochs))
        self.best_dtw          = np.inf

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        pred_deg, gt_deg = rollout_recursive_with_measured_pv(
            self.model, self.Xv, self.yv, self.scaler_ang,
            horizon=self.horizon, pv_dim=self.pv_dim,
        )
        dtw = dtw_distance_multidim(pred_deg, gt_deg, per_dim_normalize=self.per_dim_normalize)
        if logs is not None:
            logs["val_rollout_dtw"] = dtw
        if self.verbose:
            print(f"\n[DTW] epoch {epoch + 1}: rollout DTW = {dtw:.4f}  (best {self.best_dtw:.4f})")
        if dtw < self.best_dtw:
            self.best_dtw = dtw
            self.model.save(self.save_path, include_optimizer=True)
            if self.verbose:
                print(f"[DTW] Saved best model → {self.save_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    for d in [SAVE_DIR, PRED_DIR, SCALER_DIR, PLOT_DIR]:
        os.makedirs(d, exist_ok=True)

    shift = half_stride_shift(STRIDE_LEN)
    print(f"[INFO] Right-leg half-stride shift = {shift} samples")

    # ── Load Typical ───────────────────────────────────────────────
    if not os.path.exists(TYPICAL_FILE):
        raise FileNotFoundError(
            f"Typical file not found: {TYPICAL_FILE}\n"
            "If 'Left Foot Off' / 'Right Foot Off' are absent, "
            "set TYPICAL_FILE = 'Data_Normal/randomized_data_healthy.xlsx'."
        )

    typ_df = pd.read_excel(TYPICAL_FILE).copy()
    needed = set(ANGLE_COLS + [LFO_COL, RFO_COL])
    missing = needed.difference(typ_df.columns)
    if missing:
        raise KeyError(
            f"Typical file missing columns: {sorted(missing)}\n"
            "Try TYPICAL_FILE = 'Data_Normal/randomized_data_healthy.xlsx'."
        )

    typ_df = typ_df[ANGLE_COLS + [LFO_COL, RFO_COL]].fillna(0)
    typ_df = compute_phase_variables(typ_df, STRIDE_LEN, LHIP_COL, RHIP_COL, LFO_COL, RFO_COL)

    # Drop rows where PV computation produced NaN (degenerate strides in augmented data)
    n_bad = typ_df[PV_COLS].isna().any(axis=1).sum()
    if n_bad > 0:
        print(f"[WARN] Dropping {n_bad} rows with NaN PV from typical data")
        typ_df = typ_df.dropna(subset=PV_COLS).reset_index(drop=True)

    typ_df = apply_right_leg_half_stride_offset(typ_df, STRIDE_LEN, RIGHT_ANGLE_COLS)

    plot_pv_sanity(typ_df, stride_len=STRIDE_LEN, n_strides=3, title_prefix="Typical")

    # ── Load CP (subject-level split: group by real patient ID, not by file) ──
    if not os.path.isdir(CP_FOLDER):
        raise FileNotFoundError(f"CP folder not found: {CP_FOLDER}")

    cp_files = [fn for fn in sorted(os.listdir(CP_FOLDER)) if fn.endswith(".xlsx")][:MAX_CP_FILES]
    if not cp_files:
        raise RuntimeError("No CP files found.")

    # Each patient contributes ~5 trial files (e.g. "36686-20240711-1-05-01.xlsx").
    # Split by patient ID so all of one patient's trials land in the same split.
    subj_to_files = {}
    for fn in cp_files:
        subj_to_files.setdefault(fn.split("-")[0], []).append(fn)
    subj_ids = sorted(subj_to_files.keys())

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(subj_ids))
    n_train = int(0.70 * len(subj_ids))
    n_val   = int(0.15 * len(subj_ids))
    train_subj = {subj_ids[i] for i in perm[:n_train]}
    val_subj   = {subj_ids[i] for i in perm[n_train:n_train + n_val]}
    # remaining subjects go to test

    train_idx, val_idx = set(), set()
    for i, fn in enumerate(cp_files):
        sid = fn.split("-")[0]
        if sid in train_subj:
            train_idx.add(i)
        elif sid in val_subj:
            val_idx.add(i)
    # remaining indices go to test

    cp_train_frames, cp_val_frames, cp_test_frames = [], [], []
    for i, fn in enumerate(cp_files):
        fp = os.path.join(CP_FOLDER, fn)
        try:
            df = pd.read_excel(
                fp, sheet_name=CP_SHEET,
                usecols=ANGLE_COLS + [LFO_COL, RFO_COL],
                skiprows=CP_SKIPROWS,
            ).fillna(0)
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            continue
        df = compute_phase_variables(df, STRIDE_LEN, LHIP_COL, RHIP_COL, LFO_COL, RFO_COL)
        df = apply_right_leg_half_stride_offset(df, STRIDE_LEN, RIGHT_ANGLE_COLS)
        if i in train_idx:
            cp_train_frames.append(df)
        elif i in val_idx:
            cp_val_frames.append(df)
        else:
            cp_test_frames.append(df)

    if not cp_train_frames or not cp_val_frames or not cp_test_frames:
        raise RuntimeError("CP subject-level split produced an empty train/val/test group.")

    cp_train_df = pd.concat(cp_train_frames, ignore_index=True).fillna(0)
    cp_val_df   = pd.concat(cp_val_frames,   ignore_index=True).fillna(0)
    cp_test_df  = pd.concat(cp_test_frames,  ignore_index=True).fillna(0)
    cp_df       = pd.concat([cp_train_df, cp_val_df, cp_test_df], ignore_index=True)  # for stride-aligned plots

    print(f"[INFO] CP subjects: {len(cp_train_frames)} train / {len(cp_val_frames)} val / {len(cp_test_frames)} test")

    plot_pv_sanity(cp_df, stride_len=STRIDE_LEN, n_strides=3, title_prefix="CP")

    # ── Scale (fit on Typical + CP-train, so the training distribution is represented) ──
    scaler_pv  = StandardScaler()
    scaler_ang = StandardScaler()

    fit_pv_df  = pd.concat([typ_df[PV_COLS],     cp_train_df[PV_COLS]],     ignore_index=True)
    fit_ang_df = pd.concat([typ_df[ANGLE_COLS],  cp_train_df[ANGLE_COLS]],  ignore_index=True)
    scaler_pv.fit(fit_pv_df.to_numpy())
    scaler_ang.fit(fit_ang_df.to_numpy())

    typ_pv_sc      = scaler_pv.transform(typ_df[PV_COLS].to_numpy())
    typ_ang_sc     = scaler_ang.transform(typ_df[ANGLE_COLS].to_numpy())
    cp_train_pv_sc  = scaler_pv.transform(cp_train_df[PV_COLS].to_numpy())
    cp_train_ang_sc = scaler_ang.transform(cp_train_df[ANGLE_COLS].to_numpy())
    cp_val_pv_sc    = scaler_pv.transform(cp_val_df[PV_COLS].to_numpy())
    cp_val_ang_sc   = scaler_ang.transform(cp_val_df[ANGLE_COLS].to_numpy())
    cp_test_pv_sc   = scaler_pv.transform(cp_test_df[PV_COLS].to_numpy())
    cp_test_ang_sc  = scaler_ang.transform(cp_test_df[ANGLE_COLS].to_numpy())

    joblib.dump(scaler_pv,  os.path.join(SCALER_DIR, "scaler_pv_cnn_18ch.save"))
    joblib.dump(scaler_ang, os.path.join(SCALER_DIR, "scaler_angles_cnn_18ch.save"))

    typ_feats      = np.concatenate([typ_pv_sc,      typ_ang_sc],      axis=1)  # (T, 20)
    cp_train_feats = np.concatenate([cp_train_pv_sc, cp_train_ang_sc], axis=1)
    cp_val_feats   = np.concatenate([cp_val_pv_sc,   cp_val_ang_sc],   axis=1)
    cp_test_feats  = np.concatenate([cp_test_pv_sc,  cp_test_ang_sc],  axis=1)

    # ── Rolling windows (built per split so windows never cross a split boundary) ──
    X_typ,      y_typ      = make_rolling_windows(typ_feats,      typ_ang_sc,      WINDOW)
    X_cp_train, y_cp_train = make_rolling_windows(cp_train_feats, cp_train_ang_sc, WINDOW)
    X_cp_val,   y_cp_val   = make_rolling_windows(cp_val_feats,   cp_val_ang_sc,   WINDOW)
    X_cp_test,  y_cp_test  = make_rolling_windows(cp_test_feats,  cp_test_ang_sc,  WINDOW)

    split_typ = max(1, int(0.2 * len(X_typ)))

    X_train = np.vstack([X_typ[split_typ + WINDOW:], X_cp_train])
    y_train = np.vstack([y_typ[split_typ + WINDOW:], y_cp_train])
    X_val   = np.vstack([X_typ[:split_typ], X_cp_val])
    y_val   = np.vstack([y_typ[:split_typ], y_cp_val])
    X_test  = X_cp_test
    y_test  = y_cp_test

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Input dim: {X_train.shape[2]}  Output dim: {y_train.shape[1]}")

    # ── Train ───────────────────────────────────────────────────────
    model = build_cnn_model(
        input_dim=X_train.shape[2],
        window=WINDOW,
        output_dim=N_ANGLES,
        lr=1e-3,
        dropout=0.2,
    )
    model.summary()

    best_path = os.path.join(SAVE_DIR, "PV_cnn_best_rollout_dtw_18ch.keras")
    ROLL_HORIZON = STRIDE_LEN * 3
    roll_len     = min(ROLL_HORIZON, len(X_val))

    dtw_cb = DTWRolloutCheckpoint(
        X_val[:roll_len], y_val[:roll_len],
        scaler_ang=scaler_ang,
        save_path=best_path,
        horizon=roll_len,
        pv_dim=2,
        every_n_epochs=5,
    )
    es_cb = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    lr_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[dtw_cb, es_cb, lr_cb],
        verbose=1,
    )

    final_path = os.path.join(SAVE_DIR, "PV_cnn_next_tick_18ch_final.keras")
    model.save(final_path, include_optimizer=True)
    print("Saved final model →", final_path)

    if os.path.exists(best_path):
        print("Loading best rollout-DTW model →", best_path)
        model = load_model(best_path)

    # ── Evaluate ────────────────────────────────────────────────────
    loss, mae, rmse = model.evaluate(X_test, y_test, verbose=0)
    print(f"CP Test: MSE={loss:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    T_plot = 400
    start  = max(0, len(X_test) - T_plot - 1)

    pred_sc  = model.predict(X_test[start:start + T_plot], verbose=0)
    gt_sc    = y_test[start:start + T_plot]
    pred_deg = scaler_ang.inverse_transform(pred_sc)
    gt_deg   = scaler_ang.inverse_transform(gt_sc)

    metrics = compute_metrics_deg(pred_deg, gt_deg)
    print_metrics_table(metrics)

    # ── Stride-aligned plots ────────────────────────────────────────
    typ_strides  = reshape_to_strides(typ_df[ANGLE_COLS].to_numpy(), STRIDE_LEN)
    cp_strides   = reshape_to_strides(cp_df[ANGLE_COLS].to_numpy(), STRIDE_LEN)
    pred_strides = reshape_to_strides(pred_deg, STRIDE_LEN)
    gt_strides   = reshape_to_strides(gt_deg,   STRIDE_LEN)

    Nmin = min(len(typ_strides), len(cp_strides), len(pred_strides), len(gt_strides))
    typ_strides  = typ_strides[:Nmin]
    cp_strides   = cp_strides[:Nmin]
    pred_strides = pred_strides[:Nmin]
    gt_strides   = gt_strides[:Nmin]

    plot_stride_mean_std_compare_18ch(
        typ_strides, cp_strides, pred_strides,
        title="Typical vs CP vs Predicted (from CP)"
    )
    plot_overlay_strides_18ch(cp_strides,   title="CP: strides + mean", max_strides=40)
    plot_overlay_strides_18ch(pred_strides, title="Predicted: strides + mean", max_strides=40)
    plot_phase_abs_error_18ch(gt_strides, pred_strides)
    plot_residual_hist_18ch(pred_deg, gt_deg)
    plot_pred_vs_gt_18ch(pred_deg, gt_deg, title=f"PV CNN next-tick (WINDOW={WINDOW})")

    # ── Save predictions ────────────────────────────────────────────
    pv_seg   = scaler_pv.inverse_transform(X_test[start:start + T_plot, -1, :2])
    out_cols = PV_COLS + ANGLE_COLS

    pred_out = np.concatenate([pv_seg, pred_deg], axis=1)
    gt_out   = np.concatenate([pv_seg, gt_deg],   axis=1)

    pd.DataFrame(pred_out, columns=out_cols).to_excel(
        os.path.join(PRED_DIR, "rolling_pred_next_tick_with_pv_cnn_18ch.xlsx"), index=False
    )
    pd.DataFrame(gt_out, columns=out_cols).to_excel(
        os.path.join(PRED_DIR, "rolling_gt_next_tick_with_pv_cnn_18ch.xlsx"), index=False
    )
    print("Saved predictions to", PRED_DIR)

    # ── Training curves ─────────────────────────────────────────────
    for metric, label in [("loss", "MSE Loss"), ("mae", "MAE")]:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history[metric],          label=f"Train {label}")
        plt.plot(history.history[f"val_{metric}"], label=f"Val {label}")
        plt.title(f"PV CNN 18ch — {label}")
        plt.xlabel("Epoch"); plt.ylabel(label)
        plt.grid(True); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"pv_cnn_18ch_{metric}.png"), dpi=200)
        plt.show()


if __name__ == "__main__":
    main()
