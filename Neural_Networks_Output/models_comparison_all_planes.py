#!/usr/bin/env python3
"""
models_comparison_all_planes.py
Compare all 4 trained 18-channel models (Timestamp CNN/LSTM, PV CNN/LSTM)
against multiple different held-out CP test SUBJECTS, not a single fixed
saved segment.

For each held-out CP test file, this script:
  - rebuilds that subject's own scaled rolling windows (per-model feature
    representation), so windows never cross a subject/file boundary
  - runs teacher-forced next-tick prediction across the whole file
  - computes per-channel metrics (MAE, RMSE, R2)

It then reports:
  - a per-subject x per-model summary table (mean MAE / mean R2 over the
    18 channels), so you can see how consistent each model is across
    different real patients
  - optional full neural_networks_outputs/plots (pred vs gt, phase-binned error) for one chosen
    subject across all 4 models

Run from project root:
  python models_comparison_all_planes.py --list-subjects
  python models_comparison_all_planes.py                      # table over all held-out subjects
  python models_comparison_all_planes.py --subject 3 --plot    # neural_networks_outputs/plots for one subject
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ============================================================
# SETTINGS (must match the *_all_planes.py training scripts)
# ============================================================
STRIDE_LEN = 51
WINDOW = 51

CP_FOLDER = "Data_CP/"
CP_SHEET = "Data"
CP_SKIPROWS = [1, 2]
MAX_CP_FILES = 500
TRAIN_FRAC, VAL_FRAC, SPLIT_SEED = 0.70, 0.15, 42

SAVE_DIR = "Neural_Networks_Outputs/Saved_Models"
NEURAL_NETWORKS_OUTPUTS/SCALER_DIR = "Neural_Networks_Outputs/Scaler"
PLOT_DIR = "Neural_Networks_Outputs/Plots"

ANGLE_COLS = [
    "LHipAngles (1)", "LHipAngles (2)", "LHipAngles (3)",
    "LKneeAngles (1)", "LKneeAngles (2)", "LKneeAngles (3)",
    "LAnkleAngles (1)", "LAnkleAngles (2)", "LAnkleAngles (3)",
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]
RIGHT_ANGLE_COLS = [
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]
LHIP_COL, RHIP_COL = "LHipAngles (1)", "RHipAngles (1)"
LFO_COL, RFO_COL = "Left Foot Off", "Right Foot Off"
PV_COLS = ["PhaseVariable_Left", "PhaseVariable_Right"]

SIDES, JOINTS, PLANES = ["Left", "Right"], ["Hip", "Knee", "Ankle"], ["Sagittal (1)", "Frontal (2)", "Transverse (3)"]
SIDE_BASE = [0, 9]
GT_COLOR, PRED_COLOR = "#1f77b4", "#d62728"

MODEL_NAMES = ["timestamp_cnn", "timestamp_lstm", "pv_cnn", "pv_lstm"]
MODEL_LABELS = {
    "timestamp_cnn": "Timestamp CNN",
    "timestamp_lstm": "Timestamp LSTM",
    "pv_cnn": "PV CNN",
    "pv_lstm": "PV LSTM",
}


# ============================================================
# Helpers shared with the training scripts
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


def apply_right_leg_half_stride_offset(df: pd.DataFrame, stride_len: int, right_angle_cols, pv_right_col=None) -> pd.DataFrame:
    shift = half_stride_shift(stride_len)
    out = df.copy()
    for col in right_angle_cols:
        out[col] = roll_stridewise_1d(out[col].to_numpy(), stride_len, shift)
    if pv_right_col is not None and pv_right_col in out.columns:
        out[pv_right_col] = roll_stridewise_1d(out[pv_right_col].to_numpy(), stride_len, shift)
    return out


def compute_pv_stride(q: np.ndarray, c: float, enforce_monotonic: bool = True) -> np.ndarray:
    q = q.astype(np.float64)
    N = q.shape[0]
    c = float(np.clip(c, 0.05, 0.95))
    q0 = float(q[0])
    idx_min = int(np.argmin(q))
    qmin = float(q[idx_min])
    denom_stance = q0 - qmin
    s = np.zeros(N, dtype=np.float64)
    if abs(denom_stance) < 1e-6:
        s[:idx_min + 1] = np.linspace(0.0, c, idx_min + 1)
    else:
        s[:idx_min + 1] = ((q0 - q[:idx_min + 1]) / denom_stance) * c
    sm = s[idx_min]
    denom_swing = q0 - qmin
    if abs(denom_swing) < 1e-6:
        s[idx_min:] = np.linspace(sm, 1.0, N - idx_min)
    else:
        s[idx_min:] = 1.0 + ((1.0 - sm) / denom_swing) * (q[idx_min:] - q0)
    s = np.clip(s, 0.0, 1.0)
    if enforce_monotonic:
        for i in range(1, N):
            if s[i] < s[i - 1]:
                s[i] = s[i - 1]
    return s


def compute_phase_variables(df: pd.DataFrame, stride_len: int) -> pd.DataFrame:
    n_strides = len(df) // stride_len
    out = df.iloc[:n_strides * stride_len].copy()
    pvL = np.zeros(len(out), dtype=np.float32)
    pvR = np.zeros(len(out), dtype=np.float32)
    for s in range(n_strides):
        a, b = s * stride_len, (s + 1) * stride_len
        cL = float(out[LFO_COL].iloc[a]) / 100.0
        cR = float(out[RFO_COL].iloc[a]) / 100.0
        pvL[a:b] = compute_pv_stride(out[LHIP_COL].values[a:b], c=cL)
        pvR[a:b] = compute_pv_stride(out[RHIP_COL].values[a:b], c=cR)
    out["PhaseVariable_Left"] = pvL
    out["PhaseVariable_Right"] = pvR
    return out


def make_rolling_windows(features: np.ndarray, targets: np.ndarray, window: int):
    T = features.shape[0]
    if T <= window:
        return None, None
    D, K = features.shape[1], targets.shape[1]
    X = np.zeros((T - window, window, D), dtype=np.float32)
    y = np.zeros((T - window, K), dtype=np.float32)
    for i in range(T - window):
        X[i] = features[i:i + window]
        y[i] = targets[i + window]
    return X, y


# ============================================================
# Recover the held-out CP test file list (identical split logic
# to the 4 *_all_planes.py training scripts)
# ============================================================
def get_cp_test_subjects():
    """
    Returns {subject_id: [trial files...]} for the held-out test subjects,
    using the identical patient-level split as the 4 training scripts.
    Each CP file is a single 51-row stride; each real patient has ~5 trial
    files, so a subject's trials must be concatenated to get enough rows
    to form even one rolling window.
    """
    cp_files = [fn for fn in sorted(os.listdir(CP_FOLDER)) if fn.endswith(".xlsx")][:MAX_CP_FILES]
    subj_to_files = {}
    for fn in cp_files:
        subj_to_files.setdefault(fn.split("-")[0], []).append(fn)
    subj_ids = sorted(subj_to_files.keys())

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(subj_ids))
    n_train = int(TRAIN_FRAC * len(subj_ids))
    n_val = int(VAL_FRAC * len(subj_ids))
    test_subj = sorted(subj_ids[i] for i in perm[n_train + n_val:])
    return {sid: subj_to_files[sid] for sid in test_subj}


def load_cp_subject(files) -> pd.DataFrame:
    """Concatenate one patient's trial files (each a single 51-row stride)."""
    frames = []
    for fn in files:
        fp = os.path.join(CP_FOLDER, fn)
        df = pd.read_excel(
            fp, sheet_name=CP_SHEET,
            usecols=ANGLE_COLS + [LFO_COL, RFO_COL],
            skiprows=CP_SKIPROWS,
        ).fillna(0)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ============================================================
# Model + neural_networks_outputs/scaler loading
# ============================================================
def load_all_models():
    models = {}

    models["timestamp_cnn"] = dict(
        model=load_model(os.path.join(SAVE_DIR, "Timestamp_cnn_next_tick_model_18.keras")),
        neural_networks_outputs/scaler=joblib.load(os.path.join(NEURAL_NETWORKS_OUTPUTS/SCALER_DIR, "standard_neural_networks_outputs/scaler_typical_cnn_next_tick_18.save")),
        kind="timestamp",
    )
    models["timestamp_lstm"] = dict(
        model=load_model(os.path.join(SAVE_DIR, "Timestamp_lstm_next_tick_model_18.keras")),
        neural_networks_outputs/scaler=joblib.load(os.path.join(NEURAL_NETWORKS_OUTPUTS/SCALER_DIR, "standard_neural_networks_outputs/scaler_typical_lstm_next_tick_18.save")),
        kind="timestamp",
    )
    models["pv_cnn"] = dict(
        model=load_model(os.path.join(SAVE_DIR, "PV_cnn_best_rollout_dtw_18ch.keras")),
        neural_networks_outputs/scaler_pv=joblib.load(os.path.join(NEURAL_NETWORKS_OUTPUTS/SCALER_DIR, "neural_networks_outputs/scaler_pv_cnn_18ch.save")),
        neural_networks_outputs/scaler_ang=joblib.load(os.path.join(NEURAL_NETWORKS_OUTPUTS/SCALER_DIR, "neural_networks_outputs/scaler_angles_cnn_18ch.save")),
        kind="pv",
    )
    models["pv_lstm"] = dict(
        model=load_model(os.path.join(SAVE_DIR, "PV_lstm_best_rollout_dtw_18ch.keras")),
        neural_networks_outputs/scaler_pv=joblib.load(os.path.join(NEURAL_NETWORKS_OUTPUTS/SCALER_DIR, "neural_networks_outputs/scaler_pv_lstm_18ch.save")),
        neural_networks_outputs/scaler_ang=joblib.load(os.path.join(NEURAL_NETWORKS_OUTPUTS/SCALER_DIR, "neural_networks_outputs/scaler_angles_lstm_18ch.save")),
        kind="pv",
    )
    return models


def predict_next_tick(model_entry, df_subject: pd.DataFrame):
    """
    Builds this subject's own rolling windows (no cross-file leakage) for the
    given model's feature representation, runs next-tick prediction, and
    returns (pred_deg, gt_deg) both (N,18), or (None, None) if too short.
    """
    df = df_subject.copy()
    df = apply_right_leg_half_stride_offset(df, STRIDE_LEN, RIGHT_ANGLE_COLS)

    if model_entry["kind"] == "timestamp":
        neural_networks_outputs/scaler = model_entry["neural_networks_outputs/scaler"]
        ang_sc = neural_networks_outputs/scaler.transform(df[ANGLE_COLS].to_numpy()).astype(np.float32)
        X, y = make_rolling_windows(ang_sc, ang_sc, WINDOW)
        if X is None:
            return None, None
        pred_sc = model_entry["model"].predict(X, verbose=0)
        pred_sc = np.asarray(pred_sc)
        if pred_sc.ndim == 3:        # multi-step head -> take immediate next-tick
            pred_sc = pred_sc[:, 0, :]
        pred_deg = neural_networks_outputs/scaler.inverse_transform(pred_sc)
        gt_deg = neural_networks_outputs/scaler.inverse_transform(y)
        return pred_deg, gt_deg

    else:  # PV models
        df = compute_phase_variables(df, STRIDE_LEN)
        df = apply_right_leg_half_stride_offset(df, STRIDE_LEN, RIGHT_ANGLE_COLS, pv_right_col="PhaseVariable_Right")
        neural_networks_outputs/scaler_pv, neural_networks_outputs/scaler_ang = model_entry["neural_networks_outputs/scaler_pv"], model_entry["neural_networks_outputs/scaler_ang"]
        pv_sc = neural_networks_outputs/scaler_pv.transform(df[PV_COLS].to_numpy())
        ang_sc = neural_networks_outputs/scaler_ang.transform(df[ANGLE_COLS].to_numpy())
        feats = np.concatenate([pv_sc, ang_sc], axis=1)
        X, y = make_rolling_windows(feats, ang_sc, WINDOW)
        if X is None:
            return None, None
        pred_sc = model_entry["model"].predict(X, verbose=0)
        pred_deg = neural_networks_outputs/scaler_ang.inverse_transform(pred_sc)
        gt_deg = neural_networks_outputs/scaler_ang.inverse_transform(y)
        return pred_deg, gt_deg


# ============================================================
# Metrics
# ============================================================
def compute_metrics_deg(pred: np.ndarray, gt: np.ndarray):
    eps = 1e-12
    err = pred - gt
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((gt - gt.mean(axis=0)) ** 2, axis=0) + eps
    r2 = 1.0 - ss_res / ss_tot
    return mae, rmse, r2


# ============================================================
# Neural_Networks_Outputs/Plots (one subject, all 4 models)
# ============================================================
def _grid_fig(side, title, ylabel):
    fig, axes = plt.subneural_networks_outputs/plots(3, 3, figsize=(15, 9), sharex=True)
    fig.suptitle(f"{title} — {side} Side", fontsize=12)
    for row, joint in enumerate(JOINTS):
        for col, plane in enumerate(PLANES):
            ax = axes[row, col]
            ax.set_title(f"{side} {joint} – {plane}", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.35)
    return fig, axes


def plot_subject_all_models(preds: dict, gt_ref: np.ndarray, subject_name: str, save_dir=None):
    T = min(300, gt_ref.shape[0])
    t = np.arange(T)
    colors = {"timestamp_cnn": "#d62728", "timestamp_lstm": "#2ca02c", "pv_cnn": "#9467bd", "pv_lstm": "#ff7f0e"}

    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = _grid_fig(side, f"Subject {subject_name}: Predicted vs Ground Truth", "Angle (deg)")
        for row in range(3):
            for col in range(3):
                idx = base + row * 3 + col
                ax = axes[row, col]
                ax.plot(t, gt_ref[:T, idx], color=GT_COLOR, lw=1.8, label="Ground truth")
                for name, pred in preds.items():
                    if pred is None:
                        continue
                    ax.plot(t[:min(T, len(pred))], pred[:T, idx], color=colors[name], lw=1.1, ls="--", label=MODEL_LABELS[name])
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)
        for col in range(3):
            axes[2, col].set_xlabel("Sample", fontsize=8)
        plt.tight_layout()
        if save_dir:
            fname = os.path.join(save_dir, f"compare4_{side.lower()}_subject_{subject_name}.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"Saved: {fname}")
        plt.show()


def plot_phase_error_all_models(preds: dict, gt_ref: np.ndarray, subject_name: str, save_dir=None):
    n_strides = gt_ref.shape[0] // STRIDE_LEN
    T_use = n_strides * STRIDE_LEN
    x = np.linspace(0, 100, STRIDE_LEN)
    colors = {"timestamp_cnn": "#d62728", "timestamp_lstm": "#2ca02c", "pv_cnn": "#9467bd", "pv_lstm": "#ff7f0e"}

    gt_s = gt_ref[:T_use].reshape(n_strides, STRIDE_LEN, 18)

    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = _grid_fig(side, f"Subject {subject_name}: Mean |Pred-GT| vs Gait Phase", "|Error| (deg)")
        for row in range(3):
            for col in range(3):
                idx = base + row * 3 + col
                ax = axes[row, col]
                for name, pred in preds.items():
                    if pred is None:
                        continue
                    n2 = min(n_strides, len(pred) // STRIDE_LEN)
                    pred_s = pred[:n2 * STRIDE_LEN].reshape(n2, STRIDE_LEN, 18)
                    err = np.abs(pred_s[:, :, idx] - gt_s[:n2, :, idx]).mean(axis=0)
                    ax.plot(x, err, color=colors[name], lw=1.6, label=MODEL_LABELS[name])
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)
        for col in range(3):
            axes[2, col].set_xlabel("Gait cycle (%)", fontsize=8)
        plt.tight_layout()
        if save_dir:
            fname = os.path.join(save_dir, f"compare4_phase_error_{side.lower()}_subject_{subject_name}.png")
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"Saved: {fname}")
        plt.show()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Compare 4 all-planes models across multiple held-out CP subjects")
    parser.add_argument("--list-subjects", action="store_true", help="List held-out CP test files with their index and exit")
    parser.add_argument("--subject", type=int, default=None, help="Index (from --list-subjects) of one subject to inspect")
    parser.add_argument("--plot", action="store_true", help="Show neural_networks_outputs/plots for the chosen --subject")
    parser.add_argument("--no_save", action="store_true", help="Don't save PNGs when plotting")
    args = parser.parse_args()

    test_subjects = get_cp_test_subjects()
    subject_ids = list(test_subjects.keys())

    if args.list_subjects:
        for i, sid in enumerate(subject_ids):
            print(f"[{i:3d}] {sid}  ({len(test_subjects[sid])} trials)")
        print(f"\n{len(subject_ids)} held-out CP test subjects.")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)
    save_dir = None if args.no_save else PLOT_DIR

    print(f"Loading 4 models...")
    models = load_all_models()

    if args.subject is not None:
        targets = [subject_ids[args.subject]]
    else:
        targets = subject_ids

    rows = []
    last_preds, last_gt = None, None
    for sid in targets:
        df_subject = load_cp_subject(test_subjects[sid])
        preds, gts = {}, {}
        for name in MODEL_NAMES:
            pred_deg, gt_deg = predict_next_tick(models[name], df_subject)
            preds[name] = pred_deg
            gts[name] = gt_deg
            if pred_deg is None:
                continue
            mae, rmse, r2 = compute_metrics_deg(pred_deg, gt_deg)
            rows.append({
                "subject": sid, "model": MODEL_LABELS[name],
                "mean_MAE": mae.mean(), "mean_RMSE": rmse.mean(), "mean_R2": r2.mean(),
            })

        if args.subject is not None:
            last_preds, last_gt = preds, gts["timestamp_cnn"] if gts["timestamp_cnn"] is not None else next(g for g in gts.values() if g is not None)

    summary = pd.DataFrame(rows)
    print("\n=== Per-subject metrics (mean over 18 channels) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    print("\n=== Cross-subject summary per model (mean ± std across held-out subjects) ===")
    agg = summary.groupby("model")[["mean_MAE", "mean_RMSE", "mean_R2"]].agg(["mean", "std"])
    print(agg.to_string(float_format=lambda v: f"{v:7.3f}"))

    if args.subject is not None and args.plot and last_preds is not None:
        subject_name = targets[0]
        plot_subject_all_models(last_preds, last_gt, subject_name, save_dir=save_dir)
        plot_phase_error_all_models(last_preds, last_gt, subject_name, save_dir=save_dir)


if __name__ == "__main__":
    main()
