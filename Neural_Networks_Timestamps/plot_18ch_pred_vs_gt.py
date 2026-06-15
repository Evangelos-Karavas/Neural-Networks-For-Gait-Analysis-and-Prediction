#!/usr/bin/env python3
"""
plot_18ch_pred_vs_gt.py
Standalone visualization of 18-channel angle predictions vs ground truth.

Layout: two 3x3 figures per plot type (Left side / Right side).
  Rows    = Hip / Knee / Ankle
  Columns = Sagittal (1) / Frontal (2) / Transverse (3)

Loads saved Excel predictions produced by the training scripts.
Run from project root:
  python Neural_Networks_Timestamps/plot_18ch_pred_vs_gt.py --model lstm
  python Neural_Networks_Timestamps/plot_18ch_pred_vs_gt.py --model cnn
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
ANGLE_COLS = [
    "LHipAngles (1)", "LHipAngles (2)", "LHipAngles (3)",
    "LKneeAngles (1)", "LKneeAngles (2)", "LKneeAngles (3)",
    "LAnkleAngles (1)", "LAnkleAngles (2)", "LAnkleAngles (3)",
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]

PRED_DIR   = "Predictions"
PLOT_DIR   = "Plots"
STRIDE_LEN = 51

SIDES      = ["Left",  "Right"]
JOINTS     = ["Hip",   "Knee", "Ankle"]
PLANES     = ["Sagittal (1)", "Frontal (2)", "Transverse (3)"]
SIDE_BASE  = [0, 9]   # starting index in ANGLE_COLS for each side

GT_COLOR   = "#1f77b4"
PRED_COLOR = "#d62728"


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_lstm_preds():
    pred = pd.read_excel(os.path.join(PRED_DIR, "timestamp_lstm_rolling_pred_next_tick_18ch.xlsx"))
    gt   = pd.read_excel(os.path.join(PRED_DIR, "timestamp_lstm_rolling_gt_next_tick_18ch.xlsx"))
    return pred[ANGLE_COLS].to_numpy(), gt[ANGLE_COLS].to_numpy()


def load_cnn_preds():
    df   = pd.read_excel(os.path.join(PRED_DIR, "timestamp_cnn_next_tick_cp_segment_18ch.xlsx"))
    pred = df[[f"CNN_{c}" for c in ANGLE_COLS]].to_numpy()
    gt   = df[[f"GT_{c}"  for c in ANGLE_COLS]].to_numpy()
    return pred, gt


def load_pv_lstm_preds():
    pred = pd.read_excel(os.path.join(PRED_DIR, "rolling_pred_next_tick_with_pv_lstm_18ch.xlsx"))
    gt   = pd.read_excel(os.path.join(PRED_DIR, "rolling_gt_next_tick_with_pv_lstm_18ch.xlsx"))
    return pred[ANGLE_COLS].to_numpy(), gt[ANGLE_COLS].to_numpy()


def load_pv_cnn_preds():
    pred = pd.read_excel(os.path.join(PRED_DIR, "rolling_pred_next_tick_with_pv_cnn_18ch.xlsx"))
    gt   = pd.read_excel(os.path.join(PRED_DIR, "rolling_gt_next_tick_with_pv_cnn_18ch.xlsx"))
    return pred[ANGLE_COLS].to_numpy(), gt[ANGLE_COLS].to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(pred, gt):
    eps  = 1e-12
    err  = pred - gt
    mae  = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    ss_res = np.sum((gt - pred) ** 2, axis=0)
    ss_tot = np.sum((gt - gt.mean(axis=0)) ** 2, axis=0) + eps
    r2 = 1 - ss_res / ss_tot

    print(f"\n{'Channel':22s}  {'MAE (deg)':>9}  {'RMSE (deg)':>10}  {'R2':>7}")
    print("-" * 58)
    for j, col in enumerate(ANGLE_COLS):
        print(f"{col:22s}  {mae[j]:9.3f}  {rmse[j]:10.3f}  {r2[j]:7.3f}")
    print("-" * 58)
    print(f"{'Mean':22s}  {mae.mean():9.3f}  {rmse.mean():10.3f}  {r2.mean():7.3f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plot: predicted vs ground truth (time series)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pred_vs_gt(pred, gt, title_prefix="", max_samples=300, save_dir=None):
    """
    Two 3x3 figures (Left side / Right side).
    Shows a time-series segment of max_samples samples.
    """
    T = min(max_samples, len(pred))
    t = np.arange(T)

    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
        fig.suptitle(f"{title_prefix} — {side} Side: Predicted vs Ground Truth", fontsize=13)

        for row, joint in enumerate(JOINTS):
            for col, plane in enumerate(PLANES):
                idx = base + row * 3 + col
                ax  = axes[row, col]

                ax.plot(t, gt[:T, idx],   color=GT_COLOR,   lw=1.6, label="Ground truth")
                ax.plot(t, pred[:T, idx], color=PRED_COLOR, lw=1.2, ls="--", label="Predicted")
                ax.set_title(f"{side} {joint} – {plane}", fontsize=9)
                ax.set_ylabel("Angle (deg)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.35)

                if row == 0 and col == 0:
                    ax.legend(fontsize=8)

        for col in range(3):
            axes[2, col].set_xlabel("Sample", fontsize=8)

        plt.tight_layout()

        if save_dir:
            fname = os.path.join(
                save_dir,
                f"pred_vs_gt_18ch_{side.lower()}_{title_prefix.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            )
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"Saved: {fname}")

        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Plot: stride mean ± 1 SD
# ─────────────────────────────────────────────────────────────────────────────

def plot_stride_mean_std(pred, gt, stride_len=51, title_prefix="", save_dir=None):
    """
    Two 3x3 figures (Left side / Right side).
    Shows mean ± 1 SD envelope over all strides (x-axis = gait cycle %).
    """
    n       = min(len(pred), len(gt))
    n_strides = n // stride_len
    T_use   = n_strides * stride_len

    pred_s  = pred[:T_use].reshape(n_strides, stride_len, 18)
    gt_s    = gt[:T_use].reshape(n_strides, stride_len, 18)
    x       = np.linspace(0, 100, stride_len)

    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
        fig.suptitle(
            f"{title_prefix} — {side} Side: Mean ± 1 SD  ({n_strides} strides)",
            fontsize=13
        )

        for row, joint in enumerate(JOINTS):
            for col, plane in enumerate(PLANES):
                idx = base + row * 3 + col
                ax  = axes[row, col]

                gt_mu = gt_s[:, :, idx].mean(axis=0)
                gt_sd = gt_s[:, :, idx].std(axis=0)
                pr_mu = pred_s[:, :, idx].mean(axis=0)
                pr_sd = pred_s[:, :, idx].std(axis=0)

                ax.plot(x, gt_mu, color=GT_COLOR,   lw=2.0, label="GT mean")
                ax.fill_between(x, gt_mu - gt_sd, gt_mu + gt_sd,
                                color=GT_COLOR, alpha=0.20, label="GT ± 1 SD")
                ax.plot(x, pr_mu, color=PRED_COLOR, lw=2.0, ls="--", label="Pred mean")
                ax.fill_between(x, pr_mu - pr_sd, pr_mu + pr_sd,
                                color=PRED_COLOR, alpha=0.20, label="Pred ± 1 SD")

                ax.set_title(f"{side} {joint} – {plane}", fontsize=9)
                ax.set_ylabel("Angle (deg)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.35)

                if row == 0 and col == 0:
                    ax.legend(fontsize=7, ncol=2)

        for col in range(3):
            axes[2, col].set_xlabel("Gait cycle (%)", fontsize=8)

        plt.tight_layout()

        if save_dir:
            fname = os.path.join(
                save_dir,
                f"stride_mean_std_18ch_{side.lower()}_{title_prefix.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            )
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"Saved: {fname}")

        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Plot: mean absolute error vs gait phase
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase_abs_error(pred, gt, stride_len=51, title_prefix="", save_dir=None):
    """
    Two 3x3 figures (Left side / Right side).
    Shows mean |pred - gt| at each gait cycle % bin.
    """
    n         = min(len(pred), len(gt))
    n_strides = n // stride_len
    T_use     = n_strides * stride_len

    pred_s = pred[:T_use].reshape(n_strides, stride_len, 18)
    gt_s   = gt[:T_use].reshape(n_strides, stride_len, 18)
    err_s  = np.abs(pred_s - gt_s).mean(axis=0)   # (stride_len, 18)
    x      = np.linspace(0, 100, stride_len)

    for side_i, side in enumerate(SIDES):
        base = SIDE_BASE[side_i]
        fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
        fig.suptitle(
            f"{title_prefix} — {side} Side: Mean |Pred − GT| vs Gait Phase",
            fontsize=13
        )

        for row, joint in enumerate(JOINTS):
            for col, plane in enumerate(PLANES):
                idx = base + row * 3 + col
                ax  = axes[row, col]
                ax.plot(x, err_s[:, idx], color="#2ca02c", lw=1.8)
                ax.set_title(f"{side} {joint} – {plane}", fontsize=9)
                ax.set_ylabel("|Error| (deg)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.35)

        for col in range(3):
            axes[2, col].set_xlabel("Gait cycle (%)", fontsize=8)

        plt.tight_layout()

        if save_dir:
            fname = os.path.join(
                save_dir,
                f"phase_abs_error_18ch_{side.lower()}_{title_prefix.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            )
            plt.savefig(fname, dpi=200, bbox_inches="tight")
            print(f"Saved: {fname}")

        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

LOADERS = {
    "lstm":    load_lstm_preds,
    "cnn":     load_cnn_preds,
    "pv_lstm": load_pv_lstm_preds,
    "pv_cnn":  load_pv_cnn_preds,
}

LABELS = {
    "lstm":    "Timestamp LSTM (18ch)",
    "cnn":     "Timestamp CNN (18ch)",
    "pv_lstm": "Phase Variable LSTM (18ch)",
    "pv_cnn":  "Phase Variable CNN (18ch)",
}


def main():
    parser = argparse.ArgumentParser(description="18-channel gait angle prediction visualizer")
    parser.add_argument(
        "--model",
        choices=list(LOADERS.keys()),
        default="lstm",
        help="Which model's predictions to load (default: lstm)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=300,
        help="Number of time-series samples to show in pred-vs-gt plot",
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="Show plots without saving PNGs",
    )
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)
    save_dir = None if args.no_save else PLOT_DIR

    pred, gt = LOADERS[args.model]()
    label    = LABELS[args.model]

    print(f"Loaded {len(pred)} prediction samples.")
    print_metrics(pred, gt)

    plot_pred_vs_gt(
        pred, gt,
        title_prefix=label,
        max_samples=args.max_samples,
        save_dir=save_dir,
    )
    plot_stride_mean_std(
        pred, gt,
        stride_len=STRIDE_LEN,
        title_prefix=label,
        save_dir=save_dir,
    )
    plot_phase_abs_error(
        pred, gt,
        stride_len=STRIDE_LEN,
        title_prefix=label,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
