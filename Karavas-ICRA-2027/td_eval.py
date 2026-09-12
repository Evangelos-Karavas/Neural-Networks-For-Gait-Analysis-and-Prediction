#!/usr/bin/env python3
"""Readout modes, metrics and figures for the TD model comparison.

The same one-step-ahead predictor can be read out in three ways, and they
measure different things:

  teacher_forced    - the model always sees measured data; best-case per-step
                      accuracy, no error build-up.
  recursive_rollout - the model is fed its own output; the only mode in which
                      drift and phase misalignment appear.
  phase_binned      - teacher-forced errors grouped by position in the gait
                      cycle, which shows failures locked to specific sub-phases.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from td_pipeline import (
    ANGLE_COLS,
    CHANNEL_LABELS,
    HORIZON,
    N_ANGLES,
    PV_COLS,
    SAGITTAL_IDX,
    SAGITTAL_LABELS,
    STRIDE_LEN,
    WINDOW,
    make_windows,
)

# Categorical slots 1-4 of the reference palette, in their documented order.
MODEL_COLORS = {
    "timestamp_lstm": "#2a78d6",
    "pv_lstm": "#eb6834",
    "timestamp_cnn": "#1baf7a",
    "pv_cnn": "#eda100",
}
GT_COLOR = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dcdcd8"

GAIT_PHASES = [
    "Loading\nResponse",
    "Mid\nStance",
    "Terminal\nStance",
    "Pre-\nSwing",
    "Initial\nSwing",
    "Mid\nSwing",
    "Terminal\nSwing",
]


def _style_axis(ax) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


# ============================================================
# Readout modes
# ============================================================
def teacher_forced(model, fold, frame: pd.DataFrame, kind: str,
                   horizon: int = HORIZON):
    """Slide over the measured signal, predicting `horizon` samples each time.

    The model always sees real measurements, so no error can accumulate. Only
    the furthest predicted sample is returned: that is the one the horizon is
    chosen for, and the one the persistence and linear baselines are computed
    against, so the comparison stays like for like.
    """
    feats, targets = fold.features(frame, kind)
    x, y = make_windows(feats, targets, horizon=horizon)
    if x is None:
        return None, None

    pred_scaled = np.asarray(model.predict(x, verbose=0))
    if pred_scaled.ndim == 3:
        pred_scaled = pred_scaled[:, -1, :]
    gt_scaled = y[:, -1, :] if y.ndim == 3 else y

    return (
        fold.scaler_ang.inverse_transform(pred_scaled),
        fold.scaler_ang.inverse_transform(gt_scaled),
    )


def baseline_predictions(frame: pd.DataFrame, horizon: int = HORIZON):
    """Persistence and linear extrapolation over the same horizon.

    Persistence repeats the last observed sample; linear extrapolation
    continues the last observed slope. Both are parameter-free, and a learned
    model that cannot beat them is not predicting anything.
    """
    ang = frame[ANGLE_COLS].to_numpy(float)
    last = ang[WINDOW - 1:len(ang) - horizon]
    slope = last - ang[WINDOW - 2:len(ang) - horizon - 1]
    gt = ang[WINDOW + horizon - 1:]
    n = min(len(last), len(gt))
    return {
        "persistence": last[:n],
        "linear": (last + horizon * slope)[:n],
    }, gt[:n]


def recursive_rollout(model, fold, frame: pd.DataFrame, kind: str,
                      n_strides: int = 6, start: int = 0):
    """Feed the model its own output for `n_strides` consecutive strides.

    A phase-variable model still receives a correct phase value at every step,
    because the phase variable is computed from measured hip kinematics rather
    than from the model's own prediction. That self-correction is exactly what
    the representation is meant to provide, and what this readout tests.
    """
    feats, targets = fold.features(frame, kind)
    total = min(n_strides * STRIDE_LEN, len(feats) - start - WINDOW)
    if total <= 0:
        return None, None

    window = feats[start:start + WINDOW].copy()
    preds = np.zeros((total, N_ANGLES), dtype=np.float32)

    t = 0
    while t < total:
        block = np.asarray(model.predict(window[None, ...], verbose=0))[0]
        if block.ndim == 1:
            block = block[None, :]
        take = min(len(block), total - t)

        for k in range(take):
            preds[t + k] = block[k]
            if kind == "timestamp":
                next_row = block[k]
            else:
                next_row = np.concatenate([feats[start + WINDOW + t + k, :2], block[k]])
            window = np.vstack([window[1:], next_row[None, :]])
        t += take

    gt = targets[start + WINDOW:start + WINDOW + total]
    if gt.ndim == 3:
        gt = gt[:, 0, :]
    return (
        fold.scaler_ang.inverse_transform(preds),
        fold.scaler_ang.inverse_transform(gt),
    )


def phase_bounds(toe_off: float) -> np.ndarray:
    """The seven classical gait phases, anchored to this subject's own toe-off."""
    stance, swing = toe_off, 100.0 - toe_off
    return np.array([
        0.0,
        stance / 6.0,
        stance / 2.0,
        stance * 5.0 / 6.0,
        stance,
        stance + swing / 3.0,
        stance + 2.0 * swing / 3.0,
        100.0,
    ])


def phase_binned(pred: np.ndarray, gt: np.ndarray, toe_off_l: float,
                 toe_off_r: float, offset: int = WINDOW + HORIZON - 1,
                 right_shift: int = STRIDE_LEN // 2) -> np.ndarray:
    """Mean absolute error per gait phase, shape (7, 18).

    `offset` is the absolute time of the first prediction, so a sample's
    position inside its stride stays known even though teacher forcing only
    starts one window in.

    The right leg carries `right_shift`: the pipeline rolls the right channels
    half a stride, so the right heel strike sits at that index rather than at 0,
    and the right side has to be binned against its own cycle origin.
    """
    absolute = np.arange(len(gt)) + offset
    err = np.abs(pred - gt)

    out = np.full((len(GAIT_PHASES), N_ANGLES), np.nan)
    sides = (
        (toe_off_l, range(0, 9), 0),
        (toe_off_r, range(9, 18), right_shift),
    )
    for toe_off, channels, shift in sides:
        positions = ((absolute - shift) % STRIDE_LEN) / STRIDE_LEN * 100.0
        bins = np.clip(
            np.searchsorted(phase_bounds(toe_off)[1:-1], positions, side="right"), 0, 6
        )
        for p in range(len(GAIT_PHASES)):
            mask = bins == p
            if mask.any():
                for ch in channels:
                    out[p, ch] = err[mask, ch].mean()
    return out


# ============================================================
# Metrics
# ============================================================
def channel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-12
    err = pred - gt
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((gt - gt.mean(axis=0)) ** 2, axis=0) + eps
    return {
        "MAE": np.mean(np.abs(err), axis=0),
        "RMSE": np.sqrt(np.mean(err ** 2, axis=0)),
        "R2": 1.0 - ss_res / ss_tot,
    }


def phase_lag_samples(pred: np.ndarray, gt: np.ndarray, channel: int = 0,
                      max_lag: int = STRIDE_LEN // 2) -> float:
    """Signed lag, in samples, that best aligns the predicted waveform with the
    measured one over the final stride of a rollout. Positive means the
    prediction runs ahead of the measurement.

    This is the drift recursive rollout is run to expose: a model that keeps the
    right waveform shape but loses its timing scores a large lag while its plain
    MAE can still look acceptable.
    """
    if len(gt) < STRIDE_LEN:
        return float("nan")

    a = pred[-STRIDE_LEN:, channel]
    b = gt[-STRIDE_LEN:, channel]
    a = a - a.mean()
    b = b - b.mean()
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return float("nan")

    lags = np.arange(-max_lag, max_lag + 1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    scores = [float(np.dot(np.roll(a, int(lag)), b) / denom) for lag in lags]
    return float(lags[int(np.argmax(scores))])


# ============================================================
# Figures
# ============================================================
def plot_pv_strides(frame: pd.DataFrame, path: Path, n_strides: int = 5) -> None:
    """Left and right phase variable over consecutive strides.

    Sized for a narrow two-column subfigure: the figure is drawn at roughly the
    width it is placed at, so the labels reach the page at the point size set
    here instead of being shrunk to illegibility by LaTeX.
    """
    pv = frame[PV_COLS].to_numpy(float)[: n_strides * STRIDE_LEN]
    t = np.arange(len(pv))

    fig, ax = plt.subplots(figsize=(2.35, 1.30), dpi=400)
    ax.plot(t, pv[:, 0], color=MODEL_COLORS["timestamp_lstm"], linewidth=1.1, label="Left")
    ax.plot(t, pv[:, 1], color=MODEL_COLORS["pv_lstm"], linewidth=1.1, label="Right")

    for boundary in range(STRIDE_LEN, len(pv), STRIDE_LEN):
        ax.axvline(boundary, color=GRID_COLOR, linewidth=0.6, zorder=0)

    ax.set_xlabel("Sample", fontsize=7, labelpad=1.5)
    ax.set_ylabel("$s$", fontsize=8, labelpad=1.5)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.legend(frameon=False, fontsize=6, loc="lower right", handlelength=1.2,
              borderaxespad=0.2, handletextpad=0.4)
    _style_axis(ax)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_td_variability(frames: dict[str, pd.DataFrame], path: Path) -> None:
    """Sagittal hip/knee/ankle over every TD stride in the dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    x = np.linspace(0, 100, STRIDE_LEN)

    for row, side in enumerate(("L", "R")):
        for col, joint in enumerate(("Hip", "Knee", "Ankle")):
            ax = axes[row, col]
            ch = ANGLE_COLS.index(side + joint + "Angles (1)")
            curves = []
            for frame in frames.values():
                arr = frame[ANGLE_COLS].to_numpy(float)
                n = len(arr) // STRIDE_LEN
                strides = arr[:n * STRIDE_LEN].reshape(n, STRIDE_LEN, -1)[:, :, ch]
                curves.append(strides)
                for stride in strides:
                    ax.plot(x, stride, color=MODEL_COLORS["timestamp_lstm"],
                            alpha=0.13, linewidth=0.8)
            mean = np.concatenate(curves).mean(axis=0)
            ax.plot(x, mean, color=GT_COLOR, linewidth=2.0, label="Mean")

            side_name = "Left" if side == "L" else "Right"
            ax.set_title(side_name + " " + joint, fontsize=9, color=TEXT_SECONDARY)
            _style_axis(ax)
            if col == 0:
                ax.set_ylabel("Angle (deg)", fontsize=8)
            if row == 1:
                ax.set_xlabel("Gait cycle (%)", fontsize=8)

    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("All strides, all subjects (sagittal plane)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rollout(results: dict[str, np.ndarray], gt: np.ndarray, path: Path,
                 title: str, model_labels: dict[str, str]) -> None:
    """Recursive rollout, six sagittal channels, all models against ground truth."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    t = np.arange(len(gt))

    for k, ch in enumerate(SAGITTAL_IDX):
        ax = axes[k % 3, k // 3]
        ax.plot(t, gt[:, ch], color=GT_COLOR, linewidth=2.0, label="Ground truth", zorder=5)
        for key, pred in results.items():
            ax.plot(t, pred[:, ch], color=MODEL_COLORS[key], linewidth=1.6,
                    label=model_labels[key], alpha=0.9)
        for boundary in range(STRIDE_LEN, len(gt), STRIDE_LEN):
            ax.axvline(boundary, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_ylabel(SAGITTAL_LABELS[k] + "\n(deg)", fontsize=8)
        _style_axis(ax)

    for ax in axes[2, :]:
        ax.set_xlabel("Sample (rollout step)", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subject_cycle(results: dict[str, np.ndarray], gt: np.ndarray, path: Path,
                       title: str, model_labels: dict[str, str],
                       offset: int = WINDOW + HORIZON - 1) -> None:
    """Stride-averaged prediction vs ground truth over one normalized cycle,
    left side, all nine joint-plane channels."""
    positions = (np.arange(len(gt)) + offset) % STRIDE_LEN

    def cycle_stat(arr, fn):
        return np.stack([fn(arr[positions == p], axis=0) for p in range(STRIDE_LEN)])

    gt_mu = cycle_stat(gt, np.mean)
    gt_sd = cycle_stat(gt, np.std)
    pred_mu = {k: cycle_stat(v, np.mean) for k, v in results.items()}

    x = np.linspace(0, 100, STRIDE_LEN)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True)

    for ch in range(9):
        ax = axes[ch // 3, ch % 3]
        ax.fill_between(x, gt_mu[:, ch] - gt_sd[:, ch], gt_mu[:, ch] + gt_sd[:, ch],
                        color=GT_COLOR, alpha=0.12, linewidth=0)
        ax.plot(x, gt_mu[:, ch], color=GT_COLOR, linewidth=2.0, label="Ground truth", zorder=5)
        for key, mu in pred_mu.items():
            ax.plot(x, mu[:, ch], color=MODEL_COLORS[key], linewidth=1.6,
                    label=model_labels[key])
        ax.set_title(CHANNEL_LABELS[ch], fontsize=9, color=TEXT_SECONDARY)
        _style_axis(ax)
        if ch % 3 == 0:
            ax.set_ylabel("Angle (deg)", fontsize=8)
        if ch // 3 == 2:
            ax.set_xlabel("Gait cycle (%)", fontsize=8)

    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_per_phase_bars(per_phase: dict[str, np.ndarray], path: Path, title: str,
                        model_labels: dict[str, str]) -> None:
    """Mean absolute error per gait phase, one group of bars per phase."""
    keys = list(per_phase.keys())
    n = len(keys)
    x = np.arange(len(GAIT_PHASES))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, key in enumerate(keys):
        ax.bar(x + (i - (n - 1) / 2) * width, np.nanmean(per_phase[key], axis=1),
               width * 0.9, color=MODEL_COLORS[key], label=model_labels[key])

    ax.set_xticks(x)
    ax.set_xticklabels(GAIT_PHASES, fontsize=8)
    ax.set_ylabel("Mean |error| (deg)", fontsize=9)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=8, ncol=n)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subject_scatter(summary: pd.DataFrame, path: Path,
                         model_labels: dict[str, str], metric: str = "MAE") -> None:
    """Per-held-out-subject error, one column per model, so the spread across
    subjects stays visible next to the mean."""
    keys = [k for k in model_labels if k in set(summary["model"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rng = np.random.RandomState(0)

    for i, key in enumerate(keys):
        vals = summary.loc[summary["model"] == key, metric].to_numpy(float)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=34, color=MODEL_COLORS[key],
                   alpha=0.75, edgecolor="white", linewidth=0.8, zorder=3,
                   label=model_labels[key])
        ax.hlines(vals.mean(), i - 0.3, i + 0.3, color=GT_COLOR, linewidth=2.0, zorder=4)
        ax.annotate(f"{vals.mean():.2f}", (i + 0.33, vals.mean()), fontsize=8,
                    color=TEXT_SECONDARY, va="center")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([model_labels[k] for k in keys], fontsize=9)
    ax.set_ylabel(metric + " (deg), one point per held-out subject", fontsize=9)
    _style_axis(ax)
    ax.set_title("Teacher-forced " + metric + " on held-out subjects (black bar = mean)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
