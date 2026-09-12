#!/usr/bin/env python3
"""TD-only gait prediction pipeline.

Loading, subject-level splitting, augmentation, phase-variable computation and
rolling-window construction for the four 18-channel models compared in
run_comparison.py.

Every stride file in Data_Normal/ is one 51-sample gait cycle of one typically
developed subject. Subjects are split whole, augmentation is generated from the
training subjects of each split only, and windows are built inside a single
subject's own trials, so no held-out subject can reach the training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
DATA_DIR = ROOT / "Data_Normal"
OUTPUT_DIR = PKG_DIR / "outputs"

STRIDE_LEN = 51
WINDOW = 51

# Prediction horizon, in samples. A stride is resampled to 51 samples, so at a
# typical stride time of about one second a sample is roughly 20 ms and the
# default horizon of 10 samples is roughly 200 ms ahead -- the order of the
# actuation and sensing delay a wearable-robot controller has to cover.
HORIZON = 10

SHEET = "Data"
SKIPROWS = [1, 2]

# Aggregate files in Data_Normal/ that are not single-subject stride recordings.
EXCLUDED_FILES = {
    "dynamics_total_augmented.xlsx",
    "randomized_data_healthy.xlsx",
    "Healthy_Data_1_Person.xlsx",
}

ANGLE_COLS = [
    "LHipAngles (1)", "LHipAngles (2)", "LHipAngles (3)",
    "LKneeAngles (1)", "LKneeAngles (2)", "LKneeAngles (3)",
    "LAnkleAngles (1)", "LAnkleAngles (2)", "LAnkleAngles (3)",
    "RHipAngles (1)", "RHipAngles (2)", "RHipAngles (3)",
    "RKneeAngles (1)", "RKneeAngles (2)", "RKneeAngles (3)",
    "RAnkleAngles (1)", "RAnkleAngles (2)", "RAnkleAngles (3)",
]
RIGHT_ANGLE_COLS = ANGLE_COLS[9:]
N_ANGLES = len(ANGLE_COLS)

LHIP_COL, RHIP_COL = "LHipAngles (1)", "RHipAngles (1)"
LFO_COL, RFO_COL = "Left Foot Off", "Right Foot Off"
PV_COLS = ["PhaseVariable_Left", "PhaseVariable_Right"]
PV_FEATURE_COLS = PV_COLS + ANGLE_COLS

LOAD_COLS = ANGLE_COLS + [LFO_COL, RFO_COL]

CHANNEL_LABELS = [
    "L Hip Sag", "L Hip Fro", "L Hip Tra",
    "L Knee Sag", "L Knee Fro", "L Knee Tra",
    "L Ankle Sag", "L Ankle Fro", "L Ankle Tra",
    "R Hip Sag", "R Hip Fro", "R Hip Tra",
    "R Knee Sag", "R Knee Fro", "R Knee Tra",
    "R Ankle Sag", "R Ankle Fro", "R Ankle Tra",
]
SAGITTAL_IDX = [0, 3, 6, 9, 12, 15]
SAGITTAL_LABELS = ["L Hip", "L Knee", "L Ankle", "R Hip", "R Knee", "R Ankle"]


# ============================================================
# Loading
# ============================================================
def list_subjects(data_dir: Path = DATA_DIR) -> dict[str, list[Path]]:
    """Map subject id (the 'NV031' prefix) to that subject's stride files."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"TD data folder not found: {data_dir}")

    subjects: dict[str, list[Path]] = {}
    for path in sorted(data_dir.glob("*.xlsx")):
        if path.name in EXCLUDED_FILES:
            continue
        subjects.setdefault(path.name.split("-")[0], []).append(path)

    if not subjects:
        raise RuntimeError(f"No TD stride files found in {data_dir}")
    return subjects


def _fix_stride_endpoints(block: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Average the two endpoints of a stride when they disagree by more than
    `threshold` degrees, so concatenated strides do not step at the boundary."""
    out = block.copy()
    gap = np.abs(out[-1] - out[0])
    bad = gap > threshold
    if bad.any():
        mid = (out[-1, bad] + out[0, bad]) / 2.0
        out[0, bad] = mid
        out[-1, bad] = mid
    return out


def load_subject(files: list[Path]) -> pd.DataFrame:
    """Concatenate one subject's stride files into a raw (unshifted) frame."""
    frames = []
    for path in files:
        df = pd.read_excel(path, sheet_name=SHEET, usecols=LOAD_COLS, skiprows=SKIPROWS)
        df = df[LOAD_COLS].fillna(0.0)
        if len(df) < STRIDE_LEN:
            continue
        df = df.iloc[:STRIDE_LEN].reset_index(drop=True)
        df.loc[:, ANGLE_COLS] = _fix_stride_endpoints(df[ANGLE_COLS].to_numpy(float))
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No usable strides in {[p.name for p in files]}")
    return pd.concat(frames, ignore_index=True)


def load_all_subjects(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    return {sid: load_subject(files) for sid, files in list_subjects(data_dir).items()}


# ============================================================
# Phase variable
# ============================================================
def compute_pv_stride(q: np.ndarray, c: float, enforce_monotonic: bool = True) -> np.ndarray:
    """Phase variable over one stride of sagittal hip angle `q`.

    Stance/pushoff use the descending branch scaled to reach `c` at the hip
    minimum; preswing/swing use the ascending branch running from there to 1.
    `c` is this stride's measured foot-off fraction.
    """
    q = np.asarray(q, dtype=np.float64)
    n = q.shape[0]
    c = float(np.clip(c, 0.05, 0.95))

    q0 = float(q[0])
    idx_min = int(np.argmin(q))
    qmin = float(q[idx_min])
    denom = q0 - qmin

    s = np.zeros(n, dtype=np.float64)
    if abs(denom) < 1e-6:
        s[:idx_min + 1] = np.linspace(0.0, c, idx_min + 1)
        sm = c
        s[idx_min:] = np.linspace(sm, 1.0, n - idx_min)
    else:
        s[:idx_min + 1] = ((q0 - q[:idx_min + 1]) / denom) * c
        sm = s[idx_min]
        s[idx_min:] = 1.0 + ((1.0 - sm) / denom) * (q[idx_min:] - q0)

    s = np.clip(s, 0.0, 1.0)
    if enforce_monotonic:
        s = np.maximum.accumulate(s)
    return s


def add_phase_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Append the left/right phase variables, one stride block at a time."""
    n_strides = len(df) // STRIDE_LEN
    out = df.iloc[:n_strides * STRIDE_LEN].copy().reset_index(drop=True)

    pv_l = np.zeros(len(out), dtype=np.float32)
    pv_r = np.zeros(len(out), dtype=np.float32)
    for s in range(n_strides):
        a, b = s * STRIDE_LEN, (s + 1) * STRIDE_LEN
        c_l = float(out[LFO_COL].iloc[a]) / 100.0
        c_r = float(out[RFO_COL].iloc[a]) / 100.0
        pv_l[a:b] = compute_pv_stride(out[LHIP_COL].to_numpy()[a:b], c=c_l)
        pv_r[a:b] = compute_pv_stride(out[RHIP_COL].to_numpy()[a:b], c=c_r)

    out[PV_COLS[0]] = pv_l
    out[PV_COLS[1]] = pv_r
    return out


def roll_right_leg(df: pd.DataFrame) -> pd.DataFrame:
    """Shift the right-leg channels half a stride so each leg is expressed from
    its own heel strike. Applied inside each stride block, and to the right
    phase variable together with the right angles."""
    shift = STRIDE_LEN // 2
    n_strides = len(df) // STRIDE_LEN
    out = df.iloc[:n_strides * STRIDE_LEN].copy().reset_index(drop=True)

    cols = RIGHT_ANGLE_COLS + [PV_COLS[1]]
    arr = out[cols].to_numpy(float).reshape(n_strides, STRIDE_LEN, len(cols))
    out.loc[:, cols] = np.roll(arr, shift, axis=1).reshape(-1, len(cols))
    return out


def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw stride frame -> model-ready frame (phase variables, right leg shifted)."""
    return roll_right_leg(add_phase_variables(raw))


def mean_toe_off(raw: pd.DataFrame) -> tuple[float, float]:
    """Mean measured toe-off (% of stride) per side.

    The value is recorded only on the first row of each stride, so it is read at
    the stride starts rather than averaged over the zero-filled column.
    """
    n_strides = max(1, len(raw) // STRIDE_LEN)

    def _side(col: str, default: float = 60.0) -> float:
        if col not in raw.columns:
            return default
        vals = [
            float(raw[col].iloc[s * STRIDE_LEN])
            for s in range(n_strides)
            if s * STRIDE_LEN < len(raw)
        ]
        vals = [v for v in vals if 5.0 <= v <= 95.0]
        return float(np.mean(vals)) if vals else default

    return _side(LFO_COL), _side(RFO_COL)


# ============================================================
# Augmentation (training subjects only)
# ============================================================
@dataclass
class AugmentConfig:
    amplitude_range: tuple[float, float] = (0.90, 1.10)
    warp_strength: float = 0.06
    noise_frac: float = 0.03
    smooth_window: int = 9
    smooth_polyorder: int = 3


def _random_warp(rng: np.random.RandomState, strength: float) -> np.ndarray:
    """A monotone map of normalized stride time onto itself, w(0)=0, w(1)=1."""
    n_ctrl = 4
    ctrl_t = np.linspace(0.0, 1.0, n_ctrl + 2)
    offsets = np.concatenate([[0.0], rng.uniform(-strength, strength, n_ctrl), [0.0]])
    ctrl_w = np.maximum.accumulate(np.clip(ctrl_t + offsets, 0.0, 1.0))
    ctrl_w[0], ctrl_w[-1] = 0.0, 1.0

    t = np.linspace(0.0, 1.0, STRIDE_LEN)
    return np.clip(np.maximum.accumulate(np.interp(t, ctrl_t, ctrl_w)), 0.0, 1.0)


def _sg_smooth(block: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing within a stride, circularly padded so the
    endpoints are not bent by edge effects."""
    if window % 2 == 0 or window > block.shape[0]:
        return block
    pad = window // 2
    padded = np.concatenate([block[-pad:], block, block[:pad]], axis=0)
    smoothed = savgol_filter(padded, window_length=window, polyorder=polyorder, axis=0)
    return smoothed[pad:-pad]


def augment_frame(
    raw: pd.DataFrame,
    rng: np.random.RandomState,
    cfg: AugmentConfig = AugmentConfig(),
) -> pd.DataFrame:
    """One augmented copy of a subject's strides.

    Each stride is time-warped, amplitude-scaled, given Gaussian sensor noise and
    smoothed. The measured foot-off percentage is carried through the same warp,
    so the phase variable recomputed afterwards stays consistent with the warped
    kinematics.
    """
    n_strides = len(raw) // STRIDE_LEN
    channel_std = raw[ANGLE_COLS].to_numpy(float).std(axis=0)
    t = np.linspace(0.0, 1.0, STRIDE_LEN)

    blocks = []
    for s in range(n_strides):
        block = raw.iloc[s * STRIDE_LEN:(s + 1) * STRIDE_LEN].copy().reset_index(drop=True)
        ang = block[ANGLE_COLS].to_numpy(float)

        warp = _random_warp(rng, cfg.warp_strength)
        ang = np.stack([np.interp(warp, t, ang[:, j]) for j in range(ang.shape[1])], axis=1)

        ang *= rng.uniform(*cfg.amplitude_range, size=ang.shape[1])

        sigma = cfg.noise_frac * channel_std
        noise = np.clip(rng.normal(0.0, 1.0, size=ang.shape), -3.0, 3.0) * sigma
        ang = _sg_smooth(ang + noise, cfg.smooth_window, cfg.smooth_polyorder)

        block.loc[:, ANGLE_COLS] = _fix_stride_endpoints(ang)
        # An event at normalized time p now falls at w^-1(p).
        for col in (LFO_COL, RFO_COL):
            fo = float(block[col].iloc[0]) / 100.0
            block.loc[0, col] = float(np.interp(fo, warp, t)) * 100.0
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


# ============================================================
# Splitting
# ============================================================
@dataclass
class Split:
    repeat: int
    train: list[str]
    val: list[str]
    test: list[str]


def make_repeated_splits(
    subject_ids: list[str],
    n_repeats: int = 5,
    n_val: int = 2,
    n_test: int = 2,
    seed: int = 42,
) -> list[Split]:
    """Repeated random subject-level splits, each with its own seed."""
    subject_ids = sorted(subject_ids)
    if n_val + n_test >= len(subject_ids):
        raise ValueError(
            f"{len(subject_ids)} subjects cannot give {n_val} val + {n_test} test "
            "and leave any for training"
        )

    splits = []
    for r in range(n_repeats):
        perm = np.random.RandomState(seed + r).permutation(subject_ids)
        splits.append(
            Split(
                repeat=r,
                test=sorted(perm[:n_test].tolist()),
                val=sorted(perm[n_test:n_test + n_val].tolist()),
                train=sorted(perm[n_test + n_val:].tolist()),
            )
        )
    return splits


# ============================================================
# Windows and scaling
# ============================================================
def make_windows(features: np.ndarray, targets: np.ndarray, window: int = WINDOW,
                 horizon: int = HORIZON):
    """(T,D) -> X (N, window, D) and y (N, horizon, K).

    Window i spans samples [i, i+window-1] and is asked for the `horizon`
    samples that follow it, so y[i, h] is the target at time i+window+h. A
    horizon of one sample is the degenerate case: predicting it is trivially
    solved by repeating the last observed sample, which is why the study uses
    a horizon matched to a real actuation delay instead.
    """
    n = features.shape[0]
    count = n - window - horizon + 1
    if count <= 0:
        return None, None

    idx = np.arange(count)[:, None] + np.arange(window)[None, :]
    tgt = np.arange(count)[:, None] + window + np.arange(horizon)[None, :]
    return features[idx].astype(np.float32), targets[tgt].astype(np.float32)


@dataclass
class FoldData:
    """Everything one repeated split needs, for both feature representations."""
    split: Split
    scaler_ang: StandardScaler
    scaler_pv: StandardScaler
    train_frames: list[pd.DataFrame]
    val_frames: list[pd.DataFrame]
    test_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    test_raw: dict[str, pd.DataFrame] = field(default_factory=dict)

    def features(self, frame: pd.DataFrame, kind: str) -> tuple[np.ndarray, np.ndarray]:
        ang = self.scaler_ang.transform(frame[ANGLE_COLS].to_numpy(float))
        if kind == "pv":
            pv = self.scaler_pv.transform(frame[PV_COLS].to_numpy(float))
            return np.concatenate([pv, ang], axis=1), ang
        return ang, ang

    def xy(self, frames: list[pd.DataFrame], kind: str,
           horizon: int = HORIZON) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for frame in frames:
            feats, targets = self.features(frame, kind)
            x, y = make_windows(feats, targets, horizon=horizon)
            if x is not None:
                xs.append(x)
                ys.append(y)
        if not xs:
            raise RuntimeError("No usable windows -- subjects are shorter than the window")
        return np.concatenate(xs), np.concatenate(ys)

    def train_xy(self, kind: str, horizon: int = HORIZON):
        return self.xy(self.train_frames, kind, horizon)

    def val_xy(self, kind: str, horizon: int = HORIZON):
        return self.xy(self.val_frames, kind, horizon)


def build_fold(
    split: Split,
    raw_frames: dict[str, pd.DataFrame],
    aug_rounds: int = 7,
    seed: int = 0,
    cfg: AugmentConfig = AugmentConfig(),
) -> FoldData:
    """Prepare one split: augment the training subjects, fit the scalers on the
    training data only, and keep the validation and test subjects untouched."""
    train_frames: list[pd.DataFrame] = []
    for i, sid in enumerate(split.train):
        raw = raw_frames[sid]
        train_frames.append(prepare_frame(raw))
        rng = np.random.RandomState(seed * 1000 + split.repeat * 100 + i)
        for _ in range(aug_rounds):
            train_frames.append(prepare_frame(augment_frame(raw, rng, cfg)))

    val_frames = [prepare_frame(raw_frames[sid]) for sid in split.val]
    test_frames = {sid: prepare_frame(raw_frames[sid]) for sid in split.test}

    train_concat = pd.concat(train_frames, ignore_index=True)
    scaler_ang = StandardScaler().fit(train_concat[ANGLE_COLS].to_numpy(float))
    scaler_pv = StandardScaler().fit(train_concat[PV_COLS].to_numpy(float))

    return FoldData(
        split=split,
        scaler_ang=scaler_ang,
        scaler_pv=scaler_pv,
        train_frames=train_frames,
        val_frames=val_frames,
        test_frames=test_frames,
        test_raw={sid: raw_frames[sid] for sid in split.test},
    )
